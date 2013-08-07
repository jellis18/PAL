#!/usr/bin/env python

from __future__ import division
import numpy as np
import PALLikelihoods
import PALutils
import PALpulsarInit
import h5py as h5
from ptsampler import PTSampler
import scipy.linalg as sl
import argparse
import os, time

parser = argparse.ArgumentParser(description = 'Run time domain single source MCMC')

# options
parser.add_argument('--h5File', dest='h5file', action='store', type=str, required=True,
                   help='Full path to hdf5 file containing PTA data')
parser.add_argument('--outDir', dest='outDir', action='store', type=str, default='./',
                   help='Full path to output directory (default = ./)')
parser.add_argument('--best', dest='best', action='store', type=int, default=0,
                   help='Only use best pulsars based on weighted rms (default = 0, use all)')
parser.add_argument('--block', dest='block', action='store', type=int, default=0,
                   help='How many parameters to update at each iteration (default = 0, use all)')
parser.add_argument('--scale', dest='scale', action='store', type=float, default=1,
                   help='Scale factor on jump covariance matrix(default = 1, native scaling)')
parser.add_argument('--ntemps', dest='ntemps', action='store', type=int, default=1,
                   help='Number of parallel temperature chains to run (default = 1)')
parser.add_argument('--nprocs', dest='nprocs', action='store', type=int, default=1,
                   help='Number of processors to use with parallel tempering (default = 1)')




# parse arguments
args = parser.parse_args()

##### Begin Code #####

if not os.path.exists(args.outDir):
    try:
        os.makedirs(args.outDir)
    except OSError:
        pass

print 'Reading in HDF5 file' 

# import hdf5 file
pfile = h5.File(args.h5file, 'r')

# define the pulsargroup
pulsargroup = pfile['Data']['Pulsars']

# fill in pulsar class
global psr
psr = [PALpulsarInit.pulsar(pulsargroup[key],addGmatrix=True, addNoise=True) for key in pulsargroup]

if args.best != 0:
    print 'Using best {0} pulsars'.format(args.best)
    rms = np.array([p.rms() for p in psr])
    ind = np.argsort(rms)

    psr = [psr[ii] for ii in ind[0:args.best]]

    for p in psr:
        print 'Pulsar {0} has {1} ns weighted rms'.format(p.name,p.rms()*1e9)

npsr = len(psr)

pfile.close()

# make sure all pulsar have same reference time
tt=[] 
for p in psr:
    tt.append(np.min(p.toas))

# find reference time
tref = np.min(tt)

# now scale pulsar time
for p in psr:
    p.toas -= tref

# get G matrices
for p in psr:
    p.G = PALutils.createGmatrix(p.dmatrix)

# run Fp statistic to determine starting frequency
print 'Running initial Fpstat search'
fsearch = np.logspace(-9, -7, 200)
fpstat = np.zeros(len(fsearch))
for ii in range(len(fsearch)):
    fpstat[ii] = PALLikelihoods.fpStat(psr, fsearch[ii])

# determine maximum likelihood frequency
fmaxlike = fsearch[np.argmax(fpstat)]
print 'Maximum likelihood from f-stat search = {0}\n'.format(fmaxlike)

# get determinant of covariance matrix for use in likelihood
logdetTerm = []
invCov = []
for ct, p in enumerate(psr):

    efac = p.efac
    equad = p.equad
    Amp = p.Amp
    gam = p.gam
    
    #efac = 1.0
    #equad = 0
    #Amp = 0
    #gam = 4.33


    # get white noise covariance matrix
    white = PALutils.createWhiteNoiseCovarianceMatrix(p.err, efac, equad)

    # get red noise covariance matrix
    tm = PALutils.createTimeLags(p.toas, p.toas)
    red = PALutils.createRedNoiseCovarianceMatrix(tm, Amp, gam)

    C = np.dot(p.G.T, np.dot(red+white, p.G))
    cf = sl.cho_factor(C)
    logdetTerm.append(np.sum(2*np.log(np.diag(cf[0])))) #+ p.G.shape[1]*np.log(2*np.pi))
    invterm = sl.cho_solve(cf, np.eye(cf[0].shape[1]))
    invterm = np.linalg.inv(C)
    invCov.append(np.dot(p.G, np.dot(invterm, p.G.T)))

# get null model log-likelihood
nullLike = 0
for ct, p in enumerate(psr):
    nullLike += -0.5 * logdetTerm[ct]
    nullLike += -0.5 * np.dot(p.res, np.dot(p.invCov, p.res))

print 'Null Like = {0}'.format(nullLike)


# prior ranges
thmin = phasemin = incmin = psimin = 0
thmax = incmax = np.pi
psimax = np.pi/2
phimin = 0
phimax = phasemax = 2*np.pi
ldmin = -4
ldmax = 4
lmmin = 7
lmmax = 9
lfmin = -8.5
lfmax = -7.5
lhmin = -17
lhmax = -12

# set minimum and maximum parameter ranges
pmin = [thmin, phimin, lfmin, lhmin, psimin, incmin, phasemin]
pmax = [thmax, phimax, lfmax, lhmax, psimax, incmax, phasemax]

# add prior ranges for pulsar phase
for ii in range(npsr):
    pmin.append(phasemin)
    pmax.append(phasemax)

# make into array
pmin = np.array(pmin)
pmax = np.array(pmax)

# log prior function
def logprior(x):

    if np.all(x < pmax) and np.all(x > pmin):
        return np.sum(np.log(1/(pmax-pmin)))
    else:
        return -np.inf
    

# define log-likelihood
def loglike(x):

    #tstart = time.time()

    theta = x[0]
    phi = x[1]
    f = 10**x[2]
    h = 10**x[3]
    psi = x[4]
    inc = x[5]
    phase = x[6]

    # get pulsar phase
    pphase = np.zeros(npsr)
    for ii in range(npsr):
        pphase[ii] = x[(ndim-npsr) + ii]

    # pick a distance and mass from the strain. Doesnt matter since non-evolving
    mc = 5e8
    dist = 4 * np.sqrt(2/5) * (mc*4.9e-6)**(5/3) * (np.pi*f)**(2/3) / h
    dist /= 1.0267e14

    loglike = 0
    for ct, p in enumerate(psr):
    
        # solve for the pulsr distance to use in the waveform generator
        fplus, fcross, cosMu = PALutils.createAntennaPatternFuncs(p, theta, phi)
        pdist = pphase[ct]/(2*np.pi*f*(1-cosMu)) / 1.0267e11

        # make waveform with no frequency evolution
        s = PALutils.createResiduals(p, theta, phi, mc, dist, f, phase, psi, inc,\
                                     pdist=pdist, evolve=False)

        diff = p.res - s
        loglike += -0.5 * logdetTerm[ct]
        loglike += -0.5 * np.dot(diff, np.dot(p.invCov, diff))

    #print 'Evaluation time = {0} s'.format(time.time() - tstart)

    if np.isnan(loglike):
        print 'NaN log-likelihood. Not good...'
        return -np.inf
    else:
        return loglike



# function to update covariance matrix for jump proposals
def updateRecursive(chain, M2, mu, iter, mem):

    iter -= mem

    if iter == 0:
        M2 = np.zeros((ndim,ndim))
        mu = np.zeros(ndim)

    for ii in range(mem):
        diff=np.zeros(ndim)
        iter += 1
        for jj in range(ndim):
            
            diff[jj] = chain[ii,jj] - mu[jj]
            mu[jj] += diff[jj]/iter

        M2 += np.outer(diff, (chain[ii,:]-mu))

    c = M2/(iter-1)

    # compute svd
    try:
        u, s, v = np.linalg.svd(c)
    except np.linalg.linalg.LinAlgError:
        print 'Warning: SVD did not converge, not updating covariance matrix'

    return c, M2, mu, u, s

# define jump proposal function for use in MCMC
def jumpProposals(x, iter, beta):

    # how often to update covariance matrix
    mem = 1010

     # get scale
    scale = args.scale

    # get block size
    if args.block:
        block = args.block
    else:
        block = ndim

    # medium size jump every 1000 steps
    if np.random.rand() < 1/1000:
        scale = 5
    
    # large size jump every 10000 steps
    if np.random.rand() < 1/10000:
        scale = 50

    global cov, M2, mu, U, S

    # get parmeters in new diagonalized basis
    y = np.dot(U.T, x)

    # update covarinace matrix
    if (iter-1) % mem == 0 and (iter-1) != 0 and beta == 1:
        cov, M2, mu, U, S = updateRecursive(sampler.chain[0,(iter-mem-1):(iter-1),:], M2, mu, iter-1, mem)

    # make correlated componentwise adaptive jump
    ind = np.unique(np.random.randint(0, ndim, block))
    neff = len(ind)
    cd = 2.4 * np.sqrt(1/beta) / np.sqrt(neff) * scale
    y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(S[ind])
    q = np.dot(U, y)

    return q

# number of dimensions our problem has
ndim = 7 + npsr

# copied from emcee. 25% tswap acceptance rate for gaussian distributions
# if this tstep is used in a geometrically increasing beta distribution
tstep = np.array([25.2741, 7., 4.47502, 3.5236, 3.0232,
                  2.71225, 2.49879, 2.34226, 2.22198, 2.12628,
                  2.04807, 1.98276, 1.92728, 1.87946, 1.83774,
                  1.80096, 1.76826, 1.73895, 1.7125, 1.68849,
                  1.66657, 1.64647, 1.62795, 1.61083, 1.59494,
                  1.58014, 1.56632, 1.55338, 1.54123, 1.5298,
                  1.51901, 1.50881, 1.49916, 1.49, 1.4813,
                  1.47302, 1.46512, 1.45759, 1.45039, 1.4435,
                  1.4369, 1.43056, 1.42448, 1.41864, 1.41302,
                  1.40761, 1.40239, 1.39736, 1.3925, 1.38781,
                  1.38327, 1.37888, 1.37463, 1.37051, 1.36652,
                  1.36265, 1.35889, 1.35524, 1.3517, 1.34825,
                  1.3449, 1.34164, 1.33847, 1.33538, 1.33236,
                  1.32943, 1.32656, 1.32377, 1.32104, 1.31838,
                  1.31578, 1.31325, 1.31076, 1.30834, 1.30596,
                  1.30364, 1.30137, 1.29915, 1.29697, 1.29484,
                  1.29275, 1.29071, 1.2887, 1.28673, 1.2848,
                  1.28291, 1.28106, 1.27923, 1.27745, 1.27569,
                  1.27397, 1.27227, 1.27061, 1.26898, 1.26737,
                  1.26579, 1.26424, 1.26271, 1.26121,
                  1.25973])

tstep = tstep[ndim-1]

# set up temperature ladder
ntemps = args.ntemps
nthreads = args.nprocs

# geometrically spaced betas
betas = np.exp(np.linspace(0, -(ntemps-1)*np.log(tstep), ntemps))

# pick starting values
p0 = np.zeros((ntemps, ndim))
for ii in range(ntemps):
    p0[ii,:] = pmin + np.random.rand(ndim) * (pmax - pmin)

# set initial frequency to be maximum likelihood value
p0[:,2] = np.log10(fmaxlike)

# initialize covariance matrix for jumps
global cov, M2, mu, U, S
M2 = np.zeros((ndim, ndim))
mu = np.zeros(ndim)
cov_diag = np.zeros(ndim)
cov_diag[0] = 0.1
cov_diag[1] = 0.1
cov_diag[2] = 0.005
cov_diag[3] = 0.1
cov_diag[4:] = 0.01
cov = np.diag(cov_diag**2)
U, S, V = np.linalg.svd(cov)

# no cyclic variables
cyclic = np.zeros(ndim)

# add in cyclic values for phase and phi
cyclic[1] = 2*np.pi
cyclic[6] = 2*np.pi
cyclic[7:] = 2*np.pi

# initialize MH sampler
sampler=PTSampler(ntemps, ndim, loglike, logprior, jumpProposals, threads=nthreads, betas=betas, cyclic=cyclic)

# set output file
chainfile = args.outDir + '/chain.npy'
lnlikefile = args.outDir + '/lnlike.npy'
lnprobfile = args.outDir + '/lnprob.npy'
covfile = args.outDir + '/cov.npy'

# run sampler
N = 1000000
isave = 5000   # save every isave iterations
thin = 1
ct = 0
print 'Beginning Sampling in {0} dimensions\n'.format(ndim)
tstart = time.time()
for pos, prob, state in sampler.sample(p0, iterations=N):
    if ct % isave == 0 and ct>0:

        tstep = time.time() - tstart

        np.save(chainfile, sampler.chain[:,0:ct,:])
        np.save(lnlikefile,sampler.lnlikelihood[:,0:ct])
        np.save(lnprobfile,sampler.lnprobability[:,0:ct])
        np.save(covfile, cov)

        print 'Finished {0} of {1} iterations.'.format(ct, N)
        print 'Acceptance fraction = {0}'.format(sampler.acceptance_fraction)
        print 'Time elapsed: {0} s'.format(tstep)
        print 'Approximate time remaining: {0} hr\n'.format(tstep/ct * (N-ct)/3600)

    # update counter
    ct += 1



