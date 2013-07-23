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
parser.add_argument('--nmodes', dest='nmodes', action='store', type=int, default=10,
                   help='number of fourier modes to use (default=10)')
parser.add_argument('--powerlaw', dest='powerlaw', action='store_true', default=False,
                   help='Use power law model (default = False)')
parser.add_argument('--nored', dest='nored', action='store_true', default=False,
                   help='Only use GWB, no additional red noise (default = False)')
parser.add_argument('--efac', dest='efac', action='store_true', default=False,
                   help='Include EFAC as search parameter (default = False, ie. EFAC = 1)')
parser.add_argument('--equad', dest='equad', action='store_true', default=False,
                   help='Include EQUAD as search parameter (default = False, ie. EQUAD = 0)')
parser.add_argument('--best', dest='best', action='store', type=int, default=0,
                   help='Only use best pulsars based on weighted rms (default = 0, use all)')



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

# get Tmax
Tmax = np.max([p.toas.max() - p.toas.min() for p in psr])

# initialize fourier design matrix
F = [PALutils.createfourierdesignmatrix(p.toas, args.nmodes, Tspan=Tmax) for p in psr]

if args.powerlaw:
    tmp, f = PALutils.createfourierdesignmatrix(p.toas, args.nmodes, Tspan=Tmax, freq=True)

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

# now we don't need lots of inverse covariance matrices floating around
for p in psr:
    p.invCov = None 


# pre compute diagonalized efac + equad white noise model
Diag = []
projList = []
print 'Pre-Computing white noise covariances'
for ct, p in enumerate(psr):
    efac = np.dot(p.G.T, np.dot(np.diag(p.err**2), p.G))
    equad = np.dot(p.G.T, p.G)
    L = np.linalg.cholesky(equad)
    Linv = np.linalg.inv(L)
    sand = np.dot(Linv, np.dot(efac, Linv.T))
    u,s,v = np.linalg.svd(sand)
    Diag.append(s)
    proj = np.dot(u.T, np.dot(Linv, p.G.T))
    projList.append(proj)

    # project residuals onto new basis
    p.res = np.dot(proj, p.res)
    F[ct] = np.dot(proj, F[ct])


# get ORF matrix
ORF = PALutils.computeORFMatrix(psr)/2

# fill in kappa with [] TODO: this is a hack for now to not include additional red noise
if args.nored:
    kappa = [ [] for jj in range(npsr)]
    Ared = np.zeros(npsr)
    gred = np.zeros(npsr)
    
# prior ranges
emin = 0.1
emax = 5
qmin = -8
qmax = -5
lAmin = -17
lAmax = -11
Amin = 0
Amax = 5
gamMin = 0
gamMax = 7
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

# set minimum and maximum parameter ranges
pmin = np.array([thmin, phimin, lfmin, lmmin, psimin, incmin, phasemin])
pmax = np.array([thmax, phimax, lfmax, lmmax, psimax, incmax, phasemax])

# log prior function
def logprior(x):

    theta = x[0]
    phi = x[1]
    lf = x[2]
    lmc = x[3]
    psi = x[4]
    inc = x[5]
    phase = x[6]

    if np.all(x < pmax) and np.all(x > pmin):
        return np.sum(np.log(1/(pmax-pmin)))
    else:
        return -np.inf

# define log-likelihood
def loglike(x):


    efac = np.ones(npsr)
    equad = np.zeros(npsr)
    theta = x[0]
    phi = x[1]
    f = 10**x[2]
    mc = 10**x[3]
    psi = x[4]
    inc = x[5]
    phase = x[6]
    dist = 1
        
    loglike = 0
    for ct, p in enumerate(psr):

        # make waveform with no pulsar term
        s = PALutils.createResiduals(p, theta, phi, mc, dist, f, phase, psi, inc, \
                 psrTerm=False)

        # project onto white noise basis
        s = np.dot(projList[ct], s)

        loglike += np.sum(p.res*s/(efac[ct]*Diag[ct] + equad[ct]**2))
        loglike -= 0.5 * np.sum(s**2/(efac[ct]*Diag[ct] + equad[ct]**2))
   
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
            if cyclic[jj] != 0:
                dd = chain[ii,jj] - mu[jj]
                if dd < 0:
                    dd *= -1
                
                if dd > np.pi:
                    diff[jj] = 2*np.pi - dd
                else:
                    diff[jj] = dd
            else:
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
    scale = 1/5

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
    ind = np.unique(np.random.randint(0, ndim, ndim))
    neff = len(ind)
    cd = 2.4 * np.sqrt(1/beta) / np.sqrt(neff) * scale
    y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(S[ind])
    q = np.dot(U, y)

    return q

# number of dimensions our problem has
ndim = 7

# set up temperature ladder
ntemps = 4
nthreads = 4
Tmax = 10
Tmin = 1

# exponentially spaced betas
T = np.logspace(np.log(Tmin), np.log(Tmax), ntemps, base=np.exp(1))
betas = 1/T

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
cov_diag[2] = 0.01
cov_diag[3] = 0.05
cov_diag[4:] = 0.01
cov = np.diag(cov_diag**2)
U, S, V = np.linalg.svd(cov)

cov = np.diag(cov_diag**2)
U, S, V = np.linalg.svd(cov)

# no cyclic variables
cyclic = np.zeros(ndim)

# add in cyclic values for phase and phi
cyclic[1] = 2*np.pi
cyclic[6] = 2*np.pi

# initialize MH sampler
sampler=PTSampler(ntemps, ndim, loglike, logprior, jumpProposals, threads=nthreads, betas=betas, cyclic=cyclic)

# set output file
chainfile = args.outDir + '/chain.npy'
lnlikefile = args.outDir + '/lnlike.npy'
covfile = args.outDir + '/cov.npy'

# run sampler
N = 1000000
isave = 10000   # save every isave iterations
thin = 1
ct = 0
print 'Beginning Sampling in {0} dimensions\n'.format(ndim)
tstart = time.time()
for pos, prob, state in sampler.sample(p0, iterations=N):
    if ct % isave == 0 and ct>0:

        tstep = time.time() - tstart

        np.save(chainfile, sampler.chain[:,0:ct,:])
        np.save(lnlikefile,sampler.lnlikelihood[:,0:ct])
        np.save(covfile, cov)

        print 'Finished {0} of {1} iterations.'.format(ct, N)
        print 'Acceptance fraction = {0}'.format(sampler.acceptance_fraction)
        print 'Time elapsed: {0} s'.format(tstep)
        print 'Approximate time remaining: {0} hr\n'.format(tstep/ct * (N-ct)/3600)

    # update counter
    ct += 1



