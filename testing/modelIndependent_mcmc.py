#!/usr/bin/env python

# Run Lentati style noise estimation on single pulsar

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

parser = argparse.ArgumentParser(description = 'Run Lentati style noise estimation')

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
parser.add_argument('--block', dest='block', action='store', type=int, default=0,
                   help='How many parameters to update at each iteration (default = 0, use all)')
parser.add_argument('--scale', dest='scale', action='store', type=float, default=1,
                   help='Scale factor on jump covariance matrix(default = 1, native scaling)')



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
psr = [PALpulsarInit.pulsar(pulsargroup[key],addGmatrix=True) for key in pulsargroup]

if args.best != 0:
    print 'Using best {0} pulsars'.format(args.best)
    rms = np.array([p.rms() for p in psr])
    ind = np.argsort(rms)

    psr = [psr[ii] for ii in ind[0:args.best]]

    for p in psr:
        print 'Pulsar {0} has {1} ns weighted rms'.format(p.name,p.rms()*1e9)

npsr = len(psr)

pfile.close()

# get Tmax
Tmax = np.max([p.toas.max() - p.toas.min() for p in psr])

# initialize fourier design matrix
F = [PALutils.createfourierdesignmatrix(p.toas, args.nmodes, Tspan=Tmax) for p in psr]

# get frequency
tmp, f = PALutils.createfourierdesignmatrix(p.toas, args.nmodes, Tspan=Tmax, freq=True)

# get G matrices
for p in psr:
    p.G = PALutils.createGmatrix(p.dmatrix)

# pre compute diagonalized efac + equad white noise model
Diag = []
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

    return c, M2, mu

# define jump proposal function for use in MCMC
def jumpProposals(x, iter, beta):

    # how often to update covariance matrix
    mem = 1001

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

    # update covarinace matrix
    if (iter-1) % mem == 0 and (iter-1) != 0 and beta == 1:
        cov, M2, mu = updateRecursive(sampler.chain[0,(iter-mem-1):(iter-1),:], M2, mu, iter-1, mem)
        
        # compute svd
        try:
            U, S, v = np.linalg.svd(cov)
        except np.linalg.linalg.linalgerror:
            print 'warning: svd did not converge, not updating covariance matrix'

    # get parmeters in new diagonalized basis
    y = np.dot(U.T, x)

    # make correlated componentwise adaptive jump
    ind = np.unique(np.random.randint(0, ndim, block))
    neff = len(ind)
    cd = 2.4 * np.sqrt(1/beta) / np.sqrt(neff) * scale
    y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(S[ind])
    q = np.dot(U, y)

    return q

# number of dimensions our problem has
ndim = 2 + 4*npsr

# set up temperature ladder
ntemps = 1
nthreads = 1
tstep = 1.4
Tmin = 1

# exponentially spaced betas
#T = np.logspace(np.log(Tmin), np.log(Tmax), ntemps, base=np.exp(1))
#betas = 1/T

# geometrically spaced betas
betas = np.exp(np.linspace(0, -(ntemps-1)*np.log(tstep), ntemps))

# parameterize GWB as powerlaw
if args.powerlaw:
    print 'Parameterizing GWB as power law'

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

    #TODO: add in constant prior

    def logprior(x):


        efac = x[2:(2+npsr)]
        lequad = x[(2+npsr):(2+2*npsr)]
        lAred = x[(2+2*npsr):(2+3*npsr)]
        gred = x[(2+3*npsr):(2+4*npsr)]

        if x[0] > Amin and x[1] < gamMax and \
           x[1] > gamMin and np.all(efac < emax) and np.all(efac > emin) \
           and np.all(lequad < qmax) and np.all(lequad > qmin) \
           and np.all(lAred > lAmin) and \
           np.all(gred < gamMax) and np.all(gred > gamMin):

            return 0
        else:
            return -np.inf

    # define log-likelihood
    def loglike(x):


        A = x[0]*1e-14
        gam = x[1]
        efac = x[2:(2+npsr)]
        equad = 10**x[(2+npsr):(2+2*npsr)]
        Ared = 10**x[(2+2*npsr):(2+3*npsr)]
        gred = x[(2+3*npsr):(2+4*npsr)]

        
        loglike = PALLikelihoods.modelIndependentFullPTAPL(psr, F, Diag, f, A, gam, \
                                                           Ared, gred, efac, equad, ORF)

        return loglike

    # number of dimensions our problem has
    ndim = 2 + 4*npsr

    # pick starting values
    p0 = np.zeros((ntemps,ndim))
    p0[:,0] = np.random.uniform(1, 10, ntemps)
    p0[:,1] = np.random.uniform(4.33, 4.334, ntemps)
    for ii in range(npsr):
        p0[:,ii+2] = np.random.uniform(0.95, 1.05, ntemps)
        p0[:,ii+2+npsr] = np.random.uniform(-8, -6, ntemps)
        p0[:,ii+2+2*npsr] = np.random.uniform(-15, -12, ntemps)
        p0[:,ii+2+3*npsr] = np.random.uniform(0, 3, ntemps)


    # initialize covariance matrix for jumps
    global cov, U, S, M2, mu
    M2 = np.zeros((ndim, ndim))
    mu = np.zeros(ndim)

    cov_diag = np.zeros(ndim)
    cov_diag[0] = 0.1 # GW amplitude initial jump size
    cov_diag[1] = 0.1   # GW spectral index initial jump size
    for ii in range(npsr):
        cov_diag[2+ii] = 0.1        # EFAC initial jump size
        cov_diag[2+npsr+ii] = 0.1   # log EQUAD initial jump size
        cov_diag[2+2*npsr+ii] = 0.1 # log Ared initial jump size
        cov_diag[2+3*npsr+ii] = 0.1 # log gred initial jump size

    cov = np.diag(cov_diag**2)
    U, S, V = np.linalg.svd(cov)

    # no cyclic variables
    cyclic = np.zeros(ndim)

    # initialize MH sampler
    sampler = PTSampler(ntemps, ndim, loglike, logprior, jumpProposals, \
                        threads=nthreads, betas=betas, cyclic=cyclic)

# parameterize GWB as with independent coefficients
else:
    print 'Parameterizing GWB with {0} spectral coefficients'.format(args.nmodes)

    # prior ranges
    emin = 0.1
    emax = 5
    qmin = -8
    qmax = -5
    lAmin = -17
    lAmax = -11
    rhomin = -20
    rhomax = -10
    gamMin = 0
    gamMax = 7

    #TODO: add in constant prior

    def logprior(x):

        rho = x[0:args.nmodes]
        efac = x[args.nmodes:(args.nmodes+npsr)]
        lequad = x[(args.nmodes+npsr):(args.nmodes+2*npsr)]
        lAred = x[(args.nmodes+2*npsr):(args.nmodes+3*npsr)]
        gred = x[(args.nmodes+3*npsr):(args.nmodes+4*npsr)]

        if np.all(rho < rhomax) and np.all(rho > rhomin) \
           and np.all(efac < emax) and np.all(efac > emin) \
           and np.all(lequad < qmax) and np.all(lequad > qmin) \
           and np.all(lAred > lAmin) and \
           np.all(gred < gamMax) and np.all(gred > gamMin):

            return 0
        else:
            return -np.inf

    # define log-likelihood
    def loglike(x):

        rho = x[0:args.nmodes]
        efac = x[args.nmodes:(args.nmodes+npsr)]
        equad = 10**x[(args.nmodes+npsr):(args.nmodes+2*npsr)]
        Ared = 10**x[(args.nmodes+2*npsr):(args.nmodes+3*npsr)]
        gred = x[(args.nmodes+3*npsr):(args.nmodes+4*npsr)]
        
        loglike = PALLikelihoods.modelIndependentFullPTANoisePL(psr, F, s, f, rho, \
                                                    Ared, gred, efac, equad, ORF)

        return loglike

    # number of dimensions our problem has
    ndim = args.nmodes + 4*npsr

    # pick starting values
    p0 = np.zeros((ntemps,ndim))

    # pick starting values for rho based on current upper limit
    Aup = 7e-15
    gup = 13/3
    f1yr = 1/3.16e7
    p0[:,0:args.nmodes] = np.log10(Aup**2/12/np.pi**2 * f1yr**(gup-3) * f**(-gup)/Tmax)
    for ii in range(npsr):
        p0[:,ii+args.nmodes] = np.random.uniform(0.95, 1.05, ntemps)
        p0[:,ii+args.nmodes+npsr] = np.random.uniform(-8, -6, ntemps)
        p0[:,ii+args.nmodes+2*npsr] = np.random.uniform(-15, -12, ntemps)
        p0[:,ii+args.nmodes+3*npsr] = np.random.uniform(0, 3, ntemps)


    # initialize covariance matrix for jumps
    global cov, U, S, M2, mu
    M2 = np.zeros((ndim, ndim))
    mu = np.zeros(ndim)

    cov_diag = np.zeros(ndim)
    cov_diag[0:args.nmodes] = 0.1
    for ii in range(npsr):
        cov_diag[args.nmodes+ii] = 0.01        # EFAC initial jump size
        cov_diag[args.nmodes+npsr+ii] = 0.05   # log EQUAD initial jump size
        cov_diag[args.nmodes+2*npsr+ii] = 0.1 # log Ared initial jump size
        cov_diag[args.nmodes+3*npsr+ii] = 0.1 # log gred initial jump size

    cov = np.diag(cov_diag**2)
    U, S, V = np.linalg.svd(cov)

    # no cyclic variables
    cyclic = np.zeros(ndim)

    # initialize MH sampler
    sampler = PTSampler(ntemps, ndim, loglike, logprior, jumpProposals, \
                        threads=nthreads, betas=betas, cyclic=cyclic)

# set output file
chainfile = args.outDir + '/chain.npy'
lnlikefile = args.outDir + '/lnlike.npy'
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
        np.save(covfile, cov)

        print 'Finished {0} of {1} iterations.'.format(ct, N)
        print 'Acceptance fraction = {0}'.format(sampler.acceptance_fraction)
        print 'Time elapsed: {0} s'.format(tstep)
        print 'Approximate time remaining: {0} hr\n'.format(tstep/ct * (N-ct)/3600)

    # update counter
    ct += 1



