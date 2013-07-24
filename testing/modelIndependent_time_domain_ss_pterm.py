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
pmin = np.array([thmin, phimin, lfmin, lmmin, ldmin, psimin, incmin, phasemin])
pmax = np.array([thmax, phimax, lfmax, lmmax, ldmax, psimax, incmax, phasemax])

# log prior function
def logprior(x):

    logp = 0

    if np.all(x[0:8] <= pmax) and np.all(x[0:8] >= pmin):
        logp += np.sum(np.log(1/(pmax-pmin)))
    else:
        logp += -np.inf

    # pulsar distance prior
    for ct, p in enumerate(psr):
        pdist = x[8+ct]
        m = p.dist
        sig = p.distErr

        if pdist < 0:
            logp += -np.inf
        else:
            logp += -0.5 * (np.log(2*np.pi*sig**2) + (m-pdist)**2/2/sig**2)
    
    return logp

# define log-likelihood
def loglike(x):

    efac = np.ones(npsr)
    equad = np.zeros(npsr)
    theta = x[0]
    phi = x[1]
    f = 10**x[2]
    mc = 10**x[3]
    dist = 10**x[4]
    psi = x[5]
    inc = x[6]
    phase = x[7]
    pdist = x[8:(8+npsr)]
        
    loglike = 0
    for ct, p in enumerate(psr):

        # make waveform with pulsar term
        s = PALutils.createResiduals(p, theta, phi, mc, dist, f, phase, psi, inc, pdist=pdist[ct], \
                 psrTerm=True)

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

    return c, M2, mu

# define jump proposal function for use in MCMC
def jumpProposals(x, iter, beta):

    # how often to update covariance matrix
    mem = 1010

    global cov, M2, mu, U, S

    # update covarinace matrix
    if (iter-1) % mem == 0 and (iter-1) != 0 and beta == 1:
        cov, M2, mu = updateRecursive(sampler.chain[0,(iter-mem-1):(iter-1),:], M2, mu, iter-1, mem)
        
        # compute svd
        c = cov[0:8, 0:8]
        try:
            U, S, v = np.linalg.svd(c)
        except np.linalg.linalg.linalgerror:
            print 'warning: svd did not converge, not updating covariance matrix'


    # call jump proposal
    q = jumps(x, iter, beta)

    return q

def covarianceJumpProposal(x, iter, beta):

    # get scale
    scale = 1/5

    # medium size jump every 1000 steps
    if np.random.rand() < 1/1000:
        scale = 5
    
    # large size jump every 10000 steps
    if np.random.rand() < 1/10000:
        scale = 50

    # get parmeters in new diagonalized basis
    subx = x[0:8]
    y = np.dot(U.T, subx)

    # make correlated componentwise adaptive jump
    ind = np.unique(np.random.randint(0, 8, 8))
    neff = len(ind)
    cd = 2.4 * np.sqrt(1/beta) / np.sqrt(neff) * scale
    y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(S[ind])
    q = x.copy()
    q[0:8] = np.dot(U, y)

    # need to make sure that we keep the pulsar phase constant, plus small offset

    # get parameters before jump
    freq0 = 10**x[2]
    pdist = q[8:]
    phi0 = x[1]
    theta0 = x[0]
    
    freq1 = 10**q[2]
    phi1 = q[1]
    theta1 = q[0]

    # put pulsar distance in correct units
    pdist *= 1.0267e11  

    # get cosMu
    cosMu0 = np.zeros(npsr)
    cosMu1 = np.zeros(npsr)
    for ii in range(npsr):
        tmp1, temp2, cosMu0[ii] = PALutils.createAntennaPatternFuncs(psr[ii], theta0, phi0)
        tmp1, temp2, cosMu1[ii] = PALutils.createAntennaPatternFuncs(psr[ii], theta1, phi1)
    
    # construct new pulsar distances to keep the pulsar phases constant
    sigma = np.sqrt(1/beta) * 0.0
    L_new = (freq0*pdist*(1-cosMu0) + np.random.randn(npsr)*sigma)/(freq1*(1-cosMu1))

    # convert back to Kpc
    L_new /= 1.0267e11 

    q[8:] = L_new

    return q

def smallPulsarPhaseJump(x, iter, beta):

    # get old parameters
    q = x.copy()

    # jump size
    sigma = np.sqrt(beta) * 0.1

    # pick pulsar index at random
    ind = np.random.randint(0, npsr, npsr)
    ind = np.unique(ind)
    
    # get relevant parameters
    freq = 10**x[2]
    pdist = x[8+ind]
    phi = x[1]
    theta = x[0]

    # put pulsar distance in correct units
    pdist *= 1.0267e11  

    # get cosMu
    cosMu = np.zeros(len(ind))
    for ii in range(len(ind)):
        tmp1, temp2, cosMu[ii] = PALutils.createAntennaPatternFuncs(psr[ind[ii]], theta, phi)


    # construct pulsar phase
    phase_old = 2*np.pi*freq*pdist*(1-cosMu)

    # gaussian jump 
    phase_new = phase_old + np.random.randn(np.size(pdist))*sigma

    # solve for new pulsar distances from phase_new
    L_new = phase_new/(2*np.pi*freq*(1-cosMu))

    # convert back to Kpc
    L_new /= 1.0267e11 

    q[8+ind] = L_new

    return q

def bigPulsarPhaseJump(x, iter, beta):

    # get old parameters
    q = x.copy()

    # pick pulsar index at random
    ind = np.random.randint(0, npsr, npsr)
    ind = np.unique(ind)
    
    # get relevant parameters
    freq = 10**x[2]
    pdist = x[8+ind]
    pdistErr = np.array([psr[ii].distErr for ii in list(ind)])
    phi = x[1]
    theta = x[0]

    # put pulsar distance in correct units
    pdist *= 1.0267e11  
    pdistErr *= 1.0267e11  

    # get cosMu
    cosMu = np.zeros(len(ind))
    for ii in range(len(ind)):
        tmp1, temp2, cosMu[ii] = PALutils.createAntennaPatternFuncs(psr[ind[ii]], theta, phi)
    
    # construct pulsar phase
    phase_old = 2*np.pi*freq*pdist*(1-cosMu)

    # gaussian jump 
    phase_jump = np.random.randn(np.size(pdist))*pdistErr*freq*(1-cosMu)

    # make jump multiple of 2 pi
    phase_jump = np.array([int(phase_jump[ii]) \
                    for ii in range(np.size(pdist))])

    # new phase
    phase_new = phase_old + 2*np.pi*phase_jump

    # solve for new pulsar distances from phase_new
    L_new = phase_new/(2*np.pi*freq*(1-cosMu))

    # convert back to Kpc
    L_new /= 1.0267e11  

    q[8+ind] = L_new

    return q


# class that manages jump proposals
class JumpProposals(object):
    """
    Class that manages jump proposal distributions for use in MCMC or Nested Sampling.

    """

    def __init__(self):

        self.propCycle = []

    # add jump proposal distribution functions
    def addProposalToCycle(self, func, weight):
        """
        Add jump proposal distributions to cycle with a given weight.

        @param func: jump proposal function
        @param weight: jump proposal function weight in cycle

        """

        # get length of cycle so far
        length = len(self.propCycle)

        # check for 0 weight
        if weight == 0:
            print 'ERROR: Can not have 0 weight in proposal cycle!'
            sys.exit()

        # add proposal to cycle
        for ii in range(length, length + weight):
            self.propCycle.append(func)


    # randomized proposal cycle
    def randomizeProposalCycle(self):
        """
        Randomize proposal cycle that has already been filled

        """

        # get length of full cycle
        length = len(self.propCycle)

        # get random integers
        index = np.random.randint(low=0, high=(length-1), size=length)

        # randomize proposal cycle
        self.randomizedPropCycle = [self.propCycle[index[ii]] for ii in range(len(index))]


    # call proposal functions from cycle
    def __call__(self, x, iter, beta):
        """
        Call Jump proposals

        """

        # get length of cycle
        length = len(self.propCycle)

        # call function
        q = self.randomizedPropCycle[np.mod(iter,length)](x, iter, beta)
        #print self.randomizedPropCycle[np.mod(iter,length)]

        # increment proposal cycle counter and re-randomize if at end of cycle
        if iter % length == 0: self.randomizeProposalCycle()

        return q


###### MAKE JUMP PROPOSAL CYCLE #######

# define weights
BIG = 20
SMALL = 5
TINY = 1

# initialize jumps
jumps = JumpProposals()

# add covariance jump proposals
jumps.addProposalToCycle(covarianceJumpProposal, BIG)

# add small jump in pulsar phase
jumps.addProposalToCycle(smallPulsarPhaseJump, BIG)

# add large jump in pulsar phase
jumps.addProposalToCycle(bigPulsarPhaseJump, BIG)

# randomize cycle
jumps.randomizeProposalCycle()

########################################

    

# number of dimensions our problem has
ndim = 8 + npsr

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
    p0[ii,0:8] = pmin + np.random.rand(ndim-npsr) * (pmax - pmin)

# start at measured pulsar distance
for ct, p in enumerate(psr):
    p0[:,8+ct] = p.dist

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
cov_diag[4] = 0.05
cov_diag[5:8] = 0.01
for ct, p in enumerate(psr):
    cov_diag[8+ct] = p.distErr/5
cov = np.diag(cov_diag**2)
U, S, V = np.linalg.svd(cov[0:8,0:8])

# no cyclic variables
cyclic = np.zeros(ndim)

# add in cyclic values for phase and phi
#cyclic[1] = 2*np.pi
#cyclic[6] = 2*np.pi

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



