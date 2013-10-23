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
import scipy.stats as ss
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
parser.add_argument('--scale', dest='scale', action='store', type=float, default=0,
                   help='Scale factor on jump covariance matrix(default = 0, native scaling)')
parser.add_argument('--ntemps', dest='ntemps', action='store', type=int, default=1,
                   help='Number of parallel temperature chains to run (default = 1)')
parser.add_argument('--nprocs', dest='nprocs', action='store', type=int, default=1,
                   help='Number of processors to use with parallel tempering (default = 1)')
parser.add_argument('--freq', dest='freq', action='store', type=float, default=None,
                   help='Frequency at which to compute upper limit (default = None)')
parser.add_argument('--inj', dest='inj', action='store_true', default=False,
                   help='Start chain at injected parameters (default = False)')
parser.add_argument('--ti', dest='ti', action='store', type=int, default=None,
                   help='Thermodynamic Integration option (0 = cold chains, 1 = hot chains) (default = None)')




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

# get injection parameters
if args.inj:

    # data group in hdf5 file
    injgroup = pfile['Data']['Injection']

    # get injected parameters
    theta = np.pi/2 - injgroup['GWDEC'].value
    phi = injgroup['GWRA'].value
    freq = injgroup['GWFREQUENCY'].value
    mc = injgroup['GWCHIRPMASS'].value
    dist = injgroup['GWDISTANCE'].value
    psi = injgroup['GWPOLARIZATION'].value
    inc = injgroup['GWINC'].value
    phase = injgroup['GWPHASE'].value

    true = [theta, phi, np.log10(freq), np.log10(mc), np.log10(dist), psi, inc, phase]


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
if args.freq is None:
    print 'Running initial Fpstat search'
    fsearch = np.logspace(-9, -7, 1000)
    fpstat = np.zeros(len(fsearch))
    for ii in range(len(fsearch)):
        fpstat[ii] = PALLikelihoods.fpStat(psr, fsearch[ii])

    # determine maximum likelihood frequency
    fmaxlike = fsearch[np.argmax(fpstat)]
    print 'Maximum likelihood from f-stat search = {0}\n'.format(fmaxlike)

# get determinant of covariance matrix for use in likelihood
logdetTerm = []
for ct, p in enumerate(psr):

    efac = p.efac
    equad = p.equad
    Amp = p.Amp
    gam = p.gam
    fH = p.fH

    print p.name, efac, equad, Amp, gam
    
    # get white noise covariance matrix
    white = PALutils.createWhiteNoiseCovarianceMatrix(p.err, efac, equad)

    # get red noise covariance matrix
    tm = PALutils.createTimeLags(p.toas, p.toas, round=True)
    red = PALutils.createRedNoiseCovarianceMatrix(tm, Amp, gam, fH=fH)

    C = np.dot(p.G.T, np.dot(red+white, p.G))
    cf = sl.cho_factor(C)
    logdetTerm.append(np.sum(2*np.log(np.diag(cf[0])))) #+ p.G.shape[1]*np.log(2*np.pi))

# get null model log-likelihood
nullLike = 0
for ct, p in enumerate(psr):
    nullLike += -0.5 * logdetTerm[ct]
    nullLike += -0.5 * np.dot(p.res, np.dot(p.invCov, p.res))

print 'Null Like = {0}'.format(nullLike)


# prior ranges
thmin = phasemin = incmin = psimin = 0
thmax = np.pi
psimax = np.pi
incmax = np.pi
phimin = 0
phimax = phasemax = 2*np.pi
ldmin = -3
ldmax = 4
lmmin = 6
lmmax = 10
if args.freq is None:
    lfmin = np.log10(6.329113924050633e-09)
    lfmax = np.log10(4.113924050632911e-07)
elif args.freq is not None:
    lfmin = np.log10(args.freq) - 0.01
    lfmax = np.log10(args.freq) + 0.01
    print 'Using reduced frequency prior range {0} - {1}'.format(lfmin, lfmax)
lhmin = -16
lhmax = -11
hmin = 0
hmax = 1000

# set minimum and maximum parameter ranges
pmin = [thmin, phimin, lfmin, lmmin, ldmin, psimin, incmin, phasemin]
pmax = [thmax, phimax, lfmax, lmmax, ldmax, psimax, incmax, phasemax]


# make into array
pmin = np.array(pmin)
pmax = np.array(pmax)

# log prior function
def logprior(x):

    logp = 0
    if np.all(x[:8] < pmax) and np.all(x[:8] > pmin):

        logp += -np.sum(np.log(pmax-pmin))

        # include flat prior in \cos\theta and \cos\iota
        logp += np.log(pmax[0] - pmin[0]) + np.log(pmax[6] - pmin[6]) + np.log(np.sin(x[0])) + np.log(np.sin(x[6]))

    else:
        logp += -np.inf

    # add prior for pulsar distance
    for ct, p in enumerate(psr):
        m = p.dist
        sig = p.distErr
        pdist = x[8+ct]

        if pdist > 0:

            # guassian prior
            logp += -0.5 * (np.log(2*np.pi*sig**2) + (pdist-m)**2/sig**2)

        else:

            logp += -np.inf

    return logp
    

# define log-likelihood
def loglike(x):

    start = time.time()

    # parameter values
    theta = x[0]
    phi = x[1]
    freq = 10**x[2]
    mc = 10**x[3]
    dist = 10**x[4]
    psi = x[5]
    inc = x[6]
    phase = x[7]

    pdist = x[8:]

    # generate list of waveforms for all puslars
    s = PALutils.createResidualsFast(psr, theta, phi, mc, dist, freq, phase, psi, inc,\
                                     pdist=pdist, evolve=False, phase_approx=True)
    
    loglike = 0
    for ct, p in enumerate(psr):

        diff = p.res - s[ct]
        loglike += -0.5 * (logdetTerm[ct] + np.dot(diff, np.dot(p.invCov, diff)))


    if np.isnan(loglike):
        print 'NaN log-likelihood. Not good...'
        return -np.inf
    else:
        return loglike
    #return 0

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
    
    return c, M2, mu

# define jump proposal function for use in MCMC
def jumpProposals(x, iter, beta):

    # how often to update covariance matrix
    mem = 1000

    global cov, M2, mu, U, S

    # update covarinace matrix
    if (iter-1) % mem == 0 and (iter-1) != 0 and beta == 1:
        cov, M2, mu = updateRecursive(sampler.chain[0,(iter-mem-1):(iter-1),:], \
                                      M2, mu, iter-1, mem)
        
        # compute svd
        try:
            U, S, v = np.linalg.svd(cov)
        except np.linalg.linalg.linalgerror:
            print 'warning: svd did not converge, not updating covariance matrix'


    # call burn-in jump proposal
    if iter < 100000: q = jumps(x, iter, beta)

    # post burn in jump proposal
    else: q = jumps2(x, iter, beta)



    return q

def covarianceJumpProposal(x, iter, beta):

    # number of parameters to update at once 
    prob = np.random.rand()
    if prob > (1 - 1/ndim):
        block = ndim

    elif prob > (1 - 2/ndim):
        block = np.ceil(ndim/2)

    elif prob > 0.8:
        block = 5

    else:
        block = 1

    # Using constant block size from command line
    if args.block:
        block = args.block

    # adjust step size
    prob = np.random.rand()

    # very small jump
    if prob > 0.9:
        scale = 0.01
        
    # small jump
    elif prob > 0.7:
        scale = 0.2

    # large jump
    elif prob > 0.97:
        scale = 10
    
    # small-medium jump
    elif prob > 0.6:
        scale = 0.5

    # standard medium jump
    else:
        scale = 1.0

    # get scale from command line
    if args.scale:
        scale = args.scale

    # get parmeters in new diagonalized basis
    y = np.dot(U.T, x)

    # make correlated componentwise adaptive jump
    ind = np.unique(np.random.randint(0, ndim, block))
    neff = len(ind)
    if beta > 1/100:
        cd = 2.4 * np.sqrt(1/beta) / np.sqrt(2*neff) * scale
    else:
        cd = 2.4 * 10 / np.sqrt(2*neff) * scale

    #y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(S[ind])
    #q = np.dot(U, y)
    cd = 2.4/np.sqrt(2*ndim) * scale * np.sqrt(1/beta)
    q = np.random.multivariate_normal(x, cd**2*cov)

    # need to make sure that we keep the pulsar phase constant, plus small offset
    for ct, p in enumerate(psr):
    
        L_new = pulsarPhaseFix(x[8+ct], p.cosMu(x[0], x[1]), x[7], 10**x[3], \
                        np.pi*10**x[2], q[8+ct], p.cosMu(q[0], q[1]), \
                        q[7], 10**q[3], np.pi*10**q[2], beta, \
                        phaseJump=True)

        q[8+ct] = L_new  
    
    return q

#mass-distance correlated jump
def massDistanceJump(x, iter, beta):

    # get old parameters
    q = x.copy()

    # initial values
    mc0 = 10**x[3] * 4.9e-6
    dist0 = 10**x[4] * 1.0267e14 

    # draw distance uniformly from prior
    dist1 = 10**np.random.uniform(-3, 4) * 1.0267e14 

    # find chirp mass value that keeps M^5/3/D constant
    mc1 = (dist1 * mc0**(5/3)/dist0)**(3/5)

    q[4] = np.log10(dist1/1.0267e14)
    q[3] = np.log10(mc1/4.9e-6)

    return q



# Differential evolution jump
def DEJump(x, iter, beta):

    # get old parameters
    q = x.copy()

    # draw a random integer from 0 - iter
    mm = np.random.randint(0, iter)
    nn = np.random.randint(0, iter)

    # make sure mm and nn are not the same iteration
    while mm == nn: nn = np.random.randint(0, iter)

    # only jump in subset of parameters
    prob = np.random.rand()

    # jump in all parameters
    if prob > 0.9:
        ind = [ii for ii in range(ndim)]
        neff = len(ind)

    # sky location jump
    elif prob < 0.3:
        ind = [0, 1]
        neff = len(ind)
    
    # mass-distance jump
    elif prob > 0.7:
        ind = [3, 4]
        neff = len(ind)
        
    else:
        ind = np.unique(np.random.randint(0, ndim, 1))
        while ind == 0 or ind == 1 or ind == 3 or ind == 4:
            ind = np.unique(np.random.randint(0, ndim, 1))
        neff = len(ind)

    # get jump scale size
        prob = np.random.rand()

    # mode jump
    if prob > 0.5:
        scale = 1.0

    else:
        scale = np.random.rand()


    for ii in ind:
        
        # jump size
        sigma = sampler.chain[0, mm, ii] - sampler.chain[0, nn, ii]

        # jump
        q[ii] += scale * sigma

    # fix pulsar phase
    for ct, p in enumerate(psr):
        
        L_new = pulsarPhaseFix(x[8+ct], p.cosMu(x[0], x[1]), x[7], 10**x[3], \
                        np.pi*10**x[2], q[8+ct], p.cosMu(q[0], q[1]), \
                        q[7], 10**q[3], np.pi*10**q[2], beta, \
                        phaseJump=True)

        q[8+ct] = L_new  


    return q



def pulsarPhaseJump(x, iter, beta):

    # get old parameters
    q = x.copy()

    for ct, p in enumerate(psr):
        
        L_new = pulsarPhaseFix(x[8+ct], p.cosMu(x[0], x[1]), x[7], 10**x[3], \
                        np.pi*10**x[2], q[8+ct], p.cosMu(q[0], q[1]), \
                        q[7], 10**q[3], np.pi*10**q[2], beta, \
                        phaseJump=True)

        q[8+ct] = L_new  

    return q


def pulsarDistanceJump(x, iter, beta):

    # get old parameters
    q = x.copy()

    # draw jump from prior
    for ct, p in enumerate(psr):
        
        prob = np.random.rand()

        if prob > 0.75:
            sig = 0.5*p.distErr

        else: 
            sig = p.distErr

        # TODO: maybe add scale = 2.4/np.sqrt(npsr)?

        L_new = x[8+ct] + np.random.randn() * sig

        # phase jump
        phaseJump = False
        if np.random.rand() > 0.5: phaseJump = True
    
        L_new = pulsarPhaseFix(x[8+ct], p.cosMu(x[0], x[1]), x[7], 10**x[3], \
                                np.pi*10**x[2], L_new, p.cosMu(q[0], q[1]), \
                                q[7], 10**q[3], np.pi*10**q[2], beta, \
                                phaseJump=phaseJump)
    
        q[8+ct] = L_new

    return q

# TODO: make single component adaptive proposal


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
        index = np.random.randint(0, (length-1), length)

        # randomize proposal cycle
        self.randomizedPropCycle = [self.propCycle[ind] for ind in index]


    # call proposal functions from cycle
    def __call__(self, x, iter, beta):
        """
        Call Jump proposals

        """

        # get length of cycle
        length = len(self.propCycle)

        # call function
        q = self.randomizedPropCycle[np.mod(iter, length)](x, iter, beta)

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
jumps.addProposalToCycle(pulsarPhaseJump, SMALL)

# add large jump in pulsar phase
jumps.addProposalToCycle(pulsarDistanceJump, 2*SMALL)

# add correlated mass-distance jumps
jumps.addProposalToCycle(massDistanceJump, SMALL)

# randomize cycle
jumps.randomizeProposalCycle()


### post burn-in jump proposal cyclc ###

# initialize jumps
jumps2 = JumpProposals()

# add covariance jump proposals
jumps2.addProposalToCycle(covarianceJumpProposal, BIG)

# add small jump in pulsar phase
jumps2.addProposalToCycle(pulsarPhaseJump, SMALL)

# add large jump in pulsar phase
jumps2.addProposalToCycle(pulsarDistanceJump, 2*SMALL)

# add correlated mass-distance jumps
jumps2.addProposalToCycle(massDistanceJump, SMALL)

# add differential evolution jumps
jumps2.addProposalToCycle(DEJump, 2*SMALL)

# randomize cycle
jumps2.randomizeProposalCycle()


## function to put pulsar change pulsar distance to put back in phase
def pulsarPhaseFix(L0, cosMu0, p0, mc0, omega0, L1, cosMu1, p1, mc1, omega1, \
                        beta, phaseJump=True):

    # get in correct units
    L0 *= 1.0267e11
    L1 *= 1.0267e11
    mc0 *= 4.9e-6
    mc1 *= 4.9e-6

    # pulsar frequency
    omegap0 = omega0 * (1+256/5*mc0**(5/3)*omega0**(8/3)*L0*(1-cosMu0))**(-3/8)
    omegap1 = omega1 * (1+256/5*mc1**(5/3)*omega1**(8/3)*L1*(1-cosMu1))**(-3/8)

    # pulsar phase
    phase0 = 1/32/mc0**(5/3) * (omega0**(-5/3) - omegap0**(-5/3))
    phase1 = 1/32/mc1**(5/3) * (omega1**(-5/3) - omegap1**(-5/3))

    # make new phase multiple of 2 pi
    N = np.int16(np.round(phase1/2/np.pi))

    # compute new phase with small jump
    prob = np.random.rand()
    if prob > 0.8:
        sigma = 0.1 * np.random.randn()

    elif prob > 0.6:
        sigma = 0.02 * np.random.randn()

    elif prob > 0.4:
        sigma = 0.002 * np.random.randn()

    else:
        sigma = 0.005 * np.random.randn()

    if phaseJump == False: sigma = 0.0
    
    phase_new = np.mod(phase0, 2*np.pi) + p0 - p1 + 2*np.pi*N + sigma * np.sqrt(1/beta)


    # now compute pulsar distance
    L_new = 5/256 * (omega1**(8/3) * (omega1**(-5/3) - 32*mc1**(5/3)*phase_new)**(8/5) \
                                - 1)/(mc1**(5/3)*omega1**(8/3)*(1-cosMu1))

    return L_new / 1.0267e11

# number of dimensions our problem has
ndim = 8 + npsr

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

Tmin = 1
tstep = tstep[ndim-1]

# set up temperature ladder
ntemps = args.ntemps
nthreads = args.nprocs

# low temperature chain
if args.ti == 0:
    Tmax = 100
    Tmin = 1
    tstep = np.exp(np.log(Tmax)/ntemps)

# high temperature chains
elif args.ti == 1:
    Tmax = 1e4
    Tmin = 100
    tstep = np.exp(np.log(Tmax)/ntemps)

# geometrically spaced betas
betas = np.exp(np.linspace(-np.log(Tmin), -(ntemps)*np.log(tstep), ntemps))

# pick starting values
p0 = np.zeros((ntemps, ndim))
for ii in range(ntemps):
    p0[ii,0:8] = pmin + np.random.rand(ndim-npsr) * (pmax - pmin)
    p0[ii,8:] = np.array([p.dist for p in psr])

# set initial frequency to be maximum likelihood value
if args.freq is None:
    p0[0,2] = np.log10(fmaxlike)

# if using injected values
if args.inj:
    p0[:,0:8] = true + np.random.randn(8)*true/100

# if using strain, start at 0
#p0[:,4] = 0.0

# initialize covariance matrix for jumps
global cov, M2, mu, U, S, savephase, jumpsize
jumpsize = np.zeros(npsr)
savephase = np.zeros((1000000,npsr))
M2 = np.zeros((ndim, ndim))
mu = np.zeros(ndim)
cov_diag = np.zeros(ndim)
cov_diag[0] = 0.2
cov_diag[1] = 0.2
cov_diag[2] = 0.005
cov_diag[3] = 0.2
cov_diag[4] = 0.2
cov_diag[5:7] = 0.2
cov_diag[7] = 0.1
cov_diag[8:] = np.array([p.distErr for p in psr])
cov = np.diag((cov_diag/10)**2)
U, S, V = np.linalg.svd(cov)


# no cyclic variables
cyclic = np.zeros(ndim)

# add in cyclic values for phase and phi
cyclic[1] = 2*np.pi
cyclic[7] = 2*np.pi

# initialize MH sampler
sampler=PTSampler(ntemps, ndim, loglike, logprior, jumpProposals, threads=nthreads,\
                  betas=betas, cyclic=cyclic, Tskip=100)

# set output file
for ii in range(args.ntemps):
    chainname = args.outDir + '/chain_{0}.dat'.format(betas[ii])
    chainfile = open(chainname, 'w')
    chainfile.close()

covname = args.outDir + '/cov.npy'

# run sampler
N = 1000000
isave = 5000   # save every isave iterations
thin = 10
ct = 0
print 'Beginning Sampling in {0} dimensions\n'.format(ndim)
tstart = time.time()
times = np.zeros(N)
times[0] = time.time()
for pos, prob, state in sampler.sample(p0, iterations=N):
    #print 'Total MCMC one step time = {0}'.format(time.time() - times[ct])
    if ct % isave == 0 and ct > 0:

        tstep = time.time() - tstart

        # write files
        for ii in range(args.ntemps):
            chainname = args.outDir + '/chain_{0}.dat'.format(betas[ii])
            chainfile = open(chainname, 'a+')
            for jj in range((ct-isave), ct, thin):
                chainfile.write('%e\t %e\t'%(sampler.lnprobability[ii,jj], sampler.lnlikelihood[ii,jj]))
                chainfile.write('\t'.join([str(sampler.chain[ii,jj,kk]) for kk in range(ndim)]))
                chainfile.write('\n')
            chainfile.close()
       

        #np.save(chainfile, sampler.chain[:,0:ct,:])
        #np.save(lnlikefile,sampler.lnlikelihood[:,0:ct])
        #np.save(lnprobfile,sampler.lnprobability[:,0:ct])
        np.save(covname, cov)
        #np.save(phasefile, savephase[0:ct,:])

        print 'Finished {0} of {1} iterations.'.format(ct, N)
        print 'Acceptance fraction = {0}'.format(sampler.acceptance_fraction)
        print 'Tswap Acceptance fraction = {0}'.format(sampler.tswap_acceptance_fraction)
        print 'Time elapsed: {0} s'.format(tstep)
        print 'Approximate time remaining: {0} hr\n'.format(tstep/ct * (N-ct)/3600)

    # update counter
    ct += 1
    times[ct] = time.time()



