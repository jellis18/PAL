#!/usr/bin/env python

# compute frequentist upper limits with the Fp statistic

from __future__ import division
import numpy as np
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
import PALutils
import PALLikelihoods
import PALpulsarInit
import h5py as h5
import optparse
import os, glob

parser = optparse.OptionParser(description = 'Run Optimal Statistic Upper Limit')

# options
parser.add_option('--h5File', dest='h5file', action='store', type=str,
                   help='Full path to hdf5 file containing PTA data')
parser.add_option('--nreal', dest='nreal', action='store', type=int, default=1000,
                   help='Number of realizations to use for each amplitude (default = 1000)')
parser.add_option('--outdir', dest='outdir', action='store', type=str, default='./',
                   help='Full path to output directory(default = ./)')

# global arguments to keep track of injections
global injAmp
global injDetProb

# parse arguments
(args, x) = parser.parse_args()

##### PREPARE DATA STRUCTURES #####

# import hdf5 file
pfile = h5.File(args.h5file)

# define the pulsargroup
pulsargroup = pfile['Data']['Pulsars']

# fill in pulsar class
psr = [PALpulsarInit.pulsar(pulsargroup[key],addNoise=True, addGmatrix=True) \
            for key in pulsargroup]

# number of pulsars
npsr = len(psr)

# create list of reference residuals
res = [p.res for p in psr]

# get list of R matrices
R = [PALutils.createRmatrix(p.dmatrix, p.err) for p in psr]

# pre-compute noise covariance matrices

ct = 0
D = []
Dmatrix = []
print 'Computing diagonalized auto-covariance matrices'
for key in pulsargroup:

    # get noise values from file TODO: change this to read directly from pulsar class
    Amp = pulsargroup[key]['Amp'].value
    gam = pulsargroup[key]['gam'].value
    efac = pulsargroup[key]['efac'].value
    equad = pulsargroup[key]['equad'].value
    try:
        fH = pulsargroup[key]['fH'].value
    except KeyError:
        fH = None

    # make covariance matrix
    tm = PALutils.createTimeLags(psr[ct].toas, psr[ct].toas, round=True)

    red = PALutils.createRedNoiseCovarianceMatrix(tm, Amp, gam, fH=fH)
    white = PALutils.createWhiteNoiseCovarianceMatrix(psr[ct].err, efac, equad)

    # sandwich with G matrices
    cov = red + white
    noiseCov = np.dot(psr[ct].G.T, np.dot(cov, psr[ct].G))

    # diagonalize GW covariance matrix and noise covariance matrix
    L = np.linalg.cholesky(noiseCov)
    Linv = np.linalg.inv(L)

    # get generalized GW covariance matrix with Amp = 1
    redGW = PALutils.createRedNoiseCovarianceMatrix(tm, 1, 4.33333)
    redGW = np.dot(psr[ct].G.T, np.dot(redGW, psr[ct].G))

    # sandwich with Linv matrices
    redSand = np.dot(Linv, np.dot(redGW, Linv.T))

    # diagonalize
    u, s, v = np.linalg.svd(redSand) 

    # store diagonal terms
    D.append(s)

    # keep diagonalizing matrix
    tmp = np.dot(u.T, np.dot(Linv, psr[ct].G.T))
    Dmatrix.append(tmp)

    ct += 1

# pre-compute cross covariance matrices
print 'Pre-computing cross covariance matrices...'
SIJ = []
for ii in range(npsr):
    for jj in range(ii+1, npsr):

        # matrix of time lags
        tm = PALutils.createTimeLags(psr[ii].toas, psr[jj].toas, round=True)

        # cross covariance matrix
        crossCov = PALutils.createRedNoiseCovarianceMatrix(tm, 1, 4.33333)

        #print Dmatrix[ii].shape, Dmatrix[jj].shape, crossCov.shape
        SIJ.append(np.dot(Dmatrix[ii], np.dot(crossCov, Dmatrix[jj].T)))



# compute ORF
print 'Computing ORF...'
ORF = PALutils.computeORF(psr)

# close hdf5 file
pfile.close()


#############################################################################################

#### DEFINE UPPER LIMIT FUNCTION #####
def upperLimitFunc(A, optstat_ref, nreal):
    """
    Compute the value of the Optimal Statistic for different signal realizations
    
    @param A: value of GWB amplitude
    @param optstat_ref: value of optimal statistic with no injection 
    @param nreal: number of realizations

    """
    count = 0
    for ii in range(nreal):
        
        # create residuals
        inducedRes = PALutils.createGWB(psr, A, 4.3333)

        Pinvr = []
        Pinv = []
        for ct, p in enumerate(psr):

            # replace residuals in pulsar object
            p.res = res[ct] + np.dot(R[ct], inducedRes[ct])

            # determine injected amplitude by minimizing Likelihood function
            c = np.dot(Dmatrix[ct], p.res)
            f = lambda x: -PALutils.twoComponentNoiseLike(x, D[ct], c)
            fbounded = minimize_scalar(f, bounds=(0, 1e-14, 3.0e-13), method='Brent')
            Amp = np.abs(fbounded.x)
            #print Amp
            #Amp = A

            # construct P^-1 r
            Pinvr.append(c/(Amp**2 * D[ct] + 1))
            Pinv.append(1/(Amp**2 * D[ct] + 1))

        # construct optimal statstic
        k = 0
        top = 0
        bot = 0
        for ll in range(npsr):
            for kk in range(ll+1, npsr):

                # compute numerator of optimal statisic
                top += ORF[k]/2 * np.dot(Pinvr[ll], np.dot(SIJ[k], Pinvr[kk]))

                # compute trace term
                bot += (ORF[k]/2)**2 * np.trace(np.dot((Pinv[ll]*SIJ[k].T).T, (Pinv[kk]*SIJ[k]).T))
                # iterate counter 
                k += 1

        # get optimal statistic and SNR
        optStat = top/bot
        snr = top/np.sqrt(bot)
        
        # check to see if larger than in real data
        if optStat > optstat_ref:
            count += 1


    # now get detection probability
    detProb = count/nreal

    print A, detProb
    injAmp.append(A)
    injDetProb.append(detProb)

    return detProb - 0.95


#############################################################################################


# now compute bound with scalar minimization function using Brent's method
Ahigh = 1e-13
Alow = 5e-15
xtol = 1e-16
nreal = args.nreal

# initiate global variables
injAmp = []
injDetProb = []

# get reference optimal-statistic
print 'Getting reference Optimal Statistic Value'
optStat_ref = PALLikelihoods.optStat(psr, ORF)[0]

# perfrom upper limit calculation
inRange = False
while inRange == False:

    try:    # try brentq method
        A_up = brentq(upperLimitFunc, Alow, Ahigh, xtol=xtol, \
              args=(optStat_ref, nreal))
        inRange = True
    except ValueError:      # bounds not in range
        if Ahigh < 1e-11:   # don't go too high
            Ahigh *= 2      # double high strain
        else:
            A_up = Ahigh    # if too high, just set to upper bound
            inRange = True

# output data
if not os.path.exists(args.outdir):
    try:
        os.makedirs(args.outdir)
    except OSError:
        pass

fname = 'optStatupper_{0}.txt'.format(args.nreal)
fout = open(args.outdir + fname, 'w')
for ii in range(len(injAmp)):
    fout.write('%g %g\n'%(injAmp[ii], injDetProb[ii]))
