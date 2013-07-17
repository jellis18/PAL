#!/usr/bin/env python

# compute frequentist upper limits with the Fp statistic

from __future__ import division
import numpy as np
from scipy.optimize import brentq
import libstempo as t2
import PALutils
import PALLikelihoods
import PALpulsarInit
import h5py as h5
import argparse
import os, glob

parser = argparse.ArgumentParser(description = 'Simulate Fake Data (Under Construction)')

# options
parser.add_argument('--h5File', dest='h5file', action='store', type=str, required=True,
                   help='Full path to hdf5 file containing PTA data')
parser.add_argument('--parDir', dest='parDir', action='store', type=str, required=True,
                   help='Full path to par files')
parser.add_argument('--timDir', dest='timDir', action='store', type=str, required=True,
                   help='Full path to tim files')
parser.add_argument('--freq', dest='freq', action='store', type=float, required=True,
                   help='Frequency at which to compute upper limit')
parser.add_argument('--nreal', dest='nreal', action='store', type=int, required=1000,
                   help='Number of realizations to use for each amplitude (default = 1000)')

# parse arguments
args = parser.parse_args()

##### PREPARE DATA STRUCTURES #####

# import hdf5 file
pfile = h5.File(args.h5file)

# define the pulsargroup
pulsargroup = pfile['Data']['Pulsars']

# fill in pulsar class
psr = [PALpulsarInit.pulsar(pulsargroup[key],addNoise=True) for key in pulsargroup]

# close hdf5 file
pfile.close()

# number of pulsars
npsr = len(psr)

# make sure all pulsar have same reference time
tt=[] 
for p in psr:
    tt.append(np.min(p.toas))

# find reference time
tref = np.min(tt)

# now scale pulsar time
for p in psr:
    p.toas -= tref


# read in tim and par files
parFile = glob.glob(args.parDir + '/*.par')
timFile = glob.glob(args.timDir + '/*.tim')

# sort
parFile.sort()
timFile.sort()

# check to make sure same number of tim and par files
if len(parFile) != len(timFile):
    raise IOError, "Need same number of par and tim files!"

# check to make sure same number of tim/par files as was in hdf5 file
if len(parFile) != npsr:
    raise IOError, "Different number of pulsars in par directory and hdf5 file!"

# run tempo2
pp = [t2.tempopulsar(parFile[ii],timFile[ii]) for ii in range(npsr)]

# finally check to make sure that they are the same pulsars
for ct,p in enumerate(psr):
    if p.name not in  [ps.name for ps in pp]:
        raise IOError, "PSR {0} not found in hd5f file!".format(p.name)

# make sure pulsar names are in correct order
# TODO: is this a very round about way to do this?
index = []
for ct,p in enumerate(pp):
    
    if p.name == psr[ct].name:
        index.append(ct)
    else:
        for ii in range(npsr):
            if pp[ii].name == psr[ct].name:
                index.append(ii)

pp = [pp[ii] for ii in index]

#############################################################################################

#### DEFINE UPPER LIMIT FUNCTION #####

def upperLimitFunc(h):
    """
    Compute the value of the fstat for a range of parameters, with fixed
    amplitude over many realizations.

    @param h: value of the strain amplitude to keep constant
    @param fstat_ref: value of fstat for real data set
    @param freq: GW frequency
    @param nreal: number of realizations

    """
    
    Tmaxyr = np.array([(p.toas.max() - p.toas.min())/3.16e7 for p in psr]).max()
    count = 0
    for ii in range(nreal):

        # draw parameter values
        gwtheta = np.arccos(np.random.uniform(-1, 1))
        gwphi = np.random.uniform(0, 2*np.pi)
        gwphase = np.random.uniform(0, 2*np.pi)
        gwinc = np.arccos(np.random.uniform(0, 1))
        gwpsi = np.random.uniform(-np.pi/4, np.pi/4)

        # check to make sure source has not coalesced during observation time
        coal = True
        while coal:
            gwmc = 10**np.random.uniform(7, 10)
            tcoal = 2e6 * (gwmc/1e8)**(-5/3) * (freq/1e-8)**(-8/3)
            if tcoal > Tmaxyr:
                coal = False

        # determine distance in order to keep strain fixed
        gwdist = 4 * np.sqrt(2/5) * (gwmc*4.9e-6)**(5/3) * (np.pi*freq)**(2/3) / h

        # convert back to Mpc
        gwdist /= 1.0267e14

        # create residuals and refit for all pulsars
        for ct,p in enumerate(psr):
            inducedRes = PALutils.createResiduals(p, gwtheta, gwphi, gwmc, gwdist, \
                            freq, gwphase, gwpsi, gwinc)
 
            # add to site arrival times of pulsar
            pp[ct].stoas[:] += np.longdouble(inducedRes/86400)

            # refit
            pp[ct].fit(iters=3)

            # replace residuals in pulsar object
            p.res = pp[ct].residuals()

        # compute f-statistic
        fpstat = PALLikelihoods.fpStat(psr, freq)

        # check to see if larger than in real data
        if fpstat > fstat_ref:
            count += 1

    # now get detection probability
    detProb = count/nreal

    print h, detProb

    return detProb - 0.95


#############################################################################################

# compute reference f-statistic
fstat_ref = PALLikelihoods.fpStat(psr, args.freq)

# now compute bound with scalar minimization function using Brent's method
hhigh = 5e-14
hlow = 1e-15
xtol = 1e-16
freq = args.freq
nreal = args.nreal
h_up = brentq(upperLimitFunc, hlow, hhigh, xtol=xtol)
#fbounded = minimize_scalar(upperLimitFunc, args=(fstat_ref, args.freq, args.nreal), \
#                           bounds=(hlow, hmid, hhigh), method='Brent')


    
