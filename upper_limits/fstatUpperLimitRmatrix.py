#!/usr/bin/env python

# compute frequentist upper limits with the Fp statistic

from __future__ import division
import numpy as np
from scipy.optimize import brentq
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
parser.add_argument('--freq', dest='freq', action='store', type=float, default=None,
                   help='Frequency at which to compute upper limit (default = None)')
parser.add_argument('--nreal', dest='nreal', action='store', type=int, default=1000,
                   help='Number of realizations to use for each amplitude (default = 1000)')
parser.add_argument('--nfreqs', dest='nfreqs', action='store', type=int, default=40,
                   help='Number of frequencies to compute upper limits  (default = 40)')
parser.add_argument('--outdir', dest='outdir', action='store', type=str, default='./',
                   help='Full path to output directory(default = ./)')

# parse arguments
args = parser.parse_args()

##### PREPARE DATA STRUCTURES #####

# import hdf5 file
pfile = h5.File(args.h5file)

# define the pulsargroup
pulsargroup = pfile['Data']['Pulsars']

# fill in pulsar class
psr = [PALpulsarInit.pulsar(pulsargroup[key],addNoise=True) for key in pulsargroup]

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

# create list of reference residuals
res = [p.res for p in psr]

# get list of R matrices
R = [PALutils.createRmatrix(p.dmatrix, p.err) for p in psr]


#############################################################################################

#### DEFINE UPPER LIMIT FUNCTION #####

def upperLimitFunc(h, fstat_ref, freq, nreal):
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

        # create residuals 
        for ct,p in enumerate(psr):
            inducedRes = PALutils.createResiduals(p, gwtheta, gwphi, gwmc, gwdist, \
                            freq, gwphase, gwpsi, gwinc, evolve=True)
 
            # replace residuals in pulsar object
            p.res = res[ct] + np.dot(R[ct], inducedRes)

        # compute f-statistic
        fpstat = PALLikelihoods.fpStat(psr, freq)
        
        # check to see if larger than in real data
        if fpstat > fstat_ref:
            count += 1

    # now get detection probability
    detProb = count/nreal

    print freq, h, detProb

    return detProb - 0.95


#############################################################################################


# now compute bound with scalar minimization function using Brent's method
hhigh = 1e-13
hlow = 1e-15
xtol = 1e-16
freq = args.freq
nreal = args.nreal

if freq is not None:

    # get reference f-statistic
    fstat_ref = PALLikelihoods.fpStat(psr, freq)

    # perfrom upper limit calculation
    inRange = False
    while inRange == False:

        try:    # try brentq method
            h_up = brentq(upperLimitFunc, hlow, hhigh, xtol=xtol, \
                  args=(fstat_ref, freq, nreal))
            inRange = True
        except ValueError:      # bounds not in range
            if hhigh < 1e-11:   # don't go too high
                hhigh *= 2      # double high strain
            else:
                h_up = hhigh    # if too high, just set to upper bound
                inRange = True

elif freq is None:

    # set upper and lower bounds on frequency
    flow = 1e-9
    fhigh = 4e-7

    # evenly spaced in log
    if args.nfreqs is not None:
        freqs = np.logspace(np.log10(flow), np.log10(fhigh), args.nfreqs)
    else:
        freqs = np.logspace(np.log10(flow), np.log10(fhigh), 40)

    # compute reference f-statistic
    fstat_ref = [PALLikelihoods.fpStat(psr, freqs[ii]) for ii in range(len(freqs))]

    h_up = np.zeros(len(freqs))
    for ii in range(len(freqs)):
        
        hhigh = 1e-13
        hlow = 1e-15

        inRange = False
        while inRange == False:
        
            try:
                h_up[ii] = brentq(upperLimitFunc, hlow, hhigh, xtol=xtol, \
                      args=(fstat_ref[ii], freqs[ii], nreal))
                inRange = True
            except ValueError:
                if hhigh < 1e-11:
                    hhigh *= 2
                else:
                    h_up[ii] = hhigh
                    inRange = True


# output data
if not os.path.exists(args.outdir):
    try:
        os.makedirs(args.outdir)
    except OSError:
        pass

if args.freq is None:   # save entire list
    fout = open(args.outdir + 'upperLimits.txt', 'w')
    for ii in range(len(freqs)):
        fout.write('%g %g\n'%(freqs[ii], h_up[ii]))

else:   # only one frequency
    fname = 'upper_{0}_{1}.txt'.format(args.freq, args.nreal)
    fout = open(args.outdir + fname, 'w')
    fout.write('%g %g\n'%(args.freq, h_up))
