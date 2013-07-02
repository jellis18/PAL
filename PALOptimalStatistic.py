#!/usr/bin/env python

# Run the Optimal-statistic stochastic BG code as defined in Chamberlin, Creighton, Demorest et al (2013)

from __future__ import division
import numpy as np
import PALLikelihoods
import PALutils
import PALpulsarInit
import h5py as h5
import argparse
import scipy.special as ss
import os

parser = argparse.ArgumentParser(description = 'Run the Optimal-statistic stochastic BG code as \
                                 defined in Chamberlin, Creighton, Demorest et al (2013)')

# options
parser.add_argument('--h5File', dest='h5file', action='store', type=str, required=True,
                   help='Full path to hdf5 file containing PTA data')
parser.add_argument('--outDir', dest='outDir', action='store', type=str, default='./',
                   help='Full path to output directory (default = ./)')
parser.add_argument('--spectralIndex', dest='gam', action='store', type=float, default=4.3333,
                   help='Power spectral index of stochastic background (default = 4.3333 (SMBHBs))')


# parse arguments
args = parser.parse_args()

##### Begin Code #####

print 'Reading in HDF5 file' 

# import hdf5 file
pfile = h5.File(args.h5file, 'r')

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

#import glob
#invmat = glob.glob('/Users/Justin/Work/nanograv/nanograv/data_products/joris/*invCov*')
#
## get list of R matrices
#R = [PALutils.createRmatrix(p.dmatrix, p.err) for p in psr]
#
#for ct,p in enumerate(psr):
#    p.invCov = np.dot(R[ct].T, np.dot(p.invCov, R[ct]))


# compute pairwise overlap reduction function values
print 'Computing Overlap Reduction Function Values'
ORF = PALutils.computeORF(psr)

# compute optimal statistic
print 'Running Optimal Statistic on {0} Pulsars'.format(npsr)
Opt, sigma, snr = PALLikelihoods.optStat(psr, ORF, gam=args.gam)

print 'Results of Search\n'

print '------------------------------------\n'

print 'A_gw^2 = {0}'.format(Opt)
print 'std. dev. = {0}'.format(sigma)
print 'SNR = {0}'.format(snr)

if snr > 3.0:
    print 'SNR of {0} is above threshold!'.format(snr)
else:
    up = np.sqrt(Opt + np.sqrt(2)*sigma*ss.erfcinv(2*(1-0.95)))
    print '2-sigma upper limit based on variance of estimators is A_gw < {0}'.format(up)






