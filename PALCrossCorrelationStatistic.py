#!/usr/bin/env python

# Run the Cross correlation statistic defined in Demorest et al (2012) 

from __future__ import division
import numpy as np
import PALLikelihoods
import PALutils
import PALpulsarInit
import h5py as h5
import optparse
import scipy.special as ss
import os

parser = optparse.OptionParser(description = 'Run the Cross Correlation statistic defined in \
                                                Demorest et al. (2012)')

# options
parser.add_option('--h5File', dest='h5file', action='store', type=str,
                   help='Full path to hdf5 file containing PTA data')
parser.add_option('--outDir', dest='outDir', action='store', type=str, default='./',
                   help='Full path to output directory (default = ./)')
parser.add_option('--spectralIndex', dest='gam', action='store', type=float, default=4.3333,
                   help='Power spectral index of stochastic background (default = 4.3333 (SMBHBs))')


# parse arguments
(args, x) = parser.parse_args()

##### Begin Code #####

print 'Reading in HDF5 file' 

# import hdf5 file
pfile = h5.File(args.h5file, 'r')

# define the pulsargroup
pulsargroup = pfile['Data']['Pulsars']

# fill in pulsar class
psr = [PALpulsarInit.pulsar(pulsargroup[key],addNoise=True, addGmatrix=True) for key in pulsargroup]

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


# compute pairwise overlap reduction function values
print 'Computing Overlap Reduction Function Values'
ORF = PALutils.computeORF(psr)

# since we have defined our ORF to be normalized to 1
hdcoeff = ORF/2

# compute optimal statistic
print 'Running Cross correlation Statistic on {0} Pulsars'.format(npsr)
crosspower, crosspowererr = PALLikelihoods.crossPower(psr, args.gam)

# angular separation
xi = []
for ll in range(npsr):
    for kk in range(ll+1, npsr):
        xi.append(PALutils.angularSeparation(psr[ll].theta, psr[ll].phi, \
                                            psr[kk].theta, psr[kk].phi))

# Perform chi-squared fit to determine best fit amplituded to HD curve
hc_sqr = np.sum(crosspower*hdcoeff / (crosspowererr*crosspowererr)) / \
            np.sum(hdcoeff*hdcoeff / (crosspowererr*crosspowererr))

hc_sqrerr = 1.0 / np.sqrt(np.sum(hdcoeff * hdcoeff / (crosspowererr * crosspowererr)))

# get reduced chi-squared value
chisqr = np.sum(((crosspower - hc_sqr*hdcoeff) / crosspowererr)**2)
redchisqr = np.sum(chisqr) / len(crosspower)


print 'Results of Search\n'

print '------------------------------------\n'

print 'A_gw^2 = {0}'.format(hc_sqr)
print 'std. dev. = {0}'.format(hc_sqrerr)
print 'Reduced Chi-squared = {0}'.format(redchisqr)

up = np.sqrt(hc_sqr + np.sqrt(2)*hc_sqrerr*ss.erfcinv(2*(1-0.95)))
print '2-sigma upper limit based on chi-squared fit is A_gw < {0}'.format(up)







