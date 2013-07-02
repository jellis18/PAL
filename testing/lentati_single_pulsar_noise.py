#!/usr/bin/env python

# Run Lentati style noise estimation on single pulsar

from __future__ import division
import numpy as np
import PALLikelihoods
import PALutils
import PALpulsarInit
import h5py as h5
import pymultinest
import scipy.linalg as sl
import argparse
import os

parser = argparse.ArgumentParser(description = 'Run Lentati style noise estimation')

# options
parser.add_argument('--h5File', dest='h5file', action='store', type=str, required=True,
                   help='Full path to hdf5 file containing PTA data')
parser.add_argument('--outDir', dest='outDir', action='store', type=str, default='./',
                   help='Full path to output directory (default = ./)')
parser.add_argument('--pulsar', dest='pname', action='store', type=str, required=True,
                   help='name of pulsar to use')
parser.add_argument('--nmodes', dest='nmodes', action='store', type=int, required=10,
                   help='number of fourier modes to use (default=10)')



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
pulsargroup = pfile['Data']['Pulsars'][args.pname]

# fill in pulsar class
psr = PALpulsarInit.pulsar(pulsargroup, addGmatrix=True)

# initialize fourier design matrix
F = PALutils.createfourierdesignmatrix(psr.toas, args.nmodes)

# get pre-constructed quantities
N = np.dot(psr.G.T, np.dot(np.diag(psr.err**2), psr.G))
cf = sl.cho_factor(N)
logdet_N = np.sum(2*np.log(np.diag(cf[0])))
Ninv = np.dot(psr.G, np.dot(np.linalg.inv(N), psr.G.T))
FNF = np.dot(F.T, np.dot(Ninv, F))
d = np.dot(F.T, np.dot(Ninv, psr.res))
dtNdt = np.dot(psr.res, np.dot(Ninv, psr.res))


# multinest prior function
def myprior(cube, ndim, nparams):
    # define parameter ranges
    emin = 0.1
    emax = 10
    rhomin = -20
    rhomax = 0

    # convert from hypercube
    cube[0] = emin + cube[0] * (emax - emin)
    for ii in range(1, ndim):
        cube[ii] = rhomin + cube[ii] * (rhomax - rhomin)

def myloglike(cube, ndim, nparams):

    efac = cube[0]
    rho = np.zeros(ndim-1)
    for ii in range(ndim-1):
        rho[ii] = cube[ii+1]
   
    loglike = PALLikelihoods.lentatiMarginalizedLike(psr, \
                            d, FNF, dtNdt, logdet_N, rho, efac)

    #print efac, rho, loglike

    return loglike

# number of dimensions our problem has
n_params = args.nmodes + 1
nlive = 500

# run MultiNest
pymultinest.run(myloglike, myprior, n_params, resume = False, \
                verbose = True, sampling_efficiency = 0.8, \
                outputfiles_basename =  args.outDir+'/test', \
                n_iter_before_update=5, n_live_points=nlive)



