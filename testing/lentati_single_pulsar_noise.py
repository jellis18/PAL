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
parser.add_argument('--nmodes', dest='nmodes', action='store', type=int, default=10,
                   help='number of fourier modes to use (default=10)')
parser.add_argument('--powerlaw', dest='powerlaw', action='store_true', default=False,
                   help='Use power law model (default = False)')
parser.add_argument('--fc', dest='fc', action='store_true', default=False,
                   help='Use power law model with cross over frequency (default = False)')
parser.add_argument('--broken', dest='broken', action='store_true', default=False,
                   help='Use power law with two spectral indices and a cross over frequency (default = False)')


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
if args.nmodes != 0:
    F, f = PALutils.createfourierdesignmatrix(psr.toas, args.nmodes, freq=True)

# get G matrices
psr.G = PALutils.createGmatrix(psr.dmatrix)

# pre compute diagonalized efac + equad white noise model
efac = np.dot(psr.G.T, np.dot(np.diag(psr.err**2), psr.G))
equad = np.dot(psr.G.T, psr.G)
L = np.linalg.cholesky(equad)
Linv = np.linalg.inv(L)
sand = np.dot(Linv, np.dot(efac, Linv.T))
u,s,v = np.linalg.svd(sand)
proj = np.dot(u.T, np.dot(Linv, psr.G.T))

# project residuals onto new basis
psr.res = np.dot(proj, psr.res)

if args.nmodes != 0:
    F = np.dot(proj, F)


# parameterize by power law
if args.powerlaw and args.nmodes != 0:
    print 'Parameterizing Power spectrum coefficients by a power law'

    def myprior(cube, ndim, nparams):
        # define parameter ranges
        emin = 0.1
        emax = 10
        qmin = -10
        qmax = -5
        lAmin = -20
        lAmax = -10
        gamMin = -7
        gamMax = 7

        # convert from hypercube
        cube[0] = emin + cube[0] * (emax - emin)
        cube[1] = qmin + cube[1] * (qmax - qmin)
        cube[2] = lAmin + cube[2] * (lAmax - lAmin)
        cube[3] = gamMin + cube[3] * (gamMax - gamMin)

    def myloglike(cube, ndim, nparams):

        efac = cube[0]
        equad = 10**cube[1]
        A = 10**cube[2] 
        gam = cube[3] 
        
        loglike = PALLikelihoods.lentatiMarginalizedLikePL(psr, F, s, A, f, gam, efac, equad)

        #print efac, rho, loglike

        return loglike

    # number of dimensions our problem has
    n_params = 4
    nlive = 500

if args.powerlaw == False and args.nmodes != 0 and args.fc and args.broken == False:
    print 'Parameterizing Power spectrum coefficients by a power law with a cross over frequency'

    def myprior(cube, ndim, nparams):
        # define parameter ranges
        emin = 0.1
        emax = 10
        qmin = -10
        qmax = -5
        lAmin = -20
        lAmax = -10
        gamMin = -7
        gamMax = 7
        lfcmin = -12
        lfcmax = -7

        # convert from hypercube
        cube[0] = emin + cube[0] * (emax - emin)
        cube[1] = qmin + cube[1] * (qmax - qmin)
        cube[2] = lAmin + cube[2] * (lAmax - lAmin)
        cube[3] = gamMin + cube[3] * (gamMax - gamMin)
        cube[4] = lfcmin + cube[4] * (lfcmax - lfcmin)

    def myloglike(cube, ndim, nparams):

        efac = cube[0]
        equad = 10**cube[1]
        A = 10**cube[2] 
        gam = cube[3] 
        fc = 10**cube[4]

        loglike = PALLikelihoods.lentatiMarginalizedLikePL(psr, F, s, A, f, gam, efac, equad, fc=fc)

        #print efac, rho, loglike

        return loglike

    # number of dimensions our problem has
    n_params = 5
    nlive = 500

if args.powerlaw == False and args.nmodes != 0 and args.broken:
    print 'Parameterizing Power spectrum coefficients by a broken power law'

    def myprior(cube, ndim, nparams):
        # define parameter ranges
        emin = 0.1
        emax = 10
        qmin = -10
        qmax = -5
        lAmin = -20
        lAmax = -10
        gamMin = -7
        gamMax = 7
        lfcmin = -12
        lfcmax = -7

        # convert from hypercube
        cube[0] = emin + cube[0] * (emax - emin)
        cube[1] = qmin + cube[1] * (qmax - qmin)
        cube[2] = lAmin + cube[2] * (lAmax - lAmin)
        cube[3] = gamMin + cube[3] * (gamMax - gamMin)
        cube[4] = lfcmin + cube[4] * (lfcmax - lfcmin)
        cube[5] = gamMin + cube[5] * (gamMax - gamMin)

    def myloglike(cube, ndim, nparams):

        efac = cube[0]
        equad = 10**cube[1]
        A = 10**cube[2] 
        gam = cube[3] 
        fc = 10**cube[4]
        beta = cube[5]

        loglike = PALLikelihoods.lentatiMarginalizedLikePL(psr, F, s, A, f, gam, efac, equad, fc=fc, beta=beta)

        #print efac, rho, loglike

        return loglike

    # number of dimensions our problem has
    n_params = 6
    nlive = 500

    
# parameterize by independent coefficienta
if args.powerlaw == False and args.nmodes != 0 and args.fc == False and args.broken == False:

    print 'Parameterizing Power spectrum coefficients by {0} independent coefficients'.format(args.nmodes)
    
    def myprior(cube, ndim, nparams):
        # define parameter ranges
        emin = 0.1
        emax = 10
        qmin = -10
        qmax = -5
        rhomin = -20
        rhomax = 0

        # convert from hypercube
        cube[0] = emin + cube[0] * (emax - emin)
        cube[1] = qmin + cube[1] * (qmax - qmin)
        for ii in range(2, ndim):
            cube[ii] = rhomin + cube[ii] * (rhomax - rhomin)

    def myloglike(cube, ndim, nparams):

        efac = cube[0]
        equad = 10**cube[1]
        rho = np.zeros(ndim-2)
        for ii in range(ndim-2):
            rho[ii] = cube[ii+2]
       
        loglike = PALLikelihoods.lentatiMarginalizedLike(psr, F, s, rho, efac, equad)

        #print efac, rho, loglike

        return loglike

    # number of dimensions our problem has
    n_params = args.nmodes + 2
    nlive = 500

if args.powerlaw == False and args.nmodes == 0 and args.fc == False and args.broken == False:

    print 'Only using white noise model!'

    def myprior(cube, ndim, nparams):
        # define parameter ranges
        emin = 0.1
        emax = 10
        qmin = -10
        qmax = -5

        # convert from hypercube
        cube[0] = emin + cube[0] * (emax - emin)
        cube[1] = qmin + cube[1] * (qmax - qmin)

    def myloglike(cube, ndim, nparams):

        efac = cube[0]
        equad = 10**cube[1]
       
        loglike = -0.5 * (np.sum(np.log(efac*s + equad**2)) + np.sum(psr.res**2/(efac*s + equad**2)))

        #print efac, rho, loglike

        return loglike

    # number of dimensions our problem has
    n_params = args.nmodes + 2
    nlive = 500


# run MultiNest
pymultinest.run(myloglike, myprior, n_params, resume = False, \
                verbose = True, sampling_efficiency = 0.1, \
                outputfiles_basename =  args.outDir+'/test', \
                n_iter_before_update=5, n_live_points=nlive)



