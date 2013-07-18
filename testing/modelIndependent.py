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

if args.powerlaw:
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


# parameterize by power law
if args.powerlaw:
    print 'Parameterizing Power spectrum coefficients by a power law'

    if args.efac == False and args.equad == False and args.nored:
        print 'Automatically setting EFAC = 1 and EQUAD = 0 for all pulsars'

        def myprior(cube, ndim, nparams):
            # define parameter ranges
            emin = 0.1
            emax = 10
            qmin = -10
            qmax = -5
            lAmin = -17
            lAmax = -10
            gamMin = 0
            gamMax = 7

            # convert from hypercube
            cube[0] = lAmin + cube[0] * (lAmax - lAmin)
            cube[1] = gamMin + cube[1] * (gamMax - gamMin)

        def myloglike(cube, ndim, nparams):

            efac = np.ones(npsr)
            equad = np.zeros(npsr)
            A = 10**cube[0] 
            gam = cube[1] 
            
            loglike = PALLikelihoods.modelIndependentFullPTAPL(psr, F, Diag, f, A, gam, \
                                                               Ared, gred, efac, equad, ORF)

            #print efac, rho, loglike

            return loglike

        # number of dimensions our problem has
        n_params = 2
        nlive = 500
    
    if args.efac == False and args.equad and args.nored:
        print 'Using EQUAD as a search parameter. Automatically setting EFAC = 1.'

        def myprior(cube, ndim, nparams):
            # define parameter ranges
            emin = 0.1
            emax = 10
            qmin = -10
            qmax = -5
            lAmin = -17
            lAmax = -10
            gamMin = 0
            gamMax = 7

            # convert from hypercube
            #cube[0] = emin + cube[0] * (emax - emin)
            #cube[1] = qmin + cube[1] * (qmax - qmin)
            cube[0] = lAmin + cube[0] * (lAmax - lAmin)
            cube[1] = gamMin + cube[1] * (gamMax - gamMin)
            for ii in range(npsr):
                cube[ii+2] = qmin + cube[ii+2] * (qmax - qmin)

        def myloglike(cube, ndim, nparams):

            efac = np.ones(npsr)
            equad = np.zeros(npsr)
            A = 10**cube[0] 
            gam = cube[1] 
            for ii in range(npsr):
                equad[ii] = 10**cube[ii+2]
            
            loglike = PALLikelihoods.modelIndependentFullPTAPL(psr, F, Diag, f, A, gam, \
                                                               Ared, gred, efac, equad, ORF)

            #print efac, rho, loglike

            return loglike

        # number of dimensions our problem has
        n_params = 2 + npsr
        nlive = 500
    
    if args.efac and args.equad == False and args.nored:
        print 'Using EFAC as a search parameter. Automatically setting EQUAD = 0.'

        def myprior(cube, ndim, nparams):
            # define parameter ranges
            emin = 0.1
            emax = 10
            qmin = -10
            qmax = -5
            lAmin = -17
            lAmax = -10
            gamMin = 0
            gamMax = 7

            # convert from hypercube
            #cube[0] = emin + cube[0] * (emax - emin)
            #cube[1] = qmin + cube[1] * (qmax - qmin)
            cube[0] = lAmin + cube[0] * (lAmax - lAmin)
            cube[1] = gamMin + cube[1] * (gamMax - gamMin)
            for ii in range(npsr):
                cube[ii+2] = emin + cube[ii+2] * (emax - emin)

        def myloglike(cube, ndim, nparams):

            efac = np.zeros(npsr)
            equad = np.zeros(npsr)
            A = 10**cube[0] 
            gam = cube[1] 
            for ii in range(npsr):
                efac[ii] = cube[ii+2]
            
            loglike = PALLikelihoods.modelIndependentFullPTAPL(psr, F, Diag, f, A, gam, \
                                                               Ared, gred, efac, equad, ORF)

            #print efac, rho, loglike

            return loglike

        # number of dimensions our problem has
        n_params = 2 + npsr
        nlive = 500
    
    if args.efac and args.equad and args.nored:
        print 'Using both EFAC and EQUAD as search parameters'

        def myprior(cube, ndim, nparams):
            # define parameter ranges
            emin = 0.1
            emax = 10
            qmin = -10
            qmax = -5
            lAmin = -17
            lAmax = -10
            gamMin = 0
            gamMax = 7

            # convert from hypercube
            cube[0] = lAmin + cube[0] * (lAmax - lAmin)
            cube[1] = gamMin + cube[1] * (gamMax - gamMin)
            for ii in range(npsr):
                cube[ii+2] = emin + cube[ii+2] * (emax - emin)
                cube[ii+npsr+2] = qmin + cube[ii+npsr+2] * (qmin - qmax)

        def myloglike(cube, ndim, nparams):

            efac = np.zeros(npsr)
            equad = np.zeros(npsr)
            A = 10**cube[0] 
            gam = cube[1] 
            for ii in range(npsr):
                efac[ii] = cube[ii+2]
                equad[ii] = 10**cube[ii+2+npsr]
            
            loglike = PALLikelihoods.modelIndependentFullPTAPL(psr, F, Diag, f, A, gam, \
                                                               Ared, gred, efac, equad, ORF)

            #print efac, rho, loglike

            return loglike

        # number of dimensions our problem has
        n_params = 2 + 2*npsr
        nlive = 500
    
    if args.efac and args.equad and args.nored == False:
        print 'Using both EFAC and EQUAD as search parameters. Also parameterizing intrinsic red noise as powerlaw.'

        def myprior(cube, ndim, nparams):
            # define parameter ranges
            emin = 0.1
            emax = 5
            qmin = -8
            qmax = -5
            lAmin = -17
            lAmax = -11
            gamMin = 0
            gamMax = 7

            # convert from hypercube
            cube[0] = lAmin + cube[0] * (lAmax - lAmin)
            cube[1] = gamMin + cube[1] * (gamMax - gamMin)
            for ii in range(npsr):
                cube[ii+2] = emin + cube[ii+2] * (emax - emin)
                cube[ii+npsr+2] = qmin + cube[ii+npsr+2] * (qmin - qmax)
                cube[ii+2*npsr+2] = lAmin + cube[ii+2*npsr+2] * (lAmin - lAmax)
                cube[ii+3*npsr+2] = gamMin + cube[ii+3*npsr+2] * (gamMin - gamMax)

        def myloglike(cube, ndim, nparams):

            efac = np.zeros(npsr)
            equad = np.zeros(npsr)
            Ared = np.zeros(npsr)
            gred = np.zeros(npsr)
            A = 10**cube[0] 
            gam = cube[1] 
            for ii in range(npsr):
                efac[ii] = cube[ii+2]
                equad[ii] = 10**cube[ii+2+npsr]
                Ared[ii] = 10**cube[ii+2+2*npsr]
                gred[ii] = cube[ii+2+3*npsr]
            
            loglike = PALLikelihoods.modelIndependentFullPTAPL(psr, F, Diag, f, A, gam, \
                                                               Ared, gred, efac, equad, ORF)

            #print efac, rho, loglike

            return loglike

        # number of dimensions our problem has
        n_params = 2 + 4*npsr
        nlive = 500
    
    # parameterize by independent coefficienta
else:
    print 'Parameterizing Power spectrum coefficients by {0} independent coefficients'.format(args.nmodes)

    if args.equad and args.efac:
        
        print 'Using both EFAC and EQUAD as search parameters'

        def myprior(cube, ndim, nparams):
            # define parameter ranges
            emin = 0.1
            emax = 10
            qmin = -10
            qmax = -5
            rhomin = -20
            rhomax = -10

            # convert from hypercube
            for ii in range(npsr):
                cube[ii] = emin + cube[ii] * (emax - emin)
                cube[ii+npsr] = qmin + cube[ii+npsr] * (qmax - qmin)
            for ii in range(2*npsr, ndim):
                cube[ii] = rhomin + cube[ii] * (rhomax - rhomin)

        def myloglike(cube, ndim, nparams):

            efac = np.zeros(npsr)
            equad = np.zeros(npsr)
            for ii in range(npsr):
                efac[ii] = cube[ii]
                equad[ii] = 10**cube[ii+npsr]
            rho = np.zeros(ndim-2*npsr)
            for ii in range(ndim-2*npsr):
                rho[ii] = cube[ii+2*npsr]
           
            loglike = PALLikelihoods.modelIndependentFullPTA(psr, F, Diag, rho, kappa, efac, equad, ORF)

            #print efac, rho, loglike

            return loglike

        # number of dimensions our problem has
        n_params = args.nmodes + 2*npsr
        nlive = 500
    
    if args.equad and args.efac == False:
        
        print 'Using only EQUAD as search parameter. Setting EFAC = 1 for all pulsars'
        
        def myprior(cube, ndim, nparams):
            # define parameter ranges
            emin = 0.1
            emax = 10
            qmin = -10
            qmax = -5
            rhomin = -20
            rhomax = 0

            # convert from hypercube
            for ii in range(npsr):
                cube[ii] = qmin + cube[ii] * (qmax - qmin)
            for ii in range(npsr, ndim):
                cube[ii] = rhomin + cube[ii] * (rhomax - rhomin)

        def myloglike(cube, ndim, nparams):

            efac = np.ones(npsr)
            equad = np.zeros(npsr)
            for ii in range(npsr):
                equad[ii] = 10**cube[ii]
            rho = np.zeros(ndim-npsr)
            for ii in range(ndim-npsr):
                rho[ii] = cube[ii+npsr]
           
            loglike = PALLikelihoods.modelIndependentFullPTA(psr, F, Diag, rho, kappa, efac, equad, ORF)

            #print efac, rho, loglike

            return loglike

        # number of dimensions our problem has
        n_params = args.nmodes + npsr
        nlive = 500
    
    if args.equad == False and args.efac:
        
        print 'Using only EFAC as search parameter. Setting EQUAD = 0 for all pulsars'
        
        def myprior(cube, ndim, nparams):
            # define parameter ranges
            emin = 0.1
            emax = 10
            qmin = -10
            qmax = -5
            rhomin = -20
            rhomax = -10

            # convert from hypercube
            for ii in range(npsr):
                cube[ii] = emin + cube[ii] * (emax - emin)
            for ii in range(npsr, ndim):
                cube[ii] = rhomin + cube[ii] * (rhomax - rhomin)

        def myloglike(cube, ndim, nparams):

            efac = np.zeros(npsr)
            equad = np.zeros(npsr)
            for ii in range(npsr):
                efac[ii] = cube[ii]
            rho = np.zeros(ndim-npsr)
            for ii in range(ndim-npsr):
                rho[ii] = cube[ii+npsr]
           
            loglike = PALLikelihoods.modelIndependentFullPTA(psr, F, Diag, rho, kappa, efac, equad, ORF)

            #print efac, rho, loglike

            return loglike

        # number of dimensions our problem has
        n_params = args.nmodes + npsr
        nlive = 500
    
    if args.equad == False and args.efac == False:
        
        print 'Setting EQUAD = 0  and EFAC = 1 for all pulsars'
        
        def myprior(cube, ndim, nparams):
            # define parameter ranges
            emin = 0.1
            emax = 10
            qmin = -10
            qmax = -5
            rhomin = -20
            rhomax = -10

            # convert from hypercube
            for ii in range(0, ndim):
                cube[ii] = rhomin + cube[ii] * (rhomax - rhomin)

        def myloglike(cube, ndim, nparams):

            efac = np.ones(npsr)
            equad = np.zeros(npsr)
            rho = np.zeros(ndim)
            for ii in range(ndim):
                rho[ii] = cube[ii]
           
            loglike = PALLikelihoods.modelIndependentFullPTA(psr, F, Diag, rho, kappa, efac, equad, ORF)

            #print efac, rho, loglike

            return loglike

        # number of dimensions our problem has
        n_params = args.nmodes 
        nlive = 500

# start time
tstart = time.time()

# run MultiNest
#pymultinest.run(myloglike, myprior, n_params, resume = False, \
#                verbose = True, sampling_efficiency = 0.1, \
#                outputfiles_basename =  args.outDir+'/test', \
#                n_iter_before_update=5, n_live_points=nlive, \
#                const_efficiency_mode=False)

nlive = 250
pymultinest.run(myloglike, myprior, n_params, resume = False, \
                verbose = True, sampling_efficiency = 0.01, \
                outputfiles_basename =  args.outDir+'/test', \
                n_iter_before_update=5, n_live_points=nlive, \
                const_efficiency_mode=True, importance_nested_sampling=True)

print 'Total Elapsed Time: {0} s'.format(time.time() - tstart)



