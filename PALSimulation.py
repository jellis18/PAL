#!/usr/bin/env python

# Create Simulated Datasets by either injecting signals into real data or simulating ideal datasets.


from __future__ import division
import numpy as np
import libstempo as t2
import PALutils
import PALpulsarInit
import h5py as h5
import optparse 
import os, glob

parser = optparse.OptionParser(description = 'Simulate Fake Data (Under Construction)')

# options
parser.add_option('--h5File', dest='h5file', action='store', type=str,
                   help='Full path to hdf5 file containing PTA data')
parser.add_option('--outFile', dest='outFile', action='store', type=str,
                   help='Full path to output filename')
parser.add_option('--single', dest='single', action='store_true', default=False,
                   help='Add single source? (default = False)')
parser.add_option('--nopterm', dest='nopterm', action='store_true', default=False,
                   help='Dont include pulsar term in single source waveform? (default = False)')
parser.add_option('--gwra', dest='gwra', action='store', type=float, default=1.0,
                   help='GW Right Ascension (default = 1.0 radian)')
parser.add_option('--gwdec', dest='gwdec', action='store', type=float, default=0.5,
                   help='GW Declination (default = 0.5 radian)')
parser.add_option('--gwinc', dest='gwinc', action='store', type=float, default=0.5,
                   help='GW inclination angle (default = 0.5 radian)')
parser.add_option('--gwphase', dest='gwphase', action='store', type=float, default=0.5,
                   help='GW initial phase (default = 0.5 radian)')
parser.add_option('--gwpolarization', dest='gwpolarization', action='store', type=float, default=0.5,
                   help='GW polarization angle (default = 0.5 radian)')
parser.add_option('--gwchirpmass', dest='gwchirpmass', action='store', type=float, default=5e8,
                   help='GW chirp mass (default = 5e8 Solar Masses)')
parser.add_option('--gwmass1', dest='gwmass1', action='store', type=float, default=None,
                   help='GW SMBMB mass 1 (default = None)')
parser.add_option('--gwmass2', dest='gwmass2', action='store', type=float, default=None,
                   help='GW SMBMB mass 2 (default = None)')
parser.add_option('--gwdist', dest='gwdist', action='store', type=float, default=100,
                   help='GW luminosity distance (default = 100 Mpc)')
parser.add_option('--gwredshift', dest='gwredshift', action='store', type=float, default=None,
                   help='GW redshift of source (default = None)')
parser.add_option('--gwfreq', dest='gwfreq', action='store', type=float, default=1e-8,
                   help='GW initial frequency (default = 1e-8 Hz)')
parser.add_option('--snr', dest='snr', action='store', type=float, default=None,
                   help='Single source SNR (default = None, use input GW distnace)')
parser.add_option('--gwb', dest='gwb', action='store_true', default=False,
                   help='Add stochastic background? (default = False)')
parser.add_option('--gwbAmp', dest='gwbAmp', action='store', type=float, default=5e-15,
                   help='GWB amplitude (default = 5e-15)')
parser.add_option('--gwbIndex', dest='gwbIndex', action='store', type=float, default=4.33,
                   help='GWB amplitude (default = 4.33)')
parser.add_option('--noise', dest='noise', action='store_true', default=False,
                   help='Add noise based on real data values? (default = False)')
parser.add_option('--seed', dest='seed', action='store', type=int, default=0,
                   help='Random number seed for noise realizations (default = 0, no seed)')
parser.add_option('--DM', dest='DM', action='store_true', default=False,
                   help='Add DM based on real data values? (default = False)')
parser.add_option('--tim', dest='tim', action='store', type=str, default=None,
                   help='Output new tim files (default = None, dont output tim files)')


# parse arguments
(args, x) = parser.parse_args()

# compute chirp mass if mass m1 and m2 are given
if args.gwmass1 is not None and args.gwmass2 is not None:
    args.gwchirpmass = (args.gwmass1*args.gwmass2)**(3/5) / (args.gwmass1+args.gwmass2)**(1/5) 

# compute luminosity distance from redshift if given
if args.gwredshift is not None:
    args.gwdist = PALutils.computeLuminosityDistance(args.gwredshift)

# make copy of h5file TODO: maybe better way than using os.system?
h5copy = args.outFile
print 'Saving file to {0}'.format(h5copy)
os.system('cp {0} {1}'.format(args.h5file, h5copy))

# import hdf5 file
pfile = h5.File(h5copy, 'r+')

# define the pulsargroup
pulsargroup = pfile['Data']['Pulsars']

# fill in pulsar class
psr = [PALpulsarInit.pulsar(pulsargroup[key],addNoise=True, addGmatrix=True) for key in pulsargroup]

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
parFile = [pulsargroup[key]['parFile'].value for key in pulsargroup]
timFile = [pulsargroup[key]['timFile'].value for key in pulsargroup]

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


################## SIMULATED RESIDUALS ########################

# create idealized TOAs
for p in pp:
    p.stoas[:] -= p.residuals()/86400
    p.fit()

# add gwb
if args.gwb:
    
    print 'Simulating GWB with Amp = {0} and gamma = {1}'.format(args.gwbAmp, args.gwbIndex)

    inducedRes = PALutils.createGWB(psr, args.gwbAmp, args.gwbIndex)
        
    # add to site arrival times of pulsar
    for ct, p in enumerate(pp):
        p.stoas[:] += np.longdouble(inducedRes[ct]/86400)

# add DM variations
if args.DM:

    print 'Simulating DM using values in hdf5 file'

    for ct, p in enumerate(psr):

        # get values from hdf5 file
        try:
            DMAmp = pfile['Data']['Pulsars'][p.name]['DMAmp'].value
            DMgam = pfile['Data']['Pulsars'][p.name]['DMgam'].value
            inducedRes = np.squeeze(np.array(PALutils.createGWB([p], DMAmp, DMgam, True)))

            # add to site arrival times of pulsar
            pp[ct].stoas[:] += np.longdouble(inducedRes/86400)

        except KeyError:
            print 'No DM values for pulsar {0}'.format(p.name)


# add noise based on values in hdf5 file
if args.noise:
    
    print 'Simulating noise based on values in hdf5 file'

    for ct, p in enumerate(psr):

        try:
            fH = pfile['Data']['Pulsars'][p.name]['fH'].value
        except KeyError:
            fH = None


        # get values from hdf5 file
        try:
            Amp = pfile['Data']['Pulsars'][p.name]['Amp'].value
            gam = pfile['Data']['Pulsars'][p.name]['gam'].value
            efac = pfile['Data']['Pulsars'][p.name]['efac'].value
            equad = pfile['Data']['Pulsars'][p.name]['equad'].value
            tm = PALutils.createTimeLags(p.toas, p.toas)

            red = PALutils.createRedNoiseCovarianceMatrix(tm, Amp, gam, fH=fH)
            white = PALutils.createWhiteNoiseCovarianceMatrix(p.err, efac, equad)

            cov = red + white

            # cholesky decomp
            L = np.linalg.cholesky(cov)

            # set random number seed
            if args.seed:
                print 'Using fixed random number seed!'
                np.random.seed(seed=args.seed*(ct+1))
            
            # zero mean unit variance
            w = np.random.randn(p.ntoa)

            # get induced residuals
            inducedRes = np.dot(L, w)

            pp[ct].stoas[:] += np.longdouble(inducedRes/86400)

        except KeyError:
            print 'No noise values for pulsar {0}'.format(p.name)

# no options, just white noise
if args.noise == False:

    print 'Using only white noise based on error bars'
    # add to site arrival times of pulsar
    for ct,p in enumerate(pp):
        
        # set random number seed
        if args.seed:
            print 'Using fixed random number seed!'
            np.random.seed(args.seed*(ct+1))

        p.stoas[:] += p.toaerrs*1e-6 * np.random.randn(p.nobs)/86400

        # add correct "inverse covariance matrix" to hdf5 file
        white = PALutils.createWhiteNoiseCovarianceMatrix(p.toaerrs*1e-6, 1, 0)
        tmp = np.dot(psr[ct].G.T, np.dot(white, psr[ct].G))
        invCov = np.dot(psr[ct].G, np.dot(np.linalg.inv(tmp), psr[ct].G.T))
        pulsargroup[psr[ct].name]['invCov'][...] = invCov
        psr[ct].invCov = invCov

# add single source put after all noise simulation so we can get accurate snr
if args.single:

    print 'Simulating single source'

    # check for pterms
    if args.nopterm:
        print 'Not including pulsar term in single source waveform!'
        pterm = False
    else:
        pterm = True
    
    if args.snr is not None:
          
        print 'Scaling distance to give SNR = {0}'.format(args.snr)

        snr2 = 0
        args.gwdist = 1
        for ct, p in enumerate(pp):

            inducedRes = (PALutils.createResiduals(psr[ct], np.pi/2-args.gwdec, args.gwra, \
                            args.gwchirpmass, args.gwdist, args.gwfreq, args.gwphase, \
                            args.gwpolarization, args.gwinc, psrTerm=pterm))

            # compute snr
            snr2 += PALutils.calculateMatchedFilterSNR(psr[ct], inducedRes, inducedRes)**2

        # get total snr
        snr = np.sqrt(snr2)

        # scale distance appropiately
        args.gwdist = snr/args.snr

        print 'Scaled GW distance = {0} for SNR = {1}'.format(args.gwdist, args.snr)
    
    # make residuals
    for ct, p in enumerate(pp):

        inducedRes = (PALutils.createResiduals(psr[ct], np.pi/2-args.gwdec, args.gwra, \
                        args.gwchirpmass, args.gwdist, args.gwfreq, args.gwphase, \
                        args.gwpolarization, args.gwinc, psrTerm=pterm))

        # add to site arrival times of pulsar
        p.stoas[:] += np.longdouble(inducedRes/86400)

# refit
for p in pp:
    p.fit(iters=10)


# write tim file if option is given
if args.tim is not None:

    # make directory if it doesn't exist
    if not os.path.exists(args.tim):
        try:
            os.makedirs(args.tim)
            print 'Making output tim file directory {0}'.format(args.tim)
        except OSError:
            pass

    for ct,p in enumerate(pp):

        pname = timFile[ct].split('/')[-1].split('.')[0]
        p.savetim(args.tim + '/' + pname + '_sim.tim')

#TODO: add this for all injected sources

# write data to hdf5 file
for ct, key in enumerate(pulsargroup):

    # get residual group
    res = pulsargroup[key]['residuals']

    # write
    if pulsargroup[key]['pname'].value == pp[ct].name:

        # add simulated residuals
        res[...] = np.double(pp[ct].residuals())

    else:
        raise IOError, 'hdf5 pulsar name and tempo2 name do not agree!'

# add injection group to hdf5 file
datagroup = pfile['Data']
if "Injection" in datagroup:
    del datagroup['Injection']
    injectiongroup = datagroup.create_group('Injection')
else:
    injectiongroup = datagroup.create_group('Injection')

# add injection parameters
injectiongroup.create_dataset('GWRA', data = args.gwra)
injectiongroup.create_dataset('GWDEC', data = args.gwdec)
injectiongroup.create_dataset('GWPHASE', data = args.gwphase)
injectiongroup.create_dataset('GWINC', data = args.gwinc)
injectiongroup.create_dataset('GWPOLARIZATION', data = args.gwpolarization)
injectiongroup.create_dataset('GWCHIRPMASS', data = args.gwchirpmass)
injectiongroup.create_dataset('GWDISTANCE', data = args.gwdist)
injectiongroup.create_dataset('GWFREQUENCY', data = args.gwfreq)


# close hdf5 file
pfile.close()

print 'Done.'
