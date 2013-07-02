#!/usr/bin/env python

# Create Simulated Datasets by either injecting signals into real data or simulating ideal datasets.

#TODO: Add lots more options, for now we just have injections of continuous signals into real data

from __future__ import division
import numpy as np
import libstempo as t2
import PALutils
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
parser.add_argument('--outDir', dest='outDir', action='store', type=str, default='./',
                   help='Full path to output directory (default = ./)')
parser.add_argument('--single', dest='single', action='store_true', default=True,
                   help='Add single source? (default = True)')
parser.add_argument('--gwra', dest='gwra', action='store', type=float, default=1.0,
                   help='GW Right Ascension (default = 1.0 radian)')
parser.add_argument('--gwdec', dest='gwdec', action='store', type=float, default=0.5,
                   help='GW Declination (default = 0.5 radian)')
parser.add_argument('--gwinc', dest='gwinc', action='store', type=float, default=0.5,
                   help='GW inclination angle (default = 0.5 radian)')
parser.add_argument('--gwphase', dest='gwphase', action='store', type=float, default=0.5,
                   help='GW initial phase (default = 0.5 radian)')
parser.add_argument('--gwpolarization', dest='gwpolarization', action='store', type=float, default=0.5,
                   help='GW polarization angle (default = 0.5 radian)')
parser.add_argument('--gwchirpmass', dest='gwchirpmass', action='store', type=float, default=5e8,
                   help='GW chirp mass (default = 5e8 Solar Masses)')
parser.add_argument('--gwmass1', dest='gwmass1', action='store', type=float, default=None,
                   help='GW SMBMB mass 1 (default = None)')
parser.add_argument('--gwmass2', dest='gwmass2', action='store', type=float, default=None,
                   help='GW SMBMB mass 2 (default = None)')
parser.add_argument('--gwdist', dest='gwdist', action='store', type=float, default=100,
                   help='GW luminosity distance (default = 100 Mpc)')
parser.add_argument('--gwredshift', dest='gwredshift', action='store', type=float, default=None,
                   help='GW redshift of source (default = None)')
parser.add_argument('--gwfreq', dest='gwfreq', action='store', type=float, default=1e-8,
                   help='GW initial frequency (default = 1e-8 Hz)')
parser.add_argument('--gwb', dest='gwb', action='store_true', default=True,
                   help='Add stochastic background? (default = True)')
parser.add_argument('--gwbAmp', dest='gwbAmp', action='store', type=float, default=5e-15,
                   help='GWB amplitude (default = 5e-15)')
parser.add_argument('--gwbIndex', dest='gwbIndex', action='store', type=float, default=4.33,
                   help='GWB amplitude (default = 4.33)')
parser.add_argument('--noise', dest='noise', action='store_true', default=False,
                   help='Add noise based on real data values? (default = True)')

# parse arguments
args = parser.parse_args()

# compute chirp mass if mass m1 and m2 are given
if args.gwmass1 is not None and args.gwmass2 is not None:
    args.gwchirpmass = (args.gwmass1*args.gwmass2)**(3/5) / (args.gwmass1+args.gwmass2)**(1/5) 

# compute luminosity distance from redshift if given
if args.gwredshift is not None:
    args.gwdist = PALutils.computeLuminosityDistance(args.gwredshift)

# make copy of h5file TODO: maybe better way than using os.system?
h5copy = args.h5file.split('.')[0] + '_sim.hdf5'
print 'Saving file to {0}'.format(h5copy)
os.system('cp {0} {1}'.format(args.h5file, h5copy))

# import hdf5 file
pfile = h5.File(h5copy, 'r+')

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


################## SIMULATED RESIDUALS ########################

# create idealized TOAs
for p in pp:
    p.stoas[:] -= p.residuals()/86400
    p.fit()

# add single source
if args.single:

    print 'Simulating single source'

    for ct, p in enumerate(pp):

        inducedRes = (PALutils.createResiduals(psr[ct], np.pi/2-args.gwdec, args.gwra, args.gwchirpmass, \
                                args.gwdist, args.gwfreq, args.gwphase, args.gwpolarization, \
                                args.gwinc))

        # add to site arrival times of pulsar
        p.stoas[:] += np.longdouble(inducedRes/86400)

# add gwb
if args.gwb:
    
    print 'Simulating GWB with Amp = {0} and gamma = {1}'.format(args.gwbAmp, args.gwbIndex)

    inducedRes = PALutils.createGWB(psr, args.gwbAmp, args.gwbIndex)
        
    # add to site arrival times of pulsar
    for ct, p in enumerate(pp):
        p.stoas[:] += np.longdouble(inducedRes[ct]/86400)

# add noise based on values in hdf5 file
if args.noise:
    
    print 'Simulating noise based on values in hdf5 file'

    for ct, p in enumerate(psr):

        # get values from hdf5 file
        Amp = pfile['Data']['Pulsars'][p.name]['Amp'].value
        gam = pfile['Data']['Pulsars'][p.name]['gam'].value
        efac = pfile['Data']['Pulsars'][p.name]['efac'].value
        equad = pfile['Data']['Pulsars'][p.name]['equad'].value

        try:
            fH = pfile['Data']['Pulsars'][p.name]['fH'].value
        except KeyError:
            fH = None


        tm = PALutils.createTimeLags(p.toas, p.toas)

        red = PALutils.createRedNoiseCovarianceMatrix(tm, Amp, gam, fH=fH)
        white = PALutils.createWhiteNoiseCovarianceMatrix(p.err, efac, equad)

        cov = red + white

        # cholesky decomp
        L = np.linalg.cholesky(cov)

        # zero mean unit variance 
        w = np.random.randn(p.ntoa)

        # get induced residuals
        inducedRes = np.dot(L, w)

        pp[ct].stoas[:] += np.longdouble(inducedRes/86400)

# no options, just white noise
if args.noise == False:

    print 'No injection, just using white noise based on error bars'
    # add to site arrival times of pulsar
    for p in pp:
        p.stoas[:] += p.toaerrs*1e-6 * np.random.randn(p.nobs)/86400


# refit
for p in pp:
    p.fit(iters=10)

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
