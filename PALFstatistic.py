#!/usr/bin/env python

# Run the F-statistic search code as defined in Ellis, Siemens, Creighton (2012)

from __future__ import division
import numpy as np
import PALLikelihoods
import PALutils
import PALpulsarInit
import h5py as h5
import optparse
import os

parser = optparse.OptionParser(description = 'Run F-statistic search as defined in Ellis, Siemens, Creighton (2012)')

# options
parser.add_option('--h5File', dest='h5file', action='store', type=str,
                   help='Full path to hdf5 file containing PTA data')
parser.add_option('--outDir', dest='outDir', action='store', type=str, default='./',
                   help='Full path to output directory (default = ./)')
parser.add_option('--runFpStat', dest='fpFlag', action='store_true', default=True,
                   help='Option to run Incoherent Fp Statistic (default = True)')
parser.add_option('--runFeStat', dest='feFlag', action='store_true', default=False,
                   help='Option to run Earth term Fe Statistic (default = False)')
parser.add_option('--fhigh', dest='fhigh', action='store', type=float, default=5e-7,
                   help='Highest frequency to search (default = 5e-7 Hz)')
parser.add_option('--nfreqs', dest='nfreqs', action='store', type=int, default=200,
                   help='Number of frequencies to search (default = 200)')
parser.add_option('--logsample', dest='logsample', action='store_true', default=False,
                   help='Sample in log frequency (default = False)')
parser.add_option('--best', dest='best', action='store', type=int, default=0,
                   help='Only use best pulsars based on weighted rms (default = 0, use all)')


# parse arguments
(args, x) = parser.parse_args()

##### Begin Code #####

print 'Reading in HDF5 file' 

# import hdf5 file
pfile = h5.File(args.h5file, 'r')

# define the pulsargroup
pulsargroup = pfile['Data']['Pulsars']

# fill in pulsar class
psr = [PALpulsarInit.pulsar(pulsargroup[key],addNoise=True) for key in pulsargroup]

if args.best != 0:
    print 'Using best {0} pulsars'.format(args.best)
    rms = np.array([p.rms() for p in psr])
    ind = np.argsort(rms)

    psr = [psr[ii] for ii in ind[0:args.best]]

    for p in psr:
        print 'Pulsar {0} has {1} ns weighted rms'.format(p.name,p.rms()*1e9)


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


# set up frequencies to search with low frequency 1/2*Tmax and upper frequency 5e-7

Tmax = np.array([p.toas.max() - p.toas.min() for p in psr]).max()
flow = 0.5/Tmax
fhigh = args.fhigh

# set up frequency vector
if args.logsample:
    f = np.logspace(np.log10(flow), np.log10(fhigh), args.nfreqs)
else:
    f = np.linspace(flow, fhigh, args.nfreqs)

# carry out Fp search
if args.fpFlag:

    print 'Beginning Fp Search with {0} pulsars, with frequency range {1} -- {2}'.format(npsr, f[0], f[-1])
    
    fpstat = np.zeros(args.nfreqs)
    for ii in range(args.nfreqs):

        fpstat[ii] = PALLikelihoods.fpStat(psr, f[ii])


    print 'Done Search. Computing False Alarm Probability'
    
    # single template FAP
    pf = np.array([PALutils.ptSum(npsr, fpstat[ii]) for ii in range(np.alen(f))])

    # get total false alarm probability with trials factor
    pfT = 1 - (1-pf)**np.alen(f)

    # write results to file
    if not os.path.exists(args.outDir):
        os.makedirs(args.outDir)

    # get filename from hdf5 file
    fname = args.outDir + '/' + args.h5file.split('/')[-1].split('.')[0] + '.txt'
    fout = open(fname, 'w')
    print 'Writing results to file {0}'.format(fname)
    for ii in range(np.alen(f)):

        fout.write('%g %g %g\n'%(f[ii], fpstat[ii], pfT[ii]))
    
    fout.close()

    #TODO: add Fe statistic
