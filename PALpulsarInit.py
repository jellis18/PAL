#!/usr/bin/env python

import numpy as np
import sys, os, glob
import h5py as h5
import libstempo as t2
import PALutils as pal_utils

# class defninitions to initialize PTA data in the hdf5 file format.

class PulsarFile(object):

    def __init__(self, filename=None):
        # Open the hdf5 file?
        self.filename = filename

    def __del__(self):
        # Delete the instance, and close the hdf5 file?
        pass

    def addpulsar(self, parfile, timfile, DMOFF=None, dailyAverage=False):

        """
        Add another pulsar to the HDF5 file, given a tempo2 par and tim file.

        @param parfile: tempo2 par file
        @param timfile: tempo2 tim file
        @param DMOFF: Option to turn off DMMODEL fitting
        @param dailyAverage: Option to perform daily averaging to reduce the number
                             of points by consructing daily averaged TOAs that have
                             one residual per day per frequency band. (This has only
                             been tested on NANOGrav data thus far.)

        """

        # Check whether the two files exist
        if not os.path.isfile(parfile) or not os.path.isfile(timfile):
            raise IOError, "Cannot find parfile (%s) or timfile (%s)!" % (parfile, timfile)
        assert(self.filename != None), "ERROR: HDF5 file not set!"

        # 'a' means: read/write if exists, create otherwise
        self.h5file = h5.File(self.filename, 'a')

        if "Model" in self.h5file:
            self.h5file.close()
            self.h5file = None
            raise IOError, "model already available in '%s'. Refusing to add data" % (self.filename)

        # Create the data subgroup if it does not exist
        if "Data" in self.h5file:
            datagroup = self.h5file["Data"]
        else:
            datagroup = self.h5file.create_group("Data")

        # Load pulsar data from the JPL Cython tempo2 library
        t2pulsar = t2.tempopulsar(parfile, timfile)

        # do multiple fits
        t2pulsar.fit(iters=10)

        # turn off DMMODEL fitting
        if DMOFF is not None:
            t2pulsar['DMMODEL'].fit = False

        # refit 5 times to make sure we are converged
        t2pulsar.fit(iters=5)

        # Create the pulsar subgroup if it does not exist
        if "Pulsars" in datagroup:
            pulsarsgroup = datagroup["Pulsars"]
        else:
            pulsarsgroup = datagroup.create_group("Pulsars")

        # Look up the name of the pulsar, and see if it exist
        if t2pulsar.name in pulsarsgroup:
            self.h5file.close()
            raise IOError, "%s already exists in %s!" % (t2pulsar.name, self.filename)

        pulsarsgroup = pulsarsgroup.create_group(t2pulsar.name)

        # Read the data from the tempo2 structure.
        designmatrix = np.double(t2pulsar.designmatrix())
        residuals = np.double(t2pulsar.residuals())    
        toas = np.double(t2pulsar.toas())
        errs = np.double(t2pulsar.toaerrs*1e-6)
        pname = t2pulsar.name

        try:    # if tim file has frequencies
            freqs = np.double(t2pulsar.freqs)
        except AttributeError: 
            freqs = 0

        try:    # if tim file has frequency band flags
            bands = t2pulsar.flags['B']
        except KeyError:
            bands = 0

        # if doing daily averaging
        if dailyAverage:

            # get average quantities
            toas, qmatrix, errs, dmatrix, freqs, bands = pal_utils.dailyAverage(t2pulsar)

            # construct new daily averaged residuals and designmatrix
            residuals = np.dot(qmatrix, residuals)
            designmatrix = np.dot(qmatrix, dmatrix)

        
        # Write the TOAs, residuals, and uncertainties.
        spd = 24.0*3600     # seconds per day
        pulsarsgroup.create_dataset('TOAs', data = toas*spd)           # days (MJD) * sec per day
        pulsarsgroup.create_dataset('residuals', data = residuals)     # seconds
        pulsarsgroup.create_dataset('toaErr', data = errs)             # seconds
        pulsarsgroup.create_dataset('freqs', data = freqs*1e6)             # Hz
        pulsarsgroup.create_dataset('bands', data = bands)             # Hz

        # add tim and par file paths
        pulsarsgroup.create_dataset('parFile', data = parfile)             # string
        pulsarsgroup.create_dataset('timFile', data = timfile)             # string


        # Write the full design matrix
        pulsarsgroup.create_dataset('designmatrix', data = designmatrix)

        # Obtain the timing model parameters
        tmpname = np.array(t2pulsar.pars)
        tmpvalpre = np.double([t2pulsar.prefit[parname].val for parname in t2pulsar.pars])
        tmpvalpost = np.double([t2pulsar[parname].val for parname in t2pulsar.pars])
        tmperrpre = np.double([t2pulsar.prefit[parname].err for parname in t2pulsar.pars])
        tmperrpost = np.double([t2pulsar[parname].err for parname in t2pulsar.pars])


        # Write the timing model parameter (TMP) descriptions
        pulsarsgroup.create_dataset('pname', data=pname)            # pulsar name
        pulsarsgroup.create_dataset('tmp_name', data=tmpname)       # TMP name
        pulsarsgroup.create_dataset('tmp_valpre', data=tmpvalpre)   # TMP pre-fit value
        pulsarsgroup.create_dataset('tmp_valpost', data=tmpvalpost) # TMP post-fit value
        pulsarsgroup.create_dataset('tmp_errpre', data=tmperrpre)   # TMP pre-fit error
        pulsarsgroup.create_dataset('tmp_errpost', data=tmperrpost) # TMP post-fit error

        # Close the hdf5 file
        self.h5file.close()

    # add inverse covariance matrix G (G.T C G)^-1 G.T
    def addInverseCovFromNoiseFile(self, parfile, timfile, noisefile, DMOFF=None, dailyAverage=False):
        """
        
        Add noise covariance matrix after timing model subtraction.

        """

        # Check whether the two files exist
        if not os.path.isfile(parfile) or not os.path.isfile(timfile):
            raise IOError, "Cannot find parfile (%s) or timfile (%s)!" % (parfile, timfile)
        assert(self.filename != None), "ERROR: HDF5 file not set!"

        # 'a' means: read/write if exists, create otherwise
        self.h5file = h5.File(self.filename, 'a')

        # Create the data subgroup if it does not exist
        if "Data" in self.h5file:
            datagroup = self.h5file["Data"]
        else:
            raise IOError, "Cannot add noise parameters if Data group does not exist!"

        # Load pulsar data from the JPL Cython tempo2 library
        t2pulsar = t2.tempopulsar(parfile, timfile)
        
        # turn off DMMODEL fitting
        if DMOFF is not None:
            t2pulsar['DMMODEL'].fit = False

        # refit 5 times to make sure we are converged
        t2pulsar.fit(iters=5)

        # Create the pulsar subgroup if it does not exist
        if "Pulsars" in datagroup:
            pulsarsgroup = datagroup["Pulsars"]
        else:
            raise IOError, "Cannot add noise parameters if pulsar group does not exist!"

        # Look up the name of the pulsar, and see if it exist
        if t2pulsar.name in pulsarsgroup:
            pass
        else:
            raise IOError, "%s must already exists in %s to add noise parameters!"\
                    % (t2pulsar.name, self.filename)

        pulsarsgroup = pulsarsgroup[t2pulsar.name]

        # first create G matrix from design matrix and toas
        designmatrix = np.double(t2pulsar.designmatrix())
        toas = np.double(t2pulsar.toas()*86400)
        errs = np.double(t2pulsar.toaerrs*1e-6)

        # if doing daily averaging
        if dailyAverage:

            # get average quantities
            toas, qmatrix, errs, dmatrix, freqs, bands = pal_utils.dailyAverage(t2pulsar)

            # construct new daily averaged residuals and designmatrix
            toas *= 86400
            designmatrix = np.dot(qmatrix, dmatrix)
        
        G = pal_utils.createGmatrix(designmatrix)

        # create matrix of time lags
        tm = pal_utils.createTimeLags(toas, toas, round=True)

        # now read noise file to get model and parameters
        file = open(noisefile,'r')

        fH = None
        tau = None
        DMAmp = None
        DMgam = None
 
        for line in file.readlines():
            # default parameters for different models other than pure PL
            key=line.split()[0] 
            # get amplitude
            if "Amp" == key:
                Amp = float(line.split()[-1])

            # get spectral index
            elif "gam" == key:
                gam = float(line.split()[-1])
            
            # get efac
            elif "efac" == key:
                efac = float(line.split()[-1])
            
            # get quad
            elif "equad" == key:
                equad = float(line.split()[-1])
            
            # get high frequency cutoff if available
            elif "fH" == key:
                fH = float(line.split()[-1])
            
            # get correlation time scale if available
            elif "tau" == key:
                tau = float(line.split()[-1])

            # get DM Amplitude if available
            elif "DMAmp" == key:
                DMAmp = float(line.split()[-1])

            # get DM Spectral Index if available
            elif "DMgam" == key:
                DMgam = float(line.split()[-1])

        # cosstruct red and white noise covariance matrices
        red = pal_utils.createRedNoiseCovarianceMatrix(tm, Amp, gam, fH=fH)
        white = pal_utils.createWhiteNoiseCovarianceMatrix(errs, efac, equad, tau=tau)

        # construct post timing model marginalization covariance matrix
        cov = red + white
        pcov = np.dot(G.T, np.dot(cov, G))

        # finally construct "inverse"
        invCov = np.dot(G, np.dot(np.linalg.inv(pcov), G.T))

        # create dataset for inverse covariance matrix
        pulsarsgroup.create_dataset('invCov', data = invCov) 

        # create dataset for G matrix
        pulsarsgroup.create_dataset('Gmatrix', data = G) 

        # record noise parameter values
        pulsarsgroup.create_dataset('Amp', data = Amp)
        pulsarsgroup.create_dataset('gam', data = gam)
        pulsarsgroup.create_dataset('efac', data = efac)
        pulsarsgroup.create_dataset('equad', data = equad)
        if fH is not None:
            pulsarsgroup.create_dataset('fH', data = fH)
        if tau is not None:
            pulsarsgroup.create_dataset('tau', data = tau)
        if DMAmp is not None:
            pulsarsgroup.create_dataset('DMAmp', data = DMAmp)
        if DMgam is not None:
            pulsarsgroup.create_dataset('DMgam', data = DMgam)


        # Close the hdf5 file
        self.h5file.close()

    def addDistance(self, parfile, timfile, distfile):
        """
        
        Add puslar distance and uncertainty from file.

        """

        # Check whether the two files exist
        if not os.path.isfile(parfile) or not os.path.isfile(timfile):
            raise IOError, "Cannot find parfile (%s) or timfile (%s)!" % (parfile, timfile)
        assert(self.filename != None), "ERROR: HDF5 file not set!"

        # 'a' means: read/write if exists, create otherwise
        self.h5file = h5.File(self.filename, 'a')

        # Create the data subgroup if it does not exist
        if "Data" in self.h5file:
            datagroup = self.h5file["Data"]
        else:
            raise IOError, "Cannot add noise parameters if Data group does not exist!"

        # Load pulsar data from the JPL Cython tempo2 library
        t2pulsar = t2.tempopulsar(parfile, timfile)

        # Create the pulsar subgroup if it does not exist
        if "Pulsars" in datagroup:
            pulsarsgroup = datagroup["Pulsars"]
        else:
            raise IOError, "Cannot add noise parameters if pulsar group does not exist!"

        # Look up the name of the pulsar, and see if it exist
        if t2pulsar.name in pulsarsgroup:
            pass
        else:
            raise IOError, "%s must already exists in %s to add noise parameters!"\
                    % (t2pulsar.name, self.filename)

        pulsarsgroup = pulsarsgroup[t2pulsar.name]

        # find distance and uncertainty from file
        dfile = open(distfile,'r')

        dist = None
        distErr = None
        for line in dfile.readlines():

            if t2pulsar.name in line or 'J' + t2pulsar.name in line or 'B' + t2pulsar.name in line:
                dist = float(line.split()[1])
                distErr = float(line.split()[2])

        # add distance to file if found
        if dist is not None and distErr is not None:
            pulsarsgroup.create_dataset('dist', data=dist)
            pulsarsgroup.create_dataset('distErr', data=distErr)
        else:
            print "Cannot find PSR {0} in distance file {1}. Using dist = 1kpc with 10% uncertainty".format(t2pulsar.name, distfile)
            
            pulsarsgroup.create_dataset('dist', data=1.0)
            pulsarsgroup.create_dataset('distErr', data=0.1)
            
        self.h5file.close()
        dfile.close()




class pulsar(object):

    """
    Pulsar object that is initialized with a pulsargroup from the hdf5 file

    TODO: add noise estimation values and dictionary for efacs for different backends
    
    """

    def __init__(self,pulsargroup, addNoise=False, addGmatrix=True):


        # loop though keys in pulsargroup and fill in psr attributes that are needed for GW analysis
        self.dist = None
        self.distErr = None

        for key in pulsargroup:

            # look for TOAs
            if key == "TOAs":
                self.toas = pulsargroup[key].value

            # residuals
            elif key == "residuals":
                self.res = pulsargroup[key].value

            # toa error bars
            elif key == "toaErr":
                self.err = pulsargroup[key].value
            
            # frequencies in Hz
            elif key == "freqs":
                self.freqs = pulsargroup[key].value
            
            # design matrix
            elif key == "designmatrix":
                self.dmatrix = pulsargroup[key].value
                self.ntoa, self.nfit = self.dmatrix.shape
            
            # design matrix
            elif key == "pname":
                self.name = pulsargroup[key].value
            
            # pulsar distance in kpc
            elif key == "dist":
                self.dist = pulsargroup[key].value 
            
            # pulsar distance uncertainty in kpc
            elif key == "distErr":
                self.distErr = pulsargroup[key].value 

            # right ascension and declination
            elif key == 'tmp_name':
                par_names = list(pulsargroup[key].value)
                for ct,name in enumerate(par_names):

                    # right ascension and phi
                    if name == "RAJ":
                        self.ra = pulsargroup["tmp_valpost"].value[ct]
                        self.phi = self.ra
                    
                    # right ascension
                    if name == "DECJ":
                        self.dec = pulsargroup["tmp_valpost"].value[ct]
                        self.theta = np.pi/2 - self.dec

            # inverse covariance matrix
            elif key == "invCov":
                if addNoise:
                    self.invCov = pulsargroup[key].value

            # G matrix
            elif key == "Gmatrix":
                if addGmatrix:
                    self.G = pulsargroup[key].value

        if self.dist is None:
            print 'WARNING: No distance info, using d = 1 kpc'
            self.dist = 1.0

        if self.distErr is None:
            print 'WARNING: No distance error info, using sigma_d = 0.1 kpc'
            self.distErr = 0.1


    def rms(self):

        """
        Return weighted RMS in seconds

        """

        W = 1/self.err**2

        return np.sqrt(np.sum(self.res**2*W)/np.sum(W))

   
def createPulsarHDF5File(parDir, timDir, noiseDir=None, distFile=None, \
                         saveDir=None, DMOFF=None, dailyAverage=False):
    """

    Utility function to fill in our hdf5 pulsar file

    @param parDir: Full path to par files
    @param timDir: Full path to tim files
    @param noiseDir: Full path to noise files [optional]
    @param distFile: Full path to pulsar distance file [optional]
    @param saveDir: Full path to output hdf5 file [optional]. 
                    Save pulsar.hdf5 to current directory if 
                    not specified.
    @param DMOFF: Option to turn off DMMODEL fitting if applicable 
    @param dailyAverage: Option to to daily averaging  
    
    """

    # get par, tim and noise files
    parFile = glob.glob(parDir + '/*.par')
    timFile = glob.glob(timDir + '/*.tim')

    # sort 
    parFile.sort()
    timFile.sort()

    if noiseDir is not None:
        noiseFile = glob.glob(noiseDir + '/*.noise')
        noiseFile.sort()

    if noiseDir is not None:
        # make sure pulsar names are in correct order
        # TODO: is this a very round about way to do this?
        index = []
        for ct,p in enumerate(parFile):

            # get prefix of pulsar name from par file
            prefix = p.split('/')[-1].split('.')[0].split('_')[0][0]
            if prefix == 'J' or prefix =='B':
                pname = p.split('/')[-1].split('.')[0].split('_')[0][1:]
            else:
                pname = p.split('/')[-1].split('.')[0].split('_')[0]

            # do same for noise file
            prefix = noiseFile[ct].split('/')[-1].split('.')[0].split('_')[0][0]
            if prefix == 'J' or prefix =='B':
                nname = noiseFile[ct].split('/')[-1].split('.')[0].split('_')[0][1:]
            else:
                nname = noiseFile[ct].split('/')[-1].split('.')[0].split('_')[0]
           
            if pname == nname:
                index.append(ct)
            else:
                for ii in range(len(parFile)):
                    # get prefix of pulsar name from par file
                    prefix = parFile[ct].split('/')[-1].split('.')[0].split('_')[0][0]
                    if prefix == 'J' or prefix =='B':
                        pname = parFile[ct].split('/')[-1].split('.')[0].split('_')[0][1:]
                    else:
                        pname = parFile[ct].split('/')[-1].split('.')[0].split('_')[0]

                    # do same for noise file
                    prefix = noiseFile[ii].split('/')[-1].split('.')[0].split('_')[0][0]
                    if prefix == 'J' or prefix =='B':
                        nname = noiseFile[ii].split('/')[-1].split('.')[0].split('_')[0][1:]
                    else:
                        nname = noiseFile[ii].split('/')[-1].split('.')[0].split('_')[0]
                    
                    if nname == pname:
                        index.append(ii)

        # reorder noise files
        noiseFile = [noiseFile[ii] for ii in index]

    if noiseDir is not None:
        for ii in range(len(parFile)):
            print parFile[ii].split('/')[-1],\
                    timFile[ii].split('/')[-1], \
                    noiseFile[ii].split('/')[-1]

    # check to make sure we have same number of files
    if len(parFile) != len(timFile):
        raise IOError, "Must have same number of par and tim files!"

    # initialize pulsar hdf5 class
    if saveDir is None:
        saveDir = 'pulsar.hdf5'

    pulsar = PulsarFile(saveDir)

    # add pulsars
    [pulsar.addpulsar(parFile[ii], timFile[ii], DMOFF=DMOFF, dailyAverage=dailyAverage) \
            for ii in range(len(parFile))]

    # add noise covaraiance matrices
    if noiseDir is not None:
        [pulsar.addInverseCovFromNoiseFile(parFile[ii], timFile[ii], noiseFile[ii], DMOFF=DMOFF, \
                                           dailyAverage=dailyAverage) for ii in range(len(parFile))]

    # add pulsar distances and uncertainties
    if distFile is not None:
        [pulsar.addDistance(parFile[ii], timFile[ii], distFile) \
         for ii in range(len(parFile))]

    # done
if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()

    parser.add_option('--parDir', dest='parDir', action='store', type=str,
                       help='Full path to par files (required)', default = './')
    parser.add_option('--timDir', dest='timDir', action='store', type=str, 
                       help='Full path to tim files (required)', default='./')
    parser.add_option('--noiseDir', dest='noiseDir', action='store', type=str, default=None,
                       help='Full path to noise files')
    parser.add_option('--outFile', dest='outFile', action='store', type=str, 
                       help='Full path to output filename (required)')
    parser.add_option('--distFile', dest='distFile', action='store', type=str, default=None,
                       help='Full path to pulsar distance file')
    parser.add_option('--DMOFF', dest='DMOFF', action='store', type=str, default=None,
                       help='Turn on DMMODEL fitting')

    (args, x) = parser.parse_args()

    createPulsarHDF5File(args.parDir, args.timDir, args.noiseDir, args.distFile, \
                             args.outFile, args.DMOFF, dailyAverage=False)



