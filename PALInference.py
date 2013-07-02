# Defines various classes for runState of MCMC and jump proposal 
# distribution statistics as well as functional definitions of many
# different proposal distributions

from __future__ import division
import numpy as np
import sys,os

class PALInferenceVariables(object):
    """
    Class that contains the MCMC (or nested sampling) variables
    and corresponding information about those variables.

    @param name: The name of the parameter [string], the standard names are:
                 Right Ascension:       ra              [0, 2*pi]
                 Declination:           dec             [-pi/2, pi/2]
                 Initial GW phase:      phase_0         [0, 2*pi]
                 Polarization Angle:    polarization    [-pi/4, pi/4]
                 Inclination Angle:     inclination     [0, pi/2]
                 Luminosity Distance:   dist            [0, 1e4] (Mpc)
                 Chirp Mass:            chirp_mass      [1e7, 1e10] (Solar Masses)
                 Intitial GW frequency: frequency       [1e-10, 1e-7] (Hz)
                 Pulsar Distance:       pdist           [0.01, 10.0] (kpc)
                

    @param value: The value of the parameter at the current iteration [float]
                  All values are floats, however the pulsar distance is a 
                  vector of length N_psr
    
    
    @param vary: The type of parameter [string]
                 PTA_INFERENCE_LINEAR:      A linear parameter that only has a max and min value
                 PTA_INFERENCE_CYCLIC:      A cyclic parameter that varies from 0 -> 2pi
                 PTA_INFERENCE_REFLECTIVE:  A reflective parameter that is not cyclic
                 PTA_INFERENCE_FIXED:       A fixed paramter that is not varied
    
    @param proposed: The proposed value of the parameter [float]
    
    @param sigma: The jump size for the proposal distibution [float]

    @param max: Maximum value of parameter (will default to value above if None)

    @param min: Minimum value of parameter (will default to value above if None)

    """

    def __init__(self, name, value, vary, proposed=None, sigma=None, max=None, min=None):

        self.name = name
        self.value = value
        self.proposed = proposed
        self.sigma = sigma
        self.vary = vary

        # set default max and min values if None
        if self.min is None and self.max is None:

            if self.name == 'ra':
                self.min = 0.0
                self.max = 2*np.pi
            
            elif self.name == 'dec':
                self.min = np.pi
                self.max = -np.pi
            
            elif self.name == 'phase_0':
                self.min = 0.0
                self.max = 2*np.pi
            
            elif self.name == 'polarization':
                self.min = -np.pi/4
                self.max = np.pi/4
            
            elif self.name == 'inclination':
                self.min = 0.0
                self.max = np.pi/2

            elif self.name == 'inclination':
                self.min = 0.0
                self.max = np.pi/2
            
            elif self.name == 'dist':
                self.min = 0.0
                self.max = 1e4
            
            elif self.name == 'chirp_mass':
                self.min = 1e7
                self.max = 1e10
            
            elif self.name == 'frequency':
                self.min = 1e-10
                self.max = 5e-6
            
            elif self.name == 'pdist':
                self.min = 0.01
                self.max = 10.0

        else:
            self.max = max
            self.min = min

 
    def cyclicReflectiveBounds(self):
        """
        Apply cyclic and reflective bounds to bring parameters back
        into prior range.

        """

        # cyclic parameters
        if self.vary == 'PTA_INFERENCE_CYCLIC':

            # mod parameter by 2*pi
            self.proposed = np.mod(self.proposed, 2*np.pi)

        # reflective parameter
        elif self.vary == 'PTA_INFERENCE_REFLECTIVE':
             
            # loop over until in range
            outRange = True
            while outRange:
                
                if self.proposed > self.max:
                    self.proposed = 2*self.max - self.proposed
                
                elif self.proposed < self.max:
                    self.proposed = 2*self.min - self.proposed
                
                else:
                    # in range
                    outRange = False


# create a dictioary of parameter variables
def PALInferenceMakeDictionary(PALInferenceVariables):
    """
    Function that reads in list of PALInferenceVariables objects and returns
    a Python dictionary with the parameter names and keys

    TODO: make this part of runState

    """
    
    pardict = {}
    for var in PALInferenceVariables:
        pardict[var.name] = var

    return pardict


class runState(object):
    """

    Run State object that holds all options and information about the MCMC
    (or nested sampling) run. Must be initialized before any other actions
    are taken in PALInference. 
    
    At the moment we have a different runState object for every temperature chain.

    """

    def __init__(self, lnlikeFunc, lnpriorFunc, ndim, temperature=None, nthreads=1, \
                 N=2e6, resumefile=None, pulsarPhaseSigma=None, covarianceJumpScale=None, \
                 propIter=0, cov=None, ntemps=1, Tmax=None, Tmin=1):

        self.lnlikeFunc = lnlikeFunc
        self.lnpriorFunc = lnpriorFunc
        self.ndim = ndim
        self.temperature = temperature
        self.nthreads = nthreads
        self.N = N
        self.resumefile = resumefile
        self.pulsarPhaseSigma = pulsarPhaseSigma
        self.covarianceJumpScale = covarianceJumpScale
        self.propIter = propIter
        self.ntemps = ntemps
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.cov = cov

        # initialize chain, lnprob and lnlike
        self.chain = np.zeros(N, ndim)
        self.lnprob = np.zeros(N)
        self.lnlike = np.zeros(N)

        # initialize acceptance rate
        self.acc = np.zeros(N)

