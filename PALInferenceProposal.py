#!/usr/bin/env python

from __future__ import division
import numpy as np
import sys,os


####################################################################
#                                                                  #
#                                                                  #
#                                                                  #
#              Jump Proposal Distribution functions.               #
#     All call the parameter dictionary, runState and psr object   #
#                                                                  #
#                                                                  #
#                                                                  #
####################################################################

                                                                     
def PALInferencePulsarPhaseJumpSmall(psr, runState, pardict):
    """
    Proposal distribution that makes small jumps the unwrapped
    phase \phi_p = \omega L (1-\cos\mu).

    """

    # record current params into proposed attribute
    for key in pardict:
        pardict[key].proposed = pardict[key].value

    # get temparature if doing parallel tempering
    if runState.temperature:
        T = runState.temperature

    # get relevant parameters
    freq = pardict['frequency'].value
    pdist = pardict['pdist'].value
    ra = pardict['ra'].value
    dec = pardict['dec'].value

    # put pulsar distance in correct units
    pdist *= utils.KPC2S

    # get cosMu
    cosMu = utils.calculateCosMu(psr)

    # construct pulsar phase
    phase_old = 2*np.pi*freq*pdist*(1-cosMu)

    # check for sigma parameter in runState
    if runState.pulsarPhaseSigma:
        sigma = runState.pulsarPhaseSigma
    else:
        sigma = 0.2    # radians

    # account for different temperature chains
    sigma *= np.sqrt(T)

    # gaussian jump 
    phase_new = phase_old + np.random.randn(len(pdist))*sigma

    # solve for new pulsar distances from phase_new
    L_new = phase_new/(2*np.pi*freq*(1-cosMu))

    # convert back to Kpc
    L_new /= utils.KPC2S

    # record new values into pardict
    pardict['pdist'].proposed = L_new



def PALInferencePulsarPhaseJumpBig(psr, runState, pardict):
    """
    Proposal distribution that makes large jumps the unwrapped
    phase \phi_p = \omega L (1-\cos\mu). All jumps are modded
    such that the jump is a multiple of 2*pi.

    We accomplish this by choosing to jump in phase proportional 
    to \omega \Delta L (1-\cos\mu), where \Delta L is the 1-sigma
    uncertainty on the distance measurement

    """

    # record current params into proposed attribute
    for key in pardict:
        pardict[key].proposed = pardict[key].value

    # get relevant parameters
    freq = pardict['frequency'].value
    pdist = pardict['pdist'].value
    ra = pardict['ra'].value
    dec = pardict['dec'].value
    deltaL = np.array([p.distErr for p in psr])

    # put pulsar distance in correct units
    pdist *= utils.KPC2S

    # get cosMu
    cosMu = utils.calculateCosMu(psr, ra, dec)

    # construct pulsar phase
    phase_old = 2*np.pi*freq*pdist*(1-cosMu)

    # gaussian jump 
    phase_jump = np.random.randn(len(pdist))*deltaL*freq(1-cosMu)

    # make jump multiple of 2 pi
    phase_jump = np.array([int(phase_jump[ii]) \
                    for ii in range(len(pdist))])

    # new phase
    phase_new = phase_old + 2*np.pi*phase_jump

    # solve for new pulsar distances from phase_new
    L_new = phase_new/(2*np.pi*freq*(1-cosMu))

    # convert back to Kpc
    L_new /= utils.KPC2S

    # record new values into pardict
    pardict['pdist'].proposed = L_new


def PALInferenceMassDistanceJump(psr, runStata, pardict):
    """
    Jumps in fixed quantity \mathcal{M}^{5/3}/D_L

    """
    # record current params into proposed attribute
    for key in pardict:
        pardict[key].proposed = pardict[key].value

    # get temparature if doing parallel tempering
    if runState.temperature:
        T = runState.temperature

    
    # get mass and distance
    mc = pardict['chirp_mass'].value
    dl = pardict['dist'].value

    # conserved quantity
    const = (mc*utils.SOLAR2S)**(5/3) / (dl*utils.MPC2S)

    # jump in distance
    new_dl = dl + np.random.randn()*10.0

    # get corresponding mass
    new_mc = const*dl*utils.MPC2S/(utils.SOLAR2S)**(5/3)

    # update pardict
    pardict['chirp_mass'].proposed = new_mc
    pardict['dist'].proposed = new_dl


def PALInferenceSingleProposal(psr, runState, pardict):
    """
    
    Proposal to update single parameter. Does not update pulsar
    distance.

    TODO: make smarter about jumps in log
    
    """

    # set large jumps
    big_sigma = 1.0

    # jump 10 times further every 1000 iterations
    if np.random.rand() < 1e-3: big_sigma = 10.0
    
    # jump 100 times further every 10000 iterations
    if np.random.rand() < 1e-4: big_sigma = 100.0

    # record current params into proposed attribute
    for key in pardict:
        pardict[key].proposed = pardict[key].value

    # get temparature if doing parallel tempering
    if runState.temperature:
        T = runState.temperature

    # draw parameter at random, that is not pulsar distance
    ndim = len(pardict.keys()) - 1

    found = False
    while found = False:
        index = np.random.randint(low=0, high=ndim)

        # check to make sure parameter is not pulsar distance or fixed
        if pardict.keys()[index].name != 'pdist' and pardict.keys()[index].vary != 'PTA_INFERENCE_FIXED':
            
            # choose param
            param_name = pardict.keys()[index].name
            found = True

    # check pardict to see if jump size is set, if not use default sizes
    logJump = False
    if pardict[param_name].sigma:
        sigma = pardict[param_name].sigma 

    else:

        if param_name == 'ra' or param_name == 'dec':
            sigma = 0.2
        
        elif param_name == 'phase_0' or param_name == 'polarization' or param_name == 'inclination':
            sigma = 0.3

        # will default to jumping in log
        elif param_name == 'chirp_mass':    
            sigma = 0.1
            logJump = True
        
        # will default to jumping in log
        elif param_name == 'dist':    
            sigma = 0.2
            logJump = True
            
        # will default to jumping in log
        elif param_name == 'frequency':    
            sigma = 0.01
            logJump = True

        else:
            sigma = pardict[param_name].value/10

    # take into account temperatrue and periodic big jumps
    sigma *= np.sqrt(T) * big_sigma

    # update proposed params
    if logJump:
        new_log_par = np.log10(pardict[param_name].proposed) + np.random.randn()*sigma
        pardict[param_name].proposed = 10**new_log_par
    else:
        pardict[param_name].proposed += np.random.randn()*sigma

    # cyclic reflective bounds
    [pardict[key].cyclicReflectiveBounds() for key in pardict]


def PALInferenceCovarianceJumpSourceParameters(psr, runState, pardict):
    """

    Proposal distribution that uses covariance matrix of previous
    points in chain to make a correlated jump proposal. This proposal
    only jumps in source parameters and not in pulsar distance. Here
    we do not scale jumps by sqrt(T) since this is taken into account
    in the history of the high temerature chains.

    TODO: make smarter about jumps in log

    """
    
    # record current params into proposed attribute
    for key in pardict:
        pardict[key].proposed = pardict[key].value

    # create array of parameters that are not fixed 
    params = []
    param_keys = []
    for key in pardict:
        if pardict[key].vary !='PTA_INFERENCE_FIXED':
            params.append(pardict[key].proposed)
            param_keys.append(key)

    # convert to numpy array
    params = np.array(params)

    # number of dimensions
    ndim = len(params)

    # check for covariance jump scaling, if not set default
    if runState.covarianceJumpScale:
        scale = runState.covarainceJumpScale
    else:
        scale = 2.38**2/ndim


    # get covariance matrix from runState
    if runState.cov is not None:
        pass
    else:
        runState.cov = utils.constructCovarianceMatrix(runState, pardict)

    # update covariance matrix if needed every 500 iterations
    if np.random.rand() < 1/500: runState.cov = utils.constructCovarianceMatrix(runState, pardict)

    # make jump
    new_pars = np.random.multivariate_normal(params, runState.cov)

    # update intrinsic GW parameters
    for ct,key in enumerate(param_keys):
        if pardict[key].name != 'pdist':
            pardict[key].proposed = new_pars[ct]

    
    # cyclic reflective bounds
    [pardict[key].cyclicReflectiveBounds() for key in pardict]



def PALInferenceCovarianceJumpAllParameters(psr, runState, pardict):
    """

    Proposal distribution that uses covariance matrix of previous
    points in chain to make a correlated jump proposal. This proposal
    jumps in all parameters that are not fixed. Here
    we do not scale jumps by sqrt(T) since this is taken into account
    in the history of the high temerature chains.
    
    TODO: make smarter about jumps in log

    """
    
    # record current params into proposed attribute
    for key in pardict:
        pardict[key].proposed = pardict[key].value

    # create array of parameters that are not fixed 
    params = []
    param_keys = []
    for key in pardict:
        if pardict[key].vary !='PTA_INFERENCE_FIXED':
            params.append(pardict[key].proposed)
            param_keys.append(key)

    # convert to numpy array
    params = np.array(params)

    # number of dimensions
    ndim = len(params)

    # check for covariance jump scaling, if not set default
    if runState.covarianceJumpScale:
        scale = runState.covarainceJumpScale
    else:
        scale = 2.38**2/ndim

    # get covariance matrix from runState
    if runState.cov is not None:
        pass
    else:
        runState.cov = utils.constructCovarianceMatrix(runState, pardict)

    # update covariance matrix if needed every 500 iterations
    if np.random.rand() < 1/500: runState.cov = utils.constructCovarianceMatrix(runState, pardict)

    # make jump
    new_pars = np.random.multivariate_normal(params, runState.cov)

    # update proposed parameters
    for ct,key in enumerate(param_keys):
        pardict[key].proposed = new_pars[ct]
    
    # cyclic reflective bounds
    [pardict[key].cyclicReflectiveBounds() for key in pardict]
        
    
def PALInferenceSingleComponentAdaptiveJump(psr, runState, pardict):
    """

    Proposal distribution that uses previous history to make jump
    in one parameter drawn at random based on variance of previous
    points in the chain.

    This proposal only jumps in source parameters and not in pulsar 
    distance. Here we do not scale jumps by sqrt(T) since this is
    taken into account in the history of the high temerature chains.

    TODO: make smarter about jumps in log

    """
    
    # set large jumps
    big_sigma = 1.0

    # jump 10 times further every 1000 iterations
    if np.random.rand() < 1e-3: big_sigma = 10.0
    
    # jump 100 times further every 10000 iterations
    if np.random.rand() < 1e-4: big_sigma = 100.0

    # record current params into proposed attribute
    for key in pardict:
        pardict[key].proposed = pardict[key].value

    # draw parameter at random, that is not pulsar distance
    ndim = len(pardict.keys()) - 1

    found = False
    while found = False:
        index = np.random.randint(low=0, high=ndim)

        # check to make sure parameter is not pulsar distance or fixed
        if pardict.keys()[index].name != 'pdist' and pardict.keys()[index].vary != 'PTA_INFERENCE_FIXED':
            
            # choose param
            param_name = pardict.keys()[index].name
            found = True

    # get sigma from runState
    sigma = utils.getSingleParameterAdaptiveSigma(runState, pardict[param_name])
    sigma *= big_sigma

    # cyclic and reflective bounds

    # update proposed params
    pardict[param_name] += np.random.randn()*sigma

    # cyclic reflective bounds
    [pardict[key].cyclicReflectiveBounds() for key in pardict]


# jump in prior space
def PALInferenceDrawFromPrior(psr, runState, pardict):
    """
    
    Jump in range min - max of parameter. Not for pulsar distances

    """

    # record current params into proposed attribute
    for key in pardict:
        pardict[key].proposed = pardict[key].value

    # draw parameter at random, that is not pulsar distance
    ndim = len(pardict.keys()) - 1

    found = False
    while found = False:
        index = np.random.randint(low=0, high=ndim)

        # check to make sure parameter is not pulsar distance or fixed
        if pardict.keys()[index].name != 'pdist' and pardict.keys()[index].vary != 'PTA_INFERENCE_FIXED':
            
            # choose param
            param_name = pardict.keys()[index].name
            found = True


    pardict[param_name].proposed = pardict[param_name].min + np.random.rand()* \
                                    (pardict[param_name].max - pardict[param_name].min)




##########################################################################################################


# class that manages jump proposals
class PALInfrenceJumpProposals(object):
    """
    Class that manages jump proposal distributions for use in MCMC or Nested Sampling.

    """

    def __init__(self):


    # add jump proposal distribution functions
    self.propCycle = []
    def addProposalToCycle(self, func, weight):
        """
        
        Add jump proposal distributions to cycle with a given weight.

        @param func: jump proposal function
        @param weight: jump proposal function weight in cycle

        """

        # get length of cycle so far
        length = len(self.propCycle)

        # check for 0 weight
        if weight == 0:
            break

        # add proposal to cycle
        for ii in range(length, length + weight):
            self.propCycle.append(func)


    # randomized proposal cycle
    def randomizeProposalCycle(self):
        """
        
        Randomize proposal cycle that has already been filled

        """

        # get length of full cycle
        length = len(self.propCycle)

        # get random integers
        index = np.random.randint(low=0, high=(length-1), size=length)

        # randomize proposal cycle
        self.randomizedPropCycle = [self.propCycle[index[ii]] for ii in range(len(index))]


    # call proposal functions from cycle
    def __call__(self, psr, runState, pardict):
        """
        
        Call jump proposal from cycle. Returns a dictionary with proposed parameters.
        If we have reached the end of the proposal cycle, then randomize again

        @param psr: pulsar class
        @param runState: run state object
        @param pardict: Python dictionary containing parameter information

        TODO: find a better way of doing this without creating new dictionary

        """

        # get length of cycle
        length = len(self.propCycle)

        # if no iteration counter in runState, add one
        if runState.propIter:
            iter = runState.propIter
        else:
            runState.propIter = 0

        
        # call function
        self.randomizedPropCycle[iter](psr, runState, pardict)

        # increment proposal cycle counter and re-randomize if at end of cycle
        if (iter+1) % length == 0: randomizeProposalCycle()
        runState.propIter = (iter+1) % length

        # create new output dictionary
        output = {}
        for key in pardict:
            output[key] = pardict[key].proposed

        return output



######### setup some default proposal cycles ######################

def PALInferenceDefaultContinuousPropCycle():
    """

    Setup a default proposal cycle to be used post burn-in

    """

    # define weights
    BIG = 20
    SMALL = 5
    TINY = 1

    # initialize jumps
    jumps = PALInferenceJumpProposals()

    # add adaptive single parameter updates
    jumps.addProposalToCycle(PALInferenceSingleComponentAdaptiveJump, BIG)

    # add adaptive correlated proposals in all parameters
    jumps.addProposalToCycle(PALInferenceCovarianceJumpAllParameters, SMALL)
    
    # add adaptive correlated proposals in source parameters
    jumps.addProposalToCycle(PALInferenceCovarianceJumpSourceParameters, BIG)

    # add correlated mass distance jump
    jumps.addProposalToCycle(PALInferenceMassDistanceJump, SMALL)

    # add big pulsar phase/distance jump
    jumps.addProposalToCycle(PALInferencePulsarPhaseJumpBig, SMALL)

    # add small pulsar phase/distance jump
    jumps.addProposalToCycle(PALInferencePulsarPhaseJumpSmall, BIG)

    # add random Draws from prior for source parameters
    jumps.addProposalToCycle(PALInferenceDrawFromPrior, TINY)

    # randomize cycle
    jumps.randomizeProposalCycle()

    return jumps


def PALInferenceDefaultContinuousPropCycleBurn():
    """

    Setup a default proposal cycle to be used for burn-in

    """

    # define weights
    BIG = 20
    SMALL = 5
    TINY = 1

    # initialize jumps
    jumps = PALInferenceJumpProposals()

    # standard single parameter updates
    jumps.addProposal(PALInferenceSingleProposal, BIG)

    # add adaptive single parameter updates
    jumps.addProposalToCycle(PALInferenceSingleComponentAdaptiveJump, SMALL)

    # add adaptive correlated proposals in all parameters
    jumps.addProposalToCycle(PALInferenceCovarianceJumpAllParameters, SMALL)
    
    # add adaptive correlated proposals in source parameters
    jumps.addProposalToCycle(PALInferenceCovarianceJumpSourceParameters, SMALL)

    # add correlated mass distance jump
    jumps.addProposalToCycle(PALInferenceMassDistanceJump, SMALL)

    # add big pulsar phase/distance jump
    jumps.addProposalToCycle(PALInferencePulsarPhaseJumpBig, 2*SMALL)

    # add small pulsar phase/distance jump
    jumps.addProposalToCycle(PALInferencePulsarPhaseJumpSmall, BIG)

    # add random Draws from prior for source parameters
    jumps.addProposalToCycle(PALInferenceDrawFromPrior, TINY)

    # randomize cycle
    jumps.randomizeProposalCycle()

    return jumps

