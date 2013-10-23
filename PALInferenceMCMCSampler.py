#!/usr/bin/env python

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from mpi4py import MPI
import time
import numpy as np

__all__ = ["PALInferenceMCMCSampler"]

# Parallel temperining MCMC sampler that makes use of mpi4py for parallelization

class MCMCSampler(object):
    """
    General MCMC Sampler class.

    """

    def __init__(self, ndim, jump, logl, logp, outDir):
        """

        """
            
        # input values
        self.ndim = ndim
        self.jump = jump
        self.logl = logl
        self.logp = logp
        self.outDir = outDir

        # initialize values to be updated during run
        self.naccepted = 0
        self.nswap = 0
        self.nswap_accepted = 0

    # TODO: use an options class here instead of all these arguments
    def _temperature_ladder(self, nchains, Tmin=1, Tmax=None, injSNR=None, hotSNR=3, \
                           tstep=None, evidence=False):
        """
        Constuct geometric spacing temperature ladder

        """

        # temperature ladder
        self.ladder = np.zeros(nchains)

        # assume that we have some injected SNR
        if injSNR is not None and Tmax is None:
            Tmax = (injSNR/hotSNR)**2

        # assume a (maximum) injected SNR of 10
        elif injSNR is None and Tmax is None:
            Tmax = (10/hotSNR)**2

        
        # set up temperature step size
        if tstep is None:
            tstep = (Tmax/Tmin)**(1/(nchains-1))

        
        # set up ladder
        if evidence:
            Ts = Tmax
            Tmax = 1e4

            # set temperature size
            tstep = (Ts/Tmin)**(1/(nchains/2-1))
            
            # "signal" part of temperature ladder
            for ii in range(int(nchains/2)): self.ladder[ii] = Tmin*tstep**ii

            # set new tstep
            tstep = (Tmax/Ts)**(1/(nchains/2-1))
            
            # "noise" part of temperature ladder
            for ii in range(int(nchains/2), nchains): 
                self.ladder[ii] = Ts*tstep**(ii-int(nchains/2)+1)

        else:

            for ii in range(nchains): self.ladder[ii] = Tmin*tstep**ii

        
    def sample(pars, Niter, Tskip=100):
        """
        Sample.
        
        """

        # set up MPI parameters
        comm = MPI.COMM_WORLD
        MPIrank = comm.Get_rank()
        MPIsize = comm.Get_size()

        # get number of chains
        nchain = MPIsize



