#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from mpi4py import MPI
import os, sys, time


class PTSampler(object):

    """
    Parallel Tempering Markov Chain Monte-Carlo (PTMCMC) sampler. 
    This implementation uses an adaptive jump proposal scheme
    by default using both standard and single component Adaptive
    Metropolis (AM) and Differential Evolution (DE) jumps.

    This implementation also makes use of MPI (mpi4py) to run
    the parallel chains.

    Along with the AM and DE jumps, the user can add custom 
    jump proposals with the ``addProposalToCycle`` fuction. 

    @param ndim: number of dimensions in problem
    @param logl: log-likelihood function
    @param logp: log prior function (must be normalized for evidence evaluation)
    @param cov: Initial covariance matrix for jump proposals
    @param outDir: Full path to output directory for chain files (default = ./chains)
    @param verbose: Update current run-status to the screen (default=True)

    """

    def __init__(self, ndim, logl, logp, cov, outDir='./chains', verbose=True):

        # MPI initialization
        self.comm = MPI.COMM_WORLD
        self.MPIrank = self.comm.Get_rank()
        self.nchain = self.comm.Get_size()

        self.ndim = ndim
        self.logl = logl
        self.logp = logp
        self.outDir = outDir
        self.verbose = verbose

        # setup output file
        if not os.path.exists(self.outDir):
            try:
                os.makedirs(self.outDir)
            except OSError:
                pass

        # set up covariance matrix
        self.cov = cov
        self.U, self.S, v = np.linalg.svd(self.cov)
        self.M2 = np.zeros((self.ndim, self.ndim))
        self.mu = np.zeros(self.ndim)

        # initialize proposal cycle
        self.propCycle = []
        

    def sample(self, p0, Niter, Tmin=1, Tmax=10, Tskip=100, \
               isave=1000, covUpdate=5000, SCAMweight=20, \
               AMweight=20, DEweight=20, burn=5000):

        """
        Function to carry out PTMCMC sampling.

        @param p0: Initial parameter vector
        @param Niter: Number of iterations to use for T = 1 chain
        @param Tmin: Minimum temperature in ladder (default=1) 
        @param Tmax: Maximum temperature in ladder (default=10) 
        @param Tskip: Number of steps between proposed temperature swaps (default=100)
        @param isave: Number of iterations before writing to file (default=1000)
        @param covUpdate: Number of iterations between AM covariance updates (default=5000)
        @param SCAMweight: Weight of SCAM jumps in overall jump cycle (default=20)
        @param AMweight: Weight of AM jumps in overall jump cycle (default=20)
        @param DEweight: Weight of DE jumps in overall jump cycle (default=20)
        @param burn: Burn in time (DE jumps added after this iteration) (default=5000)

        """

        # set up arrays to store lnprob, lnlike and chain
        self._lnprob = np.zeros(Niter)
        self._lnlike = np.zeros(Niter)
        self._chain = np.zeros((Niter, self.ndim))
        self.naccepted = 0
        self.swapProposed = 0
        self.nswap_accepted = 0

        # setup default jump proposal distributions

        # add SCAM
        self.addProposalToCycle(self.covarianceJumpProposalSCAM, SCAMweight)
        
        # add AM
        self.addProposalToCycle(self.covarianceJumpProposalAM, AMweight)
        
        # randomize cycle
        self.randomizeProposalCycle()

        # setup temperature ladder
        ladder = self._temperatureLadder(Tmin)

        # temperature for current chain
        self.temp = ladder[self.MPIrank]

        ### compute lnprob for initial point in chain ###
        self._chain[0,:] = p0

        # compute prior
        lp = self.logp(p0)

        if lp == float(-np.inf):

            lnprob0 = -np.inf

        else:

            lnlike0 = self.logl(p0) 
            lnprob0 = 1/self.temp * lnlike0 + lp

        # record first values
        self._lnprob[0] = lnprob0
        self._lnlike[0] = lnlike0

        # set up output file
        fname = self.outDir + '/chain_{0}.txt'.format(self.temp)
        self._chainfile = open(fname, 'w')
        self._chainfile.close()

        self.comm.barrier()


        # start iterations
        iter = 0
        tstart = time.time()
        runComplete = False
        while runComplete == False:
            iter += 1

            # update covariance matrix
            if (iter-1) % covUpdate == 0 and (iter-1) != 0:
                self._updateRecursive(iter-1, covUpdate)

            # jump proposal
            y, qxy = self._jump(p0, iter)

            # after burn in, add DE jumps
            if iter == burn:
                self.addProposalToCycle(self.DEJump, DEweight)
                
                # randomize cycle
                self.randomizeProposalCycle()


            # compute prior and likelihood
            lp = self.logp(y)
            
            if lp == -np.inf:

                newlnprob = -np.inf

            else:

                newlnlike = self.logl(y) 
                newlnprob = 1/self.temp * newlnlike + lp

            # hastings step
            diff = newlnprob - lnprob0 + qxy

            if diff >= np.log(np.random.rand()):

                # accept jump
                p0, lnlike0, lnprob0 = y, newlnlike, newlnprob

                # update acceptance counter
                self.naccepted += 1


            # put results into arrays
            self._chain[iter,:] = p0
            self._lnlike[iter] = lnlike0
            self._lnprob[iter] = lnprob0


            ##################### TEMPERATURE SWAP ###############################
            readyToSwap = 0
            swapAccepted = 0

            # if Tskip is reached, block until next chain in ladder is ready for swap proposal
            if iter % Tskip == 0 and self.MPIrank < self.nchain-1:
                self.swapProposed += 1

                # send current likelihood for swap proposal
                self.comm.send(lnlike0, dest=self.MPIrank+1)

                # determine if swap was accepted
                swapAccepted = self.comm.recv(source=self.MPIrank+1)

                # perform swap
                if swapAccepted:
                    self.nswap_accepted += 1

                    # exchange likelihood
                    lnlike0 = self.comm.recv(source=self.MPIrank+1)

                    # exchange parameters
                    self.comm.send(p0, dest=self.MPIrank+1)
                    p0 = self.comm.recv(source=self.MPIrank+1)

                    # calculate new posterior values
                    lnprob0 = 1/self.temp * lnlike0 + self.logp(p0)


            # check if next lowest temperature is ready to swap
            elif self.MPIrank > 0:

                readyToSwap = self.comm.Iprobe(source=self.MPIrank-1)
                 # trick to get around processor using 100% cpu while waiting
                time.sleep(0.000001) 

                # hotter chain decides acceptance
                if readyToSwap:
                    newlnlike = self.comm.recv(source=self.MPIrank-1)
                    
                    # determine if swap is accepted and tell other chain
                    logChainSwap = (1/ladder[self.MPIrank-1] - 1/ladder[self.MPIrank]) \
                            * (lnlike0 - newlnlike)

                    if logChainSwap >= np.log(np.random.rand()):
                        swapAccepted = 1
                    else:
                        swapAccepted = 0

                    # send out result
                    self.comm.send(swapAccepted, dest=self.MPIrank-1)

                    # perform swap
                    if swapAccepted:

                        # exchange likelihood
                        self.comm.send(lnlike0, dest=self.MPIrank-1)
                        lnlike0 = newlnlike

                        # exchange parameters
                        self.comm.send(p0, dest=self.MPIrank-1)
                        p0 = self.comm.recv(source=self.MPIrank-1)
                    
                        # calculate new posterior values
                        lnprob0 = 1/self.temp * lnlike0 + self.logp(p0)


        ##################################################################

            # put results into arrays
            self._chain[iter,:] = p0
            self._lnlike[iter] = lnlike0
            self._lnprob[iter] = lnprob0

            # write to file
            if iter % isave == 0:
                self._writeToFile(fname, iter, isave)
                if self.MPIrank == 0 and self.verbose:
                    sys.stdout.write('\r')
                    sys.stdout.write('Finished %2.2f percent in %f s'%(iter/Niter*100, \
                                                    time.time() - tstart))
                    sys.stdout.flush()

            # stop
            if self.MPIrank == 0 and iter >= Niter-1:
                print '\nRun Complete'
                runComplete = True

            if self.MPIrank == 0 and runComplete:
                for jj in range(1, self.nchain):
                    self.comm.send(runComplete, dest=jj, tag=55)

            # check for other chains
            if self.MPIrank > 0:
                runComplete = self.comm.Iprobe(source=0, tag=55)
                time.sleep(0.000001) # trick to get around 



    def _temperatureLadder(self, Tmin):

        """
        Method to compute temperature ladder. At the moment this uses
        a geometrically spaced temperature ladder with a temperature
        spacing designed to give 25 % temperature swap acceptance rate.

        """

        #TODO: make options to do other temperature ladders

        if self.nchain > 1:
            tstep = 1 + np.sqrt(2/self.ndim)
            ladder = np.zeros(self.nchain)
            for ii in range(self.nchain): ladder[ii] = Tmin*tstep**ii
        else:
            ladder = np.array([1])

        return ladder


    def _writeToFile(self, fname, iter, isave):

        """
        Function to write chain file. File has 3+ndim columns,
        the first is log-posterior (unweighted), log-likelihood,
        and accepatence probability, followed by parameter values.
        
        @param fname: chainfile name
        @param iter: Iteration of sampler
        @param isave: Number of iterations between saves

        """

        self._chainfile = open(fname, 'a+')
        for jj in range((iter-isave), iter, 10):
            self._chainfile.write('%e\t %e\t %e\t'%(self._lnprob[jj], self._lnlike[jj],\
                                                  self.naccepted/iter))
            self._chainfile.write('\t'.join([str(self._chain[jj,kk]) \
                                            for kk in range(self.ndim)]))
            self._chainfile.write('\n')
        self._chainfile.close()



    # function to update covariance matrix for jump proposals
    def _updateRecursive(self, iter, mem):

        """ 
        Function to recursively update sample covariance matrix.

        @param iter: Iteration of sampler
        @param mem: Number of steps between updates

        """

        iter -= mem

        if iter == 0:
            self.M2 = np.zeros((self.ndim, self.ndim))
            self.mu = np.zeros(self.ndim)

        for ii in range(mem):
            diff = np.zeros(self.ndim)
            iter += 1
            for jj in range(self.ndim):
                
                diff[jj] = self._chain[iter+ii,jj] - self.mu[jj]
                self.mu[jj] += diff[jj]/iter

            self.M2 += np.outer(diff, (self._chain[iter+ii,:]-self.mu))

        self.cov = self.M2/(iter-1)  

        # do svd
        self.U, self.S, v = np.linalg.svd(self.cov)
        
    # SCAM jump
    def covarianceJumpProposalSCAM(self, x, iter, beta):

        """
        Single Component Adaptive Jump Proposal. This function will occasionally
        jump in more than 1 parameter. It will also occasionally use different
        jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

        """

        q = x.copy()
        qxy = 0

        # number of parameters to update at once 
        prob = np.random.rand()
        if prob > (1 - 1/self.ndim):
            block = self.ndim

        elif prob > (1 - 2/self.ndim):
            block = np.ceil(self.ndim/2)

        elif prob > 0.8:
            block = 5

        else:
            block = 1

        # adjust step size
        prob = np.random.rand()

        # very small jump
        if prob > 0.9:
            scale = 0.01
            
        # small jump
        elif prob > 0.7:
            scale = 0.2

        # large jump
        elif prob > 0.97:
            scale = 10
        
        # small-medium jump
        elif prob > 0.6:
            scale = 0.5

        # standard medium jump
        else:
            scale = 1.0

        # get parmeters in new diagonalized basis
        y = np.dot(self.U.T, x)

        # make correlated componentwise adaptive jump
        ind = np.unique(np.random.randint(0, self.ndim, block))
        neff = len(ind)
        cd = 2.4  / np.sqrt(2*neff) * scale 

        y[ind] = y[ind] + np.random.randn(neff) * cd * np.sqrt(self.S[ind])
        q = np.dot(self.U, y)

        return q, qxy
    
    # AM jump
    def covarianceJumpProposalAM(self, x, iter, beta):

        """
        Adaptive Jump Proposal. This function will occasionally 
        use different jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

        """

        q = x.copy()
        qxy = 0

        # adjust step size
        prob = np.random.rand()

        # very small jump
        if prob > 0.9:
            scale = 0.01
            
        # small jump
        elif prob > 0.7:
            scale = 0.2

        # large jump
        elif prob > 0.97:
            scale = 10
        
        # small-medium jump
        elif prob > 0.6:
            scale = 0.5

        # standard medium jump
        else:
            scale = 1.0

        cd = 2.4/np.sqrt(2*self.ndim) * scale
        q = np.random.multivariate_normal(x, cd**2*self.cov)

        return q, qxy


    # Differential evolution jump
    def DEJump(self, x, iter, beta):

        """
        Differential Evolution Jump. This function will  occasionally 
        use different jump sizes to ensure proper mixing.

        @param x: Parameter vector at current position
        @param iter: Iteration of sampler
        @param beta: Inverse temperature of chain

        @return: q: New position in parameter space
        @return: qxy: Forward-Backward jump probability

        """

        # get old parameters
        q = x.copy()
        qxy = 0

        # draw a random integer from 0 - iter
        mm = np.random.randint(0, iter)
        nn = np.random.randint(0, iter)

        # make sure mm and nn are not the same iteration
        while mm == nn: nn = np.random.randint(0, iter)

        # get jump scale size
        prob = np.random.rand()

        # mode jump
        if prob > 0.5:
            scale = 1.0

        else:
            scale = np.random.rand() * 2.4/np.sqrt(2*self.ndim) * np.sqrt(1/beta)


        for ii in range(self.ndim):
            
            # jump size
            sigma = self._chain[mm, ii] - self._chain[nn, ii]

            # jump
            q[ii] += scale * sigma
        
        return q, qxy


    # add jump proposal distribution functions
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
            print 'ERROR: Can not have 0 weight in proposal cycle!'
            sys.exit()

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
        index = np.random.randint(0, (length-1), length)

        # randomize proposal cycle
        self.randomizedPropCycle = [self.propCycle[ind] for ind in index]


    # call proposal functions from cycle
    def _jump(self, x, iter):
        """
        Call Jump proposals

        """

        # get length of cycle
        length = len(self.propCycle)

        # call function
        q, qxy = self.randomizedPropCycle[np.mod(iter, length)](x, iter, 1/self.temp)

        # increment proposal cycle counter and re-randomize if at end of cycle
        if iter % length == 0: self.randomizeProposalCycle()

        return q, qxy

    #TODO: add auxilary jump (for things like pulsar distances)











