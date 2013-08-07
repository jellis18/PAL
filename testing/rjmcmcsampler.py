#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["RJMCMCSampler"]

try:
    import acor
    acor = acor
except ImportError:
    acor = None

import numpy as np

class RJMCMCPost(object):
    """
    Wrapper for posterior used with the :class:`PTSampler`.

    """

    def __init__(self, logl, logp, newpar):
        """
        :param logl:
            Function returning natural log of the likelihood.

        :param logp:
            Function returning natural log of the prior.

        :param newpar:
            Number of new parameters if trans dimensional jump

        """

        self._logl = logl
        self._logp = logp
        self._newpar = newpar

    def __call__(self, x):
        """
        Returns ``lnpost(x)``, ``lnlike(x)`` (the second value will be
        treated as a blob by emcee), where

        .. math::

            \ln \pi(x) \equiv \beta \ln l(x) + \ln p(x)

        :param x:
            The position in parameter space.

        """
        if self._newpar > 0:
            lp = self._logp(x[0:-self._newpar])
        else:
            lp = self._logp(x)

        # If outside prior bounds, return 0.
        if lp == float('-inf'):
            return lp, lp

        ll = self._logl(x)

        return ll + lp, ll


class RJMCMCSampler(object):
    """
    A parallel-tempered ensemble sampler, using :class:`EnsembleSampler`
    for sampling within each parallel chain.

    :param ntemps:
        The number of temperatures.

    :param nwalkers:
        The number of ensemble walkers at each temperature.

    :param dim:
        The dimension of parameter space.

    :param logl:
        The log-likelihood function.

    :param logp:
        The log-prior function.

    :param threads: (optional)
        The number of parallel threads to use in sampling.

    :param pool: (optional)
        Alternative to ``threads``.  Any object that implements a
        ``map`` method compatible with the built-in ``map`` will do
        here.  For example, :class:`multi.Pool` will do.

    :param betas: (optional)
        Array giving the inverse temperatures, :math:`\\beta=1/T`, used in the
        ladder.  The default is for an exponential ladder, with beta
        decreasing by a factor of :math:`1/\\sqrt{2}` each rung.

    """
    def __init__(self, nmodel, logl, logp, propJump, args=[]):

        self.logl = logl
        self.logp = logp
        self.propJump = propJump
        self.nmodel = nmodel
        self.args = args


        self.naccepted = np.zeros(self.nmodel)
        self._nmod = np.zeros(self.nmodel, dtype=np.float)


    def reset(self):
        """
        Clear the ``chain``, ``lnprobability``, ``lnlikelihood``,
        ``acceptance_fraction``, ``tswap_acceptance_fraction`` stored
        properties.

        """

        self.naccepted = np.zeros(self.nmodel, dtype=np.float)
        self.nmod = np.zeros(self.nmodel, dtype=np.float)

        self._chain = []
        self._lnprob = []
        self._lnlikelihood = []

    def _get_lnprob(self, p, newpar):
        fn = RJMCMCPost(self.logl, self.logp, newpar)
        return fn(p)

    def sample(self, p0, model0, lnprob0=None, lnlike0=None, iterations=1,
               thin=1, storechain=True):
        """
        Advance the chains ``iterations`` steps as a generator.

        :param p0:
            The initial positions of the walkers.  Shape should be
            ``(ntemps, nwalkers, dim)``.

        :param lnprob0: (optional)
            The initial posterior values for the ensembles.  Shape
            ``(ntemps, nwalkers)``.

        :param lnlike0: (optional)
            The initial likelihood values for the ensembles.  Shape
            ``(ntemps, nwalkers)``.

        :param iterations: (optional)
            The number of iterations to preform.

        :param thin: (optional)
            The number of iterations to perform between saving the
            state to the internal chain.

        :param storechain: (optional)
            If ``True`` store the iterations in the ``chain``
            property.

        At each iteration, this generator yields

        * ``p``, the current position of the walkers.

        * ``lnprob`` the current posterior values for the walkers.

        * ``lnlike`` the current likelihood values for the walkers.

        """
        p = np.copy(np.array(p0))

        # If we have no lnprob or logls compute them
        if lnprob0 is None or lnlike0 is None:
            lnprob0, lnlike0 = self._get_lnprob(p, 0)

        lnprob = lnprob0
        lnlike = lnlike0
        model = model0
        
        # initialize chain variables
        self._chain = [[] for ii in range(self.nmodel)]
        self._lnprob = [[] for ii in range(self.nmodel)]
        self._lnlikelihood = [[] for ii in range(self.nmodel)]
        self._modchain = np.zeros(iterations)

        # do sampling
        self.iterations = 0
        for i in range(iterations):
            self.iterations += 1

            # propose jump in parameter space
            q, newmod = self._get_jump(p, model)
            q = np.array(q)

            # get number of new parameters (can be negative)
            newpar = np.alen(q) - np.alen(p)

            # evaluate posterior
            newlnprob, newlnlike = self._get_lnprob(q, newpar)

            # MH step
            diff = newlnprob-lnprob

            if diff < 0:
                diff = np.exp(diff) - np.random.rand()

            if diff >= 0:
                p = q
                model = newmod
                lnprob = newlnprob
                lnlike = newlnlike
                self.naccepted[model-1] += 1
               
            # add count to specific model
            self._modchain[i] = model 
            self._nmod[model-1] += 1 
            
            # save chain values
            if (i+1) % thin == 0:
                if storechain:
                    self._chain[model-1].append(p)
                    self._lnprob[model-1].append(lnprob)
                    self._lnlikelihood[model-1].append(lnlike)

            yield p, model, lnprob, lnlike


    def _get_jump(self, p, model):
        """Return the new position from the jump proposal distibution"""
        return self.propJump(p, self.iterations, model)

    


    @property
    def chain(self):
        """
        Returns the stored chain of samples; shape ``(Ntemps,
        Nwalkers, Nsteps, Ndim)``.

        """
        return self._chain
    
    @property
    def modchain(self):
        """
        Returns the stored chain of samples; shape ``(Ntemps,
        Nwalkers, Nsteps, Ndim)``.

        """
        return self._modchain
    
    @property
    def nmod(self):
        """
        Returns the stored chain of samples; shape ``(Ntemps,
        Nwalkers, Nsteps, Ndim)``.

        """
        return self._nmod

    @property
    def lnprobability(self):
        """
        Matrix of lnprobability values; shape ``(Ntemps, Nwalkers, Nsteps)``.

        """
        return self._lnprob

    @property
    def lnlikelihood(self):
        """
        Matrix of ln-likelihood values; shape ``(Ntemps, Nwalkers, Nsteps)``.

        """
        return self._lnlikelihood


    @property
    def acceptance_fraction(self):
        """
        Matrix of shape ``(Ntemps, Nwalkers)`` detailing the
        acceptance fraction for each walker.

        """
        return np.array([self.naccepted[j]/self._nmod[j] \
                        for j in range(self.nmodel)])

    @property
    def acor(self):
        """
        Returns a matrix of autocorrelation lengths for each
        parameter in each temperature of shape ``(Ntemps, Ndim)``.

        """
        if acor is None:
            raise ImportError('acor')
        else:
            acors = np.zeros((self.ntemps, self.dim))

            for i in range(self.ntemps):
                for j in range(self.dim):
                    acors[i, j] = acor.acor(self._chain[i, :, :, j])[0]

            return acors
