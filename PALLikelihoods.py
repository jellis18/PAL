from __future__ import division
import numpy as np
import scipy.linalg as sl
from scipy.optimize import minimize_scalar
import PALutils
import sys,os

########################## DETECTION STATISTICS ##################################

# compute f_p statistic
def fpStat(psr, f0):
    """ 
    Computes the Fp-statistic as defined in Ellis, Siemens, Creighton (2012)
    
    @param psr: List of pulsar object instances
    @param f0: Gravitational wave frequency

    @return: Value of the Fp statistic evaluated at f0

    """

    fstat=0.
    npsr = len(psr)

    # define N vectors from Ellis et al, 2012 N_i=(x|A_i) for each pulsar
    N = np.zeros(2)
    M = np.zeros((2, 2))
    for ii,p in enumerate(psr):

        # Define A vector
        A = np.zeros((2, p.ntoa))
        A[0,:] = 1./f0**(1./3.) * np.sin(2*np.pi*f0*p.toas)
        A[1,:] = 1./f0**(1./3.) * np.cos(2*np.pi*f0*p.toas)

        N = np.array([np.dot(A[0,:], np.dot(p.invCov, p.res)), \
                      np.dot(A[1,:], np.dot(p.invCov, p.res))]) 
        
        # define M matrix M_ij=(A_i|A_j)
        for jj in range(2):
            for kk in range(2):
                M[jj,kk] = np.dot(A[jj,:], np.dot(p.invCov, A[kk,:]))
                
        # take inverse of M
        Minv = np.linalg.inv(M)
        fstat += 0.5 * np.dot(N, np.dot(Minv, N))

    # return F-statistic
    return fstat


def optStat(psr, ORF, gam=4.33333):
    """
    Computes the Optimal statistic as defined in Chamberlin, Creighton, Demorest et al (2013)

    @param psr: List of pulsar object instances
    @param ORF: Vector of pairwise overlap reduction values
    @param gam: Power Spectral index of GBW (default = 13/3, ie SMBMBs)

    @return: Opt: Optimal statistic value (A_gw^2)
    @return: sigma: 1-sigma uncertanty on Optimal statistic
    @return: snr: signal-to-noise ratio of cross correlations

    """

    #TODO: maybe compute ORF in code instead of reading it in. Would be less
    # of a risk but a bit slower...

    k = 0
    npsr = len(psr)
    top = 0
    bot = 0
    for ll in xrange(0, npsr):
        for kk in xrange(ll+1, npsr):

            # form matrix of toa residuals and compute SigmaIJ
            tm = PALutils.createTimeLags(psr[ll].toas, psr[kk].toas)

            # create cross covariance matrix without overall amplitude A^2
            SIJ = ORF[k]/2 * PALutils.createRedNoiseCovarianceMatrix(tm, 1, gam)
            
            # construct numerator and denominator of optimal statistic
            bot += np.trace(np.dot(psr[ll].invCov, np.dot(SIJ, np.dot(psr[kk].invCov, SIJ.T))))
            top += np.dot(psr[ll].res, np.dot(psr[ll].invCov, np.dot(SIJ, \
                        np.dot(psr[kk].invCov, psr[kk].res))))
            k+=1

    # compute optimal statistic
    Opt = top/bot
    
    # compute uncertainty
    sigma = 1/np.sqrt(bot)

    # compute SNR
    snr = top/np.sqrt(bot)

    # return optimal statistic and snr
    return Opt, sigma, snr

def crossPower(psr, gam=13/3):
    """

    Compute the cross power as defined in Eq 9 and uncertainty of Eq 10 in 
    Demorest et al (2012).

    @param psr: List of pulsar object instances
    @param gam: Power spectral index of GWB

    @return: vector of cross power for each pulsar pair
    @return: vector of cross power uncertainties for each pulsar pair

    """

    # initialization
    npsr = len(psr) 

    # now compute cross power
    rho = []
    sig = []
    xi = []
    for ll in range(npsr):
        for kk in range(ll+1, npsr):
            
            # matrix of time lags
            tm = PALutils.createTimeLags(psr[ll].toas, psr[kk].toas)

            # create cross covariance matrix without overall amplitude A^2
            SIJ = PALutils.createRedNoiseCovarianceMatrix(tm, 1, gam)
            
            # construct numerator and denominator of optimal statistic
            bot = np.trace(np.dot(psr[ll].invCov, np.dot(SIJ, np.dot(psr[kk].invCov, SIJ.T))))
            top = np.dot(psr[ll].res, np.dot(psr[ll].invCov, np.dot(SIJ, \
                        np.dot(psr[kk].invCov, psr[kk].res))))

            # cross correlation and uncertainty
            rho.append(top/bot)
            sig.append(1/np.sqrt(bot))


    return np.array(rho), np.array(sig)

########################## FIRST-CUT BAYES FACTOR ##################################

# compute B_p statistic at particular frequency
def bpStatFreq(psr, f0):
    """ 
    Computes the Bp-statistic by setting uniform-priors on signal-amplitude parameters
    May lead to unphysical priors on the "physical" parameters, but intended only as first-cut
    
    @param psr: List of pulsar object instances
    @param f0: Gravitational wave frequency

    @return: Value of the Bp statistic evaluated at f0

    """

    fstat=0.
    npsr = len(psr)
    log_prod_detM = 1.0

    # define N vectors from Ellis et al, 2012 N_i=(x|A_i) for each pulsar
    N = np.zeros(2)
    M = np.zeros((2, 2))
    for ii,p in enumerate(psr):

        # Define A vector
        A = np.zeros((2, p.ntoa))
        A[0,:] = 1./f0**(1./3.) * np.sin(2*np.pi*f0*p.toas)
        A[1,:] = 1./f0**(1./3.) * np.cos(2*np.pi*f0*p.toas)

        N = np.array([np.dot(A[0,:], np.dot(p.invCov, p.res)), \
                      np.dot(A[1,:], np.dot(p.invCov, p.res))]) 
        
        # define M matrix M_ij=(A_i|A_j)
        for jj in range(2):
            for kk in range(2):
                M[jj,kk] = np.dot(A[jj,:], np.dot(p.invCov, A[kk,:]))
                
        # take inverse of M
        Minv = np.linalg.inv(M)
        fstat += 0.5 * np.dot(N, np.dot(Minv, N))

        #log_prod_detM += -0.5 * np.log(M[0,0]*M[1,1] - M[0,1]*M[1,0])
        log_prod_detM += 0.5 * np.log(Minv[0,0]*Minv[1,1] - Minv[0,1]*Minv[1,0])
        print fstat, log_prod_detM

    
    # return Bp-statistic
    return ( log_prod_detM) + np.log(np.power(2.0*np.pi,1.0*npsr))

# compute B_p statistic integrated over frequency to give crude Bayes' factor
def bpStat(psr, flow, fhigh):
    """ 
    Computes the Bp-statistic by setting uniform-priors on signal-amplitude parameters
    May lead to unphysical priors on the "physical" parameters, but intended only as first-cut
    
    @param psr: List of pulsar object instances
    @param f0: Gravitational wave frequency

    @return: Value of the Bp statistic integrated over frequency

    """

    return quad(lambda x: bpStatFreq(psr,x), 1.0*flow, 1.0*fhigh)


#def crossPower(psr):
#    """
#
#    Compute the cross power as defined in Eq 9 and uncertainty of Eq 10 in 
#    Demorest et al (2012).
#
#    @param psr: List of pulsar object instances
#
#    @return: vector of cross power for each pulsar pair
#    @return: vector of cross power uncertainties for each pulsar pair
#
#    """
#
#    # initialization
#    npsr = len(psr) 
#
#
#    for ii in range(npsr):
#
#        # matrix of time lags
#        tm = PALutils.createTimeLags(psr[ii].toas, psr[ii].toas)
#
#        # red noise covariance matrix
#        Cgw = PALutils.createRedNoiseCovarianceMatrix(tm, 1, 13/3)
#        Cgw = np.dot(psr[ii].G.T, np.dot(Cgw, psr[ii].G))
#
#        # white noise covariance matrix
#        white = PALutils.createWhiteNoiseCovarianceMatrix(psr[ii].err, 1, 0)
#        white = np.dot(psr[ii].G.T, np.dot(white, psr[ii].G))
#
#        # chlolesky decomposition of white noise
#        L = sl.cholesky(white)
#        Linv = np.linalg.inv(L)
#
#        # sandwich with Linv
#        Cgwnew = np.dot(Linv, np.dot(Cgw, Linv.T))
#
#        # get svd of matrix
#        u, s, v = sl.svd(Cgwnew)
#
#        # data written in new basis
#        c = np.dot(u.T, np.dot(Linv, np.dot(psr[ii].G.T, psr[ii].res)))
#
#        # obtain the maximum likelihood value of Agw
#        f = lambda x: -PALutils.twoComponentNoiseLike(x, s, c)
#        fbounded = minimize_scalar(f, bounds=(0, 1e-14, 3.0e-13), method='Golden')
#
#        # maximum likelihood value
#        hc_ml = np.abs(fbounded.x)
#        print 'Max like Amp = {0}'.format(hc_ml)
#
#        # create inverse covariance matrix from svd decomposition
#        tmp = hc_ml**2 * Cgw + white
#        #psr[ii].invCov = np.dot(psr[ii].G, np.dot(sl.inv(tmp), psr[ii].G.T))
#
#    # now compute cross power
#    rho = []
#    sig = []
#    xi = []
#    for ll in range(npsr):
#        for kk in range(ll+1, npsr):
#            
#            # matrix of time lags
#            tm = PALutils.createTimeLags(psr[ll].toas, psr[kk].toas)
#
#            # create cross covariance matrix without overall amplitude A^2
#            SIJ = PALutils.createRedNoiseCovarianceMatrix(tm, 1, 13/3)
#            
#            # construct numerator and denominator of optimal statistic
#            bot = np.trace(np.dot(psr[ll].invCov, np.dot(SIJ, np.dot(psr[kk].invCov, SIJ.T))))
#            top = np.dot(psr[ll].res, np.dot(psr[ll].invCov, np.dot(SIJ, \
#                        np.dot(psr[kk].invCov, psr[kk].res))))
#
#            # cross correlation and uncertainty
#            rho.append(top/bot)
#            sig.append(1/np.sqrt(bot))
#
#            # angular separation
#            xi.append(PALutils.angularSeparation(psr[ll].theta, psr[ll].phi, \
#                                                psr[kk].theta, psr[kk].phi))
#    
#
#    return np.array(rho), np.array(sig), np.array(xi)
#

            
    

######################### BAYESIAN LIKELIHOOD FUNCTIONS ####################################


def firstOrderLikelihood(psr, ORF, Agw, gamgw, Ared, gred, efac, equad, \
                        interpolate=False):
    """
    Compute the value of the first-order likelihood as defined in 
    Ellis, Siemens, van Haasteren (2013).

    @param psr: List of pulsar object instances
    @param ORF: Vector of pairwise overlap reduction values
    @param Agw: Amplitude of GWB in standard strain amplitude units
    @param gamgw: Power spectral index of GWB
    @param Ared: Vector of amplitudes of intrinsic red noise in GWB strain units
    @param gamgw: Vector of power spectral index of red noise
    @param efac: Vector of efacs 
    @param equad: Vector of equads
    @param interpolate: Boolean to perform interpolation only with compressed
                        data. (default = False)

    @return: Log-likelihood value

    """
    npsr = len(psr)
    loglike = 0
    tmp = []

    # start loop to evaluate auto-terms
    for ll in range(npsr):

       r1 = np.dot(psr[ll].G.T, psr[ll].res)

       # create time lags
       tm = PALutils.createTimeLags(psr[ll].toas, psr[ll].toas)

       #TODO: finish option to do interpolation when using compression

       # calculate auto GW covariance matrix
       SC = PALutils.createRedNoiseCovarianceMatrix(tm, Agw, gamgw)

       # calculate auto red noise covariance matrix
       SA = PALutils.createRedNoiseCovarianceMatrix(tm, Ared[ll], gred[ll])

       # create white noise covariance matrix
       #TODO: add ability to use multiple efacs for different backends
       white = PALutils.createWhiteNoiseCovarianceMatrix(psr[ll].err, efac[ll], equad[ll])

       # total auto-covariance matrix
       P = SC + SA + white

       # sandwich with G matrices
       Ppost = np.dot(psr[ll].G.T, np.dot(P, psr[ll].G))

       # do cholesky solve
       cf = sl.cho_factor(Ppost)

       # solution vector P^_1 r
       rr = sl.cho_solve(cf, r1)

       # temporarily store P^-1 r
       tmp.append(np.dot(psr[ll].G, rr))

       # add to log-likelihood
       loglike  += -0.5 * (np.sum(np.log(2*np.pi*np.diag(cf[0])**2)) + np.dot(r1, rr))

 
    # now compute cross terms
    k = 0
    for ll in range(npsr):
        for kk in range(ll+1, npsr):

            # create time lags
            tm = PALutils.createTimeLags(psr[ll].toas, psr[kk].toas)

            # create cross covariance matrix
            SIJ = PALutils.createRedNoiseCovarianceMatrix(tm, 1, gamgw)

            # carry out matrix-vetor operations
            tmp1 = np.dot(SIJ, tmp[kk])

            # add to likelihood
            loglike += ORF[k]/2 * Agw**2 * np.dot(tmp[ll], tmp1)
            
            # increment ORF counter
            k += 1

    return loglike

def lentatiMarginalizedLike(psr, d, FNF, dtNdt, logdet_N, rho, efac):
    """
    Lentati marginalized likelihood function

    """

    W = 1/efac


    arr = np.zeros(2*len(rho))
    ct = 0
    for ii in range(0, 2*len(rho), 2):
        arr[ii] = rho[ct]
        arr[ii+1] = rho[ct]
        ct += 1

    Phi = np.diag(10**arr)
    Sigma = W*FNF + np.diag(1/10**arr)

    cf = sl.cho_factor(Sigma)
    expval2 = sl.cho_solve(cf, d)
    logdet_Sigma = np.sum(2*np.log(np.diag(cf[0])))

    logdet_Phi = np.sum(np.log(10**arr))

    n = psr.G.shape[1]

    logLike = -0.5 * (logdet_N + logdet_Phi + logdet_Sigma + n*np.log(efac))\
                    - 0.5 * (W*dtNdt - W**2*np.dot(d, expval2))

    #print logdet_Sigma, logdet_Phi, W**2*np.dot(d, expval2)
  

    return logLike

