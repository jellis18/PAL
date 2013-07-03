from __future__ import division
import numpy as np
import scipy.special as ss
import scipy.linalg as sl
import scipy.integrate as si
import scipy.interpolate as interp
#import numexpr as ne
import sys,os

def createAntennaPatternFuncs(psr, gwtheta, gwphi):
    """
    Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).

    @param psr: pulsar object for single pulsar
    @param gwtheta: GW polar angle in radians
    @param gwphi: GW azimuthal angle in radians

    @return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the 
             pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = [-np.sin(gwphi), np.cos(gwphi), 0.0]
    n = [-np.cos(gwtheta)*np.cos(gwphi), -np.cos(gwtheta)*np.sin(gwphi), np.sin(gwtheta)]
    omhat = [-np.sin(gwtheta)*np.cos(gwphi), -np.sin(gwtheta)*np.sin(gwphi), -np.cos(gwtheta)]

    phat = [np.sin(psr.theta)*np.cos(psr.phi), np.sin(psr.theta)*np.sin(psr.phi), np.cos(psr.theta)]

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    return fplus, fcross, cosMu

def createResiduals(psr, gwtheta, gwphi, mc, dist, fgw, phase0, psi, inc, \
                     psrTerm=True, evolve=True):
    """
    Function to create GW incuced residuals from a SMBMB as 
    defined in Ellis et. al 2012,2013.

    @param psr: pulsar object for single pulsar
    @param gwtheta: Polar angle of GW source in celestial coords [radians]
    @param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    @param mc: Chirp mass of SMBMB [solar masses]
    @param dist: Luminosity distance to SMBMB [Mpc]
    @param fgw: Frequency of GW (twice the orbital frequency) [Hz]
    @param phase0: Initial Phase of GW source [radians]
    @param psi: Polarization of GW source [radians]
    @param inc: Inclination of GW source [radians]
    @param psrTerm: Option to include pulsar term [boolean] 
    @param evolve: Option to exclude evolution [boolean]

    @return: Vector of induced residuals

    """

    # get values from pulsar object
    toas = psr.toas
    pdist = psr.dist

    # get antenna pattern funcs and cosMu
    fplus, fcross, cosMu = createAntennaPatternFuncs(psr, gwtheta, gwphi)
    
    # convert units
    mc *= 4.9e-6         # convert from solar masses to seconds
    dist *= 1.0267e14    # convert from Mpc to seconds
    pdist *= 1.0267e11   # convert from kpc to seconds
    
    # get pulsar time
    tp = toas-pdist*(1-cosMu)
    
    # evolution
    if evolve:

        # calculate time dependent frequency at earth and pulsar
        fdot = (96/5) * np.pi**(8/3) * mc**(5/3) * (fgw)**(11/3)
        omega = 2*np.pi*fgw*(1-8/3*fdot/fgw*toas)**(-3/8)
        omega_p = 2*np.pi*fgw*(1-256/5 * mc**(5/3) * np.pi**(8/3) * fgw**(8/3) *tp)**(-3/8)

  
        # calculate time dependent phase
        phase = phase0+ 2*np.pi/(32*np.pi**(8/3)*mc**(5./3.))*\
                (fgw**(-5/3) - (omega/2/np.pi)**(-5/3))
        phase_p = phase0+ 2*np.pi/(32*np.pi**(8/3)*mc**(5./3.))*\
                (fgw**(-5/3) - (omega_p/2/np.pi)**(-5/3))
          
    # no evolution
    else: 
        
        # monochromatic
        omega = 2*np.pi*fgw
        omega_p = omega
        
        # phases
        phase = phase0 + omega * toas
        phase_p = phase0 + omega * tp
        

    # define time dependent coefficients
    At = -0.5*np.sin(phase)*(3+np.cos(2*inc))
    Bt = 2*np.cos(phase)*np.cos(inc)
    At_p = -0.5*np.sin(phase_p)*(3+np.cos(2*inc))
    Bt_p = 2*np.cos(phase_p)*np.cos(inc)

    # now define time dependent amplitudes
    alpha = mc**(5./3.)/(dist*(omega/2)**(1./3.))
    alpha_p = mc**(5./3.)/(dist*(omega_p/2)**(1./3.))


    # define rplus and rcross
    rplus = alpha*(At*np.cos(2*psi)-Bt*np.sin(2*psi))
    rcross = alpha*(At*np.sin(2*psi)+Bt*np.sin(2*psi))
    rplus_p = alpha_p*(At_p*np.cos(2*psi)-Bt_p*np.sin(2*psi))
    rcross_p = alpha_p*(At_p*np.sin(2*psi)+Bt_p*np.sin(2*psi))

    # residuals
    if psrTerm:
        res = fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross)
    else:
        res = -fplus*rplus - fcross*rcross

    return res


def computeLuminosityDistance(z):
    """

    Compute luminosity distance via gaussian quadrature.

    @param z: Redshift at which to calculate luminosity distance

    @return: Luminosity distance in Mpc

    """

    # constants
    H0 = 71 * 1000      # Hubble constant in (m/s)/Mpc
    Ol = 0.73           # Omega lambda
    Om = 0.27           # Omega matter
    G = 6.67e-11        # Gravitational constant in SI units
    c = 3.0e8           # Speed of light in SI units

    # proper distance function
    properDistance = lambda z: c/H0*np.sqrt(Ol+Om*(1+z)**3)
    
    #def properDistance(z):
    #    
    #    Ez = np.sqrt(Ol+Om*(1+z)**3)
    #    
    #    return c/H0/Ez

    # carry out numerical integration
    Dp = si.quadrature(properDistance, 0 ,z)[0]
    Dl = (1+z) * Dp

    return Dl

def createRmatrix(designmatrix, err):
    """
    Create R matrix as defined in Ellis et al (2013) and Demorest et al (2012)

    @param designmatrix: Design matrix as returned by tempo2

    @return: R matrix 
   
    """

    W = np.diag(1/err)

    u, s, v = sl.svd(np.dot(W, designmatrix),full_matrices=False)

    return np.eye(len(err)) - np.dot(np.linalg.inv(W), np.dot(u, np.dot(u.T, W))) 


def createGmatrix(designmatrix):
    """
    Return G matrix as defined in van Haasteren et al 2013

    @param designmatrix: Design matrix as returned by tempo2

    @return: G matrix as defined in van Haasteren et al 2013

    """

    nfit = designmatrix.shape[1]
    npts = designmatrix.shape[0]

    # take singular value decomp
    u, s, v = sl.svd(designmatrix, full_matrices=True)

    return u[:,-(npts-nfit):]


def createTimeLags(toa1, toa2, round=True):
    """
    Create matrix of time lags tm = |t_i - t_j|

    @param toa1: times-of-arrival in seconds for psr 1
    @param toa2: times-of-arrival in seconds for psr 2
    @param round: option to round time difference to 0 if less than 1 hr

    @return: matrix of time lags tm = |t_i - t_j|

    """

    t1, t2 = np.meshgrid(toa2, toa1)

    tm = np.abs(t1-t2)

    if round:
        hr = 3600. # hour in seconds
        tm = np.where(tm<hr, 0.0, tm)
        
    return tm


def sumTermCovarianceMatrix(tm, fL, gam, nsteps):
    """
    Calculate the power series expansion for the Hypergeometric
    function in the standard power law covariance matrix.

    @param tm: Matrix of time lags in years
    @param fL: Low frequency cutoff
    @param gam: Power Law spectral index
    @param nsteps: Number of terms in the power series expansion
    """

    sum=0
    for i in range(nsteps):

     sum += ((-1)**i)*((2*np.pi*fL*tm)**(2*i))/(ss.gamma(2*i+1)*(2.*i+1-gam))

    return sum

def sumTermCovarianceMatrix_fast(tm, fL, gam):
    """
    Calculate the power series expansion for the Hypergeometric
    function in the standard power law covariance matrix. This
    version uses the Python package numexpr and is much faster
    that using numpy. For now it is hardcoded to use only the 
    first 3 terms.

    @param tm: Matrix of time lags in years
    @param fL: Low frequency cutoff
    @param gam: Power Law spectral index
    """

    x = 2*np.pi*fL*tm

    sum = ne.evaluate("1/(1-gam) - x**2/(2*(3-gam)) + x**4/(24*(5-gam))")

    return sum



def createRedNoiseCovarianceMatrix(tm, Amp, gam, fH=None, fast=False):
    """
    Create red noise covariance matrix. If fH is None, then
    return standard power law covariance matrix. If fH is not
    none, return power law covariance matrix with high frequency 
    cutoff.

    @param tm: Matrix of time lags in seconds
    @param Amp: Amplitude of red noise in GW units
    @param gam: Red noise power law spectral index
    @param fH: Optional high frequency cutoff in yr^-1
    @param fast: Option to use Python numexpr to speed 
                    up calculation (default = True)

    @return: Red noise covariance matrix in seconds^2

    """

    # conversion from seconds to years
    s2yr = 1/3.16e7
    
    # convert tm to yr
    tm *= s2yr

    # compute high frequency cutoff
    Tspan = tm.max()
    fL = 1/(10*Tspan)


    if fH is None:

        # convert amplitude to correct units
        A = Amp**2/24/np.pi**2
        if fast:
            x = 2*np.pi*fL*tm
            corr = (2*A/(fL**(gam-1)))*((ss.gamma(1-gam)*np.sin(np.pi*gam/2)*ne.evaluate("x**(gam-1)")) \
                            -sumTermCovarianceMatrix_fast(tm, fL, gam))
        else:
            corr = (2*A/(fL**(gam-1)))*((ss.gamma(1-gam)*np.sin(np.pi*gam/2)*(2*np.pi*fL*tm)**(gam-1)) \
                            -sumTermCovarianceMatrix(tm, fL, gam, 5))

    elif fH is not None:

        alpha=(3-gam)/2.0

        # convert amplitude to correct units
        A = Amp**2
 
        EulerGamma=0.577

        x = 2*np.pi*fL*tm

        norm = (fL**(2*alpha - 2)) * 2**(alpha - 3) / (3 * np.pi**1.5 * ss.gamma(1.5 - alpha))

        # introduce a high-frequency cutoff
        xi = fH/fL
        
        # avoid the gamma singularity at alpha = 1
        if np.abs(alpha - 1) < 1e-6:
            zero = np.log(xi) + (EulerGamma + np.log(0.5 * xi)) * np.log(xi) * (alpha - 1)
        else:
            zero = norm * 2**(-alpha) * ss.gamma(1 - alpha) * (1 - xi**(2*alpha - 2))

        corr = A * np.where(x==0,zero,norm * x**(1 - alpha) * (ss.kv(1 - alpha,x) - xi**(alpha - 1) \
                                                           * ss.kv(1 - alpha,xi * x)))

    # return in s^2
    return corr / (s2yr**2)

def createWhiteNoiseCovarianceMatrix(err, efac, equad, tau=None, tm=None):
    """
    Return standard white noise covariance matrix with
    efac and equad parameters

    @param err: Error bars on toas in seconds
    @param efac: Multiplier on error bar component of covariance matrix
    @param equad: Extra toa independent white noise in seconds
    @param tau: Extra time scale of correlation if appropriate. If this
                parameter is specified must also read in matrix of time lags
    @param tm: Matrix of time lags.

    @return: White noise covariance matrix in seconds^2

    """
    
    if tau is None and tm is None:
        corr = efac * np.diag(err**2) + equad**2 * np.eye(np.alen(err)) 

    elif tau is not None and tm is not None:
        sincFunc = np.sinc(2*np.pi*tm/tau)
        corr = efac * np.diag(err**2) + equad**2 * sincFunc

    return corr

# return the false alarm probability for the fp-statistic
def ptSum(N, fp0):
    """
    Compute False alarm rate for Fp-Statistic. We calculate
    the log of the FAP and then exponentiate it in order
    to avoid numerical precision problems

    @param N: number of pulsars in the search
    @param fp0: The measured value of the Fp-statistic

    @returns: False alarm probability ad defined in Eq (64)
              of Ellis, Seiemens, Creighton (2012)

    """

    n = np.arange(0,N)

    return np.sum(np.exp(n*np.log(fp0)-fp0-np.log(ss.gamma(n+1))))

def dailyAverage(pulsar):
    """

    Function to compute daily averaged residuals such that we
    have one residual per day per frequency band.

    @param pulsar: pulsar class from Michele Vallisneri's 
                     libstempo library.

    @return: mtoas: Average TOA of a single epoch
    @return: qmatrix: Linear operator that transforms residuals to
                      daily averaged residuals
    @return: merr: Daily averaged error bar
    @return: mfreqs: Daily averaged frequency value
    @return: mbands: Frequency band for daily averaged residual

    """

    toas = pulsar.toas()        # days 
    res = pulsar.residuals()    # seconds
    err = pulsar.toaerrs * 1e-6 # seconds
    freqs = pulsar.freqs        # MHz

    # timescale to do averaging (1 day)
    t_ave = 86400    # s/day
    
    # set up array with one day spacing
    yedges = np.longdouble(np.arange(toas.min(),toas.max()+1,1))

    # unique frequency bands
    bands = list(np.unique(pulsar.flags['B']))
    flags = list(pulsar.flags['B'])

    qmatrix = []
    mtoas = []
    merr = []
    mres = []
    mfreqs = []
    mbands = []
    for ii in range(len(yedges)-1):

        # find toa indices that are in bin 
        indices = np.flatnonzero(np.logical_and(toas>=yedges[ii], toas<yedges[ii+1]))

        # loop over different frequency bands
        for band in bands:
            array = np.zeros(len(toas))

            # find indices in that band
            toainds = [ct for ct,flag in enumerate(flags) if flag == band]

            # find all indices that are within 1 day and are in frequency band
            ind = [indices[jj] for jj in range(len(indices)) if np.any(np.equal(indices[jj],toainds))]

            # construct average quantities
            if len(ind) > 0:
                weight = (np.sum(1/err[ind]**2))
                array[ind] = 1/err[ind]**2 / weight
                qmatrix.append(array)
                mtoas.append(np.mean(toas[ind]))
                merr.append(np.sqrt(1/np.sum(1/err[ind]**2)))
                mfreqs.append(np.mean(pulsar.freqs[ind]))
                mbands.append(band)

            
    # turn lists into arrays with double precision
    qmatrix = np.double(np.array(qmatrix))
    mtoas = np.double(np.array(mtoas))
    merr = np.double(np.array(merr))
    mfreqs = np.double(np.array(mfreqs))
    mbands = np.array(mbands)
    
    # construct new design matrix without inter band frequency jumps
    dmatrix = np.double(pulsar.designmatrix())[:,0:-pulsar.nJumps]

    return mtoas, qmatrix, merr, dmatrix, mfreqs, mbands

def computeORF(psr):
    """
    Compute pairwise overlap reduction function values.

    @param psr: List of pulsar object instances

    @return: Numpy array of pairwise ORF values for every pulsar
             in pulsar class

    """

    # begin loop over all pulsar pairs and calculate ORF
    k = 0
    npsr = len(psr)
    ORF = np.zeros(npsr*(npsr-1)/2.)
    phati = np.zeros(3)
    phatj = np.zeros(3)
    for ll in xrange(0, npsr):
        phati[0] = np.cos(psr[ll].phi) * np.sin(psr[ll].theta)
        phati[1] = np.sin(psr[ll].phi) * np.sin(psr[ll].theta)
        phati[2] = np.cos(psr[ll].theta)

        for kk in xrange(ll+1, npsr):
            phatj[0] = np.cos(psr[kk].phi) * np.sin(psr[kk].theta)
            phatj[1] = np.sin(psr[kk].phi) * np.sin(psr[kk].theta)
            phatj[2] = np.cos(psr[kk].theta)

            xip = (1.-np.sum(phati*phatj)) / 2.
            ORF[k] = 3.*( 1./3. + xip * ( np.log(xip) -1./6.) )
            k += 1

    return ORF

def computeORFMatrix(psr):
    """
    Compute ORF matrix.

    @param psr: List of pulsar object instances

    @return: Matrix that has the ORF values for every pulsar
             pair with 2 on the diagonals to account for the 
             pulsar term.

    """

    # begin loop over all pulsar pairs and calculate ORF
    npsr = len(psr)
    ORF = np.zeros((npsr, npsr))
    phati = np.zeros(3)
    phatj = np.zeros(3)
    for ll in xrange(0, npsr):
        phati[0] = np.cos(psr[ll].phi) * np.sin(psr[ll].theta)
        phati[1] = np.sin(psr[ll].phi) * np.sin(psr[ll].theta)
        phati[2] = np.cos(psr[ll].theta)

        for kk in xrange(0, npsr):
            phatj[0] = np.cos(psr[kk].phi) * np.sin(psr[kk].theta)
            phatj[1] = np.sin(psr[kk].phi) * np.sin(psr[kk].theta)
            phatj[2] = np.cos(psr[kk].theta)
            
            if ll != kk:
                xip = (1.-np.sum(phati*phatj)) / 2.
                ORF[ll, kk] = 3.*( 1./3. + xip * ( np.log(xip) -1./6.) )
            else:
                ORF[ll, kk] = 2.0

    return ORF


def twoComponentNoiseLike(Amp, D, c):
    """

    Likelihood function for two component noise model

    @param Amp: trial amplitude in GW units
    @param D: Vector of eigenvalues from diagonalized red noise
              covariance matrix
    @param c: Residuals written new diagonalized basis

    @return: loglike: The log-likelihood for this pulsar

    """

    loglike = -0.5 * np.sum(np.log(2*np.pi*(Amp**2*D + 1)) + c**2/(Amp**2*D + 1))

    return loglike

def angularSeparation(theta1, phi1, theta2, phi2):
    """
    Calculate the angular separation of two points on the sky.

    @param theta1: Polar angle of point 1 [radian]
    @param phi1: Azimuthal angle of point 1 [radian]
    @param theta2: Polar angle of point 2 [radian]
    @param phi2: Azimuthal angle of point 2 [radian]

    @return: Angular separation in radians

    """
    
    # unit vectors
    rhat1 = phat = [np.sin(theta1)*np.cos(phi1), np.sin(theta1)*np.sin(phi1), np.cos(theta1)]
    rhat2 = phat = [np.sin(theta2)*np.cos(phi2), np.sin(theta2)*np.sin(phi2), np.cos(theta2)]

    cosMu = np.dot(rhat1, rhat2)

    return np.arccos(cosMu)


def createfourierdesignmatrix(t, nmodes):
    """
    Construct fourier design matrix from eq 11 of Lentati et al, 2013

    @param t: vector of time series in seconds
    @param nmodes: number of fourier coefficients to use

    @return: F: fourier design matrix

    """

    N = len(t)
    F = np.zeros((N, 2*nmodes))
    T = t.max() - t.min()

    if nmodes % 2 == 0:
        print "WARNING: Number of modes should be odd!"

    # define sampling frequencies
    #f = np.logspace(np.log10(1/T), np.log10(nmodes/T), nmodes)
    f = np.linspace(1/T, nmodes/T, nmodes)

    # The sine/cosine modes
    ct = 0
    for ii in range(0, 2*nmodes-1, 2):
        
        F[:,ii] = np.cos(2*np.pi*f[ct]*t)
        F[:,ii+1] = np.sin(2*np.pi*f[ct]*t)
        ct += 1

    return F

def createGWB(psr, Amp, gam, DM=False):
    """
		Function to create GW incuced residuals from a stochastic GWB as defined
		in Chamberlin, Creighton, Demorest et al. (2013)
		
		@param psr: pulsar object for single pulsar
		@param Amp: Amplitude of red noise in GW units
		@param gam: Red noise power law spectral index
		
		@return: Vector of induced residuals
		
		"""
    print 'in createGWB', Amp, gam, DM	
    # get maximum number of points
    npts = np.max([p.ntoa for p in psr])

    Npulsars = len(psr)

    # current Hubble scale in units of 1/day
    H0=(2.27e-18)*(60.*60.*24.)

    # create simulated GW time span (start and end times). Will be slightly larger than real data span

    #gw start and end times for entire data set
    start = np.min([p.toas.min() for p in psr]) - 86400
    stop = np.max([p.toas.max() for p in psr]) + 86400
        
    # define "how much longer" or howml variable, needed because IFFT cannot quite match the value of the integral of < |r(t)|^2 > 
    howml = 10.

    # duration of the signal, spanning total time data taken in days
    dur = stop - start

    # make a vector of evenly sampled data points
    ut = np.linspace(start, stop, npts)

    # time resolution in days
    dt = dur/npts

    # compute the overlap reduction function
    ORF = computeORFMatrix(psr)

    # define frequencies spanning from DC to Nyquist. This is a vector spanning these frequencies in increments of 1/(dur*howml).
    f=np.arange(0, 1./(2.*dt), 1./(dur*howml))

    Nf=len(f)

    # Use Cholesky transform to take 'square root' of ORF
    M=np.linalg.cholesky(ORF)

    # Create random frequency series from zero mean, unit variance, Gaussian distributions
    w = np.zeros((Npulsars, Nf), complex)
    for ll in np.arange(Npulsars):
        w[ll,:] = np.random.randn(Nf) + 1j*np.random.randn(Nf)

    # Calculate strain spectral index alpha, beta
    alpha_f = -1./2.*(gam-3)

    # Value of secondary spectral index beta (note: beta = 2+2*alpha)
    beta_f=2.*alpha_f+2.

    # convert Amp to Omega
    f1yr_sec = 1./3.16e7
    Omega_beta = (2./3.)*(np.pi**2.)/(H0**2.)*float(Amp)**2*(1/f1yr_sec)**(2*alpha_f)

    # calculate GW amplitude Omega 
    Omega=Omega_beta*f**(beta_f)
    print f
    # Calculate frequency dependent pre-factor C(f)
    # could rewrite in terms of A instead of Omega for convenience.
    C=H0**2./(16.*np.pi**2)/(2.*np.pi)**2 * f**(-5.) * Omega * (dur * howml)

    ### injection residuals in the frequency domain
    Res_f=np.dot(M,w)
    for ll in np.arange(Npulsars):
        Res_f[ll] = Res_f[ll] * C**(0.5)    #rescale by frequency dependent factor
        Res_f[ll,0] = 0						#set DC bin to zero to avoid infinities
        Res_f[ll,-1] = 0					#set Nyquist bin to zero also

    # Now fill in bins after Nyquist (for fft data packing) and take inverse FT
    Res_f2 = np.zeros((Npulsars, 2*Nf-2), complex)    # make larger array for residuals
    Res_t =  np.zeros((Npulsars, 2*Nf-2))
    for ll in np.arange(Npulsars):
        for kk in np.arange(Nf):					# copies values into the new array up to Nyquist        
            Res_f2[ll,kk] = Res_f[ll,kk]
        for jj in np.arange(Nf-2):					# pads the values bigger than Nyquist with frequencies back down to 1. Biggest possible index of this array is 2*Nf-3.
            Res_f2[ll,Nf+jj] = np.conj(Res_f[ll,(Nf-2)-jj])

        ## rows: each row corresponds to a pulsar
        ## columns: each col corresponds to a value of the time series containing injection signal.
        Res_t[ll,:]=np.real(np.fft.ifft(Res_f2[ll,:])/dt)     #ifft includes a factor of 1/N, so divide by dt to effectively multiply by df=1/T

    # shorten data and interpolate onto TOAs
    Res = np.zeros((Npulsars, npts))
    res_gw = []
    for ll in range(Npulsars):
        Res[ll,:] = Res_t[ll, 100:(npts+100)]
        f = interp.interp1d(ut, Res[ll,:], kind='linear')
	if DM and len(psr)==1:
	    print 'adding DM to toas'
            res_gw.append(f(psr[ll].toas)/((2.410*1E-16)*psr[ll].freqs**2))
	else:
	    res_gw.append(f(psr[ll].toas))

    return res_gw


