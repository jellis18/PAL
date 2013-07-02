from __future__ import division
import numpy as np
import scipy.special as ss
import scipy.linalg as sl
import scipy.interpolate as interp
import sys,os
import PALutils

def createStochasticResiduals(psr, Amp, gam):
    """
		Function to create GW incuced residuals from a stochastic GWB as defined
		in Chamberlin, Creighton, Demorest et al. (2013)
		
		@param psr: pulsar object for single pulsar
		@param Amp: Amplitude of red noise in GW units
		@param gam: Red noise power law spectral index
		
		@return: Vector of induced residuals
		
		"""
	
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
    ORF = PALutils.computeORFMatrix(psr)

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
        res_gw.append(f(psr[ll].toas))

    return res_gw
