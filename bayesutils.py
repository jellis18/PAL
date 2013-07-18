import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.ndimage.filters as filter
import healpy as hp
from bayestar import plot
import matplotlib.mlab as ml
import statsmodels.api as sm
from matplotlib.ticker import FormatStrFormatter, LinearLocator, NullFormatter, NullLocator
import matplotlib.ticker
from optparse import OptionParser

"""
Given a 2D matrix of (marginalised) likelihood levels, this function returns
the 1, 2, 3- sigma levels. The 2D matrix is usually either a 2D histogram or a
likelihood scan

"""
def getsigmalevels(hist2d):
  # We will draw contours with these levels
  sigma1 = 0.68268949
  level1 = 0
  sigma2 = 0.95449974
  level2 = 0
  sigma3 = 0.99730024
  level3 = 0

  #
  lik = hist2d.reshape(hist2d.size)
  sortlik = np.sort(lik)

  # Figure out the 1sigma level
  dTotal = np.sum(sortlik)
  nIndex = sortlik.size
  dSum = 0
  while (dSum < dTotal * sigma1):
    nIndex -= 1
    dSum += sortlik[nIndex]
  level1 = sortlik[nIndex]

  # 2 sigma level
  nIndex = sortlik.size
  dSum = 0
  while (dSum < dTotal * sigma2):
    nIndex -= 1
    dSum += sortlik[nIndex]
  level2 = sortlik[nIndex]

  # 3 sigma level
  nIndex = sortlik.size
  dSum = 0
  while (dSum < dTotal * sigma3):
    nIndex -= 1
    dSum += sortlik[nIndex]
  level3 = sortlik[nIndex]

  return level1, level2, level3

def confinterval(samples, sigmalevel=1, onesided=False):
    """

    Given a list of samples, return the desired cofidence intervals.
    Returns the minimum and maximum confidence levels

    @param samples: Samples that we wish to get confidence intervals

    @param sigmalevel: Sigma level 1, 2, or 3 sigma, will return 
                       corresponding confidence limits

    @param onesided: Boolean to use onesided or twosided confidence
                     limits.

    """
    # The probabilities for different sigmas
    sigma = [0.68268949, 0.95449974, 0.99730024]

    # Create the ecdf function
    ecdf = sm.distributions.ECDF(samples)

    # Create the binning
    x = np.linspace(min(samples), max(samples), 1000)
    y = ecdf(x)

    # Find the intervals
    x2min = y[0]
    if onesided:
        bound = 1 - sigma[sigmalevel-1]
    else:
        bound = 0.5*(1-sigma[sigmalevel-1])

    for i in range(len(y)):
        if y[i] >= bound:
            x2min = x[i]
            break

    x2max = y[-1]
    if onesided:
        bound = sigma[sigmalevel-1]
    else:
        bound = 1 - 0.5 * (1 - sigma[sigmalevel-1])

    for i in reversed(range(len(y))):
        if y[i] <= bound:
            x2max = x[i]
            break

    return x2min, x2max



def makesubplot2d(ax, samples1, samples2, color=True, weights=None, smooth=True):

    xmin = np.min(samples1)
    xmax = np.max(samples1)
    ymin = np.min(samples2)
    ymax = np.max(samples2)

    hist2d,xedges,yedges = np.histogram2d(samples1, samples2, weights=weights, \
            bins=40,range=[[xmin,xmax],[ymin,ymax]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]
    
    xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])
    yedges = np.delete(yedges, -1) + 0.5*(yedges[1] - yedges[0])
    
    # gaussian smoothing
    if smooth:
        hist2d = filter.gaussian_filter(hist2d, sigma=0.75)
    
    level1, level2, level3 = getsigmalevels(hist2d)
    
    contourlevels = (level1, level2, level3)
    
    #contourcolors = ('darkblue', 'darkblue', 'darkblue')
    contourcolors = ('black', 'black', 'black')
    contourlinestyles = ('-', '--', ':')
    contourlinewidths = (1.5, 1.5, 1.5)
    contourlabels = [r'1 $\sigma$', r'2 $\sigma$',r'3 $\sigma$']
    
    contlabels = (contourlabels[0], contourlabels[1], contourlabels[2])

    c1 = ax.contour(xedges,yedges,hist2d.T,contourlevels, \
            colors=contourcolors, linestyles=contourlinestyles, \
            linewidths=contourlinewidths, zorder=2)
    if color:
        c2 = ax.imshow(np.flipud(hist2d.T), extent=extent, aspect=ax.get_aspect(), \
                  interpolation='gaussian')
    
    
def makesubplot1d(ax, samples, weights=None, interpolate=False, smooth=True):
    """ 
    Make histogram of samples

    """
    hist, xedges = np.histogram(samples, 30, normed=True)
    xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])

    # gaussian smoothing
    if smooth:
        hist = filter.gaussian_filter(hist, sigma=0.75)
        if interpolate:
            f = interp.interp1d(xedges, hist, kind='cubic')
            xedges = np.linspace(xedges.min(), xedges.max(), 1000)
            hist = f(xedges)

    # make plot
    ax.plot(xedges, hist, color='k', lw=1.5)
        

# make triangle plot of marginalized posterior distribution
def triplot(chain, color=True, weights=None, interpolate=False, smooth=True, \
           labels=None, figsize=(11,8.5), title=None, inj=None):

    """

    Make Triangle plot

    """

    # rcParams settings
    plt.rcParams['ytick.labelsize'] = 10.0
    plt.rcParams['xtick.labelsize'] = 10.0
    plt.rcParams['text.usetex'] = True
    plt.rcParams['figure.figsize'] = figsize

    # get number of parameters
    ndim = chain.shape[1]
    parameters = np.linspace(0,ndim-1,ndim)

    f, axarr = plt.subplots(nrows=len(parameters), ncols=len(parameters),figsize=figsize)

    for i in range(len(parameters)):
        # for j in len(parameters[np.where(i <= parameters)]:
        for j in range(len(parameters)):
            ii = i
            jj = len(parameters) - j - 1

            xmajorLocator = matplotlib.ticker.MaxNLocator(nbins=4,prune='both')
            ymajorLocator = matplotlib.ticker.MaxNLocator(nbins=4,prune='both')

            if j <= len(parameters)-i-1:
                axarr[jj][ii].xaxis.set_minor_locator(NullLocator())
                axarr[jj][ii].yaxis.set_minor_locator(NullLocator())
                axarr[jj][ii].xaxis.set_major_locator(NullLocator())
                axarr[jj][ii].yaxis.set_major_locator(NullLocator())

                axarr[jj][ii].xaxis.set_minor_formatter(NullFormatter())
                axarr[jj][ii].yaxis.set_minor_formatter(NullFormatter())
                axarr[jj][ii].xaxis.set_major_formatter(NullFormatter())
                axarr[jj][ii].yaxis.set_major_formatter(NullFormatter())
                xmajorFormatter = FormatStrFormatter('%g')
                ymajorFormatter = FormatStrFormatter('%g')

                if ii == jj:
                    # Make a 1D plot
                    makesubplot1d(axarr[ii][ii], chain[:,parameters[ii]], \
                                  weights=weights, interpolate=interpolate, \
                                  smooth=smooth)

                    if inj is not None:
                        axarr[ii][ii].axvline(inj[ii], lw=2, color='k')
                else:
                    # Make a 2D plot
                    makesubplot2d(axarr[jj][ii], chain[:,parameters[ii]], \
                            chain[:,parameters[jj]],color=color, weights=weights, \
                                  smooth=smooth)

                    if inj is not None:
                        axarr[jj][ii].plot(inj[ii], inj[jj], 'x', color='k', markersize=12, mew=2, mec='k')

                axarr[jj][ii].xaxis.set_major_locator(xmajorLocator)
                axarr[jj][ii].yaxis.set_major_locator(ymajorLocator)
            else:
                axarr[jj][ii].set_visible(False)
                #axarr[jj][ii].axis('off')

            if jj == len(parameters)-1:
                axarr[jj][ii].xaxis.set_major_formatter(xmajorFormatter)
                if labels:
                    axarr[jj][ii].set_xlabel(labels[ii])

            if ii == 0:
                if jj == 0:
                    axarr[jj][ii].yaxis.set_major_locator(NullLocator())
                    #axarr[jj][ii].set_ylabel('Post.')
                else:
                    axarr[jj][ii].yaxis.set_major_formatter(ymajorFormatter)
                    if labels:
                        axarr[jj][ii].set_ylabel(labels[jj])

    # overall plot title
    if title:
        f.suptitle(title, fontsize=14, y=0.90)
     
    # make plots closer together 
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0.1)


def pol2cart(lon, lat): 
    """ 
    Utility function to convert longitude,latitude on a unit sphere to 
    cartesian co-ordinates.

    """ 

    x = np.cos(lat)*np.cos(lon) 
    y = np.cos(lat)*np.sin(lon) 
    z = np.sin(lat) 

    return np.array([x,y,z])


def greedy_bin_sky(skypos, skycarts):
    """

    Greedy binning algorithm

    """

    N = len(skycarts) 
    skycarts = np.array(skycarts)
    bins = np.zeros(N) 
    for raSample, decSample in skypos: 
        sampcart = pol2cart(raSample, decSample) 
        dx = np.dot(skycarts, sampcart)
        maxdx = np.argmax(dx)
        bins[maxdx] += 1 

    # fill in skymap
    histIndices = np.argsort(bins)[::-1]    # in decreasing order
    NSamples = len(skypos)

    frac = 0.0
    skymap = np.zeros(N)
    for i in histIndices:
        frac = float(bins[i])/float(NSamples) 
        skymap[i] = frac

    return skymap


def plotSkyMap(raSample, decSample, nside=64, contours=None, colorbar=True, \
              inj=None, psrs=None):
    """

    Plot Skymap of chain samples on Mollwiede projection.

    @param raSample: Array of right ascension samples
    @param decSample: Array of declination  samples
    @param nside: Number of pixels across equator [default = 64]
    @param contours: Confidence contours to draw eg. 68%, 95% etc
                     By default this is set to none and no contours
                     will be drawn.
    @param colorbar: Boolean option to draw colorbar [default = True]
    @param inj: list of injected values [ra, dec] in radians to plot
                [default = None]
    @param psrs: Stacked array of pulsar sky locations [ra, dec] in radians
                 [default=None] Will plot as white diamonds

    """

    # clear figures
    plt.clf()

    # create stacked array of ra and dec
    skypos = np.column_stack([raSample, decSample])

    npix = hp.nside2npix(nside)    # number of pixels total


    # initialize theta and phi map coordinantes
    skycarts=[]
    for ii in range(npix):
        skycarts.append(np.array(hp.pix2vec(nside,ii)))

    # get skymap values from greedy binning algorithm
    skymap = greedy_bin_sky(skypos, skycarts)

    # smooth skymap
    skymap = hp.smoothing(skymap, 0.05)

    # make plot
    ax = plt.subplot(111, projection='astro mollweide')

    # Add contours
    if contours is not None:
        for percent in contours:
            indices = np.argsort(-skymap)
            sky = skymap[indices]
            region = np.zeros(skymap.shape)
            ind = np.min(ml.find(np.cumsum(sky) >= 0.01*percent))
            region[indices[0:ind]] = 1.0
            cs = plot.contour(lambda lon, lat: region[hp.ang2pix(nside, 0.5*np.pi - lat, lon)], \
                          colors='k', linewidths=1.0, levels=[0.5])
            #plt.clabel(cs, [0.5], fmt={0.5: '$\mathbf{%d\%%}$' % percent}, fontsize=8, inline=True)

    # plot map
    ax.grid()
    plot.outline_text(ax)
    plot.healpix_heatmap(skymap)

    # add injection
    if inj:
        ax.plot(inj[0], inj[1], 'x', color='k', markersize=8, mew=2, mec='k')

    # add pulsars
    if psrs:
        ax.plot(psrs[:,0], psrs[:,1], 'D', color='w', markersize=3, mew=1, mec='w')

    # add colorbar and title
    if colorbar:
        plt.colorbar(orientation='horizontal')
        plt.suptitle(r'$\log\, p(\alpha,\delta|d)$', y=0.1)

    # save skymap
    plt.savefig('skymap.pdf', bbox_inches='tight')



def upperlimitplot2d(x, y, sigma=2, ymin=None, ymax=None, bins=40, log=False, \
                     savename=None, labels=None, hold=False, **kwargs):

    """

    Make upper limits of a parameter as a function of another.

    @param x: Parameter we are making upper limits for
    @param y: Parameter which we will bin
    @param sigma: Sigma level of upper limit
    @param ymin: Minimum value of binning parameter [default=None]
    @param ymax: Maximum value of binning parameter [default=None]
    @param bins: Number of bins
    @param log: If True, plot on log-log scale
    @param savename: Output filename for saved figure
    @param labels: List of labels for axes [xlabel, ylabel]
    @param hold: Hold current figure?

    """

    # clear current figure
    if hold == False:
        plt.clf()

    if ymin is None:
        ymin = y.min()
    if ymax is None:
        ymax = y.max()

    yedges = np.linspace(ymin, ymax, bins+1)
    deltay = yedges[1] - yedges[0]
    yvals = np.linspace(ymin+0.5*deltay, ymax-0.5*deltay, bins)
    bin_index = []
    upper = []

    for i in range(bins):
        # Obtain the indices in the range of the bin
        indices = np.flatnonzero(np.logical_and(y>yedges[i], y<yedges[i+1]))

        # Obtain the 1-sided x-sigma upper limit
        if len(indices) > 0:
            bin_index.append(i)
            a, sigma1 = confinterval(x[indices], sigmalevel=sigma, onesided=True)
            upper.append(sigma1)

    # make bin_indes and upper into arrays
    bin_index = np.array(bin_index)
    upper = np.array(upper)

    # make plot
    if log:
        plt.loglog(10**yvals[bin_index], 10**upper, **kwargs)
        plt.grid(which='major')
        plt.grid(which='minor')
    else:
        plt.plot(yvals[bin_index], upper, **kwargs)
        plt.grid()

    # labels
    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

    if savename:
        plt.savefig(savename, bbox_inches='tight')
    else:
        plt.savefig('2dUpperLimit.pdf', bbox_inches='tight')

    










