"""
Convenience functions for collecting aperture and psf-fitting photometry, with
flux uncertainties defined using a "brute force" algorithm: dropping fake
stars around the target source and recovering their fluxes, or collecting
empty aperture flux measurements and then fitting a gaussian to the
histogram of recovered flux values.

Also includes functions for displaying images before, during and after applying
the photometry routines (esp. useful for examining residuals after psf fitting)
"""
import os
import numpy as np
try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits
from .aper import aper
from .cntrd import cntrd
from .pkfit_norecenter import pkfit_class
from .dao_value import dao_value
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats

def mkpsfimage(psfmodel, x, y, size, fluxscale=1):
    """  Construct a numpy array with the psf model appropriately scaled,
    using the gaussian parameters from the header and the residual components
    from the image array of the given fits file

    :param psfmodel: the psf model, provided as either
       (a) a fits file containing the psf model with gaussian parameters in
       the header and a lookup table of residual values in the data array, or
       (b) a tuple with the list of gaussian parameters as the first value and
       the lookup table in the second, as returned by getpsfmodel
    :param x,y: float values in the range [0,1) giving the sub-pixel position
       for the desired center of the psf in the output image
    :param size: width and height in pixels for the output image showing the
       psf realization
    :param fluxscale: scale the output psf image to have this total value for
       the flux, summed across all pixels in the output image
    :return: a numpy array holding the psf image realization
    """

    # require stampsize to be odd, so the central pixel contains
    # the peak of the psf. Define dx,dy as the half-width of the stamp on
    # either side of the central row/column
    assert isinstance(size, int)
    if size % 2 == 0:
        size += 1
    dx = dy = (size - 1) / 2

    gaussparam, lookuptable, psfmag, psfzpt = rdpsfmodel(psfmodel)

    # make a 2D data array with the realized psf image (gaussian+residuals)
    xx = np.tile(np.arange(-dx, dx + 1, 1, float), (size, 1))
    yy = np.tile(np.arange(-dy, dy + 1, 1, float), (size, 1)).T

    psfimage = dao_value(xx - x, yy - y, gaussparam, lookuptable, deriv=False)
    psfstarflux = 10 ** (-0.4 * (psfmag - psfzpt))
    psfimage *= fluxscale / psfstarflux

    return psfimage

def rdpsfmodel(psfmodelfile):
    """ Read in a psf model from a fits file. Gaussian parameters are in the
    header, and the image array has a lookup table of non-gaussian components
    sub-sampled to a half-pixel grid.
    If the user provides a 2-tuple instead of a filename, then we presume the
    user already has the psf model components, so we just return it back.

    :param psfmodelfile: a fits file containing the psf model
    :return: [gaussparam, lookuptable]
    """
    psfmodelfile = psfmodelfile
    if isinstance(psfmodelfile, str):
        assert os.path.isfile(
            os.path.abspath(os.path.expanduser(psfmodelfile)))
        # read in the psf non-gaussian components array (i.e. the lookup table)
        lookuptable = pyfits.getdata(psfmodelfile)
        # read in the gaussian parameters from the image header
        hdr = pyfits.getheader(psfmodelfile)
        scale = hdr['GAUSS1']  # , 'Gaussian Scale Factor'
        xpsf = hdr['GAUSS2']  # , 'Gaussian X Position'
        ypsf = hdr['GAUSS3']  # , 'Gaussian Y Position'
        xsigma = hdr['GAUSS4']  # , 'Gaussian Sigma: X Direction'
        ysigma = hdr['GAUSS5']  # , 'Gaussian Sigma: Y Direction'
        psfmag = hdr['PSFMAG']  # , 'aperture magnitude of the PSF star
        psfzpt = hdr['PSFZPT']  # , 'zeropoint used to set PSF star mag scaling
        gaussparam = [scale, xpsf, ypsf, xsigma, ysigma]
    elif np.iterable(psfmodelfile):
        assert len(psfmodelfile) == 4
        gaussparam, lookuptable, psfmag, psfzpt = psfmodelfile
    else:
        raise RuntimeError(
            "psfmodel must either be a filename or a 4-tuple giving:"
            "[gaussian parameters, look-up table, psf mag, zpt]")
    return gaussparam, lookuptable, psfmag, psfzpt


def addtoimarray(imagedat, psfmodel, xy, fluxscale=1):
    """  Generate a fake star and add it into the given image array at the
    specified  xy position, then return the modified image array.

   :param imagedat: numpy array holding the original image
   :param psfmodel: the psf model, provided as either
       (a) a fits file containing the psf model with gaussian parameters in
       the header and a lookup table of residual values in the data array, or
       (b) a tuple with the list of gaussian parameters as the first value and
       the lookup table in the second, as returned by getpsfmodel
    :param xy: tuple with 2 float values giving the x,y pixel position
       for the fake psf to be planted.  These should be in IDL/python
       convention (where [0,0] is the lower left corner) and not in the
       FITS convention (where [1,1] is the lower left corner)
    :param fluxscale: flux scaling factor for the psf
    :return: a numpy array showing the psf image realization
    """
    gaussparam, lookuptable, psfmag, psfzpt = rdpsfmodel(psfmodel)

    maxsize = int(np.sqrt(lookuptable.size) / 2 - 2)
    x, y = xy
    dx = dy = int((maxsize - 1) / 2)

    # make a psf postage stamp image and add it into the image array
    psfimage = mkpsfimage(psfmodel, x % 1, y % 1, dx * 2 + 1, fluxscale)

    # TODO : handle cases where the location is too close to the image edge
    # imagedat[int(y)-dy:int(y)+dy+1, int(x)-dx:int(x)+dx+1] += psfimage
    imagedat[int(y) - dy:int(y) + dy + 1, int(x) - dx:int(x) + dx + 1]\
        += psfimage

    return imagedat


def add_and_recover(imagedat, psfmodel, xy, fluxscale=1, psfradius=5,
                    skyannpix=None, skyalgorithm='sigmaclipping',
                    setskyval=None, recenter=False, ronoise=1, phpadu=1,
                    cleanup=True, verbose=False, debug=False):
    """  Add a single fake star psf model to the image at the given position
    and flux scaling, re-measure the flux at that position and report it,
    Also deletes the planted psf from the imagedat array so that we don't
    pollute that image array.

    :param imagedat: target image numpy data array
    :param psfmodel: psf model fits file or tuple with [gaussparam,lookuptable]
    :param xy: x,y position for fake psf, using the IDL/python convention
        where [0,0] is the lower left corner.
    :param fluxscale: flux scaling to apply to the planted psf
    :param recenter: use cntrd to locate the center of the added psf, instead
        of relying on the input x,y position to define the psf fitting
    :param cleanup: remove the planted psf from the input imagedat array.
    :return:
    """
    if not skyannpix:
        skyannpix = [8, 15]

    # add the psf to the image data array
    imdatwithpsf = addtoimarray(imagedat, psfmodel, xy, fluxscale=fluxscale)

    # TODO: allow for uncertainty in the x,y positions

    gaussparam, lookuptable, psfmag, psfzpt = rdpsfmodel(psfmodel)

    # generate an instance of the pkfit class for this psf model
    # and target image
    pk = pkfit_class(imdatwithpsf, gaussparam, lookuptable, ronoise, phpadu)
    x, y = xy

    if debug:
        from .photfunctions import showpkfit
        from matplotlib import pyplot as pl, cm

        fig = pl.figure(3)
        showpkfit(imdatwithpsf, psfmodel, xy, 11, fluxscale, verbose=True)

        fig = pl.figure(1)
        pl.imshow(imdatwithpsf[y - 20:y + 20, x - 20:x + 20], cmap=cm.Greys,
                  interpolation='nearest')
        pl.colorbar()

        import pdb

        pdb.set_trace()

    if recenter:
        xc, yc = cntrd(imdatwithpsf, x, y, psfradius, verbose=verbose)
        if xc > 0 and yc > 0 and abs(xc - xy[0]) < 5 and abs(yc - xy[1]) < 5:
            x, y = xc, yc
    # do aperture photometry to get the sky
    aperout = aper(imdatwithpsf, x, y, phpadu=phpadu,
                   apr=psfradius * 3, skyrad=skyannpix,
                   setskyval=(setskyval is not None and setskyval),
                   zeropoint=psfzpt, exact=False,
                   verbose=verbose, skyalgorithm=skyalgorithm,
                   debug=debug)
    apmag, apmagerr, apflux, apfluxerr, sky, skyerr, apbadflag, apoutstr\
        = aperout

    # do the psf fitting
    try:
        scale = pk.pkfit_fast_norecenter(1, x, y, sky, psfradius)
        fluxpsf = scale * 10 ** (-0.4 * (psfmag - psfzpt))
    except RuntimeWarning:
        print("photfunctions.add_and_recover failed on RuntimeWarning")
        fluxpsf = np.inf
    if cleanup:
        # remove the fake psf from the image
        imagedat = addtoimarray(imdatwithpsf, psfmodel, xy,
                                fluxscale=-fluxscale)

    return apflux[0], fluxpsf, [x, y]


def showpkfit(imagedat, psfmodelfile, xyposition, stampsize, fluxscale,
              verbose=False, **imshowkws):
    """   Display the psf fitting photometry of a star.

    :param imagedat: numpy array with the target image data
    :param psfimfile: fits image containing the psf model
    :param xyposition: tuple giving position of the center of the target
                    in imagedat pixels
    :param stampsize:  width and height of the image display in pixels
    :return:
    """
    from matplotlib import pyplot as pl, cm, ticker

    # require stampsize to be odd, so the central pixel contains
    # the peak of the psf. Define dx,dy as the half-width of the stamp on
    # either side of the central row/column
    assert isinstance(stampsize, int)
    if stampsize % 2 == 0:
        stampsize += 1
    dx = dy = (stampsize - 1) / 2

    # break up the x,y coordinates of the target into its integer
    # and decimal components
    xint, yint = int(xyposition[0]), int(xyposition[1])
    xsubpix, ysubpix = xyposition[0] % 1, xyposition[1] % 1

    # display a cutout of the target image, centered at the target location
    fig = pl.gcf()
    fig.set_size_inches([8, 3])
    pl.clf()
    ax1 = fig.add_subplot(1, 3, 1)
    stampimage = imagedat[yint - dy:yint + dy + 1, xint - dx:xint + dx + 1]
    imout1 = ax1.imshow(stampimage, cmap=cm.Greys, aspect='equal',
                        interpolation='nearest', origin='lower', **imshowkws)
    pl.colorbar(imout1, ax=ax1, use_gridspec=True, orientation='horizontal',
                ticks=ticker.MaxNLocator(5))
    ax1.set_title('Target Image')

    if verbose > 1:
        print("f_target=%.3e" % stampimage.sum())

    # construct a psf image of the same postage-stamp size, with the psf
    # appropriately shifted to the sub-pixel center of the target source
    psfmodel = rdpsfmodel(psfmodelfile)
    psfimage = mkpsfimage(psfmodel, xsubpix, ysubpix, stampsize, fluxscale)
    ax2 = fig.add_subplot(1, 3, 2)
    imout2 = ax2.imshow(psfimage,
                        cmap=cm.Greys, aspect='equal',
                        interpolation='nearest', origin='lower', **imshowkws)
    pl.colorbar(imout2, ax=ax2, use_gridspec=True, orientation='horizontal',
                ticks=ticker.MaxNLocator(5))
    ax2.set_title('Scaled PSF Model')

    if verbose:
        print('xc,yc=%.2f,%.2f' % (xyposition[0], xyposition[1]))
        # print( 'Data image shape: %s'%str(np.shape(stampimage)))
        # print( 'PSF image shape: %s'%str(np.shape(psfimage)))
    if verbose > 1:
        print("f_psf=%.3e" % psfimage.sum())

    residimage = stampimage - psfimage
    ax3 = fig.add_subplot(1, 3, 3)
    imout3 = ax3.imshow(residimage,
                        cmap=cm.Greys, aspect='equal',
                        interpolation='nearest', origin='lower', **imshowkws)
    pl.colorbar(imout3, ax=ax3, use_gridspec=True, orientation='horizontal',
                ticks=ticker.MaxNLocator(5))
    ax3.set_title('Residual')
    if verbose > 1:
        print("f_resid=%.3e" % residimage.sum())

    fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
    # pl.draw()

    return imout1, imout2, imout3


def get_flux_and_err(imagedat, psfmodel, xy, ntestpositions=100, psfradpix=3,
                     apradpix=3, skyannpix=None, skyalgorithm='sigmaclipping',
                     setskyval=None, recenter_target=True, recenter_fakes=True,
                     exptime=1, exact=True, ronoise=1, phpadu=1, showfit=False,
                     verbose=False, debug=False, weightim=None):
    """  Measure the flux and flux uncertainty for a source at the given x,y
    position using both aperture and psf-fitting photometry.

    Flux errors are measured by planting fake psfs or empty apertures into the
    sky annulus and recovering a distribution of fluxes with forced photometry.

    :param imagedat: target image numpy data array (with the star still there)
    :param psfmodel: psf model fits file or a 4-tuple with
           [gaussparam,lookuptable,psfmag,psfzpt]
    :param xy: x,y position of the center of the fake planting field
    :param ntestpositions: number of test positions for empty apertures
           and/or fake stars to use for determining the flux error empirically.
    :param psfradpix: radius to use for psf fitting, in pixels
    :param apradpix: radius of photometry aperture, in pixels
    :param skyannpix: inner and outer radius of sky annulus, in pixels
    :param skyalgorithm: algorithm to use for determining the sky value from
           the pixels within the sky annulus: 'sigmaclipping' or 'mmm'
    :param setskyval: if not None, use this value for the sky, ignoring
           the skyannulus
    :param recenter_target: use cntrd to locate the target center near the
           given xy position.
    :param recenter_fakes: recenter on each planted fake when recovering it
    :param exptime: exposure time of the image, for determining poisson noise
    :param ronoise: read-out noise, for determining aperture flux error
           analytically
    :param phpadu: photons-per-ADU, for determining aper flux err analytically
    :param verbose: turn verbosity on
    :param debug: enter pdb debugging mode
    :return: apflux, apfluxerr, psfflux, psffluxerr
             The flux measured through the given aperture and through
             psf fitting, along with associated errors.
    """
    if not np.any(skyannpix):
        skyannpix = [8, 15]

    # locate the target star center position
    x, y = xy
    if recenter_target:
        x, y = cntrd(imagedat, x, y, psfradpix, verbose=verbose)
        if x < 0 or y < 0:
            print("WARNING [photfunctions.py] : recentering failed")
            import pdb
            pdb.set_trace()

    # do aperture photometry directly on the source
    # (Note : using an arbitrary zeropoint of 25 here)
    aperout = aper(imagedat, x, y, phpadu=phpadu,
                   apr=apradpix, skyrad=skyannpix,
                   setskyval=setskyval,
                   zeropoint=25, exact=exact,
                   verbose=verbose, skyalgorithm=skyalgorithm,
                   debug=debug)
    apmag, apmagerr, apflux, apfluxerr, sky, skyerr, apbadflag, apoutstr = \
        aperout

    # define a set of test position points that uniformly samples the sky
    # annulus region, for planting empty apertures and/or fake stars
    rmin = float(skyannpix[0])
    rmax = float(skyannpix[1])
    u = np.random.uniform(rmin, rmax, ntestpositions)
    v = np.random.uniform(0, rmin + rmax, ntestpositions)
    r = np.where(v < u, u, rmax + rmin - u)
    theta = np.random.uniform(0, 2 * np.pi, ntestpositions)
    xtestpositions = r * np.cos(theta) + x
    ytestpositions = r * np.sin(theta) + y

    psfflux = psffluxerr = np.nan
    if psfmodel is not None:
        # set up the psf model realization
        gaussparam, lookuptable, psfmag, psfzpt = rdpsfmodel(psfmodel)
        psfmodel = [gaussparam, lookuptable, psfmag, psfzpt]
        pk = pkfit_class(imagedat, gaussparam, lookuptable, ronoise, phpadu, weightim)

        # do the psf fitting
        try:
            scale = pk.pkfit_fast_norecenter(1, x, y, sky, psfradpix)
            psfflux = scale * 10 ** (0.4 * (25. - psfmag))
        except RuntimeWarning:
            print("PythonPhot.pkfit_norecenter failed.")
            psfflux = np.nan

        if showfit:
            showpkfit(imagedat, psfmodel, xy, psfradpix*5, psfflux)
            from matplotlib import pyplot as plt
            plt.show()
            out = raw_input("Showing psf fit and residual image. "
                            " Return to continue.")



        if np.isfinite(psfflux):
            # remove the target star from the image
            imagedat = addtoimarray(imagedat, psfmodel, [x, y],
                                    fluxscale=-psfflux)

            # plant fakes and recover their fluxes with psf fitting
            # imdatsubarray = imagedat[y-rmax-2*psfradpix:y+rmax+2*psfradpix,
            #                x-rmax-2*psfradpix:x+rmax+2*psfradpix]
            fakecoordlist, fakefluxlist = [], []
            for xt, yt in zip(xtestpositions, ytestpositions):
                    # To ensure appropriate sampling of sub-pixel positions,
                    # we assign random sub-pixel offsets to each position.
                    xt = int(xt) + np.random.random()
                    yt = int(yt) + np.random.random()
                    fakefluxaper, fakefluxpsf, fakecoord = add_and_recover(
                        imagedat, psfmodel, [xt, yt], fluxscale=psfflux,
                        cleanup=True, psfradius=psfradpix, recenter=recenter_fakes,
                        verbose=verbose)
                    if np.isfinite(fakefluxpsf):
                        fakecoordlist.append(fakecoord)
                        fakefluxlist.append(fakefluxpsf)
            fakefluxlist = np.array(fakefluxlist)
            fakefluxmean, fakefluxsigma = gaussian_fit_to_histogram(fakefluxlist)
            if abs(fakefluxmean - psfflux) > fakefluxsigma and verbose:
                print("WARNING: psf flux may be biased. Fake psf flux tests "
                      "found a significantly non-zero sky value not accounted for "
                      "in measurement of the target flux:  \n"
                      "Mean psf flux offset in sky annulus = %.3e\n" %
                      (fakefluxmean - psfflux) +
                      "sigma of fake flux distribution = %.3e" %
                      fakefluxsigma +
                      "\nNOTE: this is included as a systematic error, added in "
                      "quadrature to the psf flux err derived from fake psf "
                      "recovery.")
            if debug:
                import pdb
                pdb.set_trace()
            psfflux_poissonerr = (poissonErr(psfflux * exptime, confidence=1) /
                                  exptime)
            if not np.isfinite(psfflux_poissonerr):
                psfflux_poissonerr = 0
            # Total flux error is the quadratic sum of the poisson noise with
            # the systematic (shift) and statistical (dispersion) errors
            # inferred from fake psf planting and recovery
            psffluxerr = np.sqrt(psfflux_poissonerr**2 +
                                 (fakefluxmean - psfflux)**2 +
                                 fakefluxsigma**2)

    # drop down empty apertures and recover their fluxes with aperture phot
    # NOTE : if the star was removed for psf fitting, then we take advantage
    # of that to get aperture flux errors with the star gone.
    emptyaperout = aper(imagedat, np.array(xtestpositions),
                        np.array(ytestpositions), phpadu=phpadu,
                        apr=apradpix, setskyval=sky, zeropoint=25,
                        exact=False, verbose=verbose,
                        skyalgorithm=skyalgorithm, debug=debug)
    emptyapflux = emptyaperout[2]
    if np.any(np.isfinite(emptyapflux)):
        emptyapmeanflux, emptyapsigma = gaussian_fit_to_histogram(emptyapflux)
        emptyapbias = abs(emptyapmeanflux) - emptyapsigma
        if np.any(emptyapbias > 0) and verbose:
            print("WARNING: aperture flux may be biased. Empty aperture flux tests"
                  " found a significantly non-zero sky value not accounted for in "
                  "measurement of the target flux:  \n"
                  "Mean empty aperture flux in sky annulus = %s\n"
                  % emptyapmeanflux +
                  "sigma of empty aperture flux distribution = %s"
                  % emptyapsigma)
        if np.iterable(apflux):
            apflux_poissonerr = np.array(
                [poissonErr(fap * exptime, confidence=1) / exptime
                 for fap in apflux])
            apflux_poissonerr[np.isfinite(apflux_poissonerr)==False] = 0
        else:
            apflux_poissonerr = (poissonErr(apflux * exptime, confidence=1) /
                                 exptime)
            if not np.isfinite(apflux_poissonerr):
                apflux_poissonerr = 0
        apfluxerr = np.sqrt(apflux_poissonerr**2 +
                            emptyapbias**2 + emptyapsigma**2)

    else:
        if np.iterable(apradpix):
            apfluxerr = [np.nan for aprad in apradpix]
        else:
            apfluxerr = np.nan

    if psfmodel is not None and np.isfinite(psfflux):
        # return the target star back into the image
        imagedat = addtoimarray(imagedat, psfmodel, [x, y],
                                fluxscale=psfflux)

    if debug:
        import pdb
        pdb.set_trace()

    return apflux, apfluxerr, psfflux, psffluxerr, sky, skyerr


def gaussian_fit_to_histogram(dataset):
    """ fit a gaussian function to the histogram of the given dataset
    :param dataset: a series of measurements that is presumed to be normally
       distributed, probably around a mean that is close to zero.
    :return: mean, mu and width, sigma of the gaussian model fit.
    """
    def gauss(x, mu, sigma):
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    if np.ndim(dataset) == 2:
        musigma = np.array([gaussian_fit_to_histogram(dataset[:, i])
                            for i in range(np.shape(dataset)[1])])
        return musigma[:, 0], musigma[:, 1]

    dataset = dataset[np.isfinite(dataset)]
    ndatapoints = len(dataset)
    stdmean, stdmedian, stderr, = sigma_clipped_stats(dataset, sigma=5.0)
    nhistbins = max(10, int(ndatapoints / 10))
    histbins = np.linspace(stdmedian - 3 * stderr, stdmedian + 3 * stderr,
                           nhistbins)
    yhist, xhist = np.histogram(dataset, bins=histbins)
    binwidth = np.mean(np.diff(xhist))
    binpeak = float(np.max(yhist))
    param0 = [stdmedian, stderr]  # initial guesses for gaussian mu and sigma
    xval = xhist[:-1] + (binwidth / 2)
    yval = yhist / binpeak
    minparam, cov = curve_fit(gauss, xval, yval, p0=param0)
    mumin, sigmamin = minparam
    return mumin, sigmamin

def poissonErr( N, confidence=1 ):
    """
    Adapted from P.K.G.Williams :
    http://newton.cx/~peter/2012/06/poisson-distribution-confidence-intervals/

    Let's say you observe n events in a period and want to compute the k
    confidence interval on the true rate - that is, 0 < k <= 1, and k =
    0.95 would be the equivalent of 2sigma. Let a = 1 - k, i.e. 0.05. The
    lower bound of the confidence interval, expressed as a potential
    number of events, is
       scipy.special.gammaincinv (n, 0.5 * a)
    and the upper bound is
       scipy.special.gammaincinv (n + 1, 1 - 0.5 * a)

    The halving of a is just because the 95% confidence interval is made
    up of two tails of 2.5% each, so the gammaincinv function is really,
    once you chop through the obscurity, exactly what you want.

    INPUTS :
      N : the number of observed events

      confidence : may either be a float <1, giving the exact
          confidence limit desired (e.g.  0.95 or 0.99)
          or it can be an integer in [1,2,3], in which case
          we set the desired confidence interval to match
          the 1-, 2- or 3-sigma gaussian confidence limits
             confidence=1 gives the 1-sigma (68.3%) confidence limits
             confidence=2  ==>  95.44%
             confidence=3  ==>  99.74%
    """
    from scipy.special import gammaincinv as ginv
    if confidence<1 : k = confidence
    elif confidence==1 : k = 0.6826
    elif confidence==2 : k = 0.9544
    elif confidence==3 : k = 0.9974
    else :
        print( "ERROR : you must choose nsigma from [1,2,3]")
        return( None )
    lower = ginv( N, 0.5 * (1-k) )
    upper = ginv( N+1, 1-0.5*(1-k) )
    mean_error = (upper-lower)/2.
    return( mean_error )
