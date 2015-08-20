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
import exceptions
import numpy as np
import pyfits
from aper import aper
from cntrd import cntrd
from pkfit_norecenter import pkfit_class
from fakestar import addtoimarray, add_and_recover
from scipy.optimize import curve_fit


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
        raise exceptions.RuntimeError(
            "psfmodel must either be a filename or a 4-tuple giving:"
            "[gaussian parameters, look-up table, psf mag, zpt]")
    return gaussparam, lookuptable, psfmag, psfzpt



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
    from .fakestar import mkpsfimage

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
        #print( 'PSF image shape: %s'%str(np.shape(psfimage)))
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
                     exact=True, ronoise=1, phpadu=1, verbose=False,
                     debug=False):
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
        x, y = cntrd(imagedat, x, y, psfradpix)
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

    psfflux = fakefluxsigma = None
    if psfmodel is not None:
        # set up the psf model realization
        gaussparam, lookuptable, psfmag, psfzpt = rdpsfmodel(psfmodel)
        psfmodel = [gaussparam, lookuptable, psfmag, psfzpt]
        pk = pkfit_class(imagedat, gaussparam, lookuptable, ronoise, phpadu)

        # do the psf fitting
        scale = pk.pkfit_norecenter_numpy(1, x, y, sky, psfradpix)
        psfflux = scale * 10 ** (0.4 * (25. - psfmag))

        # remove the target star from the image
        imagedat = addtoimarray(imagedat, psfmodel, [x, y],
                                fluxscale=-psfflux)

        # plant fakes and recover their fluxes with psf fitting
        #imdatsubarray = imagedat[y-rmax-2*psfradpix:y+rmax+2*psfradpix,
        #                x-rmax-2*psfradpix:x+rmax+2*psfradpix]
        fakecoordlist, fakefluxlist = [], []
        for xt in xtestpositions:
            for yt in ytestpositions:
                # To ensure appropriate sampling of sub-pixel positions,
                # we assign random sub-pixel offsets to each position.
                xt = int(xt) + np.random.random()
                yt = int(yt) + np.random.random()
                fakefluxaper, fakefluxpsf, fakecoord = add_and_recover(
                    imagedat, psfmodel, [xt, yt], fluxscale=psfflux,
                    cleanup=True, psfradius=psfradpix, recenter=recenter_fakes)
                fakecoordlist.append(fakecoord)
                fakefluxlist.append(fakefluxpsf)
        fakefluxmean, fakefluxsigma = gaussian_fit_to_histogram(fakefluxlist)
        if abs(fakefluxmean - psfflux) > fakefluxsigma and verbose:
            print("WARNING: psf flux may be biased. Fake psf flux tests "
                  "found a significantly non-zero sky value not accounted for "
                  "in measurement of the target flux:  \\"
                  "Mean psf flux offset in sky annulus = %.3e\\" %
                  (fakefluxmean - psfflux) +
                  "sigma of fake flux distribution = %.3e" % fakefluxsigma)

    # drop down empty apertures and recover their fluxes with aperture phot
    # NOTE : if the star was removed for psf fitting, then we take advantage
    # of that to get aperture flux errors with the star gone.
    emptyaperout = aper(imagedat, np.array(xtestpositions),
                        np.array(ytestpositions), phpadu=phpadu,
                        apr=apradpix, setskyval=sky, zeropoint=25,
                        exact=False, verbose=verbose,
                        skyalgorithm=skyalgorithm, debug=debug)
    emptyapflux = emptyaperout[2]
    emptyapmeanflux, emptyapsigma = gaussian_fit_to_histogram(emptyapflux)

    emptyapbias = abs(emptyapmeanflux) - emptyapsigma
    if np.any(emptyapbias > 0) and verbose:
        print("WARNING: aperture flux may be biased. Empty aperture flux tests"
              " found a significantly non-zero sky value not accounted for in "
              "measurement of the target flux:  \\"
              "Mean empty aperture flux in sky annulus = %s\\"
              % emptyapmeanflux +
              "sigma of empty aperture flux distribution = %s"
              % emptyapsigma)

    if psfmodel is not None:
        # return the target star back into the image
        imagedat = addtoimarray(imagedat, psfmodel, [x, y],
                                fluxscale=psfflux)

    if debug > 1:
        import pdb
        pdb.set_trace()

    return apflux, emptyapsigma, psfflux, fakefluxsigma, sky, skyerr


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

    ndatapoints = len(dataset)
    stderr = np.std(dataset)
    stdmedian = np.median(dataset)
    nhistbins = max(10, int(ndatapoints / 10))
    histbins = np.linspace(stdmedian - 3 * stderr, stdmedian + 3 * stderr,
                           nhistbins)
    yhist, xhist = np.histogram(dataset, bins=histbins)
    binwidth = np.mean(np.diff(xhist))
    binpeak = float(np.max(yhist))
    param0 = [stdmedian, stderr]  # initial guesses for gaussian mu and sigma
    xval = xhist[:-1] + (binwidth / 2)
    # noinspection PyTypeChecker
    yval = yhist / binpeak
    minparam, cov = curve_fit(gauss, xval, yval, p0=param0)
    mumin, sigmamin = minparam
    return mumin, sigmamin
