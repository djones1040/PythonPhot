def mkpsfimage(psfmodel, x, y, size, fluxscale=1):
    """  Construct a numpy array showing the psf model appropriately scaled,
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
    import numpy as np
    from dao_value import dao_value
    # from .dophotometry import rdpsfmodel
    from dophotometry import rdpsfmodel

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

    #psfimage = dao_value(xx+x, yy+y, gaussparam, lookuptable, deriv=False)
    psfimage = dao_value(xx - x, yy - y, gaussparam, lookuptable, deriv=False)
    psfstarflux = 10 ** (-0.4 * (psfmag - psfzpt))
    psfimage *= fluxscale / psfstarflux

    return psfimage


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
    import numpy as np
    # from .dophotometry import rdpsfmodel
    from dophotometry import rdpsfmodel

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


def addtofits(fitsin, fitsout, psfmodelfile, position, fluxscale,
              coordsys='xy', verbose=False):
    """ Create a psf, add it into the given fits image at the

    :param fitsin: fits file name for the target image
    :param psfmodelfile: fits file holding the psf model
    :param position: Scalar or array, giving the psf center positions
    :param fluxscale: Scalar or array, giving the psf flux scaling factors
    :param coordsys: Either 'radec' or 'xy'
    :return:
    """
    import numpy as np
    from .hstphot import radec2xy
    import pyfits
    from .dophotometry import rdpsfmodel

    if not np.iterable(fluxscale):
        fluxscale = np.array([fluxscale])
        position = np.array([position])

    if coordsys == 'radec':
        ra = position[:, 0]
        dec = position[:, 1]
        position = radec2xy(fitsin, ra, dec, ext=0)

    image = pyfits.open(fitsin, mode='readonly')
    imagedat = image[0].data

    psfmodel = rdpsfmodel(psfmodelfile)

    for i in range(len(fluxscale)):
        imagedat = addtoimarray(imagedat, psfmodel, position[i], fluxscale[i])
    image[0].data = imagedat

    image.writeto(fitsout, clobber=True)
    if verbose:
        print("Wrote updated image to %s" % fitsout)
    return


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
    from pkfit_norecent import pkfit_class
    from aper import aper
    from cntrd import cntrd
    from dophotometry import rdpsfmodel
    # from .aper import aper
    #from .cntrd import cntrd
    #from .dophotometry import rdpsfmodel

    # add the psf to the image data array
    imdatwithpsf = addtoimarray(imagedat, psfmodel, xy, fluxscale=fluxscale)

    #TODO: allow for uncertainty in the x,y positions

    gaussparam, lookuptable, psfmag, psfzpt = rdpsfmodel(psfmodel)

    # generate an instance of the pkfit class for this psf model
    # and target image
    pk = pkfit_class(imdatwithpsf, gaussparam, lookuptable, ronoise, phpadu)
    x, y = xy

    if debug:
        from .dophotometry import showpkfit
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
        if xc > 0 and yc > 0:
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
    errmag, chi, sharp, niter, scale = pk.pkfit_norecent(1, x, y, sky,
                                                         psfradius)
    fluxpsf = scale * 10 ** (-0.4 * (psfmag - psfzpt))

    if cleanup:
        imagedat = addtoimarray(imdatwithpsf, psfmodel, xy,
                                fluxscale=-fluxscale)

    return apflux[0], fluxpsf, [x, y]


def testgrid(imagefile, psfmodelfile, psfradius=5, fluxscale=100):
    import pyfits
    import numpy as np
    import os
    from .dophotometry import rdpsfmodel

    image = pyfits.open(imagefile, 'readonly')
    imagedat = image[0].data
    psfmodel = rdpsfmodel(psfmodelfile)

    nx, ny = imagedat.shape

    psffluxlist, apfluxlist = [], []
    xshiftlist, yshiftlist = [], []

    for x in np.arange(200, nx - 200 + 1, 100.1):
        for y in np.arange(200, nx - 200 + 1, 100.1):
            xy = [x, y]
            apflux, psfflux, xyc = add_and_recover(
                imagedat, psfmodel, xy, fluxscale=fluxscale,
                psfradius=psfradius, skyannpix=[15, 30],
                recenter=True,
                ronoise=1, phpadu=1, cleanup=False, debug=False)
            psffluxlist.append(psfflux)
            apfluxlist.append(apflux)
            xshiftlist.append(xyc[0] - x)
            yshiftlist.append(xyc[1] - y)

    if os.path.exists('test.fits'):
        os.remove('test.fits')
    image.writeto('test.fits')
    image.close()
    return np.array(psffluxlist), np.array(apfluxlist), np.array(
        xshiftlist), np.array(yshiftlist)



