import pyfits
import numpy as np
import os
from .photfunctions import rdpsfmodel, add_and_recover


def testgrid(imagefile, psfmodelfile, psfradius=5, fluxscale=100):
    """ Plant a grid of fake psfs into an image, and recover the flux
    of each fake star.

    :param imagefile:  image filename
    :param psfmodelfile: fits file with the psf model
    :param psfradius: size of psf subarray for planting
    :param fluxscale: brightness scaling applied to all fake stars
    :return: arrays of recovered fluxes and x,y shifts
    """
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



