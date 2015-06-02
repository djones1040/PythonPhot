"""
Functions for displaying images before, during and after applying
the photometry routines (esp. useful for examining residuals
after psf fitting)
"""

def showpkfit( imagedat, psfmodelfile, xyposition, stampsize, fluxscale,
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
    import numpy as np

    # display a cutout of the target image, centered at the target location
    fig = pl.figure(1,figsize=[9,3])
    pl.clf()

    xint,yint = int(xyposition[0]),int(xyposition[1])
    dx = dy = int(stampsize/2.)

    ax1 = fig.add_subplot(1,3,1)
    stampimage = imagedat[yint-dy:yint+dy, xint-dx:xint+dx]
    imout1 = ax1.imshow( stampimage,
                        cmap=cm.gray, aspect='equal',
                        interpolation='nearest', origin='lower', **imshowkws)
    pl.colorbar(imout1, ax=ax1, use_gridspec=True, orientation='horizontal',
                ticks=ticker.MaxNLocator(5))
    ax1.set_title('Target Image')

    psfimage = mkpsfimage( psfmodelfile, dx*2 )
    psfimage *= fluxscale / psfimage.sum()
    ax2 = fig.add_subplot(1,3,2)
    imout2 = ax2.imshow( psfimage,
                        cmap=cm.gray, aspect='equal',
                        interpolation='nearest', origin='lower', **imshowkws)
    pl.colorbar(imout2, ax=ax2, use_gridspec=True, orientation='horizontal',
                ticks=ticker.MaxNLocator(5))
    ax2.set_title('Scaled PSF Model')

    if verbose :
        print('xc,yc=%.2f,%.2f'%(xyposition[0],xyposition[1]))
        #print( 'Data image shape: %s'%str(np.shape(stampimage)))
        #print( 'PSF image shape: %s'%str(np.shape(psfimage)))

    residimage = stampimage - psfimage
    ax3 = fig.add_subplot(1,3,3)
    imout3 = ax3.imshow( residimage,
                        cmap=cm.gray, aspect='equal',
                        interpolation='nearest', origin='lower', **imshowkws)
    pl.colorbar(imout3, ax=ax3, use_gridspec=True, orientation='horizontal',
                ticks=ticker.MaxNLocator(5))
    ax3.set_title('Residual')

    fig.subplots_adjust(left=0.05,right=0.95,top=0.85,bottom=0.05)
    pl.draw()

    return imout1,imout2,imout3


def mkpsfimage( psfmodelfile, size ):
    """  Construct a numpy array showing the psf model appropriately scaled,
    using the gaussian parameters from the header and the residual components
    from the image array of the given fits file

    :param psfmodelfile: fits file containing the psf model
       with gaussian parameters in the header and a lookup table of residual
       values in the data array.
    :param size: width and height in pixels for the output image showing the
       psf realization
    :return: a numpy array showing the psf image realization
    """
    import pyfits
    import numpy as np
    from dao_value import dao_value

    hdr = pyfits.getheader(psfmodelfile)

    # read in the gaussian parameters from the image header
    scale = hdr['GAUSS1'] # , 'Gaussian Scale Factor'
    xpsf  = hdr['GAUSS2'] # , 'Gaussian X Position'
    ypsf  = hdr['GAUSS3'] # , 'Gaussian Y Position'
    xsigma = hdr['GAUSS4'] # , 'Gaussian Sigma: X Direction'
    ysigma = hdr['GAUSS5'] # , 'Gaussian Sigma: Y Direction'
    gaussparam = [scale,xpsf,ypsf,xsigma,ysigma]

    # read in the psf residuals array (i.e. the lookup table)
    psfresid = pyfits.getdata(psfmodelfile)

    # make a 2D data array with the realized psf image (gaussian+residuals)
    xx = np.tile(np.arange( -size/2., size/2.,1,float), (size,1))
    yy = np.tile(np.arange( -size/2., size/2.,1,float), (size,1)).T
    psfimage = dao_value(xx+xpsf, yy+ypsf, gaussparam, psfresid, deriv=False)
    return psfimage


def deletepsf( imagedat, psfparam, psfresid, xyposition ):
    """  remove a psf from the given image

    :param imagedat: numpy array with the target image data
    :param psfparam: parameters of the gaussian psf model
    :param psfresid: numpy array with psf residual values
    :param xyposition: tuple giving position of the center of the target
                    in imagedat pixels
    :return: a copy of the imagedat array, with the psf subtracted at the
              given location
    """

    # add the gaussian psf to the psf residuals image to get a complete
    # psf model image
    psfparam

