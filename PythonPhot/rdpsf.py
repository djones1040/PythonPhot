#!/usr.bin/env python
#D. Jones - 1/13/14
"""This code is from the IDL Astronomy Users Library"""

import numpy as np
from . import dao_value
try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits

def rdpsf(psfname):
    """Read the FITS file created by GETPSF in the DAOPHOT sequence

    Combines the Gaussian with the residuals to create an output PSF array.
    
    psf,hpsf = rdpsf.rdpsf( PSFname )
    
    INPUTS:
         PSFname - string giving the name of the FITS file containing the PSF
                    residuals
     
    RETURNS:
         psf - array containing the actual PSF
         hpsf - header associated with psf
     
    PROCEDURES CALLED:
         DAO_VALUE()
    REVISION HISTORY:
         Written                          W. Landsman              December,  1988
         Checked for IDL Version 2        J. Isensee & J. Hill     December,  1990
         Converted to IDL V5.0            W. Landsman              September, 1997
         Converted to Python              D. Jones                 January,   2014
    """

    resid=pyfits.getdata(psfname)
    hpsf = pyfits.getheader(psfname)

    gauss1 = hpsf['GAUSS1']  #Get Gaussian parameters (5)
    gauss2 = hpsf['GAUSS2']  #
    gauss3 = hpsf['GAUSS3']  #
    gauss4 = hpsf['GAUSS4']  #
    gauss5 = hpsf['GAUSS5']  #
    gauss=[gauss1,gauss2,gauss3,gauss4,gauss5]

    psfrad = hpsf['PSFRAD'] # Get PSF radius
    npsf = int(2*psfrad + 1) #hpsf['NAXIS1']            # Width of output array containing PSF

    psf = np.zeros([npsf,npsf])       # Create output array
    dx = np.arange(npsf,dtype='int') - psfrad    # Vector gives X distance from center of array
    dy = np.arange(npsf,dtype='int') - psfrad                       # Ditto for dy


    ny = len(dy)
    nx = len(dx)
    dx = dx.reshape(1,nx)
    dy = dy.reshape(ny,1)

    dx = rebin(dx, [ny, nx])
    dy = rebin(dy, [ny, nx])

    psf = psf + dao_value.dao_value(dx,dy,gauss,resid,deriv=False) #Compute DAOPHOT value at each point

    hpsf['NAXIS1'] = npsf
    hpsf['NAXIS2'] = npsf

    return(psf,hpsf)

def rebin(a, new_shape):

    M, N = a.shape
    m, n = new_shape
    if m<M:
        return a.reshape((m,M/m,n,N/n)).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)
