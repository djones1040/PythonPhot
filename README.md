PythonPhot PSF Fitting Photometry Tutorial
=========

getpsf.py : Generates a point-spread function (PSF) from observed stars at 
   specified locations.   Uses the family of  "peak fit" modules
   (pkfit, pkfit_noise, pkfit_norecent, etc) to fit a gaussian to each
   star and define an array of non-gaussian psf residuals.
   Returns a 5-element vector defining the gaussian, a 2-d array of
   psf residuals, and the magnitude of the psf.  Also writes out the
   psf model to a fits file with the gaussian parameters in the header
   and the residuals in the data array.

rdpsf.py : Read the .fits file created by getpsf.py that contains the
   psf model gaussian parameters and 2-d array of residuals. 


pkfit.py : fit a psf model to an isolated point source
pkfit_noise : fitting with an input noise image
pkfit_norecent : forced photometry (fitting a peak without recentering) 
pkfit_norecent_noise : forced photometry with an input noise image

     
-----------
# EXAMPLE A :   Make a psf model 

         import getpsf
         import aper
         import numpy as np
         # load FITS image and specify PSF star coordinates
         image = pyfits.getdata(fits_filename)
         xpos,ypos = np.array([1450,1400]),np.array([1550,1600])

         # run aper to get mags and sky values for specified coords
         mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = \
                aper.aper(image,xpos,ypos,phpadu=1,apr=5,zeropoint=25,
                skyrad=[40,50],badpix=[-12000,60000],exact=True)

         # use the stars at those coords to generate a PSF model
         gauss,psf,psfmag = \
                getpsf.getpsf(image,xpos,ypos,
                              mag,sky,1,1,np.arange(len(xpos)),
                              5,'output_psf.fits')

------------
# EXAMPLE B :  fit a psf to isolated stars

     import pyfits
     from PythonPhot import pkfit

     # read in the fits images containing the target sources
     image = pyfits.getdata(fits_filename)
     noiseim = pyfits.getdata(fits_noise_filename)
     maskim = pyfits.getdata(fits_mask_filename)

     # read in the fits image containing the PSF (gaussian model
     # parameters and 2-d residuals array.
     psf = pyfits.getdata(psf_filename)
     hpsf = pyfits.getheader(psf_filename)
     gauss = [hpsf['GAUSS1'],hpsf['GAUSS2'],hpsf['GAUSS3'],hpsf['GAUSS4'],hpsf['GAUSS5']]

     # x and y points for PSF fitting
     xpos,ypos = np.array([1450,1400]),np.array([1550,1600])

     # run 'aper' on x,y coords to get sky values
     mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = \
              aper.aper(image,xpos,ypos,phpadu=1,apr=5,zeropoint=25,
              skyrad=[40,50],badpix=[-12000,60000],exact=True)

     # load the pkfit class
     pk = pkfit.pkfit_class(image,gauss,psf,1,1,noiseim,maskim)

     # do the PSF fitting
     for x,y,s in zip(xpos,ypos,sky):
          errmag,chi,sharp,niter,scale = \
              pk.pkfit_norecent_noise(1,x,y,s,5)
          flux = scale*10**(0.4*(25.-hpsf['PSFMAG']))
          dflux = errmag*10**(0.4*(25.-hpsf['PSFMAG']))
          print('PSF fit to coords %.2f,%.2f gives flux %s +/- %s'%(x,y,flux,dflux))
