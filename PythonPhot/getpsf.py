#!/usr/bin/env python
#D. Jones - 1/14/14
"""This code is from the IDL Astronomy Users Library"""


import numpy as np
from . import make_2d,daoerf, rinter, pkfit, pkfit_noise
from scipy import linalg

try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits

def getpsf(image,xc,yc,
           apmag,sky,ronois,
           phpadu,idpsf,psfrad,
           fitrad,psfname, zeropoint=0,
           debug = False,verbose=True):
    """Generates a point-spread function (PSF) from observed stars. 

    The PSF is represented as a 2-dimensional Gaussian
    (integrated over each pixel) and a lookup table of residuals.
    The lookup table and Gaussian parameters are output in a FITS
    image file.   The PSF FITS file created by GETPSF can be
    read with the procedure RDPSF.      Adapted from the 1986 STSDAS 
    version of DAOPHOT
    
    CALLING SEQUENCE:
    	 gauss,psf,psfmag = getpsf.getpsf( image, xc, yc, apmag, sky, ronois, phpadu,
    		 	                   idpsf, psfrad, fitrad, psfname)


    EXAMPLE:

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
    
    INPUTS:
         image   - input image array
    	 xc      - input vector of x coordinates (from FIND), these should be
    		    IDL (first pixel is (0,0)) convention.
    	 yc      - input vector of y coordinates (from FIND)
    	 apmag   - vector of magnitudes (from APER), used for initial estimate
    		    of gaussian intensity.  If apmag is multidimensional, (more
    		    than 1 aperture was used in APER) then the first aperture
    		    is used.
    	 sky     - vector of sky values (from APER)                    
    	 ronois  - readout noise per pixel, (in electrons, or equivalent photons)
    	 phpadu  - photons per analog digital unit, used to scale the data
    		    numbers in IMAGE into photon units
    	 idpsf   - subscripts of the list of stars created by 
    		    APER which will be used to define the PSF.  Stars whose
                    centroid does not fall within psfrad of the edge of the frame,
                    or for which a Gaussian fit requires more than 25 iterations,
    		    will be ignored when creating the final PSF.
    	 psfrad  - the scalar radius, in pixels, of the circular area within
    		    which the PSF will be defined.  This should be slightly larger
    		    than the radius of the brightest star that one will be
    		    interested in.
    	 fitrad  - the scalar radius, in pixels of the circular area used in the
    		    least-square star fits.  Stetson suggest that fitrad should
    		    approximately equal to the FWHM, slightly less for crowded
                    fields.  (fitrad must be smaller than psfrad.)
    	 psfname - Name of the FITS file that will contain the table of residuals,
    		    and the best-fit Gaussian parameters.  This file is 
    		    subsequently required for use by NSTAR or PKFIT.
    	 zeropoint - Zero point value used to define the magnitudes provided
    	        in apermag, to be written into the psf model header.
                Default value is 0.0, indicating 'unknown'.

     OPTIONAL KEYWORD INPUT:
          debug    - if this keyword is True, then the result of each
    		      fitting iteration will be displayed.
    
     RETURNS:
    	  gauss  - 5 element vector giving parameters of gaussian fit to the 
                    first PSF star
    		 gauss[0] - height of the gaussian (above sky)
    		 gauss[1] - the offset (in pixels) of the best fitting gaussian
    			     and the original X centroid
    		 gauss[2] - similiar offset from the Y centroid 
    		 gauss[3] - Gaussian sigma in X
    		 gauss[4] - Gaussian sigma in Y
    	  psf    - 2-d array of PSF residuals after a Gaussian fit.
    
     PROCEDURE:
          GETPSF fits a Gaussian profile to the core of the first PSF star 
    	  and generates a look-up table of the residuals of the
    	  actual image data from the Gaussian fit.  If desired, it will then
    	  fit this PSF to another star (using PKFIT) to determine its precise 
    	  centroid, scale the same Gaussian to the new star's core, and add the
    	  differences between the actual data and the scaled Gaussian to the
    	  table of residuals.  (In other words, the Gaussian fit is performed
    	  only on the first star.)
        
     PROCEDURES CALLED:
          DAOERF, MAKE_2D, RINTER(), PKFIT
    
     REVISON HISTORY:
    	Adapted from the 1986 version of DAOPHOT in STSDAS
    	IDL Version 2                                             W Landsman     November 1988
    	Use DEBUG keyword instead of !DEBUG                       W. Landsman    May 1996
    	Converted to IDL V5.0                                     W. Landsman    September 1997
        Converted to Python                                       D. Jones       January, 2014
    """

    s = np.shape(image)    		#Get number of rows and columns in image
    ncol = s[1] ; nrow = s[0]
    nstar = len(xc)	        #Total # of stars identified in image
    psfmag =  1000000.0 # initial values for psfmag and gauss in case all fails
    gauss = np.array([1,-1,-1,-1,-1])

    # GOT_ID:  

    if fitrad >= psfrad:
        print('ERROR - Fitting radius must be smaller than radius defining PSF')
        return(gauss,-1,-1)

    numpsf = len(idpsf)      ## of stars used to create the PSF

    smag =np.shape(apmag)     #Is APMAG multidimensional?
#    if len(apmag) != smag[0]: mag = apmag[:,0] 
#    else: mag = apmag
    mag = apmag

    n = 2*int(psfrad+0.5)+1  #(Odd) width of box that contains PSF circle
    npsf = 2*n+7             #Lookup table has half pixel interpolation
    nbox = n+7		 #(Even) Width of subarray to be extracted from image
    nhalf = nbox/2.          

    if debug:
        print('GETPSF: Fitting radius - ',str(float(fitrad)))
        print('        PSF Radius     - ',str(float(psfrad)))
        print('        Stellar IDs: ',idpsf)   ; print(' ')

    boxgen = np.arange(nbox)
    xgen,ygen = make_2d.make_2d( boxgen, boxgen)

    #               Find the first PSF star in the star list.
    nstrps = -1	#Counter for number of stars used to create PSF
    # GETSTAR: 
   
    for i in range(numpsf):
        nstrps = nstrps + 1       
        if nstrps >= numpsf:
            print('ERROR - No valid PSF stars were supplied')
            gauss=np.array([-1,-1,-1,-1,-1])
            return(gauss,-1,-1)

        istar = idpsf[nstrps]       #ID number of first PSF star
        ixcen = int(xc[istar])      
        iycen = int(yc[istar])

        #  Now a subarray F will be read in from the big image, given by 
        #  IXCEN-NBOX/2+1 <= x <= IXCEN+NBOX/2, IYCEN-NBOX/2+1 <= y <= IYCEN+NBOX/2.  
        #  (NBOX is an even number.)  In the subarray, the coordinates of the centroid
        #  of the star will lie between NBOX/2 and NBOX/2+1 in each coordinate.

        lx = ixcen-nhalf+1  ;  ux = ixcen + nhalf  #Upper ; lower bounds in X
        ly = iycen-nhalf+1  ;  uy = iycen + nhalf
        if (lx < 0) or (ly < 0) or (ux >= ncol) or (uy >= nrow):    
            print('GETPSF: Star ',str(istar),' too near edge of frame.')
            continue

        f = image[int(ly):int(uy)+1,int(lx):int(ux)+1] - sky[istar]  #Read in subarray, subtract off sky
        
        # An integrated Gaussian function will be fit to the central part of the
        # stellar profile.  Initially, a 5x5 box centered on the centroid of the 
        # star is used, but if the sigma in one coordinate drops to less than
        # 1 pixel, then the box width of 3 will be used in that coordinate.
        # If the sigma increases to over 3 pixels, then a box width of 7 will be 
        # used in that coordinate
        
        x = xc[istar] - lx    #X coordinate of stellar centroid in subarray F
        y = yc[istar] - ly    #Y coordinate of stellar centroid in subarray F
        ix = int(x+0.5)       #Index of pixel containing centroid
        iy = int(y+0.5) 

        #                     #Begin least squares
        h = np.max(f)  	      #Initial guess for peak intensity
        sigx = 2.0 ; sigy = 2.0                                       
        dxcen=0.  ;  dycen=0.
        #
        niter = 0                    #Beginning of big iteration loop
        v = np.zeros(5)
        c = np.zeros([5,5])
        #                            Print the current star

        if verbose:
            print('STAR  X  Y  MAG  SKY')
            print(istar, xc[istar], yc[istar], mag[istar], sky[istar])
            print('')

        if debug: print('GETPSF: Gaussian Fit Iteration')

        niterflag=False
        for i in range(102):		     #Begin the iterative loop

            niter = niter + 1
            if niter > 100:   #No convergence after 100 iterations?
                print('No convergence after 100 iterations for star ' + str(istar))
                gauss=np.array([-1,-1,-1,-1,-1])
                niterflag=True
                if nstrps >= numpsf-1:
                    return(gauss,-1,-1)
                break

            if sigx <= 1: nx = 1    #A default box width 
            elif sigx > 3: nx = 3
            else: nx = 2

            if sigy <= 1: ny = 1
            elif sigy > 3: ny = 3
            else: ny = 2

            a = np.array([h, x+dxcen,y+dycen,sigx,sigy])
            xin = (np.arange(2*nx+1)-nx) + ix
            yin = (np.arange(2*ny+1)-ny) + iy

            xin,yin = make_2d.make_2d( xin, yin)
            g,t = daoerf.daoerf(xin, yin, a)
            
            #  The T's are the first derivatives of the model profile with respect
            #  to the five fitting parameters H, DXCEN, DYCEN, SIGX, and SIGY.
            #  Note that the center of the best-fitting Gaussian profile is
            #  expressed as an offset from the centroid of the star.  In the case of
            #  a general, asymmetric stellar profile, the center of symmetry of the
            #  best-fitting Gaussian profile will not necessarily coincide with the
            #  centroid determined by any arbitrary centroiding algorithm.  
            
            dh = f[iy-ny:iy+ny+1, ix-nx:ix+nx+1] - g.reshape(ny*2+1,nx*2+1) #Subtract best fit Gaussian from subarray
            for kk in range(5):
                tk = t[kk,:]
                v[kk] = np.sum( dh.reshape(np.shape(dh)[0]*np.shape(dh)[1]) * tk )
                for ll in range(5): 
                    c[ll,kk] = np.sum( tk * t[ll,:] )

            try: c = linalg.inv(c)	#IDL version assumes INVERT is successful
            except: continue

            z = np.matrix(v)*c         #Multiply by vector of residuals
            z = np.array(z)[0]

            h = h + z[0]/(1.0+4.0*np.abs(z[0]/h))	#Correct the fitting parameters
            dxcen = dxcen+z[1]/(1.0+3.0*np.abs(z[1]))
            dycen = dycen+z[2]/(1.0+3.0*np.abs(z[2]))
            sigx = sigx+z[3]/(1.0+4.0*np.abs(z[3]/sigx))
            sigy = sigy+z[4]/(1.0+4.0*np.abs(z[4]/sigy))

            if debug: print(niter,h,dxcen,dycen,sigx,sigy)
            if (np.abs(z[0]/h)+np.abs(z[3]/sigx)+np.abs(z[4]/sigy) < 0.0001): break  #Test for convergence

        if niterflag: continue
        #  Now that the solution has converged, we can generate an
        #  array containing the differences between the actual stellar profile
        #  and the best-fitting Gaussian analytic profile.

        a = np.array([h, x+dxcen, y+dycen, sigx,sigy])  #Parameters for Gaussian fit
        g,pder2 = daoerf.daoerf(xgen,ygen,a)                  #Compute Gaussian
        f = f - g.reshape(np.shape(f)[0],np.shape(f)[1])                             #Residuals (Real profile - Gaussian)

        psfmag = mag[istar]
        xpsf1 = xc[istar] ; ypsf1 = yc[istar]

        # The look-up table is obtained by interpolation within the array of
        # fitting residuals.  We need to interpolate because we want the look-up
        # table to be centered accurately on the centroid of the star, which of 
        # course is at some fractional-pixel position in the original data.
            
        ncen = (npsf-1)/2.
        psfgen = (np.arange(npsf) - ncen)/2.         #Index function for PSF array
        yy = psfgen + y   ;  xx = psfgen + x
        xx,yy = make_2d.make_2d(xx,yy)
        psf = rinter.rinter(f, xx, yy, deriv=False )            #Interpolate residuals onto current star

        gauss = np.array([h,dxcen,dycen,sigx,sigy])
        goodstar = nstrps                   #Index of first good star

        # For each additional star, determine the precise  coordinates of the 
        # centroid and the relative brightness of the star
        # by least-squares fitting to the current version of the point-spread
        # function.  Then subtract off the appropriately scaled integral under
        # the analytic Gaussian function  and add the departures of the actual 
        # data from the analytic Gaussian function to the look-up table.

        # D. Jones - stop the height of the gaussian from getting messed up
 #       psfmag0 = psfmag
 #       gauss_orig0 = gauss[0]
        break # D. Jones - I think we don't need to continue the loop now

    # GETMORE:            #Loop for additional PSF stars begins here                 
    for i in range(numpsf-nstrps):    
        nstrps = nstrps+1
        if nstrps >= numpsf: continue	#Have all the stars been done?

        istar = idpsf[nstrps]
        ixcen = int(xc[istar])
        iycen = int(yc[istar])                  
        try: scale = 10.**(-0.4*(mag[istar]-psfmag))
        except: scale = np.inf

        # Fit the current version of the point-spread function to the data for
        # this star.

        lx = ixcen-nhalf+1 ; ux =ixcen + nhalf
        ly = iycen-nhalf+1 ; uy =iycen + nhalf
        if ( (lx < 0) or (ly < 0) or (ux >= ncol) or (uy >= nrow)):
            print('GETPSF: Star ',str(istar),' too near edge of frame.')
            continue    

        print('STAR  X  Y  MAG  SKY')
        print(istar, xc[istar], yc[istar], mag[istar], sky[istar])
        print('')

        f = image[int(ly):int(uy)+1,int(lx):int(ux)+1]
        x = xc[istar]-lx   ;   y = yc[istar]-ly   

        pk = pkfit.pkfit_class(f,gauss,psf,ronois,phpadu)
        errmag,chi,sharp,niter,scale,x,y = pk.pkfit(scale, x, y, 
                                                    sky[istar], fitrad,
                                                    debug = debug,
                                                    xyout = True)

        if niter == 25 or scale == 1000000.0 :	#Convergence in less than 25 iterations?
            print('GETPSF: No convergence after 25 iterations or invalid scale for star',istar)
            continue

        a = np.array([gauss[0], x+dxcen,y+dycen,sigx,sigy])  #Parameters of successful fit
        e,pder2 = daoerf.daoerf(xgen,ygen,a)
        f = f - scale*e.reshape(np.shape(f)[0],np.shape(f)[1]) -sky[istar]	           #Compute array of residuals

        # Values of the array of residuals are now interpolated to an NPSF by
        # NPSF (NPSF is an odd number) array centered on the centroid of the
        # star, and added to the existing look-up table of corrections to the 
        # analytic profile 

        xx = psfgen + x
        yy = psfgen + y 
        xx,yy = make_2d.make_2d(xx,yy)
        psftemp = rinter.rinter(f,xx,yy,deriv=False)

        # deal with bad pixels
        nanrow = np.where(psftemp != psftemp)
        goodrow = np.where(psftemp == psftemp)
        psfshape = np.shape(psf)
        for n0,n1 in zip(nanrow[0],nanrow[1]):
            ind1 = n0 - 2; ind2 = n0 + 3
            ind3 = n1 - 2; ind4 = n1 + 3
            if ind1 < 0: ind1 = 0
            if ind2 > psfshape[0]: ind2 = psfshape[0]
            if ind3 < 0: ind3 = 0
            if ind4 > psfshape[1]: ind4 = psfshape[1]

            psftempsub = psftemp[ind1:ind2,ind3:ind4]
            try:
                psftemp[n0,n1] = np.median(psftempsub[np.where(psftempsub == psftempsub)])
            except:
                psftemp[n0,n1] = np.median(psftemp[np.where(psftemp == psftemp)])
#        psftemp[nanrow[0],nanrow[1]] = 0

        psf = psf + psftemp

        # Now correct both the height of the analytic Gaussian, and the value
        # of the aperture-magnitude of the point-spread function for the
        # inclusion of the additional star.

        psfmag = -2.5*np.log10((1.+scale)*10**(-0.4*psfmag))
        gauss[0] = gauss[0]*(1.+scale)
#        try:
        goodstar = np.append(np.array(goodstar), np.array(nstrps))
#        except:
#            goodstar = np.array([goodstar,nstrps])

    # WRITEOUT:   

    # Create FITS file containing the PSF created.

    # D. Jones - restore height of gaussian to original and adjust psf array accordingly
#    psfratio = gauss[0]/gauss_orig0
#    gauss[0] = gauss_orig0
#    psfmag = psfmag0
#    psf = psf/psfratio

    hdu = pyfits.PrimaryHDU()        #Create a minimal FITS header
    hdu.header['PHPADU'] = (phpadu, 'Photons per Analog Digital Unit')
    hdu.header['RONOIS'] = (ronois, 'Readout Noise')
    hdu.header['PSFRAD'] = (psfrad, 'Radius where PSF is defined (pixels)')
    hdu.header['FITRAD'] = (fitrad, 'Fitting Radius')
    hdu.header['PSFMAG'] = (float(psfmag), 'PSF Magnitude')
    hdu.header['GAUSS1'] = (gauss[0], 'Gaussian Scale Factor')
    hdu.header['GAUSS2'] = (gauss[1], 'Gaussian X Position')
    hdu.header['GAUSS3'] = (gauss[2], 'Gaussian Y Position')
    hdu.header['GAUSS4'] = (gauss[3], 'Gaussian Sigma: X Direction')
    hdu.header['GAUSS5'] = (gauss[4], 'Gaussian Sigma: Y Direction')
    hdu.header['PSFZPT'] = (float(zeropoint), 'zeropoint used to scale psf mag')
    if type(goodstar) == np.int:
        goodstarlen = 1
        hdu.header['PSFID'] = goodstar
        print('Warning: Only one valid PSF star')
        hdu.header['NSTARS'] = goodstarlen #, '# of Stars Used to Create PSF'
        hdu.data = psf
        hdu.writeto(psfname,clobber=True)
        return(gauss,psf,psfmag)

    else: 
        goodstarlen = len(goodstar)
        hdu.header['PSFID'] = goodstar[-1]
        hdu.header['NSTARS'] = goodstarlen #, '# of Stars Used to Create PSF'
        hdu.data = psf
        hdu.writeto(psfname,clobber=True)
        return(gauss,psf,psfmag)



