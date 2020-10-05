#!/usr/bin/env python
#D. Jones - 1/15/14
#S. Rodney - 2014.07.06
"""This code is translated from the IDL Astronomy Users Library

example call:

from astropy.io import fits as pyfits
from PythonPhot import aper
import numpy as np
image = pyfits.getdata(fits_filename)
xpos,ypos = np.array([1450,1400]),np.array([1550,1600])
mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = \
     aper.aper(image,xpos,ypos,phpadu=1,apr=5,zeropoint=25,
               skyrad=[40,50],badpix=[-12000,60000],
               exact=True)

"""

import numpy as np
from scipy.stats import sigmaclip, skew
from . import pixwt
from . import mmm
where,asfarray,asarray,array,zeros,arange = np.where,np.asfarray,np.asarray,np.array,np.zeros,np.arange

def aper(image,xc,yc, phpadu=1, apr=5, zeropoint=25,
         skyrad=[40,50], badpix=[0,0], setskyval = None, minsky=[],
         skyalgorithm='mmm', exact = False, readnoise = 0,
         verbose=True, debug=False):
    """ Compute concentric aperture photometry on one ore more stars
    (adapted for IDL from DAOPHOT, then translated from IDL to Python).

    APER can compute photometry in several user-specified aperture radii.
    A separate sky value is computed for each source using specified inner
    and outer sky radii.

    By default, APER uses a magnitude system where a magnitude of
    25 corresponds to 1 flux unit. APER returns both
    fluxes and magnitudes.

     REQUIRED INPUTS:
         image  -  input image array
         xc     - scalar x value or 1D array of x coordinates.
         yc     - scalar y value or 1D array of y coordinates

     OPTIONAL KEYWORD INPUTS:
         phpadu - Photons per Analog Digital Units, numeric scalar.  Converts
                   the data numbers in IMAGE to photon units.  (APER assumes
                   Poisson statistics.)
         apr    - scalar or 1D array of photometry aperture radii in pixel units.
         zeropoint - zero point for converting flux (in ADU) to magnitudes
         skyrad - Two element list giving the inner and outer radii
                   to be used for the sky annulus
         badpix - Two element list giving the minimum and maximum value
                   of a good pix. If BADPIX[0] is equal to BADPIX[1] then
                   it is assumed that there are no bad pixels.

         exact -  By default, APER counts subpixels, but uses a polygon
                 approximation for the intersection of a circular aperture with
                 a square pixel (and normalize the total area of the sum of the
                 pixels to exactly match the circular area).   If the /EXACT
                 keyword, then the intersection of the circular aperture with a
                 square pixel is computed exactly.    The /EXACT keyword is much
                 slower and is only needed when small (~2 pixels) apertures are
                 used with very undersampled data.

         print - if set and non-zero then APER will also write its results to
                   a file aper.prt.   One can specify the output file name by
                   setting PRINT = 'filename'.
         verbose -  Print warnings, status, and ancillary info to the terminal
         setskyval - Use this keyword to force the sky to a specified value
                   rather than have APER compute a sky value.    SETSKYVAL
                   can either be a scalar specifying the sky value to use for
                   all sources, or a 3 element vector specifying the sky value,
                   the sigma of the sky value, and the number of elements used
                   to compute a sky value.   The 3 element form of SETSKYVAL
                   is needed for accurate error budgeting.
         skyalgorithm - set the algorithm by which the sky value is determined
                  Valid options are 'sigmaclipping' or 'mmm'.

     RETURNS:
         mags   -  NAPER by NSTAR array giving the magnitude for each star in
                   each aperture.  (NAPER is the number of apertures, and NSTAR
                   is the number of stars).   A flux of 1 digital unit is assigned
                   a zero point magnitude of 25.
         magerr  -  NAPER by NSTAR array giving error in magnitude
                   for each star.  If a magnitude could not be deter-
                   mined then ERRAP = 9.99.
         flux    -  NAPER by NSTAR array giving fluxes
         fluxerr -  NAPER by NSTAR array giving error in each flux
         sky  -    NSTAR element array giving sky value for each star
         skyerr -  NSTAR element array giving error in sky values
         outstr  - string for each star and aperture reporting the mag and err

     PROCEDURES USED:
           MMM, PIXWT()
     NOTES:
           Reasons that a valid magnitude cannot be computed include the following:
          (1) Star position is too close (within 0.5 pixels) to edge of the frame
          (2) Less than 20 valid pixels available for computing sky
          (3) Modal value of sky could not be computed by the procedure MMM
          (4) *Any* pixel within the aperture radius is a "bad" pixel

           APER was modified in June 2000 in two ways: (1) the /EXACT keyword was
           added (2) the approximation of the intersection of a circular aperture
           with square pixels was improved (i.e. when /EXACT is not used)
     REVISON HISTORY:
           Adapted to IDL from DAOPHOT June, 1989   B. Pfarr, STX
           Adapted for IDL Version 2,               J. Isensee, July, 1990
           Code, documentation spiffed up           W. Landsman   August 1991
           TEXTOUT may be a string                  W. Landsman September 1995
           FLUX keyword added                       J. E. Hollis, February, 1996
           SETSKYVAL keyword, increase maxsky       W. Landsman, May 1997
           Work for more than 32767 stars           W. Landsman, August 1997
           Converted to IDL V5.0                    W. Landsman   September 1997
           Don't abort for insufficient sky pixels  W. Landsman  May 2000
           Added /EXACT keyword                     W. Landsman  June 2000
           Allow SETSKYVAL = 0                      W. Landsman  December 2000
           Set BADPIX[0] = BADPIX[1] to ignore bad pixels W. L.  January 2001
           Fix chk_badpixel problem introduced Jan 01 C. Ishida/W.L. February 2001
           Converted from IDL to python             D. Jones January 2014
           Adapted for hstphot project              S. Rodney  July 2014
    """

    if verbose > 1:
        import time
        tstart = time.time()
    elif verbose: import time

    if debug :
        import pdb
        pdb.set_trace()

    # Force np arrays
    if not np.iterable( xc ):
        xc = np.array([xc])
        yc = np.array([yc])
    if isinstance(xc, list): xc = np.array(xc)
    if isinstance(yc, list): yc = np.array(yc)    
    assert len(xc) == len(yc), 'xc and yc arrays must be identical length.'

    if not np.iterable( apr ) :
        apr = np.array( [ apr ] )
    Naper = len( apr ) # Number of apertures
    Nstars = len( xc )   # Number of stars to measure

    # Set parameter limits
    if len(minsky) == 0: minsky = 20

    # Number of columns and rows in image array
    s = np.shape(image)
    ncol = s[1]
    nrow = s[0]


    if setskyval is not None :
        if not np.iterable(setskyval) :
            setskyval = [setskyval,0.,1.]
        assert len(setskyval)==3, 'Keyword SETSKYVAL must contain 1 or 3 elements'
        skyrad = [ 0., np.max(apr) + 1]    #use np.max (max function does not work on scalars)
    skyrad = asfarray(skyrad)




    # String array to display mags for all apertures in one line for each star
    outstr = [ '' for star in range(Nstars)]

    # Declare arrays
    mag = zeros( [ Nstars, Naper])
    magerr =  zeros( [ Nstars, Naper])
    flux = zeros( [ Nstars, Naper])
    fluxerr =  zeros( [ Nstars, Naper])
    badflag = zeros( [ Nstars, Naper])
    sky = zeros( Nstars )
    skyerr = zeros( Nstars )
    area = np.pi*apr*apr  # Area of each aperture

    if exact:
        bigrad = apr + 0.5
        smallrad = apr/np.sqrt(2) - 0.5 

    if setskyval is None :
        rinsq =  skyrad[0]**2
        routsq = skyrad[1]**2

    #  Compute the limits of the submatrix.   Do all stars in vector notation.
    lx = (xc-skyrad[1]).astype(int)  # Lower limit X direction
    ly = (yc-skyrad[1]).astype(int)  # Lower limit Y direction
    ux = (xc+skyrad[1]).astype(int)  # Upper limit X direction
    uy = (yc+skyrad[1]).astype(int)  # Upper limit Y direction

    lx[where(lx < 0)[0]] = 0
    ux[where(ux > ncol-1)[0]] = ncol-1
    nx = ux-lx+1                         # Number of pixels X direction

    ly[where(ly < 0)[0]] = 0
    uy[where(uy > nrow-1)[0]] = nrow-1
    ny = uy-ly +1                      # Number of pixels Y direction

    dx = xc-lx                         # X coordinate of star's centroid in subarray
    dy = yc-ly                         # Y coordinate of star's centroid in subarray

    # Find the edge of the subarray that is closest to each star
    # and then flag any stars that are too close to the edge or off-image
    edge = zeros(len(dx))
    for i,dx1,nx1,dy1,ny1 in zip(range(len(dx)),dx,nx,dy,ny):
        edge[i] = min([(dx[i]-0.5),(nx[i]+0.5-dx[i]),(dy[i]-0.5),(ny[i]+0.5-dy[i])])
    badstar = np.where( (xc<0.5) | (xc>ncol-1.5) |
                        (yc<0.5) | (yc>nrow-1.5), 1, 0 )
    if np.any( badstar ) :
        nbad = badstar.sum()
        print('WARNING [aper.py] - ' + str(nbad) + ' star positions outside image')

    if verbose :
        tloop = time.time()
    for i in range(Nstars):  # Compute magnitudes for each star
        while True :
            # mimic GOTO statements : break out of this while block whenever
            # we decide this star is bad
            apflux = asarray([np.nan]*Naper)
            apfluxerr = asarray([np.nan]*Naper)
            apmag = asarray([np.nan]*Naper)
            apmagerr = asarray([np.nan]*Naper)
            skymod = 0.  # Sky mode
            skysig = 0.  # Sky sigma
            skyskw = 0.  # Sky skew
            error1 = asarray([np.nan]*Naper)
            error2 = asarray([np.nan]*Naper)
            error3 = array([np.nan]*Naper)
            apbad = np.ones( Naper )
            if badstar[i]: # star is bad, return NaNs for all values
                break

            rotbuf = image[ ly[i]:uy[i]+1,lx[i]:ux[i]+1 ] #Extract subarray from image
            shapey,shapex = np.shape(rotbuf)[0],np.shape(rotbuf)[1]
            #  RSQ will be an array, the same size as ROTBUF containing the square of
            #      the distance of each pixel to the center pixel.

            dxsq = ( arange( nx[i] ) - dx[i] )**2
            rsq = np.ones( [ny[i], nx[i]] )
            for ii  in range(ny[i]):
                rsq[ii,:] = dxsq + (ii-dy[i])**2

            if exact:
                nbox = range(nx[i]*ny[i])
                xx = (nbox % nx[i]).reshape( ny[i], nx[i])
                yy = (nbox/nx[i]).reshape(ny[i],nx[i])
                x1 = np.abs(xx-dx[i]) 
                y1 = np.abs(yy-dy[i])
            else:
                r = np.sqrt(rsq) - 0.5    #2-d array of the radius of each pixel in the subarray

            rsq,rotbuf = rsq.reshape(shapey*shapex),rotbuf.reshape(shapey*shapex)
            if setskyval is None :
                # skypix will be 1-d array of sky pixels
                skypix = np.zeros( rsq.shape )

                #  Select pixels within sky annulus,
                skypix[where(( rsq >= rinsq ) &
                             ( rsq <= routsq ))[0]] = 1
                if badpix[0]!=badpix[1] :
                    # Eliminate pixels above or below the badpix threshold vals
                    skypix[where(((rotbuf < badpix[0]) | (rotbuf > badpix[1])) &
                                 (skypix == 1))[0]] = 0
                sindex =  where(skypix)[0]
                nsky = len(sindex)

                if ( nsky < minsky ):   # Insufficient sky pixels?
                    if verbose:
                        print("ERROR: nsky=%i is fewer than minimum %i valid pixels in the sky annulus."%(nsky,minsky))
                    break

                skybuf = rotbuf[ sindex[0:nsky] ]
                if skyalgorithm.startswith('sigmaclip'):
                    # The sky annulus is (nearly) empty of stars, (as in a diff image)
                    # so we can simply compute the sigma-clipped mean of all pixels in
                    # the annulus
                    skybufclipped,lothresh,hithresh = sigmaclip( skybuf, low=4.0, high=4.0)
                    skymod = np.mean( skybufclipped )
                    skysig = np.std( skybufclipped )
                    skyskw = skew( skybufclipped )

                else:
                    # Compute the sky mode, sigma and skewness using the
                    # mean/median/mode algorithm in mmm.py, which assumes that
                    # most of the outlier pixels are positive.
                    skymod, skysig, skyskw = mmm.mmm(skybuf,readnoise=readnoise,minsky=minsky)

                skyvar = skysig**2    #Variance of the sky brightness
                sigsq = skyvar/nsky  #Square of standard error of mean sky brightness
             
                if ( skysig < 0.0 ):
                    # If the modal sky value could not be determined, then all
                    # apertures for this star are bad.  So skip to the next.
                    break

                if skysig > 999.99: skysig = 999      #Don't overload output formats
                if skyskw < -99: skyskw = -99
                if skyskw > 999.9: skyskw = 999.9

            else:
                skymod = setskyval[0]
                skysig = setskyval[1]
                nsky = setskyval[2]
                skyvar = skysig**2
                sigsq = skyvar/nsky
                skyskw = 0

            for k in range(Naper): # Find pixels within each aperture
                if ( edge[i] >= apr[k] ):   #Does aperture extend outside the image?
                    if exact:
                        mask = zeros(ny[i]*nx[i])

                        x1,y1 = x1.reshape(ny[i]*nx[i]),y1.reshape(ny[i]*nx[i])
                        igoodmag = where( ( x1 < smallrad[k] ) & (y1 < smallrad[k] ))[-1]
                        Ngoodmag = len(igoodmag)
                        if Ngoodmag > 0: mask[igoodmag] = 1
                        bad = where(  (x1 > bigrad[k]) | (y1 > bigrad[k] ))[-1]
                        mask[bad] = -1

                        gfract = where(mask == 0.0)[0] 
                        Nfract = len(gfract)
                        if Nfract > 0:
                            yygfract = yy.reshape(ny[i]*nx[i])[gfract]
                            xxgfract = xx.reshape(ny[i]*nx[i])[gfract] 

                            mask[gfract] = pixwt.Pixwt(dx[i],dy[i],apr[k],xxgfract,yygfract)
                            mask[gfract[where(mask[gfract] < 0.0)[0]]] = 0.0
                        thisap = where(mask > 0.0)[0]

                        thisapd = rotbuf[thisap]
                        fractn = mask[thisap]
                    else:
                        # approximating the circular aperture shape
                        rshapey,rshapex = np.shape(r)[0],np.shape(r)[1]
                        thisap = where( r.reshape(rshapey*rshapex) < apr[k] )[0]   # Select pixels within radius
                        thisapd = rotbuf.reshape(rshapey*rshapex)[thisap]
                        thisapr = r.reshape(rshapey*rshapex)[thisap]
                        fractn = apr[k]-thisapr 
                        fractn[where(fractn > 1)[0]] = 1
                        fractn[where(fractn < 0)[0]] = 0  # Fraction of pixels to count
                        full = zeros(len(fractn))
                        full[where(fractn == 1)[0]] = 1.0
                        gfull = where(full)[0]
                        Nfull = len(gfull)
                        gfract = where(1 - full)[0]
                        factor = (area[k] - Nfull ) / np.sum(fractn[gfract])
                        fractn[gfract] = fractn[gfract]*factor
                else:
                    if verbose :
                        print("WARNING [aper.py]: aperture extends outside the image!")
                    continue
                    # END "if exact ...  else ..."

                # Check for any bad pixel values (nan,inf) and those outside
                # the user-specified range of valid pixel values.  If any
                # are found in the aperture, raise the badflux flag.
                apbad[k] = 0
                if not np.all( np.isfinite(thisapd) ) :
                    if verbose :
                        print("WARNING : nan or inf pixels detected in aperture.\n"
                              "We're setting these to 0, but the photometry"
                              "may be biased.")
                    thisapd[np.isfinite(thisapd)==False] = 0
                    apbad[k] = 1
                    fractn = 0
                if badpix[0] < badpix[1] :
                    ibadpix = np.where((thisapd<=badpix[0]) | (thisapd>=badpix[1]))
                    if len(ibadpix[0]) > 0 :
                        if verbose :
                            print("WARNING : pixel values detected in aperture"
                                  " that are outside of the allowed range "
                                  " [%.1f , %.1f] \n"%(badpix[0],badpix[1]) +
                                  "We're treating these as 0, but the "
                                  "photometry may be biased.")
                        thisapd[ibadpix] = 0
                        apbad[k] = 1
                # Sum the flux over the irregular aperture
                apflux[k] = np.sum(thisapd*fractn)
            # END for loop over apertures

            igoodflux = where(np.isfinite(apflux))[0]
            Ngoodflux = len(igoodflux)
            if Ngoodflux > 0:
                if verbose > 2 :
                    print(" SRCFLUX   APFLUX    SKYMOD   AREA")
                    for igf in igoodflux :
                        print("%.4f   %.4f   %.4f   %.4f "%(apflux[igf]-skymod*area[igf],apflux[igf],skymod,area[igf]))
                # Subtract sky from the integrated brightnesses
                apflux[igoodflux] = apflux[igoodflux] - skymod*area[igoodflux]

            # Compute flux error
            error1[igoodflux] = area[igoodflux]*skyvar   #Scatter in sky values
            error2[igoodflux] = np.abs(apflux[igoodflux])/phpadu  #Random photon noise
            error3[igoodflux] = sigsq*area[igoodflux]**2  #Uncertainty in mean sky brightness
            apfluxerr[igoodflux] = np.sqrt(error1[igoodflux] + error2[igoodflux] + error3[igoodflux])

            igoodmag = where (apflux > 0.0)[0]  # Are there any valid integrated fluxes?
            Ngoodmag = len(igoodmag)
            if ( Ngoodmag > 0 ) : # convert valid fluxes to mags
                apmagerr[igoodmag] = 1.0857*apfluxerr[igoodmag]/apflux[igoodmag]   #1.0857 = 2.5/log(10)
                apmag[igoodmag] =  zeropoint-2.5*np.log10(apflux[igoodmag])
            break # Closing the 'while True' loop.

        # TODO : make a more informative output string
        outstr[i] = '%.3f,%.3f :'%(xc[i],yc[i]) + \
                    '  '.join( [ '%.4f+-%.4f'%(apmag[ii],apmagerr[ii])
                                 for ii in range(Naper) ] )

        sky[i] = skymod
        skyerr[i] = skysig
        mag[i,:] = apmag
        magerr[i,:]= apmagerr
        flux[i,:] = apflux
        fluxerr[i,:]= apfluxerr
        badflag[i,:] = apbad

    if Nstars == 1 :
        sky = sky[0]
        skyerr = skyerr[0]
        mag = mag[0]
        magerr = magerr[0]
        flux = flux[0]
        fluxerr = fluxerr[0]
        badflag = badflag[0]
        outstr = outstr[0]

    if verbose>1:
        print('hstphot.aper took %.3f seconds'%(time.time()-tstart))
        print('Each of %i loops took %.3f seconds'%(Nstars,(time.time()-tloop)/Nstars))

    return(mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr)
