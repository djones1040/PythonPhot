#!/usr/bin/env python
#D. Jones - 1/15/14
"""This code is from the IDL Astronomy Users Library"""

import numpy as np
import pixwt
import mmm
where,asfarray,asarray,array,zeros,arange = np.where,np.asfarray,np.asarray,np.array,np.zeros,np.arange

def aper(image,xc,yc,
         phpadu,apr,
         skyrad,badpix,
         setskyval = -99,
         verbose = False, 
         silent = False, 
         flux = False,
         exact = False,
         outfile = None,
         chk_badpix = True,
         minsky=[],
         meanback=False,
         readnoise = 0,
         nan = False,
         debug=False):
    """;+
    ; NAME:
    ;      APER
    ; PURPOSE:
    ;      Compute concentric aperture photometry (adapted from DAOPHOT) 
    ; EXPLANATION:
    ;     APER can compute photometry in several user-specified aperture radii.  
    ;     A separate sky value is computed for each source using specified inner 
    ;     and outer sky radii.   
    ;
    ; CALLING SEQUENCE:
    ;     APER, image, xc, yc, [ mags, errap, sky, skyerr, phpadu, apr, skyrad, 
    ;                       badpix, /EXACT, /FLUX, PRINT = , /SILENT, SETSKYVAL = ]
    ; INPUTS:
    ;     IMAGE -  input image array
    ;     XC     - vector of x coordinates. 
    ;     YC     - vector of y coordinates
    ;
    ; OPTIONAL INPUTS:
    ;     PHPADU - Photons per Analog Digital Units, numeric scalar.  Converts
    ;               the data numbers in IMAGE to photon units.  (APER assumes
    ;               Poisson statistics.)  
    ;     APR    - Vector of up to 12 REAL photometry aperture radii.
    ;     SKYRAD - Two element vector giving the inner and outer radii
    ;               to be used for the sky annulus
    ;     BADPIX - Two element vector giving the minimum and maximum value
    ;               of a good pix (Default [-32765,32767]).    If BADPIX[0] is
    ;               equal to BADPIX[1] then it is assumed that there are no bad
    ;               pixels.
    ;
    ; OPTIONAL KEYWORD INPUTS:
    ;     /EXACT -  By default, APER counts subpixels, but uses a polygon 
    ;             approximation for the intersection of a circular aperture with
    ;             a square pixel (and normalize the total area of the sum of the
    ;             pixels to exactly match the circular area).   If the /EXACT 
    ;             keyword, then the intersection of the circular aperture with a
    ;             square pixel is computed exactly.    The /EXACT keyword is much
    ;             slower and is only needed when small (~2 pixels) apertures are
    ;             used with very undersampled data.    
    ;     /FLUX - By default, APER uses a magnitude system where a magnitude of
    ;               25 corresponds to 1 flux unit.   If set, then APER will keep
    ;              results in flux units instead of magnitudes.
    ;     PRINT - if set and non-zero then APER will also write its results to
    ;               a file aper.prt.   One can specify the output file name by
    ;               setting PRINT = 'filename'.
    ;     /SILENT -  If supplied and non-zero then no output is displayed to the
    ;               terminal.
    ;     SETSKYVAL - Use this keyword to force the sky to a specified value 
    ;               rather than have APER compute a sky value.    SETSKYVAL 
    ;               can either be a scalar specifying the sky value to use for 
    ;               all sources, or a 3 element vector specifying the sky value, 
    ;               the sigma of the sky value, and the number of elements used 
    ;               to compute a sky value.   The 3 element form of SETSKYVAL
    ;               is needed for accurate error budgeting.
    ;
    ; OUTPUTS:
    ;     MAGS   -  NAPER by NSTAR array giving the magnitude for each star in
    ;               each aperture.  (NAPER is the number of apertures, and NSTAR
    ;               is the number of stars).   A flux of 1 digital unit is assigned
    ;               a zero point magnitude of 25.
    ;     ERRAP  -  NAPER by NSTAR array giving error in magnitude
    ;               for each star.  If a magnitude could not be deter-
    ;               mined then ERRAP = 9.99.
    ;     SKY  -    NSTAR element vector giving sky value for each star
    ;     SKYERR -  NSTAR element vector giving error in sky values
    ;
    ; PROCEDURES USED:
    ;       DATATYPE(), GETOPT, MMM, PIXWT(), STRN(), STRNUMBER()
    ; NOTES:
    ;       Reasons that a valid magnitude cannot be computed include the following:
    ;      (1) Star position is too close (within 0.5 pixels) to edge of the frame
    ;      (2) Less than 20 valid pixels available for computing sky
    ;      (3) Modal value of sky could not be computed by the procedure MMM
    ;      (4) *Any* pixel within the aperture radius is a "bad" pixel
    ;
    ;       APER was modified in June 2000 in two ways: (1) the /EXACT keyword was
    ;       added (2) the approximation of the intersection of a circular aperture
    ;       with square pixels was improved (i.e. when /EXACT is not used) 
    ; REVISON HISTORY:
    ;       Adapted to IDL from DAOPHOT June, 1989   B. Pfarr, STX
    ;       Adapted for IDL Version 2,               J. Isensee, July, 1990
    ;       Code, documentation spiffed up           W. Landsman   August 1991
    ;       TEXTOUT may be a string                  W. Landsman September 1995
    ;       FLUX keyword added                       J. E. Hollis, February, 1996
    ;       SETSKYVAL keyword, increase maxsky       W. Landsman, May 1997
    ;       Work for more than 32767 stars           W. Landsman, August 1997
    ;       Converted to IDL V5.0                    W. Landsman   September 1997
    ;       Don't abort for insufficient sky pixels  W. Landsman  May 2000
    ;       Added /EXACT keyword                     W. Landsman  June 2000 
    ;       Allow SETSKYVAL = 0                      W. Landsman  December 2000 
    ;       Set BADPIX[0] = BADPIX[1] to ignore bad pixels W. L.  January 2001     
    ;       Fix chk_badpixel problem introduced Jan 01 C. Ishida/W.L. February 2001 
    ;-"""

    if debug:
        import time
        tstart = time.time()

    #             Set parameter limits
    if len(minsky) == 0: minsky = 20
    maxrad = 100.          #Maximum outer radius permitted for the sky annulus.
    maxsky = 10000         #Maximum number of pixels allowed in the sky annulus.
    #                                

    s = np.shape(image)
    ncol = s[1] ; nrow = s[0]           #Number of columns and rows in image array
    
    chk_badpix = badpix[0] < badpix[1]     #Ignore bad pixel checks?

    if setskyval != -99 and len(setskyval) == 1:
        setskyval = [setskyval,0.,1.]
        skyrad = [ 0., max(apr) + 1]
    elif setskyval != -99 and len(setskyval) != 3:
        print('ERROR - Keyword SETSKYVAL must contain 1 or 3 elements')
        skyrad = [ 0., max(apr) + 1]

    # Get radii of sky annulii
    skyrad = asfarray(skyrad); xc = asarray(xc); yc = asarray(yc)

    try: Naper = len( apr )                        #Number of apertures
    except: Naper = 1
    if Naper == 1: apr = float(apr)
    try: Nstars = min([ len(xc), len(yc) ])  #Number of stars to measure
    except: 
        Nstars = 1
        xc,yc = asarray(xc),asarray(yc)

    ms = array( ['']*Naper )       #String array to display mag for each aperture

# if keyword_set(flux) then $
#          fmt = '(F8.1,1x,A,F7.1)' else $           #Flux format
#          fmt = '(F9.3,A,F5.3)'                  #Magnitude format
# fmt2 = '(I5,2F8.2,F7.2,3A,3(/,28x,4A,:))'       #Screen format
# fmt3 = '(I4,5F8.2,6A,2(/,44x,9A,:))'            #Print format

    mags = zeros( [ Nstars, Naper]) ; errap =  zeros( [ Nstars, Naper])           #Declare arrays
    sky = zeros( Nstars )        ; skyerr = zeros( Nstars )     
    try: area = np.pi*apr*apr                 #Area of each aperture
    except: area = np.pi*apr*apr

    if exact:
        bigrad = apr + 0.5
        smallrad = apr/np.sqrt(2) - 0.5 
     

    if setskyval == -99:
        
        rinsq =  skyrad[0]**2 
        routsq = skyrad[1]**2

#     if verbose:      #Open output file and write header info?
#   if datatype(PRINT) NE 'STR'  then file = 'aper.prt' $
#                                   else file = print
#   message,'Results will be written to a file ' + file,/INF
#   openw,lun,file,/GET_LUN
#   printf,lun,' Program: APER '+ systime(), '   User: ', $
#      getenv('USER'),'  Node: ',getenv('NODE')
#   for j = 0, Naper-1 do printf,lun, $
#               format='(a,i2,a,f4.1)','Radius of aperture ',j,' = ',apr[j]
#   printf,lun,f='(/a,f4.1)','Inner radius for sky annulus = ',skyrad[0]
#   printf,lun,f='(a,f4.1)', 'Outer radius for sky annulus = ',skyrad[1]
#   if keyword_set(FLUX) then begin
#       printf,lun,f='(/a)', $
#           'STAR   X       Y        SKY   SKYSIG    SKYSKW   FLUXES'
#      endif else printf,lun,f='(/a)', $
#           'STAR   X       Y        SKY   SKYSIG    SKYSKW   MAGNITUDES'
# endif
# print = keyword_set(PRINT)

#         Print header
# if not SILENT then begin
#    if (KEYWORD_SET(FLUX)) then begin
#       print, format="(/1X,'STAR',5X,'X',7X,'Y',6X,'SKY',8X,'FLUXES')"
#    endif else print, $ 
#       format="(/1X,'STAR',5X,'X',7X,'Y',6X,'SKY',8X,'MAGNITUDES')" 
# endif

#  Compute the limits of the submatrix.   Do all stars in vector notation.

    lx = (xc-skyrad[1]).astype(int)           #Lower limit X direction
    ly = (yc-skyrad[1]).astype(int)           #Lower limit Y direction
    ux = (xc+skyrad[1]).astype(int)    #Upper limit X direction
    uy = (yc+skyrad[1]).astype(int)             #   #Upper limit Y direction

    if Nstars == 1:
        lx,ly,ux,uy,xc,yc = asarray([lx]),asarray([ly]),asarray([ux]),asarray([uy]),asarray([xc]),asarray([yc])
    lx[where(lx < 0)[0]] = 0

    ux[where(ux > ncol-1)[0]] = ncol-1
    nx = ux-lx+1                         #Number of pixels X direction
    ly[where(ly < 0)[0]] = 0

    uy[where(uy > nrow-1)[0]] = nrow-1
    ny = uy-ly +1                        #Number of pixels Y direction
    dx = xc-lx                         #X coordinate of star's centroid in subarray
    dy = yc-ly                         #Y coordinate of star's centroid in subarray
     
    if type(dx) == np.ndarray:
        edge = zeros(len(dx))
        for i,dx1,nx1,dy1,ny1 in zip(range(len(dx)),dx,nx,dy,ny):
            edge[i] = min([(dx1-0.5),(nx1+0.5-dx1),(dy1-0.5),(ny1+0.5-dy1)]) #Closest edge to array
    else:
        edge = min([(dx-0.5),(nx+0.5-dx),(dy-0.5),(ny+0.5-dy)]) #Closest edge to array
    badstar = zeros(len(xc))
    badstar[where((xc < 0.5) | 
                     (xc > ncol-1.5) | #Stars too close to the edge
                     (yc < 0.5) | 
                     (yc > nrow-1.5))[0]] = 1
     #
    badindex = where( badstar)[0]              #Any stars outside image
    nbad = len(badindex)
    if ( nbad > 0 ):
        print('WARNING - ' + str(nbad) + ' star positions outside image')
    if flux:
        badval = np.nan
        baderr = badval
    else:
        badval = 99.999
        baderr = 9.999
    if Naper == 1: 
        apr = array([apr]); area = array([area])
        if exact:
            smallrad = array([smallrad]); bigrad = array([bigrad])

    if debug:
        tloop = time.time()
    for i in range(Nstars):           #Compute magnitudes for each star
        for v in range(1):     # bogus loop to replicate IDL GOTO
            apmag = asarray([badval]*Naper)   ; magerr = asarray([baderr]*Naper)
            skymod = 0. ; skysig = 0. ;  skyskw = 0.  #Sky mode sigma and skew
            error1 = asarray([badval]*Naper)   ; error2 = asarray([badval]*Naper)   ; error3 = array([badval]*Naper)
            if badstar[i]:    #
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
            #  Select pixels within sky annulus, and eliminate pixels falling
            #       below BADLO threshold.  SKYBUF will be 1-d array of sky pixels
            if setskyval == -99:

                skypix = rsq*0
                skypix[where(( rsq >= rinsq ) & 
                             ( rsq <= routsq ))[0]] = 1
                skypix[where(((rotbuf < badpix[0]) | 
                              (rotbuf > badpix[1])) &
                             (skypix == 1))[0]] = 0
                
                sindex =  where(skypix)[0]
                nsky = len(sindex)
                if nsky > maxsky: nsky = maxsky
                # nsky =   nsky < maxsky   #Must be less than MAXSKY pixels
             
                if ( nsky < minsky ):                       #Sufficient sky pixels?
                    if not silent:
                        print('There aren''t enough valid pixels in the sky annulus.')
                        # apmag[:] = -99.999
                    break

                skybuf = rotbuf[ sindex[0:nsky] ]
                if meanback:
                    meanclip,skybuf,skymod,skysig, \
                        CLIPSIG=clipsig, MAXITER=maxiter, CONVERGE_NUM=converge_num
                else:
                     skymod, skysig, skyskw = mmm.mmm(skybuf,
                                                      readnoise=readnoise,
                                                      minsky=minsky)

                #  Obtain the mode, standard deviation, and skewness of the peak in the
                #      sky histogram, by calling MMM.

                skyvar = skysig**2    #Variance of the sky brightness
                sigsq = skyvar/nsky  #Square of standard error of mean sky brightness
             
                if ( skysig < 0.0 ):   #If the modal sky value could not be
                    #apmag[:] = -99.999          #determined, then all apertures for
                    break                       #this star are bad.

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

            for k in range(Naper):      #Find pixels within each aperture
             
                if ( edge[i] >= apr[k] ):   #Does aperture extend outside the image?
                    if exact:
                        mask = zeros(ny[i]*nx[i])

                        x1,y1 = x1.reshape(ny[i]*nx[i]),y1.reshape(ny[i]*nx[i])
                        good = where( ( x1 < smallrad[k] ) & (y1 < smallrad[k] ))[-1]
                        Ngood = len(good)
                        if Ngood > 0: mask[good] = 1
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
                        #
                        rshapey,rshapex = np.shape(r)[0],np.shape(r)[1]
                        thisap = where( r.reshape(rshapey*rshapex) < apr[k] )[0]   #Select pixels within radius
                        thisapd = rotbuf.reshape(rshapey*rshapex)[thisap]
                        thisapr = r.reshape(rshapey*rshapex)[thisap]
                        fractn = apr[k]-thisapr 
                        fractn[where(fractn > 1)[0]] = 1
                        fractn[where(fractn < 0)[0]] = 0          # Fraction of pixels to count
                        full = zeros(len(fractn))
                        full[where(fractn == 1)[0]] = 1.0
                        gfull = where(full)[0]
                        Nfull = len(gfull)
                        gfract = where(1 - full)[0]
                        factor = (area[k] - Nfull ) / np.sum(fractn[gfract])
                        fractn[gfract] = fractn[gfract]*factor
                         

                        #     If the pixel is bad, set the total counts in this aperture to a large
                        #        negative number
                        #
                    if nan:
                        badflux =  min(np.isfinite(thisapd)) == 0
                    elif chk_badpix:
                        minthisapd = np.min(thisapd) ; maxthisapd = np.max(thisapd)
                        if (minthisapd <= badpix[0] ) or ( maxthisapd >= badpix[1]):
                            badflux = 1
                        else: badflux = 0
                    else:
                        badflux = 0
                    if not badflux:
                        apmag[k] = np.sum(thisapd*fractn)  #Total over irregular aperture


            if flux: 
                g = where(np.isfinite(apmag))[0]  
            else:
                g = where(np.abs(apmag - badval) > 0.01)[0]
            Ng = len(g)
            if Ng > 0:
                apmag[g] = apmag[g] - skymod*area[g]  #Subtract sky from the integrated brightnesses
     
            good = where (apmag > 0.0)[0]     #Are there any valid integrated fluxes?
            Ngood = len(good)
     
            if ( Ngood > 0 ):               #If YES then compute errors
                error1[g] = area[g]*skyvar   #Scatter in sky values
                error2[g] = apmag[g]/phpadu  #Random photon noise 
                if apmag[g] < 0: error2[g] = 0
                error3[g] = sigsq*area[g]**2  #Uncertainty in mean sky brightness
                magerr[g] = np.sqrt(error1[g] + error2[g] + error3[g])
                
                if not flux:
                    magerr[good] = 1.0857*magerr[g]/apmag[g]   #1.0857 = log(10)/2.5
                    apmag[good] =  25.-2.5*np.log10(apmag[g])  

# BADSTAR:   

#Print out magnitudes for this star

        for ii in range(Naper):             #Concatenate mags into a string
             
            ms[ii] = str( apmag[ii]) + '+-' + str(magerr[ii])
            #   if PRINT then  printf,lun, $      #Write results to file?
            #      form = fmt3,  i, xc[i], yc[i], skymod, skysig, skyskw, ms
            #   if not SILENT then print,form = fmt2, $       #Write results to terminal?
            #          i,xc[i],yc[i],skymod,ms

        sky[i] = skymod    ;  skyerr[i] = skysig  #Store in output variable
        mags[i,0] = apmag  ;  errap[i,0]= magerr


# if PRINT then free_lun, lun             #Close output file

    if debug:
        print('Aper took %.3f seconds'%(time.time()-tstart))
        print('Each of %i loops took %.3f seconds'%(Nstars,(time.time()-tloop)/Nstars))

    return(mags,errap,sky,skyerr)
