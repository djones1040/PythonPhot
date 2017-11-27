#!/usr/bin/env python
# D. Jones - 2/13/14
"""This code is from the IDL Astronomy Users Library"""

import numpy as np

def cntrd(img, x, y,
          fwhm, verbose = True, 
          debug = False, 
          extendbox = False, 
          keepcenter = False):
    """Compute the centroid of a star using a derivative search 
    (adapted for IDL from DAOPHOT, then translated from IDL to Python).

    CNTRD uses an early DAOPHOT "FIND" centroid algorithm by locating the 
    position where the X and Y derivatives go to zero.   This is usually a 
    more "robust"  determination than a "center of mass" or fitting a 2d 
    Gaussian  if the wings in one direction are affected by the presence
    of a neighboring star.

    xcen,ycen = cntrd.cntrd(img, x, y, fwhm)
    
    REQUIRED INPUTS:
         img  - Two dimensional image array
         x,y  - Scalar or vector integers giving approximate integer stellar 
                 center
         fwhm - floating scalar; Centroid is computed using a box of half
                 width equal to 1.5 sigma = 0.637* fwhm.

    OPTIONAL KEYWORD INPUTS:
         verbose -    Default = True.  If set, CNTRD prints an error message if it 
                       is unable to compute the centroid.
         debug -      If this keyword is set, then CNTRD will display the subarray
                       it is using to compute the centroid.
         extendbox -  {non-negative positive integer}.   CNTRD searches a box with
                       a half width equal to 1.5 sigma  = 0.637* FWHM to find the
                       maximum pixel.    To search a larger area, set extendbox to
                       the number of pixels to enlarge the half-width of the box.
                       A list/array of [X,Y] coordinates defines a rectangle.
                       Default is 0; prior to June 2004, the default was extendbox = 3
                       keepcenter - By default, CNTRD finds the maximum pixel in a box
                       centered on the input X,Y coordinates, and then extracts a new
                       box about this maximum pixel.  Set the keepcenter keyword
                       to skip then step of finding the maximum pixel, and instead use
                       a box centered on the input X,Y coordinates.

    RETURNS:
         xcen - the computed X centroid position, same number of points as X
         ycen - computed Y centroid position, same number of points as Y, 
                 floating point
    
         Values for xcen and ycen will not be computed if the computed
         centroid falls outside of the box, or if the computed derivatives
         are non-decreasing.   If the centroid cannot be computed, then a 
         xcen and ycen are set to -1 and (when verbose=True) a message is
         displayed
    
    PROCEDURE:
         Maximum pixel within distance from input pixel X, Y  determined
         from FHWM is found and used as the center of a square, within
         which the centroid is computed as the value (XCEN,YCEN) at which
         the derivatives of the partial sums of the input image over (y,x)
         with respect to (x,y) = 0.  In order to minimize contamination from
         neighboring stars stars, a weighting factor W is defined as unity in
         center, 0.5 at end, and linear in between
    
    RESTRICTIONS:
         (1) Does not recognize (bad) pixels.   Use the procedure GCNTRD.PRO
              in this situation.
         (2) DAOPHOT now uses a newer algorithm (implemented in GCNTRD.PRO) in
              which centroids are determined by fitting 1-d Gaussians to the
              marginal distributions in the X and Y directions.
         (3) The default behavior of CNTRD changed in June 2004 (from EXTENDBOX=3
              to EXTENDBOX = 0).
         (4) Stone (1989, AJ, 97, 1227) concludes that the derivative search
              algorithm in CNTRD is not as effective (though faster) as a
              Gaussian fit (used in GCNTRD.PRO).
    
    MODIFICATION HISTORY:
         Written following algorithm used by P. Stetson in DAOPHOT      J. K. Hill, S.A.S.C.   2/25/86
         Allowed input vectors                                          G. Hennessy            April,  1992
         Fixed to prevent wrong answer if floating pt. X & Y supplied   W. Landsman            March, 1993 
         Convert byte, integer subimages to float                       W. Landsman            May, 1995
         Converted to IDL V5.0                                          W. Landsman            September, 1997
         Better checking of edge of frame                               David Hogg             October, 2000
         Avoid integer wraparound for unsigned arrays                   W.Landsman             January, 2001
         Handle case where more than 1 pixel has maximum value          W.L.                   July, 2002
         Added /KEEPCENTER, EXTENDBOX (with default = 0) keywords       WL                     June, 2004
         Some errrors were returning X,Y = NaN rather than -1,-1        WL                     Aug, 2010
         Converted to Python                                            D. Jones               January, 2014
    """
    sz_image = np.shape(img)
    xsize = sz_image[1]
    ysize = sz_image[0]
    # dtype = sz_image[3]              ;Datatype

    # Compute size of box needed to compute centroid
    if not extendbox: extendbox = 0
    nhalf =  int(0.637*fwhm)  
    if nhalf < 2: nhalf = 2
    nbox = 2*nhalf+1             # Width of box to be used to compute centroid
    if not hasattr(extendbox,'__len__'):
        Xextendbox,Yextendbox = extendbox,extendbox
    elif hasattr(extendbox,'__len__'):
        Xextendbox,Yextendbox = extendbox
    nhalfbigx = nhalf + Xextendbox; nhalfbigy = nhalf + Yextendbox
    nbigx = nbox + Xextendbox*2; nbigy = nbox + Yextendbox*2 #Extend box 3 pixels on each side to search for max pixel value

    if isinstance(x,float) or isinstance(x,int): npts = 1
    else: npts = len(x) 
    if npts == 1: xcen = float(x) ; ycen = float(y)
    else: xcen = x.astype(float) ; ycen = y.astype(float)
    ix = np.round( x )          # Central X pixel        ;Added 3/93
    iy = np.round( y )          # Central Y pixel
    
    if npts == 1:
        x, y, ix, iy, xcen, ycen = [x], [y], [ix], [iy], [xcen], [ycen]
    for i in range(npts):        # Loop over X,Y vector
        pos = str(x[i]) + ' ' + str(y[i])
        if not keepcenter:
            if ((ix[i] < nhalfbigx) or ((ix[i] + nhalfbigx) > xsize-1) or
                (iy[i] < nhalfbigy) or ((iy[i] + nhalfbigy) > ysize-1)):
                xcen[i] = -1
                ycen[i] = -1
                if verbose:
                    print('Position '+ pos + ' too near edge of image')
                continue
            
            bigbox = img[int(iy[i]-nhalfbigy) : int(iy[i]+nhalfbigy+1),
                     int(ix[i]-nhalfbigx) : int(ix[i]+nhalfbigx+1)]

            # Locate maximum pixel in 'NBIG' sized subimage
            goodrow = np.where(bigbox == bigbox)
            mx = np.max( bigbox[goodrow])     #Maximum pixel value in BIGBOX
            mx_pos = np.where(bigbox == mx) #How many pixels have maximum value?
            #mx_pos = np.where(bigbox.reshape(np.shape(bigbox)[0] * np.shape(bigbox)[1]) == mx)[0]
            Nmax = len(mx_pos[0])
            idx = mx_pos[1] #% nbig          # X coordinate of Max pixel
            idy = mx_pos[0] #/ nbig          # Y coordinate of Max pixel

            if Nmax > 1:                 # More than 1 pixel at maximum?
                idx = np.round(np.sum(idx)/Nmax)
                idy = np.round(np.sum(idy)/Nmax)
            else:
                idx = idx[0]
                idy = idy[0]

            xmax = ix[i] - (nhalf+Xextendbox) + idx  #X coordinate in original image array
            ymax = iy[i] - (nhalf+Yextendbox) + idy  #Y coordinate in original image array
        else:
            xmax = ix[i]
            ymax = iy[i]

        # check *new* center location for range (added by Hogg)
        if ((xmax < nhalf) or ((xmax + nhalf) > xsize-1) or
            (ymax < nhalf) or ((ymax + nhalf) > ysize-1)):
            xcen[i] = -1
            ycen[i] = -1
            if verbose:
                print('Position '+ pos + ' moved too near edge of image')
            continue

        # Extract smaller 'STRBOX' sized subimage centered on maximum pixel
        strbox = img[int(ymax-nhalf) : int(ymax+nhalf+1), int(xmax-nhalf) : int(xmax+nhalf+1)]
        # if (dtype NE 4) and (dtype NE 5) then strbox = float(strbox)

        if debug:
            print('Subarray used to compute centroid:')
            print(strbox)

        ir = (nhalf-1)
        if ir < 1: ir = 1
        dd = np.arange(nbox-1).astype(int) + 0.5 - nhalf

        # Weighting factor W unity in center, 0.5 at end, and linear in between
        w = 1. - 0.5*(np.abs(dd)-0.5)/(nhalf-0.5)
        sumc   = np.sum(w)

        # Find X centroid
        deriv = np.roll(strbox,-1,axis=1) - strbox.astype(float)    #;Shift in X & subtract to get derivative
        deriv = deriv[nhalf-ir:nhalf+ir+1,0:nbox-1] #;Don't want edges of the array
        deriv = np.sum( deriv, 0 )                    #    ;Sum X derivatives over Y direction
        sumd   = np.sum( w*deriv )
        sumxd  = np.sum( w*dd*deriv )
        sumxsq = np.sum( w*dd**2 )
        if sumxd >= 0:    # ;Reject if X derivative not decreasing
            xcen[i]=-1
            ycen[i]=-1
            if verbose:
                print('Unable to compute X centroid around position '+ pos)
            continue
        dx = sumxsq*sumd/(sumc*sumxd)
        if np.abs(dx) > nhalf:    # Reject if centroid outside box
            xcen[i]=-1
            ycen[i]=-1
            if verbose:
                print('Computed X centroid for position '+ pos + ' out of range')
            continue
        xcen[i] = xmax - dx    # X centroid in original array

        #  Find Y Centroid
        deriv = np.roll(strbox,-1,axis=0) - strbox.astype(float)    # Shift in X & subtract to get derivative
        deriv = deriv[0:nbox-1,nhalf-ir:nhalf+ir+1]
        deriv = np.sum( deriv,1 )
        sumd =   np.sum( w*deriv )
        sumxd =  np.sum( w*deriv*dd )
        sumxsq = np.sum( w*dd**2 )
        if sumxd >= 0:  # Reject if Y derivative not decreasing
            xcen[i] = -1
            ycen[i] = -1
            if verbose:
                print('Unable to compute Y centroid around position '+ pos)
            continue
        dy = sumxsq*sumd/(sumc*sumxd)
        if np.abs(dy) > nhalf:  # Reject if computed Y centroid outside box
            xcen[i]=-1
            ycen[i]=-1
            if verbose:
                print('Computed Y centroid for position '+ pos + ' out of range')
            continue
        ycen[i] = ymax-dy

    if npts == 1:
        xcen,ycen = xcen[0],ycen[0]
    return(xcen,ycen)
