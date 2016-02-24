#!/usr/bin/env python
#D. Jones - 1/17/14
"""This code is from the IDL Astronomy Users Library"""

import numpy as np
from scipy.ndimage.filters import convolve

def find(image,
         hmin, fwhm, 
         roundlim=[-1.0,1.0], 
         sharplim=[0.2,1.0],
         doprint = False, 
         verbose = True):
    """Find positive brightness perturbations (i.e stars) in an image.
    Also returns centroids and shape parameters (roundness & sharpness).
    Adapted from 1991 version of DAOPHOT, but does not allow for bad pixels
    and uses a slightly different centroid algorithm.  Modified in March 
    2008 to use marginal Gaussian fits to find centroids.  Translated from
    IDL to Python in 2014.
    
    CALLING SEQUENCE:
         import find
         x,y,flux,sharp,round = find.find(image,hmin, fwhm, roundlim, sharplim)
    
    INPUTS:
         image -    2 dimensional image array (integer or real) for which one
    		     wishes to identify the stars present
    	 hmin  -    Threshold intensity for a point source - should generally 
                     be 3 or 4 sigma above background RMS
    	 fwhm  -    FWHM (in pixels) to be used in the convolve filter
    	 sharplim - 2 element vector giving low and high cutoff for the
    		     sharpness statistic (Default: [0.2,1.0] ).   Change this
    		     default only if the stars have significantly larger
    		     or smaller concentration than a Gaussian
    	 roundlim - 2 element vector giving low and high cutoff for the
    		     roundness statistic (Default: [-1.0,1.0] ).   Change this 
    		     default only if the stars are significantly elongated.
    
    OPTIONAL INPUT KEYWORDS:
    	verbose - set verbose = False to suppress all output display.  Default = True.
    	doprint - if set and non-zero then FIND will also write its results to
    		  a file find.prt.   Also one can specify a different output file 
    		  name by setting doprint = 'filename'.
    
     RETURNS:
    	x     -  vector containing x position of all stars identified by FIND
    	y     -  vector containing y position of all stars identified by FIND
    	flux  -  vector containing flux of identified stars as determined
    		  by a Gaussian fit.  Fluxes are NOT converted to magnitudes.
    	sharp -  vector containing sharpness statistic for identified stars
    	round -  vector containing roundness statistic for identified stars
    
    NOTES:
         (1) The sharpness statistic compares the central pixel to the mean of 
              the surrounding pixels.  If this difference is greater than the 
              originally estimated height of the Gaussian or less than 0.2 the height of the
    	      Gaussian (for the default values of SHARPLIM) then the star will be
    	      rejected. 
    
         (2) More recent versions of FIND in DAOPHOT allow the possibility of
              ignoring bad pixels.  Unfortunately, to implement this in IDL
              would preclude the vectorization made possible with the CONVOL function
              and would run extremely slowly.
    
         (3) Modified in March 2008 to use marginal Gaussian distributions to 
              compute centroid.  (Formerly, find.pro determined centroids by locating
              where derivatives went to zero -- see cntrd.pro for this algorithm.   
              This was the method used in very old (~1984) versions of DAOPHOT. )   
              As discussed in more detail in the comments to the code, the  centroid
              computation here is the same as in IRAF DAOFIND but differs slightly 
              from the current DAOPHOT.

     REVISION HISTORY:
    	Written                                                    W. Landsman, STX           February,  1987
    	ROUND now an internal function in V3.1                     W. Landsman                July,      1993
    	Change variable name DERIV to DERIVAT                      W. Landsman                February,  1996
    	Use /PRINT keyword instead of TEXTOUT                      W. Landsman                May,       1996
    	Changed loop indices to type LONG                          W. Landsman                August,    1997
        Replace DATATYPE() with size(/TNAME)                       W. Landsman                November,  2001
        Fix problem when PRINT= filename                           W. Landsman                October,   2002
        Fix problems with >32767 stars                             D. Schlegel/W. Landsman    September, 2004
        Fix error message when no stars found                      S. Carey/W. Landsman       September, 2007
        Rewrite centroid computation to use marginal Gaussians     W. Landsman                March,     2008
        Added Monitor keyword, /SILENT now suppresses all output   W. Landsman                November,  2008
        Work when threshold is negative (difference images)        W. Landsman                May,       2010
        Converted from IDL to Python                               D. Jones                   January,   2014
    """

    # Determine if hardcopy output is desired
    doprint = doprint

    image = image.astype(np.float64)
    maxbox = 13 	#Maximum size of convolution box in pixels 

    # Get information about the input image 

    type = np.shape(image)
    if len(type) != 2:
        print('ERROR - Image array (first parameter) must be 2 dimensional')
    n_x  = type[1] ; n_y = type[0]
    if verbose:
        print('Input Image Size is '+str(n_x) + ' by '+ str(n_y))

    if fwhm < 0.5:
        print('ERROR - Supplied FWHM must be at least 0.5 pixels')

    radius = 0.637*fwhm
    if radius < 2.001: radius = 2.001             #Radius is 1.5 sigma
    radsq = radius**2
    nhalf = int(radius) 
    if nhalf > (maxbox-1)/2.: nhalf = int((maxbox-1)/2.)   	#
    nbox = 2*nhalf + 1	## of pixels in side of convolution box 
    middle = nhalf          #Index of central pixel

    lastro = n_x - nhalf
    lastcl = n_y - nhalf
    sigsq = ( fwhm/2.35482 )**2
    mask = np.zeros( [nbox,nbox], dtype='int8' )   #Mask identifies valid pixels in convolution box 
    g = np.zeros( [nbox,nbox] )      #g will contain Gaussian convolution kernel

    dd = np.arange(nbox-1,dtype='int') + 0.5 - middle	#Constants need to compute ROUND
    dd2 = dd**2

    row2 = (np.arange(nbox)-nhalf)**2

    for i in range(nhalf+1):
        temp = row2 + i**2
        g[nhalf-i,:] = temp         
        g[nhalf+i,:] = temp

    g_row = np.where(g <= radsq)
    mask[g_row[0],g_row[1]] = 1     #MASK is complementary to SKIP in Stetson's Fortran
    good = np.where( mask)  #Value of c are now equal to distance to center
    pixels = len(good[0])

#  Compute quantities for centroid computations that can be used for all stars
    g = np.exp(-0.5*g/sigsq)

#  In fitting Gaussians to the marginal sums, pixels will arbitrarily be 
# assigned weights ranging from unity at the corners of the box to 
# NHALF^2 at the center (e.g. if NBOX = 5 or 7, the weights will be
#
#                                 1   2   3   4   3   2   1
#      1   2   3   2   1          2   4   6   8   6   4   2
#      2   4   6   4   2          3   6   9  12   9   6   3
#      3   6   9   6   3          4   8  12  16  12   8   4
#      2   4   6   4   2          3   6   9  12   9   6   3
#      1   2   3   2   1          2   4   6   8   6   4   2
#                                 1   2   3   4   3   2   1
#
# respectively).  This is done to desensitize the derived parameters to 
# possible neighboring, brighter stars.


    xwt = np.zeros([nbox,nbox])
    wt = nhalf - np.abs(np.arange(nbox)-nhalf ) + 1
    for i in range(nbox): xwt[i,:] = wt
    ywt = np.transpose(xwt)
    sgx = np.sum(g*xwt,1)
    p = np.sum(wt)
    sgy = np.sum(g*ywt,0)
    sumgx = np.sum(wt*sgy)
    sumgy = np.sum(wt*sgx)
    sumgsqy = np.sum(wt*sgy*sgy)
    sumgsqx = np.sum(wt*sgx*sgx)
    vec = nhalf - np.arange(nbox) 
    dgdx = sgy*vec
    dgdy = sgx*vec
    sdgdxs = np.sum(wt*dgdx**2)
    sdgdx = np.sum(wt*dgdx) 
    sdgdys = np.sum(wt*dgdy**2)
    sdgdy = np.sum(wt*dgdy) 
    sgdgdx = np.sum(wt*sgy*dgdx)
    sgdgdy = np.sum(wt*sgx*dgdy)

 
    c = g*mask          #Convolution kernel now in c      
    sumc = np.sum(c)
    sumcsq = np.sum(c**2) - sumc**2/pixels
    sumc = sumc/pixels
    c[good[0],good[1]] = (c[good[0],good[1]] - sumc)/sumcsq
    c1 = np.exp(-.5*row2/sigsq)
    sumc1 = np.sum(c1)/nbox
    sumc1sq = np.sum(c1**2) - sumc1
    c1 = (c1-sumc1)/sumc1sq

    if verbose:
        print('RELATIVE ERROR computed from FWHM ' + str(np.sqrt(np.sum(c[good[0],good[1]]**2))))


        print('Beginning convolution of image')

    h = convolve(image,c)    #Convolve image with kernel "c"

    minh = np.min(h)
    h[:,0:nhalf] = minh ; h[:,n_x-nhalf:n_x] = minh
    h[0:nhalf,:] = minh ; h[n_y-nhalf:n_y-1,:] = minh

    if verbose:
        print('Finished convolution of image')

    mask[middle,middle] = 0	#From now on we exclude the central pixel
    pixels = pixels -1      #so the number of valid pixels is reduced by 1
    good = np.where(mask)      #"good" identifies position of valid pixels
#    xx= (good % nbox) - middle	#x and y coordinate of valid pixels 
#    yy = (good/nbox).astype(int) - middle    #relative to the center
    xx= good[1] - middle	#x and y coordinate of valid pixels 
    yy = good[0] - middle    #relative to the center

    offset = yy*n_x + xx

# SEARCH: 			    #Threshold dependent search begins here

    index = np.where( h >= hmin)  #Valid image pixels are greater than hmin
    nfound = len(index)

    if nfound == 0:          #Any maxima found?

        print('ERROR - No maxima exceed input threshold of ',hmin)
        return

    for i in range(pixels):

        hy = index[0]+yy[i]; hx = index[1]+xx[i]
        hgood = np.where((hy < n_y) & (hx < n_x) & (hy >= 0) & (hx >= 0))[0]

        stars = np.where (np.greater_equal(h[index[0][hgood],index[1][hgood]],h[hy[hgood],hx[hgood]]))

        nfound = len(stars)
        if nfound == 0:  #Do valid local maxima exist?
             print('ERROR - No maxima exceed input threshold of ',hmin)
             return

        index = np.array([index[0][hgood][stars],index[1][hgood][stars]])

 
    ix = index[1] # % n_x              #X index of local maxima
    iy = index[0] # /n_x                  #Y index of local maxima
    ngood = len(index[0])

    if verbose:
        print(str(ngood)+' local maxima located above threshold')

    nstar = 0       	#NSTAR counts all stars meeting selection criteria
    badround = 0 ; badsharp=0  ;  badcntrd=0

    x = np.zeros(ngood) ; y = np.zeros(ngood)

    flux = np.zeros(ngood) ; sharp = np.zeros(ngood) ; roundness = np.zeros(ngood)

    if doprint:	#Create output file?
        import time

        if doprint == 1: file = 'find.prt'
        else: file = doprint
        if verbose:
            print('Results will be written to a file ' + file)
        fout = open(file,'w')
        print >> fout, ' Program: FIND '+ time.asctime( time.localtime(time.time()) )
        print >> fout,' Threshold above background:',hmin
        print >> fout,' Approximate FWHM:',fwhm
        print >> fout,' Sharpness Limits: Low',sharplim[0], '  High',sharplim[1]
        print >> fout,' Roundness Limits: Low',roundlim[0],'  High',roundlim[1]
        print >> fout,' No of sources above threshold',ngood

    if verbose:
        print('     STAR      X      Y     FLUX     SHARP    ROUND')

    #  Loop over star positions# compute statistics

    for i in range(ngood):

        temp = image[iy[i]-nhalf:iy[i]+nhalf+1,ix[i]-nhalf:ix[i]+nhalf+1]
        d = h[iy[i],ix[i]]                  #"d" is actual pixel intensity        

        #  Compute Sharpness statistic

        sharp1 = (temp[middle,middle] - (np.sum(mask*temp))/pixels)/d
        if ( sharp1 < sharplim[0] ) or ( sharp1 > sharplim[1] ):
            badsharp = badsharp + 1
            continue #Does not meet sharpness criteria

        #   Compute Roundness statistic

        dx = np.sum( np.sum(temp,axis=0)*c1)   
        dy = np.sum( np.sum(temp,axis=1)*c1)
        if (dx <= 0) or (dy <= 0):
            badround = badround + 1
            continue     #Cannot compute roundness

        around = 2*(dx-dy) / ( dx + dy )    #Roundness statistic
        if ( around < roundlim[0] ) or ( around > roundlim[1] ):
            badround = badround + 1
            continue     # Does not meet roundness criteria

        #
        # Centroid computation:   The centroid computation was modified in Mar 2008 and
        # now differs from DAOPHOT which multiplies the correction dx by 1/(1+abs(dx)). 
        # The DAOPHOT method is more robust (e.g. two different sources will not merge)
        # especially in a package where the centroid will be subsequently be 
        # redetermined using PSF fitting.   However, it is less accurate, and introduces
        # biases in the centroid histogram.   The change here is the same made in the 
        # IRAF DAOFIND routine (see 
        # http://iraf.net/article.php?story=7211;query=daofind )
        #    

        sd = np.sum(temp*ywt,axis=0)

        sumgd = np.sum(wt*sgy*sd)
        sumd = np.sum(wt*sd)
        sddgdx = np.sum(wt*sd*dgdx)

        hx = (sumgd - sumgx*sumd/p) / (sumgsqy - sumgx**2/p)

        # HX is the height of the best-fitting marginal Gaussian.   If this is not
        # positive then the centroid does not make sense 

        if (hx <= 0):
            badcntrd = badcntrd + 1
            continue

        skylvl = (sumd - hx*sumgx)/p
        dx = (sgdgdx - (sddgdx-sdgdx*(hx*sumgx + skylvl*p)))/(hx*sdgdxs/sigsq)
        if np.abs(dx) >= nhalf:
            badcntrd = badcntrd + 1
            continue

        xcen = ix[i] + dx    #X centroid in original array

        # Find Y centroid                 
        
        sd = np.sum(temp*xwt,axis=1)
 
        sumgd = np.sum(wt*sgx*sd)
        sumd = np.sum(wt*sd)

        sddgdy = np.sum(wt*sd*dgdy)

        hy = (sumgd - sumgy*sumd/p) / (sumgsqx - sumgy**2/p)

        if (hy <= 0):
            badcntrd = badcntrd + 1
            continue

        skylvl = (sumd - hy*sumgy)/p
        dy = (sgdgdy - (sddgdy-sdgdy*(hy*sumgy + skylvl*p)))/(hy*sdgdys/sigsq)
        if np.abs(dy) >= nhalf:
            badcntrd = badcntrd + 1
            continue

      
        ycen = iy[i] +dy    #Y centroid in original array
 

        #  This star has met all selection criteria.  Print out and save results

        x[nstar] = xcen ; y[nstar] = ycen


        flux[nstar] = d ; sharp[nstar] = sharp1 ; roundness[nstar] = around
   
        nstar = nstar+1

# REJECT: 

    nstar = nstar-1		#NSTAR is now the index of last star found

    if doprint:
        print >> fout,' No. of sources rejected by SHARPNESS criteria',badsharp
        print >> fout,' No. of sources rejected by ROUNDNESS criteria',badround
        print >> fout,' No. of sources rejected by CENTROID  criteria',badcntrd
 
    if verbose:
        print(' No. of sources rejected by SHARPNESS criteria',badsharp)
        print(' No. of sources rejected by ROUNDNESS criteria',badround)
        print(' No. of sources rejected by CENTROID  criteria',badcntrd)

    if nstar < 0: return               #Any stars found?

    x=x[0:nstar+1]  ; y = y[0:nstar+1]

    flux= flux[0:nstar+1] ; sharp=sharp[0:nstar+1]  
    roundness = roundness[0:nstar+1]


    if doprint:
        print >> fout,'     STAR       X       Y     FLUX     SHARP    ROUND'
        for i in range(nstar+1):
            print >> fout,i+1, x[i], y[i], flux[i], sharp[i], roundness[i]

# FINISH:
    if doprint:
        fout.close()
    return(x,y,flux,sharp,roundness)
