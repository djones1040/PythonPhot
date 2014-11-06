#!/usr/bin/env python
# D. Jones - 2/13/14

import numpy as np

def measure_psf_simple(image, xcen, ycen,
                       boxsize=9, scale=1.2):
    """Measure object shape from second-order moments
    using circular Gaussian weight.  Converted from IDL
    to Python
    
    INPUT PARAMETERS:
         image    -   Observed image, arbitrary units (sky-subtracted; array, float)
         xcen     -   Input X coordinates (vector, float)
         ycen     -   Input Y coordinates (vector, float)
         boxsize  -   Linear size of extraction box (integer, scalar) (defaults to 9 pixels)
         scale    -   Scale of Gaussian weight (one-dimensional sigma, pixels) (float, scalar)
    
         Note: boxsize=9, scale=1.2 to 1.5 works well for WFC; probably
          boxsize=13, scale=2.5 is needed for HRC.
    
     RETURNS:
          moments -   Vector with weighted moments (float, array; returned by function)
    
          Note: moments is an nsource*10 array, with all elements set to zero
           for sources that are too close to the edge
    
     The 10 elements are: xcen(input), ycen(input), total, scale, m0,
                          mx, my, mxx, mxy, myy
    
     Possible enhancements:
     -> Allow for adaptive weight scale
     -> Input weight/flag image
     -> Return flag array
     -> Exclude stars too close to edge
     """

    ss = np.shape (image)
    nx = ss[1]
    ny = ss[0]

    if type(xcen) == np.float or type(xcen) == np.int: nsources = 1
    else: nsources = len(xcen)
    xmat = np.zeros ([boxsize, boxsize])
    ymat = np.zeros ([boxsize, boxsize])
    for i in range(boxsize): xmat[:,i] = i
    for j in range(boxsize): ymat[j,:] = j
    
    moments = np.zeros ([10,nsources])

    if nsources == 1: xcen,ycen = [xcen],[ycen]
    for i in range(nsources):
        if nsources > 1:
            xl = (xcen[i]-boxsize/2.+0.5).astype(int) ; xh = xl + boxsize - 1
            yl = (ycen[i]-boxsize/2.+0.5).astype(int) ; yh = yl + boxsize - 1
        else:
            xl = int((xcen[i]-boxsize/2.+0.5)) ; xh = xl + boxsize - 1
            yl = int((ycen[i]-boxsize/2.+0.5)) ; yh = yl + boxsize - 1
        if ((xl >= 0) and (xh <= nx-1) and (yl >= 0) and (yh <= ny-1)):
            sub = image[yl:yh+1,xl:xh+1]

            tsub = np.sum(sub)
            for iter in range(6):
                if (iter == 0):
                    wx = xcen[i]-xl ; wy = ycen[i]-yl
                else:
                    wx = mx ; wy = my

                dsq = (xmat-wx)**2 + (ymat-wy)**2
                wei = np.exp (- 0.5*dsq/scale**2) / 2. / np.pi / scale**2
                twei = np.sum(wei)
                wsub = wei*sub
                twsub = np.sum (wsub)
                m0 = np.sum (wsub) / twei
                mx = np.sum (xmat*wsub) / twsub
                my = np.sum (ymat*wsub) / twsub
                mxx = np.sum ((xmat-mx)**2*wsub) / twsub
                mxy = np.sum ((xmat-mx)*(ymat-my)*wsub) / twsub
                myy = np.sum ((ymat-my)**2*wsub) / twsub

            moments[:,i] = [xcen[i], ycen[i], tsub, scale, m0, mx+xl, my+yl, mxx, mxy, myy]

    return moments
