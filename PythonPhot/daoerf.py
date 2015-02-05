#!/usr/bin/env python
# D. Jones - 2/13/14

import numpy as np
from scipy.special import erf

sqrt,shape,zeros,exp = np.sqrt,np.shape,np.zeros,np.exp

def daoerf(x,y,a):
    """Calulates the intensity, and derivatives, of a 2-d Gaussian PSF
    (adapted for IDL from DAOPHOT, then translated from IDL to Python).

    Corrects for the finite size of a pixel by integrating the Gaussian
    over the size of the pixel.    Used in the IDL-DAOPHOT sequence.   

    This code is from the IDL Astronomy Users Library 
    and uses GAUSSINT (called cdf here) from 
    http://www.mpia-hd.mpg.de/homes/ianc/python/_modules/spec.html
    
    CALLING SEQUENCE:
      f,pder = daoerf.daoerf(x, y, a)

    INPUTS:
         x - input scalar, vector or array, giving X coordinate values
         y - input scalar, vector or array, giving Y coordinate values, must 
             have same number of elements as XIN.
         a - 5 element parameter array describing the Gaussian
             A[0] - peak intensity
             A[1] - X position of peak intensity (centroid)
             A[2] - Y position of peak intensity (centroid)
             A[3] - X sigma of the gaussian (=FWHM/2.345)         
             A[4] - Y sigma of gaussian
    
    OUTPUTS:
         f - array containing value of the function at each (XIN,YIN).
              The number of output elements in F and PDER is identical with
              the number of elements in X and Y

    OPTIONAL OUTPUTS:
         pder - 2 dimensional array of size (NPTS,5) giving the analytic
                 derivative at each value of F with respect to each parameter A.

    REVISION HISTORY:
         Written                           W. Landsman                October,   1987
         Converted to IDL V5.0             W. Landsman                September, 1997
         Converted from IDL to Python      D. Jones                   January,   2014
    """

    norm = 2.506628275 #norm = sqrt(2*!pi)

    if len(shape(x)) > 1:
        shapex,shapey = shape(x),shape(y)
        x = x.reshape(shapex[0]*shapex[1])
        y = y.reshape(shapey[0]*shapey[1])

    npts = len(x) 

    u2 = (x[:] - a[1] + 0.5)/a[3] ; u1 = (x[:] - a[1] - 0.5)/a[3]
    v2 = (y[:] - a[2] + 0.5)/a[4] ; v1 = (y[:] - a[2] - 0.5)/a[4]
    fx = norm*a[3]*(cdf(u2) - cdf(u1))
    fy = norm*a[4]*(cdf(v2) - cdf(v1))
    f =  a[0]*fx*fy
    #Need partial derivatives ?

    pder = zeros([5,npts])
    pder[0,:] = fx*fy
    uplus = exp(-0.5*u2**2.) ; uminus = exp(-0.5*u1**2)
    pder[1,:] = a[0]*fy*(-uplus + uminus)
    vplus = exp(-0.5*v2**2.) ; vminus = exp(-0.5*v1**2)
    pder[2,:] = a[0]*fx*(-vplus + vminus)
    pder[3,:] = a[0]*fy*(fx/a[3] + u1*uminus - u2*uplus)
    pder[4,:] = a[0]*fx*(fy/a[4] + v1*vminus - v2*vplus)

    return(f,pder)

def cdf(x):
    """ 
    PURPOSE:
         Compute the integral from -inf to x of the normalized Gaussian

    INPUTS:
         x : scalar upper limit of integration

    NOTES:
         Designed to copy the IDL function of the same name.
    """
    # 2011-10-07 15:41 IJMC: Created

    scalefactor = 1./sqrt(2)
    return 0.5 + 0.5 * erf(x * scalefactor)
