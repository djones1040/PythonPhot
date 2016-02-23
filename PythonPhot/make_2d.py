#!/usr/bin/env python
# D. Jones - 2/13/14

import numpy as np
from . import rebin

def make_2d(x,y):
    """Change from 1-d indexing to 2-d indexing
    (translated from IDL to Python).

    Convert an N element X vector, and an M element Y vector, into
    N x M arrays giving all possible combination of X and Y pairs.
    Useful for obtaining the X and Y positions of each element of
    a regular grid.

    CALLING SEQUENCE:
       xx,yy = make_2d.make_2d(x,y)

    INPUTS:
         x - N element vector of X positions
         y - M element vector of Y positions

    RETURNS:
         xx - N x M element array giving the X position at each pixel
         yy - N x M element array giving the Y position of each pixel
               If only 2 parameters are supplied then X and Y will be
               updated to contain the output arrays

    EXAMPLE:
         To obtain the X and Y position of each element of a 30 x 15 array

         import make_2d
         x = numpy.arange(30)  ;  y = numpy.arange(15)     
         xx,yy = make_2d.make_2d( x, y ) 

    REVISION HISTORY:
         Written                     Wayne Landsman,ST Systems Co.    May,            1988
         Added /NOZERO keyword       W. Landsman                      March,          1991
         Converted to IDL V5.0       W. Landsman                      September,      1997
         Improved speed              P. Broos                         July,           2000
         Converted to Python         D. Jones                         January,        2014
"""


    ny = len(y)
    nx = len(x)
    xx = x.reshape(1,nx)
    yy = y.reshape(ny,1)

    xx = rebin.rebin(xx, [ny, nx])
    yy = rebin.rebin(yy, [ny, nx])

    return(xx,yy)
