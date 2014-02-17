#!/usr/bin/env python
# D. Jones - 2/13/14

import numpy as np
import rebin

def make_2d(xx,yy):

    ny = len(yy)
    nx = len(xx)
    xx = xx.reshape(1,nx)
    yy = yy.reshape(ny,1)

    xx = rebin.rebin(xx, [ny, nx])
    yy = rebin.rebin(yy, [ny, nx])

    return(xx,yy)
