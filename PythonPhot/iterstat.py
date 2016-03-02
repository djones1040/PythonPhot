#!/usr/bin/env python
# D. Jones - 2/13/14

import numpy as np

def iterstat(d,startMedian=False,sigmaclip=3.0,
             iter=6):
    """Get the sigma-clipped mean of 
    a distribution, d.

    Usage: mean,stdev = iterstat.iterstat

    Input:
         d:           the data
    Optional Inputs:
         sigmaclip:   number of standard deviations to clip
         startMedian: if True, begin with the median of the distribution
         iter:        number of iterations
    """

    clip=sigmaclip
    img=d.astype('float64')
    if startMedian:
        md=np.median(img)
    else:
        md=np.mean(img)
    n = float(len(img))
    std = np.sqrt(np.sum((img-md)**2.)/(n-1))

    for ii in range(iter):
        gd=np.where((img < md+clip*std) &
                    (img > md-clip*std))

        md=np.mean(img[gd])
        n = float(len(gd[0]))
        std = np.sqrt(np.sum((img[gd]-md)**2.)/(n-1.))

    return(md,std)
