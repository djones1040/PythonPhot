#!/usr/bin/env python
# D. Jones - 2/13/14

import numpy as np

def iterstat(d,startMedian=False):
    clip=3.0 #sigma                                                                                                         
    img=d.astype('float64')
    if startMedian:
        md=np.median(img)
    else:
        md=np.mean(img)
    n = float(len(img))
    std = np.sqrt(np.sum((img-md)**2.)/(n-1))

    for ii in range(6):
        gd=np.where((img < md+clip*std) &
                    (img > md-clip*std))

        md=np.mean(img[gd])
        n = float(len(gd[0]))
	std = np.sqrt(np.sum((img[gd]-md)**2.)/(n-1.))

    return(md,std)
