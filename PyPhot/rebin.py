#!/usr/bin/env python
# D. Jones - 2/13/14

import numpy as np

def rebin(a, new_shape):

    M, N = a.shape
    m, n = new_shape
    if m<M:
        return a.reshape((m,M/m,n,N/n)).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)
