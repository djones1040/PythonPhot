#!/usr.bin/env python
#D. Jones - 1/13/14
"""This code is from the IDL Astronomy Users Library"""
import numpy as np

def meanclip(image, 
             clipsig=3, maxiter=5,
             converge_num=0.02, verbose=False,
             returnSubs=False):
    """Computes an iteratively sigma-clipped mean on a data set
    Clipping is done about median, but mean is returned.
    Converted from IDL to Python.
    
    CALLING SEQUENCE:
        mean,sigma = meanclip( data, clipsig=, maxiter=,
                               converge_num=, verbose=,
                               returnSubs=False)
        mean,sigma,subs = meanclip( data, clipsig=, maxiter=,
                                    converge_num=, verbose=,
                                    returnSubs=True)

    INPUT PARAMETERS:
         data           -  Input data, any numeric array

    OPTIONAL INPUT PARAMETERS:
         clipsig        -  Number of sigma at which to clip.  Default=3
         maxiter        -  Ceiling on number of clipping iterations.  Default=5
         converge_num   -  If the proportion of rejected pixels is less
                            than this fraction, the iterations stop.  Default=0.02, i.e.,
                            iteration stops if fewer than 2% of pixels excluded.
         verbose        -  Set this flag to get messages.
         returnSubs     -  if True, return subscript array for pixels finally used
           
    RETURNS:
         mean           -  N-sigma clipped mean.
         sigma          -  Standard deviation of remaining pixels.
    
    MODIFICATION HISTORY:
         Written by:       RSH, RITSS, 21 Oct 98
         20 Jan 99   -     Added SUBS, fixed misplaced paren on float call, 
                            improved doc.  RSH
         Nov 2005    -     Added /DOUBLE keyword, check if all pixels are removed  
                            by clipping W. Landsman 
         Jan. 2014   -     Converted from IDL to Python by D. Jones
    """

    prf = 'MEANCLIP:  '
    
    #image = image.reshape(np.shape(image)[0]*np.shape(image)[1])
    subs = np.where(np.isfinite(image))[0]
    ct = len(subs)
    iter=0

    for i in range(maxiter):
        skpix = image[subs]
        iter = iter + 1
        lastct = ct
        medval = np.median(skpix)
        mom = [np.mean(skpix),np.std(skpix)]
        sig = mom[1]
        wsm = np.where(np.abs(skpix-medval) < clipsig*sig)[0]
        ct = len(wsm)
        if ct > 0: subs = subs[wsm]         
        if (float(np.abs(ct-lastct))/lastct <= converge_num) or \
                (iter > maxiter) or (ct == 0):
            break
    #mom = moment(image[subs],double=double,max=2)
    mean = np.mean(image[subs])
    sigma = np.std(image[subs])

    if verbose:
        print(prf+strn(clipsig)+'-sigma clipped mean')
        print(prf+'Mean computed in ',iter,' iterations')
        print(prf+'Mean = ',mean,',  sigma = ',sigma)

    if not returnSubs:
        return(mean,sigma)
    else:
        return(mean,sigma,subs)
