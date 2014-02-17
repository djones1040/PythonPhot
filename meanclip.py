#!/usr.bin/env python
#D. Jones - 1/13/14
"""This code is from the IDL Astronomy Users Library"""
import numpy as np

def meanclip(image, 
             clipsig=3, maxiter=5,
             converge_num=0.02, verbose=False):
    """;+
    ; NAME:
    ;       MEANCLIP
    ;
    ; PURPOSE:
    ;       Computes an iteratively sigma-clipped mean on a data set
    ; EXPLANATION:
    ;       Clipping is done about median, but mean is returned.
    ;       Called by SKYADJ_CUBE
    ;
    ; CATEGORY:
    ;       Statistics
    ;
    ; CALLING SEQUENCE:
    ;       MEANCLIP, Data, Mean, [ Sigma, SUBS =
    ;              CLIPSIG=, MAXITER=, CONVERGE_NUM=, /VERBOSE, /DOUBLE ]
    ;
    ; INPUT POSITIONAL PARAMETERS:
    ;       Data:     Input data, any numeric array
    ;       
    ; OUTPUT POSITIONAL PARAMETERS:
    ;       Mean:     N-sigma clipped mean.
    ;       Sigma:    Standard deviation of remaining pixels.
    ;
    ; INPUT KEYWORD PARAMETERS:
    ;       CLIPSIG:  Number of sigma at which to clip.  Default=3
    ;       MAXITER:  Ceiling on number of clipping iterations.  Default=5
    ;       CONVERGE_NUM:  If the proportion of rejected pixels is less
    ;           than this fraction, the iterations stop.  Default=0.02, i.e.,
    ;           iteration stops if fewer than 2% of pixels excluded.
    ;       /VERBOSE:  Set this flag to get messages.
    ;       /DOUBLE - if set then perform all computations in double precision.
    ;                 Otherwise double precision is used only if the input
    ;                 data is double
    ; OUTPUT KEYWORD PARAMETER:
    ;       SUBS:     Subscript array for pixels finally used.
    ;
    ;
    ; MODIFICATION HISTORY:
    ;       Written by:     RSH, RITSS, 21 Oct 98
    ;       20 Jan 99 - Added SUBS, fixed misplaced paren on float call, 
    ;                   improved doc.  RSH
    ;       Nov 2005   Added /DOUBLE keyword, check if all pixels are removed  
    ;                  by clipping W. Landsman 
    ;-"""

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

    return(mean,sigma)
