#!/usr/bin/env python
# D. Jones - 5/22/14
"""This procedure was adapted for Python from the IDL Astronomy Users Library."""

import numpy as np
from numpy import where,append,array

def group( x, y, rcrit):
    """Assign stars with non-overlapping PSF profiles into distinct groups
    (adapted for IDL from DAOPHOT, then translated from IDL to Python).

    Part of the IDL-DAOPHOT sequence

    CALLING SEQUENCE:
         ngroup = group.group(X, Y, rcrit)

    INPUTS:
          x    -  vector, giving X coordinates of a set of stars.
          y    -  vector, giving Y coordinates of a set of stars.
                   If X and Y are input as integers, then they will be converted to 
                   floating point
          rcrit - scalar, giving minimum distance between stars of two
                   distinct groups.  Stars less than this distance from
                   each other are always in the same group.    Stetson suggests
                   setting the critical distance equal to the PSF radius +
                   the Fitting radius.

    RETURNS:
          ngroup - integer vector, same number of elements as X and Y,
                    giving a group number for each star position.  Group
                    numbering begins with 0.

    METHOD:
          Each position is initially given a unique group number.  The distance
          of each star is computed against every other star.   Those distances
          less than RCRIT are assigned the minimum group number of the set.   A
          check is then made to see if any groups have merged together.
    
    REVISION HISTORY:
          Written                                             W. Landsman  STX                  April,     1988
          Major revision to properly merge groups together    W. Landsman                       September, 1991
          Work for more than 32767 points                     W. Landsman                       March,     1997
          Converted to IDL V5.0                               W. Landsman                       September, 1997
          Avoid overflow if X and Y are integers              W. Landsman                       February,  1999   
    """

    
    rcrit2 = rcrit**2.                            #Don't bother taking square roots
    npts = min( [len(x), len(y)] )    #Number of stars

    if npts < 2:
        raise RuntimeError('ERROR - Input position X,Y vectors must contain at least 2 points')

    x = 1.0*x  ;  y = 1.0*y   #Make sure at least floating point
    ngroup =  np.arange(npts)   #Initially each star in a separate group
    
    #  Whenever the positions between two stars are less than the critical
    #  distance, assign both stars the minimum group id.   The tricky part
    #  is to recognize when distinct groups have merged together.

    for i in range(npts-1):
        dis2 = (x[i] - x[i+1:])**2. + (y[i] - y[i+1:])**2.
        good =  where( dis2 <= rcrit2)[0]
        ngood = len(good)

        if ngood > 0:             #Any stars within critical radius?

            good = append(array(i),good+i+1)

            groupval = ngroup[good]
            mingroup = np.min( groupval )
            if ( mingroup < i ):      #Any groups merge?
                groupval = groupval[ where( groupval < i) ]
                nval = len(groupval)
                if nval > 1:
                    groupval = np.unique(groupval)
                nval = len(groupval)

                if nval >= 2: 
                    for j in range(1,nval):
                        redo = where ( ngroup == groupval[j] )[0]
                        ndo = len(redo)
                        if ndo > 0: ngroup[redo] = mingroup

            ngroup[good] = mingroup

#
# Star are now placed in distinct groups, but they are not ordered
# consecutively.  Remove gaps in group ordering
#
    if np.max(ngroup) == 0: return(np.zeros(len(x)))               #All stars in one group ?

    ghist = np.histogram(ngroup,bins=max(ngroup)+1)[0]
    gmax = np.max(ghist)
    val = where(ghist >= 1)[0]
    ngood = len(val)

    if ( ngood > 0 ):
        for i in range(ngood): 
            ngroup[ where( ngroup == val[i] ) ] = i

    print('Number of Groups: %s'%ngood)
    print('Largest group size %s stars'%gmax)

    return(ngroup)

