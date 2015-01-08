#!/usr/bin/env python
# D. Jones - 2/13/14
"""This code is from the IDL Astronomy Users Library"""
import numpy as np
where,abs,sqrt,greater,less_equal,less,greater_equal = \
    np.where,np.abs,np.sqrt,np.greater,np.less_equal,np.less,np.greater_equal

def Pixwt(xc, yc, r, x, y):
    """Circle-rectangle overlap area computation.

    Compute the fraction of a unit pixel that is interior to a circle.
    The circle has a radius r and is centered at (xc, yc).  The center of
    the unit pixel (length of sides = 1) is at (x, y).
    
    area = pixwt.Pixwt( xc, yc, r, x, y )

    INPUT PARAMETERS:
         xc, yc - Center of the circle, numeric scalars
    	 r      - Radius of the circle, numeric scalars
    	 x, y   - Center of the unit pixel, numeric scalar or vector

    RETURNS:
         Function value: Computed overlap area.

    EXAMPLE:
         What is the area of overlap of a circle with radius 3.44 units centered
         on the point 3.23, 4.22 with the pixel centered at [5,7]
    
         import pixwt
         pixwt.Pixwt(3.23,4.22,3.44,5,7)  ==>  0.6502

    PROCEDURE:
         Divides the circle and rectangle into a series of sectors and
    	 triangles.  Determines which of nine possible cases for the
    	 overlap applies and sums the areas of the corresponding sectors
    	 and triangles.  Called by aper.pro
    
    NOTES:
         If improved speed is needed then a C version of this routines, with
         notes on how to linkimage it to IDL is available at   
         ftp://ftp.lowell.edu/pub/buie/idl/custom/
    
    MODIFICATION HISTORY:
         Ported by Doug Loucks, Lowell Observatory, 1992 Sep, from the
         routine pixwt.c, by Marc Buie.
         converted to Python by D. Jones
    """
    return Intarea( xc, yc, r, x-0.5, x+0.5, y-0.5, y+0.5 )

def Arc( x, y0, y1, r):
    """; Function Arc( x, y0, y1, r )
    ;
    ; Compute the area within an arc of a circle.  The arc is defined by
    ; the two points (x,y0) and (x,y1) in the following manner:  The circle
    ; is of radius r and is positioned at the origin.  The origin and each
    ; individual point define a line which intersects the circle at some
    ; point.  The angle between these two points on the circle measured
    ; from y0 to y1 defines the sides of a wedge of the circle.  The area
    ; returned is the area of this wedge.  If the area is traversed clockwise
    ; then the area is negative, otherwise it is positive.
    ; ---------------------------------------------------------------------------"""
    return 0.5 * r*r * ( np.arctan( (y1).astype(float)/(x).astype(float) ) - np.arctan( (y0).astype(float)/(x).astype(float) ) )

def Chord( x, y0, y1):
    """; ---------------------------------------------------------------------------
    ; Function Chord( x, y0, y1 )
    ;
    ; Compute the area of a triangle defined by the origin and two points,
    ; (x,y0) and (x,y1).  This is a signed area.  If y1 > y0 then the area
    ; will be positive, otherwise it will be negative.
    ; ---------------------------------------------------------------------------"""
    return 0.5 * x * ( y1 - y0 )

def Oneside( x, y0, y1, r):
    """; ---------------------------------------------------------------------------
    ; Function Oneside( x, y0, y1, r )
    ;
    ; Compute the area of intersection between a triangle and a circle.
    ; The circle is centered at the origin and has a radius of r.  The
    ; triangle has verticies at the origin and at (x,y0) and (x,y1).
    ; This is a signed area.  The path is traversed from y0 to y1.  If
    ; this path takes you clockwise the area will be negative.
    ; ---------------------------------------------------------------------------"""

    true = 1
    size_x  = np.shape( x )
    if not size_x: size_x = [0]

    if size_x[ 0 ] == 0:
      if x == 0: return x
      elif abs( x ) >= r: return Arc( x, y0, y1, r )
      yh = sqrt( r*r - x*x )
      if ( y0 <= -yh ):
          if ( y1 <= -yh ) : return Arc( x, y0, y1, r )
          elif ( y1 <=  yh ) : return Arc( x, y0, -yh, r ) \
                  + Chord( x, -yh, y1 )
          else          : return Arc( x, y0, -yh, r ) \
                  + Chord( x, -yh, yh ) + Arc( x, yh, y1, r )
          
      elif ( y0 <  yh ):
          if ( y1 <= -yh ) : return Chord( x, y0, -yh ) \
                  + Arc( x, -yh, y1, r )
          elif ( y1 <=  yh ) : return Chord( x, y0, y1 )
          else : return Chord( x, y0, yh ) + Arc( x, yh, y1, r )

      else          :
          if ( y1 <= -yh ) : return Arc( x, y0, yh, r ) \
                               + Chord( x, yh, -yh ) + Arc( x, -yh, y1, r )
          elif ( y1 <=  yh ) : return Arc( x, y0, yh, r ) + Chord( x, yh, y1 )
          else          : return Arc( x, y0, y1, r )

    else :
        ans2 = x
        t0 = where( x == 0)[0]
        count = len(t0)
        if count == len( x ): return ans2

        ans = x * 0
        yh = x * 0
        to = where( abs( x ) >= r)[0]
        tocount = len(to)
        ti = where( abs( x ) < r)[0]
        ticount = len(ti)
        if tocount != 0: ans[ to ] = Arc( x[to], y0[to], y1[to], r )
        if ticount == 0: return ans
        
        yh[ ti ] = sqrt( r*r - x[ti]*x[ti] )
        
        t1 = where( np.less_equal(y0[ti],-yh[ti]) )[0]
        count = len(t1)
        if count != 0:
            i = ti[ t1 ]

            t2 = where( np.less_equal(y1[i],-yh[i]))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] =  Arc( x[j], y0[j], y1[j], r )

            t2 = where( ( greater(y1[i],-yh[i]) ) &
                        ( less_equal(y1[i],yh[i]) ))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Arc( x[j], y0[j], -yh[j], r ) \
                    + Chord( x[j], -yh[j], y1[j] )

            t2 = where( greater(y1[i], yh[i]) )[0]
            count = len(t2)

            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Arc( x[j], y0[j], -yh[j], r ) \
                    + Chord( x[j], -yh[j], yh[j] ) \
                    + Arc( x[j], yh[j], y1[j], r )
        
        t1 = where( ( greater(y0[ti],-yh[ti]) ) & 
                    ( less(y0[ti],yh[ti]) ))[0] 
        count = len(t1)
        if count != 0:
            i = ti[ t1 ]

            t2 = where( np.less_equal(y1[i],-yh[i]))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Chord( x[j], y0[j], -yh[j] ) \
                    + Arc( x[j], -yh[j], y1[j], r )
         

            t2 = where( ( greater(y1[i], -yh[i]) ) & 
                        ( less_equal(y1[i], yh[i]) ))[0]
            count = len(t2)

            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Chord( x[j], y0[j], y1[j] )

            t2 = where( greater(y1[i], yh[i]))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Chord( x[j], y0[j], yh[j] ) \
                    + Arc( x[j], yh[j], y1[j], r )

        t1 = where( greater_equal(y0[ti], yh[ti]))[0] 
        count = len(t1)
        if count != 0:
            i = ti[ t1 ]

            t2 = where ( np.less_equal(y1[i], -yh[i]))[0] 
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Arc( x[j], y0[j], yh[j], r ) \
                    + Chord( x[j], yh[j], -yh[j] ) \
                    + Arc( x[j], -yh[j], y1[j], r )

            t2 = where( ( greater(y1[i], -yh[i]) ) & 
                        ( less_equal(y1[i], yh[i]) ))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Arc( x[j], y0[j], yh[j], r ) \
                    + Chord( x[j], yh[j], y1[j] )

            t2 = where( greater(y1[i], yh[i]))[0]
            count = len(t2)
            if count != 0:
                j = ti[ t1[ t2 ] ]
                ans[j] = Arc( x[j], y0[j], y1[j], r )

        return ans

def Intarea( xc, yc, r, x0, x1, y0, y1):
    """; ---------------------------------------------------------------------------
    ; Function Intarea( xc, yc, r, x0, x1, y0, y1 )
    ;
    ; Compute the area of overlap of a circle and a rectangle.
    ;    xc, yc  :  Center of the circle.
    ;    r       :  Radius of the circle.
    ;    x0, y0  :  Corner of the rectangle.
    ;    x1, y1  :  Opposite corner of the rectangle.
    ; ---------------------------------------------------------------------------"""

#
# Shift the objects so that the circle is at the origin.
#
    x0 = x0 - xc
    y0 = y0 - yc
    x1 = x1 - xc
    y1 = y1 - yc

    return Oneside( x1, y0, y1, r ) + Oneside( y1, -x1, -x0, r ) +\
        Oneside( -x0, -y1, -y0, r ) + Oneside( -y0, x0, x1, r )
