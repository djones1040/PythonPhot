#!/usr/bin/env python
#D. Jones - 1/13/14
"""The IDL source code for this routine is at
http://spectro.princeton.edu/idlutils_doc.html#DJS_ANGLE_MATCH

Given two lists of coordinates on a sphere, find matches within an
angular distance.  For each entry in list A, find all the entries
in list B that lie within an angular distance dtheta.
Optionally output up to mmax of these matches per entry, giving
the index number of the matches in mindx, and the angular distance
in mdist.

If the lists A and B are different, then the total number of pairs
is given by total(mcount).
If the lists A and B are the same, then the total number of unique
pairs is given by (total(mcount) - N_elements(raA)) / 2.

This function loops over the objects in each list (sort of), so it's
not very fast.

CALLING SEQUENCE:
     import djs_angle_match
     ntot,mindx,mcount = djs_angle_match.djs_angle_match( raA, decA, raB, decB, dtheta=dtheta,
                                                          mmax=mmax, units=units )
     ntot,mindx,mcount,mdist = djs_angle_match.djs_angle_match( raA, decA, raB, decB, dtheta=dtheta,
                                                                mmax=mmax, units=units, returnmdist=True )

 INPUTS:
   raA:        RA of first point(s) in radians/degrees/hours
   decA:       DEC of first point(s) in radians/degrees
   raB:        RA of second point(s) in radians/degrees/hours
   decB:       DEC of second point(s) in radians/degrees
   dtheta:     Maximum angular distance for points to be considered matches

 OPTIONAL INPUTS:
   mmax:       Maximum number of matches per point.  Default to 1.
   units:      Set to
                  degrees - All angles in degrees
                  hrdeg - RA angles in hours, DEC angles and output in degrees
                  radians - All angles in radians
               Default to "degrees".

 RETURNS:
   ntot:       Total number of points A with one or more matches in B
   mcount:     For each A, number of matches in B.  Vector of length A.
   mindx:      For each A, indices of matches in B, sorted by their distance.
               If mmax > 1, this array is of dimensions (mmax, A).
               For each A, only the values (0:mcount-1,A) are relevant.
               If mmax = 1, then the return value is a vector.
               Any unused array elements are set to -1.
   mdist:      For each A, distance to matches in B, sorted by their distance.
               If mmax > 1, this array is of dimensions (mmax, A).
               For each A, only the values (0:mcount-1,A) are relevant.
               If mmax = 1, then the return value is a vector.
               Any unused array elements are set to -1.

 COMMENTS:
   By specifying only one set of coordinates (raA, decA), matches are found
   within that list, but avoiding duplicate matches (i.e., matching 1 to 2
   and then 2 to 1) and avoiding matching an object with itself (i.e.,
   matching 1 to 1).  If you wish to include self-matches and duplicates,
   then call with raB=raA and decB=decA.

 PROCEDURES CALLED:
   djs_diff_angle()

 INTERNAL PROCEDURES:
   djs_angle_1match()
   djs_angle_2match()

 REVISION HISTORY:
   27-May-1997  Written by David Schlegel, Durham
   24-Feb-1999  Converted to IDL 5 (DJS)
   05-Mar-1999  Made the internal routines for more efficient matching
                within the same coordinate list without duplicates, e.g.
                by only specifying raA, decA and not raB, decB.
   Jan. 2014    Converted from IDL to Python by D. Jones

"""

import numpy as np

def djs_angle_2match(raA, decA, 
                     raB, decB, 
                     dtheta,
                     mmax=1, 
                     units='degrees',
                     returnmdist=False):

   if units == "hrdeg":
      convRA = np.pi / 12.0
      convDEC = np.pi / 180.0
      allRA = 24.
      highDEC = 90.
   elif units == "radians":
      convRA = 1.
      convDEC = 1.
      allRA = 2. * np.pi
      highDEC = 0.5 * np.pi
   else:
      convRA = np.pi / 180.
      convDEC = np.pi / 180.
      allRA = 360.
      highDEC = 90.

   # Allocate arrays
   numA = len(raA)
   numB = len(raB)
   mcount = np.zeros(numA,dtype='int')
   mindx = np.zeros([numA, mmax],dtype='int') - 1 # Set equal to -1 for no matches
   mdist = np.zeros([numA, mmax]) - 1 # Set equal to -1 for no matches
   tempindx = np.zeros(numB,dtype='int')
   tempdist = np.zeros(numB)

   # Sort points by DEC
   # print, 'Sorting list A'
   indxA = np.argsort(decA)
   # print, 'Sorting list B'
   indxB = np.argsort(decB)

   iStart = 0
   iEnd = 0
   #print, 'Looking for duplicates'

   for iA in range(numA):

      # Limit search to declination range within "dtheta"
      while ( decB[indxB[iStart]] < decA[indxA[iA]] - dtheta \
                 and iStart < numB-1 ): iStart = iStart + 1
      while ( decB[indxB[iEnd]] < decA[indxA[iA]] + dtheta \
                 and iEnd < numB-1 ): iEnd = iEnd + 1

      nmatch = 0

      maxdec = np.abs( decA[indxA[iA]] ) + dtheta
      if maxdec > 90.: maxdec = 90.

      if (maxdec >= highDEC): dalpha = allRA + 1
      else: dalpha = (convRA/convDEC) * dtheta / np.cos(maxdec*convDEC)

      iBvec = iStart + np.arange(iEnd-iStart+1).astype(int)

      # Select objects whose RA falls in the range of "dtheta" about point A
      ii = np.where( (np.abs( raA[indxA[iA]] - raB[indxB[iBvec]] ) < dalpha) |
                     (np.abs( raA[indxA[iA]] - raB[indxB[iBvec]] + allRA ) < dalpha) |
                     (np.abs( raA[indxA[iA]] - raB[indxB[iBvec]] - allRA ) < dalpha))[0]
      cti = len(ii)
      
      if (cti > 0):
         adist = djs_diff_angle( raA[indxA[iA]], decA[indxA[iA]], \
                                    raB[indxB[iBvec[ii]]], decB[indxB[iBvec[ii]]], units=units )
         jj = np.where(adist < dtheta)[0]
         ctj = len(jj)
         # The following are matches in distances computed by djs_diff_angle.
         if (ctj > 0):
            tempindx[nmatch:nmatch+ctj] = iBvec[ii[jj]]
            tempdist[nmatch:nmatch+ctj] = adist[jj]
            nmatch = nmatch + ctj

      mcount[indxA[iA]] = min ( [mmax, nmatch] )
      if (nmatch > 0):
         # Sort the matches, and keep only the mmax closest ones
         tempsort = np.argsort ( tempdist[0:nmatch] )
         mindx[indxA[iA],0:mcount[indxA[iA]]] = \
             indxB[ tempindx[ tempsort[0:mcount[indxA[iA]]] ] ]
         mdist[indxA[iA],0:mcount[indxA[iA]]] = \
             tempdist[ tempsort[0:mcount[indxA[iA]]] ]

   if (mmax == 1):
      mindx = np.transpose(mindx)
      mdist = np.transpose(mdist)

   junk = np.where(mcount > 0)[0]
   ntot = len(junk)

   if returnmdist:
      return(ntot,mindx,mcount,mdist)
   else:
      return(ntot,mindx,mcount)

#------------------------------------------------------------------------------
def djs_angle_match(raA, decA, raB, decB, 
                    dtheta, 
                    mmax = 1,
                    units = 'degrees',
                    returnmdist = False):
   print('djs!')
   # Call with different RA,DEC
   if returnmdist:
      ntot,mindx,mcount,mdist = djs_angle_2match( raA, decA, raB, decB, dtheta,
                                                  mmax=mmax, units=units,returnmdist=returnmdist)
      return(ntot,mindx,mcount,mdist)
   else:
      ntot,mindx,mcount = djs_angle_2match( raA, decA, raB, decB, dtheta,
                                            mmax=mmax, units=units,returnmdist=returnmdist)
      return(ntot,mindx,mcount)

#------------------------------------------------------------------------------
def djs_diff_angle(ra1, dec1, 
                   ra2, dec2, 
                   units="degrees"):
   """;+
   ; NAME:
   ;   djs_diff_angle
   ;
   ; PURPOSE:
   ;   Compute the angular distance between two points on a sphere.
   ;
   ; CALLING SEQUENCE:
   ;   adist = djs_diff_angle( ra, dec, ra0, dec0, [ units=units ] )
   ;
   ; INPUTS:
   ;   ra1:        RA of first point(s) in radians/degrees/hours
   ;   dec1:       DEC of first point(s) in radians/degrees
   ;   ra2:        RA of second point(s) in radians/degrees/hours
   ;   dec2:       DEC of second point(s) in radians/degrees
   ;
   ; OPTIONAL INPUTS:
   ;   units:      Set to
   ;                  degrees - All angles in degrees
   ;                  hrdeg - RA angles in hours, DEC angles and output in degrees
   ;                  radians - All angles in radians
   ;               Default to "degrees".
   ;
   ; OUTPUTS:
   ;   adist:      Angular distance(s) in radians if UNITS is set to 'radians',
   ;               or in degrees otherwise
   ;
   ; COMMENTS:
   ;   Note that either (ra1,dec1) or (rap,decp) must be scalars or 1-element
   ;   arrays, or all must be arrays of the same length.
   ;
   ; PROCEDURES CALLED:
   ;
   ; REVISION HISTORY:
   ;   14-May-1997  Written by D. Schlegel, Durham
   ;-
   ;------------------------------------------------------------------------------
   """
   DPIBY2 = 0.5 * np.pi

   try: num1 = len(ra1)
   except: num1 = 1
   try: num2 = len(ra2)
   except: num2 = 1

   if units =='hrdeg':
      convRA = np.pi / 12.0
      convDEC = np.pi / 180.0
   elif units == 'radians':
      convRA = 1.0
      convDEC = 1.0
   elif units == 'degrees':
      convRA = np.pi / 180.0
      convDEC = np.pi / 180.0
   else : print('Unknown UNITS='+string(units))

   # The following allows the inputs to be 1-element arrays rather than
   # scalars, by recasting those 1-element arrays as scalars.
   if (num1 == 1):
      theta1 = dec1 * convDEC + DPIBY2
      theta2 = dec2 * convDEC + DPIBY2
      cosgamma= np.sin(theta1) * np.sin(theta2) \
          * np.cos((ra1 - ra2) * convRA)  + np.cos(theta1) * np.cos(theta2)
   elif (num2 == 1):
      theta1 = dec1 * convDEC + DPIBY2
      theta2 = dec2 * convDEC + DPIBY2
      cosgamma= np.sin(theta1) * np.sin(theta2) \
          * np.cos((ra1 - ra2) * convRA)  + np.cos(theta1) * np.cos(theta2)
   else:
      theta1 = dec1 * convDEC + DPIBY2
      theta2 = dec2 * convDEC + DPIBY2
      cosgamma= np.sin(theta1) * np.sin(theta2) \
          * np.cos((ra1 - ra2) * convRA)  + np.cos(theta1) * np.cos(theta2)

   out = np.arccos(cosgamma) / convDEC
   row = np.where(cosgamma > 1)[0]
   if len(row):
      out[row] = np.arccos( 1.0) / convDEC

   return(out)

#------------------------------------------------------------------------------
