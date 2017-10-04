#!/usr/bin/env python
# D. Jones - 1/10/14
"""This code is from the IDL Astronomy Users Library with
modifications from Dan Scolnic.
(adapted for IDL from DAOPHOT, then translated from IDL to Python).

Subroutine of GETPSF to perform a one-star least-squares fit, 
part of the DAOPHOT PSF photometry sequence.  This version requires
input noise and mask images.

CALLING SEQUENCE:
     from PythonPhot import pkfit_noise as pkfit
     pk = pkfit.pkfit_class(f, gauss, psf,
                            ronois, phpadu, noise_image, mask_image )
     errmag,chi,sharp,niter,scale,xnew,ynew = pk.pkfit_noise(scale,x,y,sky,radius)

PKFIT CLASS INPUTS:
     f           - NX by NY array containing actual picture data.
     ronois      - readout noise per pixel, scalar
     phpadu      - photons per analog digital unit, scalar
     gauss       - vector containing the values of the five parameters defining
                    the analytic Gaussian which approximates the core of the PSF.
     psf         - an NPSF by NPSF look-up table containing corrections from
                    the Gaussian approximation of the PSF to the true PSF.
     noise_image - the noise image corresponding to f
     mask_image  - the mask image corresponding to f.  Masked pixels are not used.

PKFIT FUNCTION INPUTS:
     x, y    - the initial estimates of the centroid of the star relative
                to the corner (0,0) of the subarray.  Upon return, the
                final computed values of X and Y will be passed back to the
                calling routine.
     sky     - the local sky brightness value, as obtained from APER
     radius  - the fitting radius-- only pixels within RADIUS of the
                instantaneous estimate of the star's centroid will be
                included in the fit, scalar

OPTIONAL PKFIT FUNCTION INPUTS:
     xyout        - if True, return new x and y positions
     maxiter      - maximum iterations (default = 25)

INPUT-OUTPUT:
     scale  - the initial estimate of the brightness of the star,
               expressed as a fraction of the brightness of the PSF.
               Upon return, the final computed value of SCALE will be
               passed back to the calling routine.

RETURNS:
     errmag - the estimated standard error of the value of SCALE
               returned by this routine.
     chi    - the estimated goodness-of-fit statistic:  the ratio
               of the observed pixel-to-pixel mean absolute deviation from
               the profile fit, to the value expected on the basis of the
               noise as determined from Poisson statistics and the
               readout noise.
     sharp  - a goodness-of-fit statistic describing how much broader
               the actual profile of the object appears than the
               profile of the PSF.
     niter  - the number of iterations the solution required to achieve
               convergence.  If NITER = 25, the solution did not converge.
               If for some reason a singular matrix occurs during the least-
               squares solution, this will be flagged by setting NITER = -1.

EXAMPLE:
     from astropy.io import fits as pyfits
     from PythonPhot import pkfit_noise as pkfit

     # read in the FITS images
     image = pyfits.getdata(fits_filename)
     noiseim = pyfits.getdata(fits_noise_filename)
     maskim = pyfits.getdata(fits__mask_filename)

     # read in the PSF image
     psf = pyfits.getdata(psf_filename)
     hpsf = pyfits.getheader(psf_filename)
     gauss = [hpsf['GAUSS1'],hpsf['GAUSS2'],hpsf['GAUSS3'],hpsf['GAUSS4'],hpsf['GAUSS5']]

     # x and y points for PSF fitting
     xpos,ypos = np.array([1450,1400]),np.array([1550,1600])

     # run 'aper' on x,y coords to get sky values
     mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = \
              aper.aper(image,xpos,ypos,phpadu=1,apr=5,zeropoint=25,
              skyrad=[40,50],badpix=[-12000,60000],exact=True)

     # load the pkfit class
     pk = pkfit.pkfit_class(image,gauss,psf,1,1,noiseim,maskim)

     # do the PSF fitting
     for x,y,s in zip(xpos,ypos,sky):
          errmag,chi,sharp,niter,scale = \
              pk.pkfit_norecent_noise(1,x,y,s,5)
          flux = scale*10**(0.4*(25.-hpsf['PSFMAG']))
          dflux = errmag*10**(0.4*(25.-hpsf['PSFMAG']))
          print('PSF fit to coords %.2f,%.2f gives flux %s +/- %s'%(x,y,flux,dflux))
               
RESTRICTIONS:
     No parameter checking is performed

REVISON HISTORY:
     Adapted from the official DAO version of 1985 January 25
     Version 2.0                              W. Landsman STX             November,  1988
     Converted to IDL V5.0                    W. Landsman                 September, 1997
     Converted to Python                      D. Jones                    January,   2014
"""
import numpy as np
from numpy import sqrt
from scipy import linalg
from . import dao_value

sqrt,where,abs,shape,zeros,array,isnan,\
    arange,matrix,exp,sum,isinf,median,ones,bool = \
    np.sqrt,np.where,np.abs,np.shape,\
    np.zeros,np.array,np.isnan,\
    np.arange,np.matrix,np.exp,\
    np.sum,np.isinf,np.median,np.ones,np.bool

class pkfit_class:

    def __init__(self,image,gauss,psf,
                 ronois,phpadu,
                 noise_image,mask_image):
        self.f = image
        self.gauss = gauss
        self.psf = psf
        self.fnoise = noise_image
        self.fmask = mask_image
        self.ronois = ronois
        self.phpadu = phpadu

    def pkfit_noise(self,scale,x,y,sky,radius,
                    maxiter=25,
                    debug=False,debug2=False,
                    xyout = False):

        f = self.f; gauss = self.gauss; psf = self.psf
        fnoise = self.fnoise; fmask = self.fmask

        if debug2:
            import time
            tstart = time.time()
        if f.dtype != 'float64': f = f.astype('float64')

#        psf1d = psf.reshape(shape(psf)[0]**2.) 
        s = shape(f) #Get array dimensions
        nx = s[1] ; ny = s[0] #Initialize a few things for the solution

        redo = 0
        pkerr = 0.027/(gauss[3]*gauss[4])**2.
        clamp = zeros(3) + 1.
        dtold = zeros(3)
        niter = 0
        chiold = 1.

        if debug:
            print('PKFIT: ITER  X      Y      SCALE    ERRMAG   CHI     SHARP')

        loop=True
        while loop:                        #Begin the big least-squares loop
            niter = niter+1

            if isnan(x) or isnan(y):
                scale=np.nan
                errmag=np.nan
                chi=np.nan
                sharp=np.nan
                if xyout:
                    return(errmag,chi,sharp,niter,scale,x,y)
                else:
                    return(errmag,chi,sharp,niter,scale)        

            ixlo = int(x-radius)
            if ixlo < 0: ixlo = 0       #Choose boundaries of subarray containing
            iylo = int(y-radius)
            if iylo < 0: iylo = 0       # 3points inside the fitting radius
            ixhi = int(x+radius) +1 
            if ixhi > (nx-1): ixhi = nx-1
            iyhi = int(y+radius) +1
            if iyhi > ny-1: iyhi = ny-1
            ixx  = ixhi-ixlo+1
            iyy  = iyhi-iylo+1
            dy   = arange(iyy) + iylo - y    #X distance vector from stellar centroid
            dysq = dy**2
            dx   = arange(ixx) + ixlo - x
            dxsq = dx**2
            rsq  = zeros([iyy,ixx])  #RSQ - array of squared

            for j in range(iyy): rsq[j,:] = (dxsq+dysq[j])/radius**2

            # The fitting equation is of the form
            #
            # Observed brightness =
            #      SCALE + delta(SCALE)  *  PSF + delta(Xcen)*d(PSF)/d(Xcen) +
            #                                           delta(Ycen)*d(PSF)/d(Ycen)
            #
            # and is solved for the unknowns delta(SCALE) ( = the correction to
            # the brightness ratio between the program star and the PSF) and
            # delta(Xcen) and delta(Ycen) ( = corrections to the program star's
            # centroid).
            #
            # The point-spread function is equal to the sum of the integral under
            # a two-dimensional Gaussian profile plus a value interpolated from
            # a look-up table.

            # D. Jones - noise edit from Scolnic
            good = where((rsq < 1.) &
                         (fnoise[iylo:iyhi+1,ixlo:ixhi+1] > 0) &
                         (fmask[iylo:iyhi+1,ixlo:ixhi+1] == 0))

            ngood = len(good[0])
            if ngood < 1: ngood = 1
            
            t = zeros([3,ngood])

            if not len(good[0]):
                scale=np.nan
                errmag=np.nan
                chi=np.nan
                sharp=np.nan
                if xyout:
                    return(errmag,chi,sharp,niter,scale,x,y)
                else:
                    return(errmag,chi,sharp,niter,scale)

            dx = dx[good[1]]
            dy = dy[good[0]]

            model,dvdx,dvdy = dao_value.dao_value(dx, dy, gauss,
                                                  psf, #psf1d=psf1d,
                                                  deriv=True)#,ps1d=True)


            if debug: 
                print('model created ')
                if xyout:
                    return(errmag,chi,sharp,niter,scale,x,y)
                else:
                    return(errmag,chi,sharp,niter,scale)

            t[0,:] = model
            sa=shape(dvdx)
            if sa[0] > ngood or len(sa) == 0:
                scale=0
                if xyout:
                    return(errmag,chi,sharp,niter,scale,x,y)
                else:
                    return(errmag,chi,sharp,niter,scale)

            t[1,:] = -scale*dvdx
            t[2,:] = -scale*dvdy
            fsub = f[iylo:iyhi+1,ixlo:ixhi+1]

            fsub = fsub[good[0],good[1]]
            # D. Jones - added for noise version from Scolnic
            fsubnoise=fnoise[iylo:iyhi+1,ixlo:ixhi+1]
            rsq = rsq[good[0],good[1]]
            # D. Jones - noise addition from Scolnic                                                                                               
            fsubnoise = fsubnoise[good[0],good[1]]
            sig=fsubnoise
            sigsq = fsubnoise**2.
        
            # D. Jones - added for noise version from Scolnic
            # Scolnic Added!!!
            #
            yx=zeros(1)
            yx[0]=sky
            skys=yx[0]
            sky=skys
            df = fsub - scale*model - sky     #Residual of the brightness from the PSF fit
        
            # The expected random error in the pixel is the quadratic sum of
            # the Poisson statistics, plus the readout noise, plus an estimated
            # error of 0.75% of the total brightness for the difficulty of flat-
            # fielding and bias-correcting the chip, plus an estimated error of
            # of some fraction of the fourth derivative at the peak of the profile,
            # to account for the difficulty of accurately interpolating within the
            # point-spread function.  The fourth derivative of the PSF is
            # proportional to H/sigma**4 (sigma is the Gaussian width parameter for
            # the stellar core); using the geometric mean of sigma(x) and sigma(y),
            # this becomes H/ sigma(x)*sigma(y) **2.  The ratio of the fitting
            # error to this quantity is estimated from a good-seeing CTIO frame to
            # be approximately 0.027 (see definition of PKERR above.)
        
            fpos = (fsub-df)   #Raw data - residual = model predicted intensity
            fposrow = where(fpos < 0.)[0]
            if len(fposrow): fpos[fposrow] = 0
            # D. Jones - noise addition from Scolnic - but ronois is never referenced, so I've omitted this
#            self.ronois=median(fsubnoise**2.-(fpos/self.phpadu + (0.0075*fpos)**2. + (pkerr*(fpos-skys))**2.))

            # D. Jones - noise addition from Scolnic
            sig=fsubnoise
            sigsq = fsubnoise**2

            relerr = df/sig
        
            # SIG is the anticipated standard error of the intensity
            # including readout noise, Poisson photon statistics, and an estimate
            # of the standard error of interpolating within the PSF.
            
            rhosq = zeros([iyy,ixx])
        
            for j in range(iyy): rhosq[j,:] = (dxsq/gauss[3]**2+dysq[j]/gauss[4]**2)

            rhosq = rhosq[good[0],good[1]]
        
            badflag = False
            if niter >= 2:    #Reject any pixel with 10 sigma residual
                badpix = where( abs(relerr/chiold) >= 10. )[0]
                nbad = len(badpix)
                # scolnic added
                sbd=shape(badpix)
                sdf=shape(df)
                if sbd[0] == sdf[0]:
                    scale=np.nan
                    errmag=np.nan
                    if xyout:
                        return(errmag,chi,sharp,niter,scale,x,y)
                    else:
                        return(errmag,chi,sharp,niter,scale)

                if nbad > 0:
                    # D. Jones - to fix a bug in the original code
                    goodind = arange(len(rsq))
                    goodind = item_remove(badpix,goodind)
                    badflag = True

                    fsub = item_remove(badpix, fsub)
                    df = item_remove(badpix,df)
                    sigsq = item_remove(badpix,sigsq)
                    sig = item_remove(badpix,sig)
                    relerr = item_remove(badpix,relerr)
                    rsq = item_remove(badpix,rsq)
                    rhosq = item_remove(badpix,rhosq)
                    fsubnoise = item_remove(badpix,fsubnoise)

                    ngood = ngood-badpix

            wt = 5./(5.+rsq/(1.-rsq))
            lilrho = where(rhosq <= 36.)[0]   #Include only pixels within 6 sigma of centroid
            if not len(lilrho):
                scale=np.nan
                errmag=np.nan
                sharp=np.nan
                chi=np.nan
                if xyout:
                    return(errmag,chi,sharp,niter,scale,x,y)
                else:
                    return(errmag,chi,sharp,niter,scale)

            rhosq[lilrho] = 0.5*rhosq[lilrho]
            dfdsig = exp(-rhosq[lilrho])*(rhosq[lilrho]-1.)
            fpos = fsub[lilrho]
            fposrow = where(fsub[lilrho]-sky < 0.)[0]
            fpos[fposrow] = sky

            # FPOS-SKY = raw data minus sky = estimated value of the stellar
            # intensity (which presumably is non-negative).

            # sig  = fpos/self.phpadu + self.ronois + (0.0075*fpos)**2 + (pkerr*(fpos-sky))**2
            # D. Jones - noise addition from Scolnic
            sig = fsubnoise[lilrho]**2
            numer = sum(dfdsig*df[lilrho]/sig)
            denom = sum(dfdsig**2/sig)

            # Derive the weight of this pixel.  First of all, the weight depends
            # upon the distance of the pixel from the centroid of the star-- it
            # is determined from a function which is very nearly unity for radii
            # much smaller than the fitting radius, and which goes to zero for
            #  radii very near the fitting radius.

            chi = sum(wt*abs(relerr))
            sumwt = sum(wt)

            wt = wt/sigsq   #Scale weight to inverse square of expected mean error
            if niter >= 2: #Reduce weight of a bad pixel
                wt = wt/(1.+(0.4*relerr/chiold)**8)

            v = zeros(3)       #Compute vector of residuals and the normal matrix.
            c = zeros([3,3])

            if not badflag:
                for kk in range(3):
                    v[kk] = sum(df*t[kk,:]*wt)
                    for ll in range(3): c[ll,kk] = sum(t[kk,:]*t[ll,:]*wt)
            else:
                for kk in range(3):
                    v[kk] = sum(df*t[kk,goodind]*wt)
                    for ll in range(3): c[ll,kk] = sum(t[kk,goodind]*t[ll,goodind]*wt)

            # Compute the (robust) goodness-of-fit index CHI.
            # CHI is pulled toward its expected value of unity before being stored
            # in CHIOLD to keep the statistics of a small number of pixels from
            # completely dominating the error analysis.

            if sumwt > 3.0:
                chi = 1.2533*chi*sqrt(1./(sumwt*(sumwt-3.)))
                chiold = ((sumwt-3.)*chi+3.)/sumwt

            if not isnan(sum(c)):
                try:
                    c = linalg.inv(c)  #Invert the normal matrix
                except:
                    scale=np.nan
                    errmag=np.nan
                    chi=np.nan
                    sharp=np.nan
                    if xyout:
                        return(errmag,chi,sharp,niter,scale,x,y)
                    else:
                        return(errmag,chi,sharp,niter,scale)

            dt = matrix(v)*c       #Compute parameter corrections
            dt = array(dt)[0]

            # In the beginning, the brightness of the star will not be permitted
            # to change by more than two magnitudes per iteration (that is to say,
            # if the estimate is getting brighter, it may not get brighter by
            # more than 525% per iteration, and if it is getting fainter, it may
            # not get fainter by more than 84% per iteration).  The x and y
            # coordinates of the centroid will be allowed to change by no more
            # than one-half pixel per iteration.  Any time that a parameter
            # correction changes sign, the maximum permissible change in that
            # parameter will be reduced by a factor of 2.
    
            div = where( dtold*dt < -1.e-38)[0]
            nbad = len(div)
            if nbad > 0: clamp[div] = clamp[div]/2.
            dtold = dt
            adt = abs(dt)

            denom2 = ( dt[0]/(5.25*scale))
            if denom2 < (-1*dt[0]/(0.84*scale)): denom2 = (-1*dt[0]/(0.84*scale))
            scale = scale+dt[0]/(1 + denom2/clamp[0])
            x = x + dt[1]/(1.+adt[1]/(0.5*clamp[1]))
            y = y + dt[2]/(1.+adt[2]/(0.5*clamp[2]))
            redo = 0

            # Convergence criteria:  if the most recent computed correction to the
            # brightness is larger than 0.1% or than 0.05 * sigma(brightness),
            # whichever is larger, OR if the absolute change in X or Y is
            # greater than 0.01 pixels, convergence has not been achieved.
        
            sharp = 2.*gauss[3]*gauss[4]*numer/(gauss[0]*scale*denom)
            errmag = chiold*sqrt(c[0,0])
            if ( adt[0] > 0.05*errmag) or (adt[0] > 0.001*scale): redo = 1
            if (adt[1] > 0.01) or (adt[2] > 0.01): redo = 1

            if debug: print(niter,x,y,scale,errmag,chiold,sharp)
        
            if niter >= 3: loop=False        #At least 3 iterations required

            # If the solution has gone 25 iterations, OR if the standard error of
            # the brightness is greater than 200%, give up.

            if (redo and (errmag <= 1.9995) and (niter < maxiter) ): loop=True
            #        if sharp < -99.999: sharp = -99.999
            #        elif sharp > 99.999: sharp = 99.999

        if xyout:
            return(errmag,chi,sharp,niter,scale,x,y)
        else:
            return(errmag,chi,sharp,niter,scale)

def item_remove(index,array):

    mask = ones(array.shape,dtype=bool)
    mask[index] = False
    smaller_array = array[mask]

    return(smaller_array)

