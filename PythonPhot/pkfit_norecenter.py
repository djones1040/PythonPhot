#!/usr/bin/env python
# D. Jones - 1/10/14
"""This code is from the IDL Astronomy Users Library
(adapted for IDL from DAOPHOT, then translated from IDL to Python).

Subroutine of GETPSF to perform a one-star least-squares fit, 
part of the DAOPHOT PSF photometry sequence.  This version requires
DOES NOT recenter the PSF. 

CALLING SEQUENCE:
     from PythonPhot import pkfit_norecenter as pkfit
     pk = pkfit.pkfit_class(f, gauss, psf, ronois, phpadu )
     errmag,chi,sharp,niter,scale,xnew,ynew = pk.pkfit_norecenter(scale,x,y,sky,radius)

PKFIT CLASS INPUTS:
     f       - NX by NY array containing actual picture data.
     ronois  - readout noise per pixel, scalar
     phpadu  - photons per analog digital unit, scalar
     gauss   - vector containing the values of the five parameters defining
                the analytic Gaussian which approximates the core of the PSF.
     psf     - an NPSF by NPSF look-up table containing corrections from
                the Gaussian approximation of the PSF to the true PSF.
     
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
     xyout   - if True, return new x and y positions
     maxiter - maximum iterations (default = 25)

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
     import pyfits
     from PyIDLPhot import pkfit_norecenter as pkfit

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
     Version 2.0                              W. Landsman STX   November,  1988
     Converted to IDL V5.0                    W. Landsman       September, 1997
     Converted to Python                      D. Jones          January,   2014
"""

import numpy as np
from scipy.optimize import leastsq
import dao_value
from numpy.ma import masked_array
from exceptions import RuntimeWarning

sqrt, where, abs, shape, zeros, array, isnan,\
    arange, matrix, exp, sum, isinf, median, ones, bool = \
    np.sqrt, np.where, np.abs, np.shape,\
    np.zeros, np.array, np.isnan,\
    np.arange, np.matrix, np.exp,\
    np.sum, np.isinf, np.median, np.ones, np.bool


class pkfit_class:

    def __init__(self, image, gauss, psf, ronois, phpadu):
        self.f = image
        self.gauss = gauss
        self.psf = psf
        self.ronois = ronois
        self.phpadu = phpadu

    def pkfit_norecenter_quick(self, scale, x, y, sky, radius,
                               debug=False, maxiter=25, sigclip=10):
        """ Fit the target star with a psf model, using quick numpy-based
        least squares fitting, with iterative sigma clipping.

        :param scale: initial guess of the optimized psf scale
        :param x: x position of the target in the data array
        :param y: y position of the target in the data array
        :param sky: fixed value of the background sky flux
        :param radius: fitting radius in pixels
        :param maxiter: max number of sigma clipping iterations
        :param sigclip: after each fitting iteration, clip pixels that have
                residuals discrepant by more than sigclip times the
                expected random flux error
        :param debug: enter the pdb debugger. Set >1 for diagnostic plots.
        :return:
        """
        # TODO : better checking for valid input data, with exceptions
        assert not (isnan(x) or isnan(y))

        f = self.f
        gauss = self.gauss
        psf = self.psf

        # psf1d = psf.reshape(shape(psf)[0]**2.)
        s = shape(f)  # Get array dimensions
        nx = s[1]
        ny = s[0]  # Initialize a few things for the solution

        redo = 0
        pkerr = 0.027/(gauss[3]*gauss[4])**2.
        clamp = zeros(3) + 1.
        dtold = zeros(3)
        niter = 0
        chiold = 1.

        if debug:
            import time
            tstart = time.time()
            print('PKFIT: ITER  X      Y      SCALE    ERRMAG   CHI     SHARP')
            import pdb
            pdb.set_trace()

        # Set the x,y pixel position boundaries for a subarray
        # containing all pixels that fall within the fitting radius
        # NOTE: in this version, with no recentering, the x,y position
        # never changes, so we can define this subarray outside the loop
        ixlo = int(x-radius)
        if ixlo < 0:
            ixlo = 0
        iylo = int(y-radius)
        if iylo < 0:
            iylo = 0
        ixhi = int(x+radius) + 1
        if ixhi > (nx-1):
            ixhi = nx-1
        iyhi = int(y+radius) + 1
        if iyhi > ny-1:
            iyhi = ny-1
        ixx = ixhi-ixlo+1
        iyy = iyhi-iylo+1
        dy = arange(iyy) + iylo - y  # Y dist. vector from stellar centroid
        dysq = dy**2
        dx = arange(ixx) + ixlo - x
        dxsq = dx**2

        # construct rsq as an array giving the square of the radial distance
        # of each pixel from the center of the target star, in units of the
        # user-defined fitting radius (i.e. rsq=1 is the circle at the
        # fitting radius)
        rsq = zeros([iyy, ixx])
        for j in range(iyy):
            rsq[j, :] = (dxsq+dysq[j])/radius**2

        # define a list of indices in the subarray for those pixels that
        # are within the fitting radius from the stellar center
        i_tofit = where(rsq.reshape(shape(rsq)[0]*shape(rsq)[1]) < 1)[0]
        n_tofit = len(i_tofit)
        if n_tofit < 1:
            n_tofit = 1

        # Extract a subarray with the observed flux for all pixels within the
        # fitting radius of the center of the target star
        flux_observed_tofit = masked_array(
            f[iylo:iyhi+1, ixlo:ixhi+1].ravel()[[i_tofit]])

        # Call the function dao_value to generate realized flux values
        # from the given psf model for each pixel position in the image
        # subarray that is within the fitting radius
        dx = dx[i_tofit % ixx]
        dy = dy[i_tofit / ixx]
        flux_model_tofit = dao_value.dao_value(dx, dy, gauss, psf, deriv=False)

        # Set the weight of each pixel for the least squares fitter based on
        # its distance from the fixed center of the target star:
        # weight ~ 1 at the center, and diminishes rapidly to ~0 at the edge
        # of the fitting radius.
        weight_tofit = (5./(5.+rsq/(1.-rsq))).ravel()[i_tofit]

        # Since we are not allowing the PSF to be recentered in this version,
        # the error function to minimize has only one free variable, SCALE:
        #    err = sum(flux_observed * weight - SCALE * flux_model * weight)
        def errfunc(psf_scale):
            """ Error function to minimize
            :param psf_scale: the psf scaling factor
            :return: vector of pixel residuals
            """
            error_vector = flux_observed_tofit * weight_tofit \
                           - psf_scale * flux_model_tofit * weight_tofit \
                           - sky
            return error_vector

        # The expected random error in the pixel is the quadratic sum of
        # the Poisson statistics, plus the readout noise, plus an estimated
        # error of 0.75% of the total brightness for the difficulty of flat
        # fielding and bias-correcting the chip, plus an estimated error
        #  of some fraction of the fourth derivative at the peak of the
        # profile, to account for the difficulty of accurately
        # interpolating within the point-spread function.  The fourth
        # derivative of the PSF is proportional to H/sigma**4 (sigma is
        # the Gaussian width parameter for the stellar core); using the
        # geometric mean of sigma(x) and sigma(y), this becomes
        # H/ sigma(x)*sigma(y) **2.  The ratio of the fitting error to
        # this quantity is estimated to be approximately 0.027
        # (see definition of PKERR above.)
        flux_observed_tofit_noneg = where(flux_observed_tofit-sky < 0,
                                          abs(sky), flux_observed_tofit)
        fluxerr2 = flux_observed_tofit_noneg/self.phpadu + \
                   self.ronois + \
                   (0.0075*flux_observed_tofit_noneg)**2 + \
                   (pkerr*(flux_observed_tofit_noneg-sky))**2
        fluxerr = sqrt(fluxerr2)

        # Solve for the SCALE factor using least squares minimization,
        # iteratively rejecting bad pixels that are more than 'sigclip'
        # sigma discrepant from the model, where sigma is the expected
        # random error (fluxerr above), separately defined for each pixel
        badpix_mask = np.zeros(n_tofit)
        for iteration in range(maxiter):
            n_badpix_beforefit = sum(badpix_mask)
            if debug:
                scale, cov, infodict, errmsg, ierr = leastsq(errfunc, scale,
                                                             full_output=True)
            else :
                scale, cov = leastsq(errfunc, scale, full_output=False)
            flux_resid_tofit = flux_observed_tofit - \
                               scale * flux_model_tofit - sky
            badpix_mask = abs(flux_resid_tofit.data/fluxerr) > sigclip
            if n_badpix_beforefit >= sum(badpix_mask):
                break
            elif debug:
                import pdb; pdb.set_trace()
            if n_badpix_beforefit > 0.5 * n_tofit:
                raise RuntimeWarning(
                    ">50pct of pixels >%.1f sigma discrepant.  " % sigclip +
                    "Disabling badpix masking in iteration %i." % iteration)
                flux_observed_tofit.mask = np.zeros(n_tofit)
            flux_observed_tofit.mask = badpix_mask

        if iteration == maxiter-1:
            raise RuntimeWarning("Max # of iterations exceeded")

        if not debug:
            return scale
        elif debug > 1:
            # Serious debugging:
            # collect the full output from the least squares fitting
            # routine, plot the observed, model and residual fluxes,
            # and enter the pdb debugger.

            flux_obs_subarray = f[iylo:iyhi+1, ixlo:ixhi+1]
            flux_model_subarray = np.zeros(flux_obs_subarray.shape).ravel()
            flux_model_subarray[i_tofit] = flux_model_tofit * scale + sky
            flux_model_subarray = flux_model_subarray.reshape(
                flux_obs_subarray.shape)
            flux_resid_subarray = np.zeros(flux_obs_subarray.shape).ravel()
            flux_resid_subarray[i_tofit] = flux_observed_tofit - \
                                           scale * flux_model_tofit - sky
            flux_resid_subarray = flux_resid_subarray.reshape(
                flux_obs_subarray.shape)

            from matplotlib import pyplot as pl, cm
            pl.figure(2, figsize=[10, 3.5])
            pl.clf()
            fig = pl.gcf()
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95)
            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)
            im1 = ax1.imshow(flux_obs_subarray, interpolation='nearest',
                             aspect='equal', cmap=cm.Greys_r)
            cb1 = pl.colorbar(im1, ax=ax1, use_gridspec=True,
                              orientation='horizontal')
            im2 = ax2.imshow(flux_model_subarray, interpolation='nearest',
                             aspect='equal', cmap=cm.Greys_r)
            cb2 = pl.colorbar(im2, ax=ax2, use_gridspec=True,
                              orientation='horizontal')
            im3 = ax3.imshow(flux_resid_subarray, interpolation='nearest',
                             aspect='equal', cmap=cm.Greys_r)
            cb3 = pl.colorbar(im3, ax=ax3, use_gridspec=True,
                              orientation='horizontal')

        return scale


def item_remove(index, inputarray):

    mask = ones(inputarray.shape, dtype=bool)
    mask[index] = False
    smaller_array = inputarray[mask]

    return smaller_array
