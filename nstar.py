#!/usr/bin/env python
# D. Jones - 4/29/14

"""
This procedure was adapted for Python from the IDL Astronomy Users Library.
Python routine also returns SHARP vector and includes 'faintlim' input
to determine faintness of stars that should be excluded
Original documentation:
;+
; NAME:
;       NSTAR
; PURPOSE:
;       Simultaneous point spread function fitting (adapted from DAOPHOT)
; EXPLANATION:
;       This PSF fitting algorithm is based on a very old (~1987) version of 
;       DAOPHOT, and much better algorithms (e.g. ALLSTAR) are now available
;       -- though not in IDL.
;       
; CALLING SEQUENCE:
;       NSTAR, image, id, xc, yc, mags, sky, group, [ phpadu, readns, psfname,
;               magerr, iter, chisq, peak, /PRINT , /SILENT, /VARSKY, /DEBUG ]
;
; INPUTS:
;       image - image array
;       id    - vector of stellar ID numbers given by FIND
;       xc    - vector containing X position centroids of stars (e.g. as found
;               by FIND)
;       yc    - vector of Y position centroids
;       mags  - vector of aperture magnitudes (e.g. as found by APER)
;               If 9 or more parameters are supplied then, upon output
;               ID,XC,YC, and MAGS will be modified to contain the new
;               values of these parameters as determined by NSTAR.
;               Note that the number of output stars may be less than 
;               the number of input stars since stars may converge, or 
;               "disappear" because they are too faint.
;       sky   - vector of sky background values (e.g. as found by APER)
;       group - vector containing group id's of stars as found by GROUP
;
; OPTIONAL INPUT:
;       phpadu - numeric scalar giving number of photons per digital unit.  
;               Needed for computing Poisson error statistics.   
;       readns - readout noise per pixel, numeric scalar.   If not supplied, 
;               NSTAR will try to read the values of READNS and PHPADU from
;               the PSF header.  If still not found, user will be prompted.
;       psfname - name of FITS image file containing the point spread
;               function residuals as determined by GETPSF, scalar string.  
;               If omitted, then NSTAR will prompt for this parameter.
;
; OPTIONAL OUTPUTS:
;       MAGERR - vector of errors in the magnitudes found by NSTAR
;       ITER - vector containing the number of iterations required for
;               each output star.  
;       CHISQ- vector containing the chi square of the PSF fit for each
;               output star.
;       PEAK - vector containing the difference of the mean residual of
;               the pixels in the outer half of the fitting circle and
;               the mean residual of pixels in the inner half of the
;               fitting circle
;
; OPTIONAL KEYWORD INPUTS:
;       /SILENT - if set and non-zero, then NSTAR will not display its results
;               at the terminal
;       /PRINT - if set and non-zero then NSTAR will also write its results to
;               a file nstar.prt.   One also can specify the output file name
;               by setting PRINT = 'filename'.
;       /VARSKY - if this keyword is set and non-zero, then the sky level of
;               each group is set as a free parameter.
;       /DEBUG - if this keyword is set and non-zero, then the result of each
;               fitting iteration will be displayed.
;
; PROCEDURES USED:
;       DAO_VALUE(), READFITS(), REMOVE, SPEC_DIR(), STRN(), SXPAR()
;
; COMMON BLOCK:
;       RINTER - contains pre-tabulated values for cubic interpolation
; REVISION HISTORY
;       W. Landsman                 ST Systems Co.       May, 1988
;       Adapted for IDL Version 2, J. Isensee, September, 1990
;       Minor fixes so that PRINT='filename' really prints to 'filename', and
;       it really silent if SILENT is set.  J.Wm.Parker HSTX 1995-Oct-31
;       Added /VARSKY option   W. Landsman   HSTX      May 1996
;       Converted to IDL V5.0   W. Landsman   September 1997
;       Replace DATATYPE() with size(/TNAME)  W. Landsman November 2001
;       Assume since V5.5, remove VMS calls W. Landsman September 2006
;-"""

import numpy as np
from numpy import where,array,sqrt,shape
import os
from scipy import linalg
import dao_value
import pyfits
import exceptions

def nstar(image,id,xc,yc,
          mags,sky,group,phpadu,
          readns,psfname,debug=False,
          doPrint=False,silent=False,
          varsky=False,faintlim=0.25):
    image = image.astype(np.float64)
    
    shapeid,shapexc,shapeyc,shapemags,shapesky,shapegroup = \
        shape(id),shape(xc),shape(yc),shape(mags),shape(sky),shape(group)
    for shapevar in [shapexc,shapeyc,shapemags,shapesky,shapegroup]:
        if shapevar != shapeid:
            raise exceptions.RuntimeError('Input variables have different shapes!')

    psf_file = psfname
    image = image.astype(np.float64)
    if not os.path.exists(psf_file):
        print('ERROR - Unable to locate PSF file %s'%psf_file)

    # Read in the FITS file containing the PSF

    s = shape(image)
    icol = s[1]-1 ; irow = s[0]-1  #Index of last row and column
    psf = pyfits.getdata(psfname)
    hpsf = pyfits.getheader(psfname)

    gauss = array([hpsf['GAUSS1'],hpsf['GAUSS2'],
                   hpsf['GAUSS3'],hpsf['GAUSS4'],hpsf['GAUSS5']])
    psfmag = hpsf['PSFMAG']
    psfrad = hpsf['PSFRAD']
    fitrad = hpsf['FITRAD']
    npsf = hpsf['NAXIS1']

#                               Compute RINTER common block arrays
    
    p_1 = np.roll(psf,1,axis=1) ; p1 = np.roll(psf,-1,axis=1) ; p2 = np.roll(psf,-2,axis=1)
    c1 = 0.5*(p1 - p_1)
    c2 = 2.*p1 + p_1 - 0.5*(5.*psf + p2)
    c3 = 0.5*(3.*(psf-p1) + p2 - p_1)
    init = 1

    ronois = readns**2.
    radsq = fitrad**2.   ;  psfrsq = psfrad**2.
    sepmin = 2.773*(gauss[3]**2.+gauss[4]**2.)

    #      PKERR will be used to estimate the error due to interpolating PSF
    #      Factor of 0.027 is estimated from good-seeing CTIO frames

    pkerr = 0.027/(gauss[3]*gauss[4])**2.     
    sharpnrm = 2.*gauss[3]*gauss[4]/gauss[0]
    if (len(group) == 1): groupid = array([group[0]])
    else: groupid = np.unique(group)

    mag = mags                        #Save original magnitude vector
    bad = where( mag > 99 )[0]     #Undefined magnitudes assigned 99.9
    nbad = len(bad)

    if nbad > 0: mag[bad] = psfmag + 7.5
    mag = 10.**(-0.4*(mag-psfmag)) #Convert magnitude to brightness, scaled to PSF
    # fmt = '(I6,2F9.2,3F9.3,I4,F9.2,F9.3)'

    if doPrint:
        if doPrint == True: 
            file = 'nstar.prt'
        else: file = doPrint
        print('Results will be written to a file %s'%file)
        fout = open(file,'w')
        print >> fout, 'NSTAR'
        print >> fout, 'PSF File:',psfname

    hdr='   ID      X       Y       MAG     MAGERR   SKY   NITER     CHI     SHARP'
    if not silent: print hdr
    if doPrint: print >> fout, hdr


    for igroup in range(len(groupid)):

        index = where(group == groupid[igroup])[0]
        nstr = len(index)

        if not silent: print 'Processing group %s    %s stars'%(int(groupid[igroup]),nstr)

        if nstr == 0:
            print 'No stars found'
            return(-1,-1,-1,-1,-1,-1)

        magerr = np.zeros(nstr)
        chiold = 1.0
        niter = 0
        clip = False
        nterm = nstr*3 + varsky
        xold = np.array([[0]*nterm])
        clamp = array([1.]*nterm)
        xb = xc[index].astype(float)  ;   yb = yc[index].astype(float)
        magg = mag[index].astype(float) ; skyg = sky[index].astype(float)
        idg = id[index]
        skybar = np.sum(skyg)/nstr
        reset = False

        doLoop = True
        restart = False
        while doLoop:
            if not restart:
                niter = niter+1

            restart=False
            if niter >= 4 : wcrit = 1
            elif niter >= 8 : wcrit = 0.4444444
            elif niter >= 12: wcrit = 0.25             
            else       : wcrit = 400                                                   

            if reset:
                xb = xg + ixmin ; yb = yg + iymin

            reset = True
            xfitmin = (xb - fitrad).astype(int)
            xfitmin[where(xfitmin < 0)] = 0
            xfitmax = (xb + fitrad).astype(int)+1 
            xfitmax[where(xfitmax > icol-1)] = icol-1
            yfitmin = (yb - fitrad).astype(int)
            yfitmin[where(yfitmin < 0)] = 0
            yfitmax = (yb + fitrad).astype(int)+1
            yfitmax[where(yfitmax > irow-1)] = irow-1
            nfitx = xfitmax - xfitmin + 1
            nfity = yfitmax - yfitmin + 1
            ixmin = np.min(xfitmin); iymin = np.min(yfitmin)
            ixmax = np.max(xfitmax); iymax = np.max(yfitmax)
            nx = ixmax-ixmin+1 ; ny = iymax-iymin+1
            dimage = image[iymin:iymax+1,ixmin:ixmax+1]
            xfitmin = xfitmin -ixmin ; yfitmin = yfitmin-iymin
            xfitmax = xfitmax -ixmin ; yfitmax = yfitmax-iymin
            #                                        Offset to the subarray
            xg = xb-ixmin ; yg = yb-iymin
            j = 0

            while (j < nstr-1):
                sep = (xg[j] - xg[j+1:])**2 + (yg[j] - yg[j+1:])**2
                bad = where(sep < sepmin)[0]
                nbad = len(bad)
                if nbad > 0:      #Do any star overlap?
                    for l in range(nbad):
                        k = bad[l] + j + 1
                        if magg[k] < magg[j]: imin = k 
                        else: imin = j #Identify fainter star
                        
                        if ( sep[l] < 0.14*sepmin) or \
                                ( magerr[imin]/magg[imin]**2. > wcrit ):
                            if  imin == j: imerge = k 
                            else: imerge = j

                            nstr = nstr - 1
                            if not silent: 
                                print 'Star %s has merged with star %s'%(idg[imin],idg[imerge])

                            totmag = magg[imerge] + magg[imin]
                            xg[imerge] = (xg[imerge]*magg[imerge] + xg[imin]*magg[imin])/totmag
                            yg[imerge] = (yg[imerge]*magg[imerge] + yg[imin]*magg[imin])/totmag
                            magg[imerge] = totmag     
                            idg = item_remove(imin,idg)
                            xg = item_remove(imin,xg)
                            yg = item_remove(imin,yg)
                            magg = item_remove(imin,magg)
                            skyg = item_remove(imin,skyg)
                            magerr = item_remove(imin,magerr)    #Remove fainter star from group
                            nterm = nstr*3 + varsky                   #Update matrix size
                            xold = [np.zeros(nterm)] 
                            clamp = array([1.]*nterm)               #Release all clamps
                            clip = False
                            niter = niter-1                           #Back up iteration counter
                            restart = True
                            break
                            #      goto, RESTART 
                    if restart: break
                if restart: break
            

                j = j+1
            if restart: continue

            xpsfmin = (xg - psfrad+1).astype(int)
            xpsfmin[where(xpsfmin < 0)] = 0
            xpsfmax = (xg + psfrad  ).astype(int)
            xpsfmax[where(xpsfmax > nx-1)] = nx-1
            ypsfmin = (yg - psfrad+1).astype(int)
            ypsfmin[where(ypsfmin < 0)] = 0
            ypsfmax = (yg + psfrad  ).astype(int)
            ypsfmax[where(ypsfmax > ny-1)] = ny-1
            npsfx = xpsfmax-xpsfmin+1 ; npsfy = ypsfmax-ypsfmin+1
            wt = np.zeros([ny,nx])
            mask = np.zeros([ny,nx],dtype='b')
            nterm = 3*nstr + varsky
            chi = np.zeros(nstr) ; sumwt = np.zeros(nstr) ; numer = np.zeros(nstr) ; denom = np.zeros(nstr)
            c = np.zeros([nterm,nterm]) ; v = np.zeros(nterm)

            for j in range(nstr):   #Mask of pixels within fitting radius of any star
                x1 = xfitmin[j]  ;  y1 = yfitmin[j]
                x2 = xfitmax[j]  ;  y2 = yfitmax[j]
                rpixsq = np.zeros([nfity[j],nfitx[j]])
                xfitgen2 = (np.arange(nfitx[j]) + x1 - xg[j])**2.
                yfitgen2 = (np.arange(nfity[j]) + y1 - yg[j])**2.
                for k in range(nfity[j]): rpixsq[k,:] = xfitgen2 + yfitgen2[k]
                temp = (rpixsq <= 0.999998*radsq)
                for ymask in range(y1,y2+1):
                    for xmask in range(x1,x2+1):
                        mask[ymask,xmask] = mask[ymask,xmask] or temp[ymask-y1,xmask-x1]
                good = where(temp)
                rsq = rpixsq[good]/radsq
                temp1 = wt[y1:y2+1,x1:x2+1] 
                temp1rows = where(np.greater(temp1[good],(5./(5.+rsq/(1.-rsq)) )))[0]
                rsqrows = where(np.less_equal(temp1[good],(5./(5.+rsq/(1.-rsq)) )))[0]
                if len(temp1rows):
                    temp1[good[0][temp1rows],good[1][temp1rows]] = temp1[good[0][temp1rows],good[1][temp1rows]]
                if len(rsqrows):
                    temp1[good[0][rsqrows],good[1][rsqrows]] = (5./(5.+rsq[rsqrows]/(1.-rsq[rsqrows])) )
                wt[y1:y2+1,x1:x2+1] = temp1

            igood = where(mask)
            ngoodpix = len(igood[0])
            x = np.zeros([nterm,ngoodpix])

            if varsky: x[nterm-1, 0] = array([-1.0]*ngoodpix)

            psfmask = np.zeros([nstr,ngoodpix],dtype='b')
            d = dimage[igood] - skybar
            for j in range(nstr):  #Masks of pixels within PSF radius of each star
                x1 = xpsfmin[j]   ;    y1 = ypsfmin[j]
                x2 = xpsfmax[j]   ;    y2 = ypsfmax[j]
                xgen = np.arange(npsfx[j]) + x1 - xg[j]
                ygen = np.arange(npsfy[j]) + y1 - yg[j]
                xgen2 = xgen**2. ; ygen2 = ygen**2.
                rpxsq = np.zeros( [npsfy[j],npsfx[j]] )
                for k in range(npsfy[j]): rpxsq[k,:] = xgen2 + ygen2[k]
                temp = np.zeros(shape(mask[y1:y2+1,x1:x2+1]))
                for ymask in range(y1,y2+1):
                    for xmask in range(x1,x2+1):
                        temp[ymask-y1,xmask-x1] = mask[ymask,xmask] and (rpxsq[ymask-y1,xmask-x1] < psfrsq)
                temp1 = np.zeros([ny,nx],dtype='b')
                temp1[y1:y2+1,x1:x2+1] = temp #is this correct? 
                goodfit = where(temp1[igood])[0]
                psfmask[j,goodfit] = 1
                good = where(temp)
                xgood = xgen[good[1]] ; ygood = ygen[good[0]]
                model,dvdx,dvdy = dao_value.dao_value(xgood,ygood,gauss,psf,psf1d=psf.reshape(shape(psf)[0]**2.),ps1d=True)
                d[goodfit] = d[goodfit] - magg[j]*model
                x[3*j,goodfit] = -model
                x[3*j+1,goodfit] = magg[j]*dvdx
                x[3*j+2,goodfit] = magg[j]*dvdy


            wt = wt[igood] ; idimage = dimage[igood]
            dpos = (idimage-d)
            dposrow = where(dpos < 0)
            dpos[dposrow] = 0

            sigsq = dpos/phpadu + ronois + (0.0075*dpos)**2 + (pkerr*(dpos-skybar))**2

            relerr = np.abs(d)/sqrt(sigsq)
            if clip:   #Reject pixels with 20 sigma errors (after 1st iteration)
                bigpix = where(relerr > 20.*chiold)[0]
                nbigpix = len(bigpix)
                if ( nbigpix > 0 ):
                    keep = np.arange(ngoodpix)
                    for i in range(nbigpix): keep = keep[ where( keep != bigpix[i])[0] ]
                    wt= wt[keep] ; d = d[keep] ; idimage = idimage[keep] ; sigsq = sigsq[keep]
                    ngoodpix = len(keep)
                    igood= (igood[0][keep],igood[1][keep])  ; relerr = relerr[keep]
                    psfmask = psfmask[:,keep]   ;  x = x[:,keep]


            sumres = np.sum(relerr*wt)
            grpwt = np.sum(wt)

            dpos = idimage
            dposrow = np.where(idimage < skybar)[0]
            if len(dposrow): dpos[dposrow] = skybar
            # dpos = ((idimage-skybar) > 0) + skybar
            sig = dpos/phpadu + ronois + (0.0075*dpos)**2. + (pkerr*(dpos-skybar))**2.
            for j in range(nstr):
                goodfit = where(psfmask[j,:])[0]
                chi[j] = np.sum(relerr[goodfit]*wt[goodfit])
                sumwt[j] = np.sum(wt[goodfit])
                xgood = igood[1][goodfit] ; ygood = igood[0][goodfit]
                rhosq = ((xg[j] - xgood)/gauss[3])**2  +  ((yg[j] - ygood)/gauss[4])**2
                goodsig = where(rhosq < 36)     #Include in sharpness index only
                rhosq = 0.5*rhosq[goodsig]       #pixels within 6 sigma of centroid
                dfdsig = np.exp(-rhosq)*(rhosq-1.)
                sigpsf = sig[goodfit[goodsig]] ; dsig = d[goodfit[goodsig]]
                numer[j] = np.sum(dfdsig*dsig/sigpsf)
                denom[j] = np.sum(dfdsig**2/sigpsf)

            wt = wt/sigsq
            if clip:  #After 1st iteration, reduce weight of a bad pixel
                wt = wt/(1.+(0.4*relerr/chiold)**8) 

            v = np.dot(x,(d*wt).reshape(ngoodpix,1))
            c = np.dot(x,np.transpose(np.ones([nterm,1])*wt * x))

            if grpwt > 3:
                chiold = 1.2533*sumres*sqrt(1./(grpwt*(grpwt-3.)))
                chiold = ((grpwt-3.)*chiold+3.)/grpwt

            i = where(sumwt > 3)
            ngood = len(i)
            if ngood > 0:
                chi[i] = 1.2533*chi[i]*sqrt(1./((sumwt[i]-3.)*sumwt[i]))
                chi[i] = ((sumwt[i]-3.)*chi[i]+3.)/sumwt[i]

            chibad = where(sumwt <= 3)[0]
            ngood = len(chibad)
            if ngood > 0: chi[chibad] = chiold

            c = linalg.inv(c)
            x = np.dot(np.transpose(v),c)

            if not clip or niter <= 1: redo = True
            else: redo = False
            if varsky:
                skybar = skybar - x[nterm-1]
                if np.abs(x[nterm-1]) >  0.01: redo = True

            clip = True
            j = 3*np.arange(nstr) ; k = j+1 ; l=j+2
            sharp = sharpnrm*numer/(magg*denom)

            if not redo:
                redovar = (0.05*chi*sqrt(c[j,j]))
                redorow = where(redovar < 0.001*magg)[0]
                if len(redorow): redovar[redorow] = 0.001*magg
                redo = np.max(np.abs(x[:,j]) > redovar)
                if redo == 0: redo = np.max( abs(np.append(x[:,k], x[:,l])) > 0.01)

            
            sgn = where( xold[0][j]*x[0][j]/magg**2 < -1e-37 )[0]  
            if len(sgn) > 0: clamp[j[sgn]] = 0.5*clamp[j[sgn]]
            sgn = where( xold[0][k]*x[0][k]        < -1e-37 )[0]
            if len(sgn) > 0: clamp[k[sgn]] = 0.5*clamp[k[sgn]]
            sgn = where( xold[0][l]*x[0][l]        < -1e-37 )[0]
            if len(sgn) == 0: clamp[l[sgn]] = 0.5*clamp[l[sgn]]

            denom1 = (x[0][j]/(0.84*magg))
            denomrow = where(denom1 < (-x[0][j]/(5.25*magg)))[0]
            if len(denomrow): denom1[denomrow] = (-x[0][j]/(5.25*magg))
            magg = magg-x[0][j] / (1.+ denom1 / clamp[j] )

            xg = xg - x[0][k]   /(1.+abs(x[0][k])/( clamp[k]*0.5))
            yg = yg - x[0][l]   /(1.+abs(x[0][l])/( clamp[l]*0.5))
            xold = x

            magerr = c[j,j]*(nstr*chi**2 + (nstr-1)*chiold**2)/(2.*nstr-1.)

            xarr = xg - nx
            yarr = yg - ny
            xrow = where(xarr < 0)[0]
            yrow = where(yarr < 0)[0]
            if len(xrow): xarr[xrow] = 0
            if len(yrow): yarr[yrow] = 0
            dx = -xg; dy = -yg
            xrow2 = where(dx < xarr)[0]
            yrow2 = where(dy < yarr)[0]
            if len(xrow2): dx[xrow2] = xarr
            if len(yrow2): dy[yrow2] = yarr

            # these two lines replaced by the lines above
            # dx = (-xg) > ( (xg - nx) > 0.) #Find stars outside subarray
            # dy = (-yg) > ( (yg-  ny) > 0.)
            # Remove stars with bad centroids
            badcen = where((dx > 0.001) | 
                           (dy > 0.001) | 
                           ( (dx+1)**2 + (dy+1)**2 >= radsq ))[0]
            nbad = len(badcen)
            if nbad > 0:
                nstr = nstr - nbad
                print '%i stars eliminated by centroid criteria'%nbad
                if nstr <= 0: 
                    # goto, DONE_GROUP 
                    break
                idg = item_remove(badcen, idg)
                xg = item_remove(badcen, xg)
                yg = item_remove(badcen, yg)
                magg = item_remove(badcen, magg)
                skyg = item_remove(badcen, skyg)
                magerr = item_remove(badcen, magerr)
                nterm = nstr*3 + varsky
                redo = True


            faint = 1
            toofaint =  where (magg <= 1.e-5)[0]
            nfaint = len(toofaint)
                              #Number of stars 12.5 mags fainter than PSF star
            if nfaint > 0:
                faint = np.min( magg[toofaint], min_pos )
                min_pos = where(faint == magg[toofaint])
                ifaint = toofaint[ min_pos ]
                magg[toofaint] = 1e-5
                faint,nfaint,nstr,idg,xg,yg,\
                    magg,skyg,magerr,nterm,\
                    xold,clamp,clip,niter,\
                    done_group,restart = rem_faint(faint,nfaint,nstr,idg,
                                                   xg,yg,magg,skyg,magerr,
                                                   nterm,xold,clamp,clip,niter,
                                                   silent=silent,faintlim=faintlim)

                 # goto, REM_FAINT                #Remove faintest star
            else:
                faint = 0.
                ifaint = -1
                if (not redo) or (niter >= 4):
                    faint = np.max(magerr/magg**2)
                    ifaint = where(faint == magerr/magg**2.)[0]
                else:
                    continue
                    # goto,START_IT 


            if debug:
                err = 1.085736*sqrt(magerr)/magg
                for i in range(nstr):
                        print idg[i],xg[i]+ixmin,yg[i]+iymin,\
                            psfmag-1.085736*alog(magg[i]),err[i],\
                            skyg[i],niter,chi[i],sharp[i]

            if redo and (niter <= 50) and (faint < wcrit): continue # goto,START_IT   
            # REM_FAINT: 
            faint,nfaint,nstr,idg,xg,yg,\
                magg,skyg,magerr,nterm,\
                xold,clamp,clip,niter,\
                done_group,restart = rem_faint(faint,nfaint,nstr,idg,
                                               xg,yg,magg,skyg,magerr,
                                               nterm,xold,clamp,clip,
                                               niter,ifaint,silent=silent,
                                               faintlim=faintlim)
            if done_group: break

            err = 1.085736*sqrt(magerr)/magg
            magg = psfmag - 1.085736*np.log(magg)
            sharprow = where(sharp > 99.99)[0]
            if len(sharprow): sharp[sharprow] = 99.99
            sharprow = where(sharp < -99.99)[0]
            if len(sharprow): sharp[sharprow] = -99.99
            # if sharp > 99.999: sharp = 99.999
            # elif sharp < -99.999: sharp = -99.999
            xg = xg+ixmin ; yg = yg+iymin

            # Print results to terminal and/or file

            if not silent: 
                for i in range(nstr): 
                    print idg[i],xg[i],yg[i],\
                        magg[i],err[i],skyg[i],niter,chi[i],sharp[i]
            if doPrint: 
                for i in range(nstr): 
                    print >> fout, idg[i],xg[i],yg[i],\
                        magg[i],err[i],skyg[i],niter,\
                        chi[i],sharp[i]

            if 'newid' not in locals():   #Initialize output vectors?
                newid = idg ;  newx = xg  ;  newy = yg ; newmag = magg
                iter = array([niter]*nstr) ; peak = sharp ; chisq = chi
                errmag = err
            else:          #Append current group to output vector
                newid = np.append(newid,idg) ; newx = np.append(newx ,xg) ; newy = np.append(newy,yg)
                newmag = np.append(newmag,magg) ; iter = np.append(iter,np.array([niter]*nstr))
                peak = np.append(peak,sharp)     ; chisq = np.append(chisq,chi) ; errmag = np.append(errmag,err)
            
            doLoop=False

#DONE_GROUP: 

    if 'newid' in locals():
        id = newid ;  xc = newx ;  yc = newy  ; mags = newmag
    else:
        print 'ERROR - There are no valid stars left, variables not updated'
        return(-1,-1,-1,-1,-1,-1)

    if doPrint: fout.close()

    return(mags,errmag,iter,chisq,sharp,peak)

def item_remove(index,array):

    mask = np.ones(array.shape,dtype=bool)
    mask[index] = False
    smaller_array = array[mask]

    return(smaller_array)

def rem_faint(faint,nfaint,nstr,
              idg,xg,yg,magg,
              skyg,magerr,nterm,
              xold,clamp,clip,
              niter,ifaint,silent=False,faintlim=0.25):
    done_group = False
    restart = False

    if (faint >= faintlim) or (nfaint > 0):
        if not silent:
            print 'Star %s is too faint'%idg[ifaint]
        nstr = nstr-1
        if nstr <= 0:
            done_group = True
        else:
            idg = item_remove(ifaint,idg)
            xg = item_remove(ifaint,xg)
            yg = item_remove(ifaint,yg)
            magg = item_remove(ifaint,magg)
            skyg = item_remove(ifaint,skyg)
            magerr = item_remove(ifaint,magerr)
            nterm = nstr*3 + varsky
            xold = np.array([[0]*nterm])
            clamp = array([1.]*nterm)
            clip = 0
            niter = niter-1
            restart = True
    return(faint,nfaint,nstr,idg,xg,yg,magg,skyg,magerr,nterm,xold,clamp,clip,niter,done_group,restart)

