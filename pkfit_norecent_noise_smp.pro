pro pkfit_norecent_noise_smp,f,scale,x,y,sky,skyerr,radius,ronois,phpadu,psf, $
                  errmag,chi,sharp,niter,fnoise, fmask,f2a,f2b,f2c,f2d, f2e,f2f, savefits
;note psf is called but not gauss,psf
stamp=100

s = size(f) 
 nx = s[1] & ny = s[2]

if abs(round(x)-double(x)) lt .001 then x=x+.01
if abs(round(y)-double(y)) lt .001 then y=y+.01

;               ;Initialize a few things for the solution

; ixlo = fix(x-radius) > 0	;Choose boundaries of subarray containing
; iylo = fix(y-radius) > 0        ;points inside the fitting radius
; ixhi = fix(x+radius) +1 < (nx-1)
; iyhi = fix(y+radius) +1 < (ny-1)
; ixx  = ixhi-ixlo+1
; iyy  = iyhi-iylo+1

ixlo = floor(x+.5-radius) > 0 
iylo = floor(y+.5-radius) > 0 
ixx=2*radius+1
iyy=2*radius+1
ixhi=ixlo+ixx
iyhi=iylo+iyy

 dy   =  findgen(iyy) + iylo - y    ;X distance vector from stellar centroid
 dysq = dy^2
 dx   = findgen(ixx) + ixlo - x
 dxsq = dx^2
 rsq  = fltarr(ixx,iyy)  ;RSQ - array of squared 
 rsq2  = fltarr(ixx,iyy)  ;RSQ - array of squared                                          

radius_loc=15.0
 for J = 0,iyy-1 do rsq[0,j] = (dxsq+dysq[j])/radius_loc^2

 for J = 0,iyy-1 do rsq2[0,j] = (dxsq+dysq[j])/(radius_loc)^2


 good_check = where(rsq lt 1. and fnoise[ixlo:ixhi,iylo:iyhi] gt 0 and fmask[ixlo:ixhi,iylo:iyhi] eq 0  ,ngood)
 good_all = where(fnoise[ixlo:ixhi,iylo:iyhi]*0+1.0 gt 0  ,ngood)

; goodx = where(rsq2*radius^2 ge 13.0^2 or fnoise[ixlo:ixhi,iylo:iyhi] eq 0 or fmask[ixlo:ixhi,iylo:iyhi] ne 0  ,ngoodx)
 bad_psf = where(rsq2*radius_loc^2 ge 13.0^2,ngoodx)
 good_psf = where(rsq2*radius_loc^2 le 13.0^2,ngoodx)

; good_pix = where(fnoise[ixlo:ixhi,iylo:iyhi] ne 0 and fmask[ixlo:ixhi,iylo:iyhi] eq 0  ,ngoodx)
 bad_pix = where(fnoise[ixlo:ixhi,iylo:iyhi] eq 0 or fmask[ixlo:ixhi,iylo:iyhi] ne 0  ,ngoodx)

 good_local = where(rsq2*radius_loc^2 lt 7.0^2 and fnoise[ixlo:ixhi,iylo:iyhi] ne 0 and fmask[ixlo:ixhi,iylo:iyhi] eq 0  ,ngoodx)


;temp=fnoise[ixlo:ixhi,iylo:iyhi] 
;if goodx[0] gt -1 then temp[goodx]=10000000000000000.0   




bigarri=fltarr(stamp,stamp)
bigarrn2=fltarr(stamp,stamp)+10000000.0
bigarrn=fltarr(stamp,stamp)+100000000.00001                                                                                                                                   
bigarrm=fltarr(stamp,stamp)
bigarr_psf=bigarrm                       

if good_check[0] eq -1 or y lt 50 or y gt ny-50 or x lt 50 or x gt nx-50 then begin

print, 'Return1'
scale=1000000.0;
errmag=100000
chi=100000
f2a=bigarrm
f2b=bigarri
f2c=(bigarrn)
f2d=(bigarr_psf)
f2e=bigarrn
f2f=bigarrn2

return
endif





;if(n_elements(dvdx) eq 0) then begin
;print,'Return'
;scale=1000000.0;                        
;errmag=100000
;f2a=bigarrm
;f2b=bigarri
;f2c=(bigarrn)
;f2d=(bigarr_psf)
;f2e=bigarrn
;f2f=bigarrn2
;return
;endif




co1=0
;Make a matrix of what psf model is at each pixel.

bigarr_psf=fltarr(stamp,stamp)
;define what cen1, cen2 is
len=(ixhi-ixlo)/2.0
cen1=stamp/2.0
cen2=stamp/2.0

model2=f[ixlo:ixhi,iylo:iyhi]*0.0
model2[good_psf]=psf

bigarrn=fltarr(stamp,stamp)+100000000.00001
bigarrm=fltarr(stamp,stamp)
bigarri=fltarr(stamp,stamp)
bigarrn2=fltarr(stamp,stamp)
bigarrd=fltarr(stamp,stamp)

z1=0
z2=0
bigarr_psf[cen1+z1-len:cen1+z1+len,cen2+z2-len:cen2+z2+len]=(model2)

temp=bigarr_psf[cen1+z1-len:cen1+z1+len,cen2+z2-len:cen2+z2+len]
if bad_psf[0] gt -1 then temp[bad_psf]=0.0
bigarr_psf[cen1+z1-len:cen1+z1+len,cen2+z2-len:cen2+z2+len]=temp

ntemp=sqrt(skyerr[0]^2+(f[ixlo:ixhi,iylo:iyhi]-sky > 0))
if bad_pix[0] gt -1 then ntemp[bad_pix]=10000000000.0
bigarrn[cen1+z1-len:cen1+z1+len,cen2+z2-len:cen2+z2+len] = ntemp
ntemp=sqrt(skyerr[0]^2+f[ixlo:ixhi,iylo:iyhi]*0)
if bad_pix[0] gt -1 then ntemp[bad_pix]=10000000000.0
bigarrn2[cen1+z1-len:cen1+z1+len,cen2+z2-len:cen2+z2+len] = ntemp

bigarrm[cen1+z1-len:cen1+z1+len,cen2+z2-len:cen2+z2+len] = fmask[ixlo:ixhi,iylo:iyhi]
bigarri[cen1+z1-len:cen1+z1+len,cen2+z2-len:cen2+z2+len] = f[ixlo:ixhi,iylo:iyhi]
;bigarrd[cen1+z1-len:cen1+z1+len,cen2+z2-len:cen2+z2+len] = diff1[ixlo:ixhi,iylo:iyhi]




;gal model * real model = gal pixel


 if keyword_set(DEBUG) then begin print,'model created ' & stop & end


 fsub = f[ixlo:ixhi,iylo:iyhi]
 fsub = fsub[good_check]

 fsub2=f[ixlo:ixhi,iylo:iyhi]


 fsubnoise=fnoise[ixlo:ixhi,iylo:iyhi]
 fsubnoise2=fsubnoise
 fsubnoise = fsubnoise[good_check]

; df = fsub - scale*model - sky     ;Residual of the brightness from the PSF fit


 expr = 'P[0]*X'
 start=[1]
if good_local[0] gt -1 then begin
 perror=100000000.0
dof=1
bestnorm=1000000000.0
 result = MPFITEXPR(expr, model2[good_local], fsub2[good_local]-sky, fsubnoise2[good_local]*0+skyerr[0], [1],perror=perror,bestnorm=bestnorm, dof=dof,/quiet)
 result2 = MPFITEXPR(expr, model2[good_local], fsub2[good_local]-sky, fsubnoise2[good_local], [1],perror=perror,bestnorm=bestnorm2, dof=dof,/quiet)
;stop
if savefits gt -1 then begin
writefits, 'gemp'+sc(savefits)+'a.fits', fsub2-sky
writefits, 'gemp'+sc(savefits)+'b.fits', model2*result[0]
writefits, 'gemp'+sc(savefits)+'c.fits', fsub2-sky-model2*result[0]
x1=50
y1=50
cntrd, model2*result[0], x1,y1,x2,y2,4
print, x2,y2
x1=50
y1=50
cntrd, fsub2-sky, x1,y1,x2,y2,4
print, x2,y2
cxy=[49,49]
fwhm_xy1 = fullwid_halfmax( fsub2-sky, CENTROID=cxy, /GAUSSIAN_FIT )
fwhm_xy2 = fullwid_halfmax(  model2*result[0], CENTROID=cxy, /GAUSSIAN_FIT )
print, fwhm_xy1, fwhm_xy2

;stop
endif                                           
;stop
errv=fltarr(51)

for h=0.0d,50.0 do errv[h]=total((fsub2[good_local]-sky-(result[0]+h/10.0*perror[0])*model2[good_local])^2/(fsubnoise2[good_local]*0+skyerr[0])^2)
err23=min(abs(errv-errv[0]-2.3),ij)
errmag=ij/10.0*perror
chi=bestnorm/dof
chi=ij/10.0
scale=result[0]

endif


f2a=bigarrm
f2b=bigarri
f2c=bigarrn
f2d=bigarr_psf
f2e=bigarrd
f2f=bigarrn2

return
 end
