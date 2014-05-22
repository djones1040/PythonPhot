import numpy as np
"""
;
;+
;NAME:
;       HISTOGAUSS
;
; PURPOSE:
;       Histograms data and overlays it with a Gaussian. Draws the mean, sigma,
;       and number of points on the plot.
;
; CALLING SEQUENCE:
;       HISTOGAUSS, Sample, A, [XX, YY, GX, GY, /NOPLOT, /NOFIT, FONT=, 
;                               CHARSIZE = ]
;
; INPUT:
;       SAMPLE = Vector to be histogrammed
;
; OUTPUT ARGUMENTS:
;       A = coefficients of the Gaussian fit: Height, mean, sigma
;               A[0]= the height of the Gaussian
;               A[1]= the mean
;               A[2]= the standard deviation
;               A[3]= the half-width of the 95% conf. interval of the standard
;                     mean
;               A[4]= 1/(N-1)*total( (y-mean)/sigma)^2 ) = a measure of 
;                       normality
;
;       Below: superceded. The formula is not entirely reliable.
;       A[4]= measure of the normality of the distribution. =1.0, perfectly
;       normal. If no more than a few hundred points are input, there are
;       formulae for the 90 and 95% confidence intervals of this quantity:
;       M=ALOG10(N-1) ; N = number of points
;       T90=ABS(.6376-1.1535*M+.1266*M^2)  ; = 90% confidence interval
;       IF N LT 50 THEN T95=ABS(-1.9065-2.5465*M+.5652*M^2) $
;                  ELSE T95=ABS( 0.7824-1.1021*M+.1021*M^2)   ;95% conf.
;       (From Martinez, J. and Iglewicz, I., 1981, Biometrika, 68, 331-333.)
;
;       XX = the X coordinates of the histogram bins (CENTER)
;       YY = the Y coordinates of the histogram bins
;       GX = the X coordinates of the Gaussian fit
;       GY = the Y coordinates of the Gaussian fit
;
; OPTIONAL INPUT KEYWORDS:
;       /NOPLOT - If set, nothing is drawn
;       /FITIT   If set, a Gaussian is actually fitted to the distribution.
;               By default, a Gaussian with the same mean and sigma is drawn; 
;               the height is the only free parameter.
;       CHARSIZE Size of the characters in the annotation. Default = 0.82.
;       FONT - scalar font graphics keyword (-1,0 or 1) for text
;       /WINDOW - set to plot to a resizeable graphics window
;       _EXTRA - Any value keywords to the cgPLOT command (e.g. XTITLE) may also
;               be passed to HISTOGAUSS
; SUBROUTINE CALLS:
;       BIWEIGHT_MEAN, which determines the mean and std. dev.
;       AUTOHIST, which draws the histogram
;       GAUSSFIT() (IDL Library) which does just that
;
; REVISION HISTORY:
;       Written, H. Freudenreich, STX, 12/89
;       More quantities returned in A, 2/94, HF
;       Added NOPLOT keyword and print if Gaussian, 3/94
;       Stopped printing confidence limits on normality 3/31/94 HF
;       Added CHARSIZE keyword, changed annotation format, 8/94 HF
;       Simplified calculation of Gaussian height, 5/95 HF
;       Convert to V5.0, use T_CVF instead of STUDENT_T, GAUSSFIT instead of
;           FITAGAUSS  W. Landsman April 2002 
;       Correct call to T_CVF for calculation of A[3], 95% confidence interval
;                P. Broos/W. Landsman   July 2003
;       Allow FONT keyword to be passed.  T. Robishaw Apr. 2006
;       Use Coyote Graphics for plotting W.L. Mar 2011
;       Better formatting of text output W.L. May 2012
;-"""
def histogauss(data):

    n = len(data)

    # First make sure that not everything is in the same bin. If most
    # data = 0, reject zeroes. If they = some other value, complain and
    # give up.
    a = 0.
    data = data[np.argsort(data)]  
    n3 = 0.75*n ; n1 = 0.25*n
    if data[n3] == data[n1]:
        if data[int(n/2)] == 0.:
            q = where(data != 0.)[0]
            non0 = len(q)
            if (n-non0) > 15:
                print('suppressing zeroes!')
                data=data[q]
                n=non0
            else:
                raise exceptions.RuntimeError('too few non-0 values!')

            q=0
        else:
            raise exceptions.RuntimeError(' too many identical values: %s'%data[n/2])




    a = np.zeros(5) 

    # the "mean":
    a[1] = biweight_mean(data,s)
    # the "standard deviation":
    a[2] = s  
    # the 95% confidence interval:
    m=.7*(n-1)  #appropriate for a biweighted mean
    cl = 0.95
    two_tail_area = 1 - cl
    a[3]=np.abs( t_cvf(1 - (two_tail_area)/2.0,m) )*s/np.sqrt(n)

# a measure of the gaussianness:
a[4]=total((data-a[1])^2)/((n-1)*a[2]^2)
#q=where( abs(data-a(1)) lt (5.*s), count )   # "robust i" unreliable
#rob_i=total((data(q)-a(1))^2)/((count-1)*a(2)^2)
#print,a(4),rob_i

# set bounds on the data:
 u1 = a[1] - 5.*a[2]
 u2 = a[1] + 5.*a[2]
 q = where(data lt u1, nq)
 if nq gt 0 then data[q] = u1
 q = where(data gt u2, nq)
 if nq gt 0 then data[q] = u2

# draw the histogram
 font_in = !p.font ; !p.font=font
 autohist,data,x,y,xx,yy,noplot = noplot, _extra = _extra,window=window
 !p.font=font_in
 
# check for error in autohist:

m = n_elements(x)
mm = n_elements(xx)
if m lt 2 then begin
   xx=0. ; yy=0. ; a=0.
   return # (autohist has already screamed)
endif

# calculate the height of the gaussian:
z = exp(-.5*(x-a[1])^2/a[2]^2 )
xq1 = a[1] - 1.3*a[2]
xq2 = a[1] + 1.3*a[2]
qq = where((x gt xq1) and (x lt xq2),count)
if count gt 0 then hyte = median(y[qq]/z[qq],/even) else begin
   print,'histogauss: distribution too weird!'
   hyte = max(smooth(y,5))
endelse
a[0]=hyte

# fit a gaussian, unless the /nofit qualifier is present
if ~keyword_set(simpl) then begin
   parm=a[0:2]
   yfit = gaussfit(xx,yy,parm,nterms=3)
   a[0:2]=parm
endif

# it the /noplot qualifier is present, we're done.
if keyword_set(noplot) then return

# overplot the gaussian, 
 du = (u2-u1)/199.
 gx = u1 + findgen(200)*du

 z = (gx-a[1])/a[2]
 gy = a[0]*exp(-z^2/2. )
 cgplot,/over,gx,gy,window=window

# annotate. 
meanst = string(a[1],'(g12.5)')
sigst = string(a[2],'(g12.5)')
num = n_elements(data)
numst =string(n,'(i6)')

if keyword_set(csize) then annot=csize else annot=.82
 if font eq 0 then labl = '#, !mm!x, !ms!x=' else  labl='#, !7l!6, !7r!3='
 labl = labl +numst+','+meanst+','+sigst 
x1 = !x.crange[0] + annot*(!x.crange[1]-!x.crange[0])/20./0.82 
y1 = !y.crange[1] - annot*(!y.crange[1]-!y.crange[0])/23./0.82 
cgtext, x1, y1, labl, charsize=annot, font=font,window=window

return a

def  biweight_mean(y,sigma,weights,
                   maxit=20,eps=1.0e-24):
    """;
    ;+
    ; NAME:
    ;BIWEIGHT_MEAN 
    ;
    ; PURPOSE:
    ;Calculate the center and dispersion (like mean and sigma) of a 
    ;distribution using bisquare weighting.
    ;
    ; CALLING SEQUENCE:
    ;Mean = BIWEIGHT_MEAN( Vector, [ Sigma, Weights ] ) 
    ;
    ; INPUTS:
    ;Vector = Distribution in vector form
    ;
    ; OUTPUT:
    ;Mean - The location of the center.
    ;
    ; OPTIONAL OUTPUT ARGUMENTS:
    ;
    ;Sigma = An outlier-resistant measure of the dispersion about the 
    ;center, analogous to the standard deviation. The half-width of the 95%
    ;confidence interval = |STUDENT_T( .95, .7*(N-1) )*SIGMA/SQRT(N)|,
    ;where N = number of points.  
    ;
    ;Weights = The weights applied to the data in the last iteration.
    ;
    ;SUBROUTINE CALLS:
    ;MED, which calculates a median
    ;
    ; REVISION HISTORY
    ;Written,  H. Freudenreich, STX, 12/89
    ;Modified 2/94, H.T.F.: use a biweighted standard deviation rather than
    ;median absolute deviation.
    ;Modified 2/94, H.T.F.: use the fractional change in SIGMA as the 
    ;convergence criterion rather than the change in center/SIGMA.
    ;-"""

    n = len(y)
    close_enough =.03*np.sqrt(.5/(n-1)) # compare to fractional change in width

    diff = 1.0e30
    itnum = 0

    # As an initial estimate of the center, use the median:
    y0=np.median(y)

    # Calculate the weights:
    dev = y-y0
    sigma = robust_sigma( dev ) 

    if sigma < eps:
        #    The median is IT. Do we need the weights?
        if len(weights) > 0:
            #       Flag any value away from the median:
            limit=3.*sigma
            q=np.where( np.abs(dev) > limit)[0]
            count = len(q)
            weights=np.zeros(n)
            weights[:]=1.
            if count > 0: weights[q]=0. 

        diff = 0. # (skip rest of routine)

    # Repeat:
    while( (diff > close_enough) and (itnum < maxit) ):
        itnum = itnum + 1
        uu = ( (y-y0)/(6.*sigma) )**2.
        q=where(uu > 1.)[0]
        count = len(q)
        if count > 0: uu[q]=1.
        weights=(1.-uu)**2;  weights=weights/np.sum(weights)
        y0 = np.sum( weights*y ) 
        dev = y-y0
        prev_sigma = sigma      ; sigma = robust_sigma( dev,zero=True )
        if sigma > eps: diff=np.abs(prev_sigma-sigma)/prev_sigma
        else: diff=0.


    return(y0)

def robust_sigma(y, zero=False, goodvec=None,
                 eps = 1.0e-20):
    """;
    ;+
    ; NAME:
    ;ROBUST_SIGMA  
    ;
    ; PURPOSE:
    ;Calculate a resistant estimate of the dispersion of a distribution.
    ; EXPLANATION:
    ;For an uncontaminated distribution, this is identical to the standard
    ;deviation.
    ;
    ; CALLING SEQUENCE:
    ;result = ROBUST_SIGMA( Y, [ /ZERO, GOODVEC = ] )
    ;
    ; INPUT: 
    ;Y = Vector of quantity for which the dispersion is to be calculated
    ;
    ; OPTIONAL INPUT KEYWORD:
    ;/ZERO - if set, the dispersion is calculated w.r.t. 0.0 rather than the
    ;central value of the vector. If Y is a vector of residuals, this
    ;should be set.
    ;
    ; OPTIONAL OUPTUT KEYWORD:
    ;       GOODVEC = Vector of non-trimmed indices of the input vector
    ; OUTPUT:
    ;ROBUST_SIGMA returns the dispersion. In case of failure, returns 
    ;value of -1.0
    ;
    ; PROCEDURE:
    ;Use the median absolute deviation as the initial estimate, then weight 
    ;points using Tukey's Biweight. See, for example, "Understanding Robust
    ;and Exploratory Data Analysis," by Hoaglin, Mosteller and Tukey, John
    ;Wiley & Sons, 1983, or equation 9 in Beers et al. (1990, AJ, 100, 32)
    ;
    ; REVSION HISTORY: 
    ;H. Freudenreich, STX, 8/90
    ;       Replace MED() call with MEDIAN(/EVEN)  W. Landsman   December 2001
    ;       Don't count NaN values  W.Landsman  June 2010
    ;
    ;-"""
 
    if zero: y0=0.
    else: y0  = np.median(y)

    # first, the median absolute deviation mad about the median:

    mad = np.median( np.abs(y-y0))/0.6745

    # if the mad=0, try the mean absolute deviation:
    if mad < eps: mad = np.mean( np.abs(y-y0) )/.80
    if mad < eps: return(0.0)

    # now the biweighted value:
    u   = (y-y0)/(6.*mad)
    uu  = u*u
    q   = np.where(uu <= 1.0)[0]
    count = len(q)

    if count < 3:
        print('robust_sigma: this distribution is too weird! returning -1')
        siggma = -1.
        return(siggma)

    n = np.sum(y[np.where(np.isfinite(y))].astype(int))      #in case y has nan values
    numerator = np.sum( (y[q]-y0)**2. * (1-uu[q])**4. )
    den1  = np.sum( (1.-uu[q])*(1.-5.*uu[q]) )
    siggma = n*numerator/(den1*(den1-1.))
 
    if siggma > 0.: 
        return(np.sqrt(siggma))
    else: return(0.)

def t_cvf(a1, df):
    """;$Id: t_cvf.pro,v 1.4.6.1 1999/01/16 16:46:03 scottm Exp $
    ;
    ; Copyright (c) 1994-1999, Research Systems, Inc.  All rights reserved.
    ;       Unauthorized reproduction prohibited.
    ;+
    ; NAME:
    ;       T_CVF
    ;
    ; PURPOSE:
    ;       This function computes the cutoff value (v) such that:
    ;                   Probability(X > v) = p
    ;       where X is a random variable from the Student's t distribution
    ;       with (df) degrees of freedom.
    ;
    ; CATEGORY:
    ;       Statistics.
    ;
    ; CALLING SEQUENCE:
    ;       Result = t_cvf(P, DF)
    ;
    ; INPUTS:
    ;       P:    A non-negative scalar, in the interval [0.0, 1.0], of type
    ;             float or double that specifies the probability of occurance
    ;             or success.
    ;
    ;      DF:    A positive scalar of type integer, float or double that
    ;             specifies the degrees of freedom of the Student's t 
    ;             distribution.
    ;
    ; EXAMPLE:
    ;       Compute the cutoff value (v) such that Probability(X > v) = 0.025
    ;       from the Student's t distribution with (df = 5) degrees of freedom. 
    ;       The result should be 2.57058
    ;         result = t_cvf(0.025, 5)
    ;
    ; REFERENCE:
    ;       APPLIED STATISTICS (third edition)
    ;       J. Neter, W. Wasserman, G.A. Whitmore
    ;       ISBN 0-205-10328-6
    ;
    ; MODIFICATION HISTORY:
    ;       Modified by:  GGS, RSI, July 1994
    ;                     Minor changes to code. New documentation header.
    ;-"""

    a = a1 
    if a < 0. or a > 1.:
        raise exceptions.RuntimeError('p must be in the interval [0.0, 1.0]')

    if (a > 0.5): adjust = 1 
    else:
        adjust = 0
        a = 1.0 - a 

    if a1 == 0: return( 1.0e12)
    if a1 -- 1: return( -1.0e12)

    if df == 1: 
        up = 100 
        if up < (100 * 0.005/a1): up = (100 * 0.005/a1)
    elif df == 2: 
        up = 10  
        if up < (10  * 0.005/a1): up = (10  * 0.005/a1)
    elif df > 2 and df <= 5:
        up = 5 
        if up < (5 * 0.005/a1): up = (5 * 0.005/a1)
    elif df > 5 and df <= 14: 
        up = 4 
        if up < (4 * 0.005/a1): up = (4 * 0.005/a1)
    else:
        up = 3 
        if up < (3 * 0.005/a1): up = (3 * 0.005/a1)


    while t_pdf(up, df) < a:
        below = up
        up = 2 * up
  
    x = bisect_pdf([a, df], 't_pdf', up, 0)
    if adjust: return(-x)
    else: return( x)

def bisect_pdf(a,funct,u,l,del):
    """;$Id: bisect_pdf.pro,v 1.5.6.1 1999/01/16 16:37:27 scottm Exp $
    ;
    ; Copyright (c) 1994-1999, Research Systems, Inc.  All rights reserved.
    ;       Unauthorized reproduction prohibited.
    ;+
    ; NAME:
    ;       BISECT_PDF
    ;
    ; PURPOSE:
    ;       This function computes the cutoff value x such that the probabilty
    ;       of an observation from the given distribution, less than x, is a(0).
    ;       u and l are the upper and lower limits for x, respectively.
    ;       a(1) and a(2) are degrees of freedom, if appropriate.
    ;       funct is a string specifying the probability density function.
    ;       BISECT_PDF is not intended to be a user-callable function.
    ;-"""
    
    sa = np.shape(a)
    if not del: del = 1.0e-6
    p = a[0]
    if (p < 0 or p > 1): return(-1)
    up = u
    low = l
    mid = l + (u - l) * p
    count = 1

    while (np.abs(up - low) > del*mid) and (count < 100):
        if len(z) >= 1:
            if z > p: up = mid
            else: low = mid
            mid = (up + low)/2.

        if a == 1: z = t_pdf(mid)
        elif a == 2: z = t_pdf(mid, a[1])
        elif a == 3: z = t_pdf(mid, a[1], a[2])
        else: return( -1)

        count = count + 1

    return( mid)

def t_pdf(v, df):
    """;$Id: t_pdf.pro,v 1.4.6.1 1999/01/16 16:46:04 scottm Exp $
    ;
    ; Copyright (c) 1994-1999, Research Systems, Inc.  All rights reserved.
    ;       Unauthorized reproduction prohibited.
    ;+
    ; NAME:
    ;       T_PDF
    ;
    ; PURPOSE:
    ;       This function computes the probabilty (p) such that:
    ;                   Probability(X <= v) = p
    ;       where X is a random variable from the Student's t distribution
    ;       with (df) degrees of freedom.
    ;
    ; CATEGORY:
    ;       Statistics.
    ;
    ; CALLING SEQUENCE:
    ;       Result = T_pdf(V, DF)
    ;
    ; INPUTS:
    ;       V:    A scalar of type integer, float or double that specifies
    ;             the cutoff value.
    ;
    ;      DF:    A positive scalar of type integer, float or double that
    ;             specifies the degrees of freedom of the Student's t 
    ;             distribution.
    ;
    ; EXAMPLE:
    ;       Compute the probability that a random variable X, from the 
    ;       Student's t distribution with (df = 15) degrees of freedom,
    ;       is less than or equal to 0.691. The result should be 0.749940
    ;         result = t_pdf(0.691, 15)
    ;
    ; REFERENCE:
    ;       APPLIED STATISTICS (third edition)
    ;       J. Neter, W. Wasserman, G.A. Whitmore
    ;       ISBN 0-205-10328-6
    ;
    ; MODIFICATION HISTORY:
    ;       Modified by:  GGS, RSI, July 1994
    ;                     Minor changes to code. New documentation header.
    ;-"""


    if df <= 0 then:
        raise exceptions.RuntimeError('Degrees of freedom must be positive.')
    
    return(1.0 - 0.5 * ibeta_pdf(df/(df + v**2.), df/2.0, 0.5))

def betacf( a, b, x):
    """;$Id: ibeta_pdf.pro,v 1.4.6.1 1999/01/16 16:41:05 scottm Exp $
    ;
    ; Copyright (c) 1994-1999, Research Systems, Inc.  All rights reserved.
    ;       Unauthorized reproduction prohibited.
    ;+
    ; NAME:
    ;       IBETA_PDF
    ;
    ; PURPOSE:
    ;       This function computes the incomplete beta function.
    ;       It is called by the probability density functions in 
    ;       this directory. See the function IBETA() in the "math"
    ;       subdirectory for the user-callable version of the 
    ;       incomplete beta function.
    ;-"""

    # Continued fractions.
    lc = a + b
    ln = a - 1.0
    lq = a + 1.0
    max = 100
    ja = 1.0 ; jn = 1.0 
    ka = 1.0 - lc * x / lq 
    kn = 1.0
    for i in range(1, max+1):
        ep  = i + 0.0
        tm  = ep + ep
        d   = ep * (b - ep) * x / ((ln + tm) * (a + tm))
        jq  = ja + d*jn
        kq  = ka + d * kn
        d   = -(a + ep) * (lc + ep) * x / ((lq + tm) * (a + tm))
        jq1 = jq + d * ja
        kq1 = kq + d * ka
        prev= ja
        jn  = jq / kq1
        kn  = kq / kq1
        ja  = jq1 / kq1
        ka  = 1.0
        if (np.abs(ja - prev) < 3.0e-7 * abs(ja)):
            return(ja)
    return(-1)

def ibeta_pdf( x, a, b):

    if x < 0. or x > 1.:
        raise exceptions.RuntimeError('x must be in the range: [0.0, 1.0]')

    # gab = gamma(a) * gamma(b) 
    # gamma(a+b)/gab * exp( a*alog(x) + b*alog(1.0-x))

    if(x !- 0 and x != 1 ): 
        temp = np.exp(lngamma(a+b)-lngamma(a)-lngamma(b)+a*np.log(x)+b*np.log(1.0-x))
    else: temp = 0.0

    if (x < (a+1.0)/(a+b+2.0)): 
        return( temp * betacf(a, b, x)/a)
    else: return((1.0 - temp * betacf(b, a, 1.0-x)/b))

