# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:17:43 2020

@author: omartin
"""

# Libraries
import numpy as nnp
from . import gpuEnabled

if not gpuEnabled:
    np = nnp
    import scipy.interpolate as interp        
    import scipy.ndimage as scnd
    import scipy.special as ssp
    import numpy.fft as fft
else:
    import cupy as cp
    import scipy.interpolate as interp        
#    import cupyx.scipy.interpolate as interp        
    import cupyx.scipy.ndimage as scnd
    import cupyx.scipy.special as ssp
    import cupy.fft as fft
    import scipy.special as spc
    np = cp

import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import matplotlib as mpl

from matplotlib.path import Path

def cpuArray(v):
    if isinstance(v,nnp.ndarray) or isinstance(v, list):
        return v
    else:
        return v.get()

#%%  FOURIER TOOLS

def cov2sf(cov):
    return 2*cov.max() - cov - np.conjugate(cov)

def fftCorrel(x,y):
    nPts = x.shape
    
    if len(nPts) == 1:
        factor = 1/nPts
        out =  fft.ifft(fft.fft(x)*np.conj(fft.fft(y)))*factor
    elif len(nPts) == 2:
        factor = 1/(nPts[0]*nPts[1])
        out =  fft.ifft2(fft.fft2(x)*np.conj(fft.fft2(y)))*factor
    return out

def fftsym(x):
    if x.ndim ==2:
        nx,ny            = x.shape
        if np.iscomplex(x):
            out              = np.zeros((nx,ny)) + complex(0,1)*np.zeros((nx,ny))
            out[0,0]         = x[0,0]
            out[0,1:ny-1]     = x[1,np.arange(ny-1,1,-1)]
            out[1:nx-1,0]     = x[np.arange(nx-1,1,-1),1]
            out[1:nx-1,1:ny-1] = x[np.arange(nx-1,1,-1),np.arange(ny-1,1,-1)]
        else:
            out              = np.zeros((nx,ny))
            out[0,0]         = x[0,0]
            out[0,1:ny-1]     = x[1,np.arange(ny-1,1,-1)]
            out[1:nx-1,0]     = x[np.arange(nx-1,1,-1),1]
            out[1:nx-1,1:ny-1] = x[np.arange(nx-1,1,-1),np.arange(ny-1,1,-1)]
            
        return out
    elif x.ndim ==1:
        return fft.fftshift(x)

def freq_array(nX, L=1, offset=1e-10):
    k2D = np.mgrid[0:nX, 0:nX].astype(float)
    k2D[0] -= nX//2
    k2D[1] -= nX//2
    k2D     *= np.asarray(L)
    k2D     += offset
    return k2D[0],k2D[1]

def getStaticOTF(tel,nOtf,samp,wvl,xStat=[],theta_ext=0,spatialFilter=1):
        
        # DEFINING THE RESOLUTION/PUPIL
        nPup = tel.pupil.shape[0]
        
        # ADDING STATIC MAP
        phaseStat = np.zeros((nPup,nPup))
        if not tel.opdMap_on is None and np.any(tel.opdMap_on):
            if theta_ext:
                tel.opdMap_on = scnd.rotate(tel.opdMap_on,theta_ext,reshape=False)
            phaseStat = (2*np.pi*1e-9/wvl) * tel.opdMap_on
            
        # ADDING USER-SPECIFIED STATIC MODES
        xStat = np.asarray(xStat)
        phaseMap = 0
        if not tel.statModes is None and np.any(tel.statModes):
            if tel.statModes.shape[2]==len(xStat):
                phaseMap = 2*np.pi*1e-9/wvl * np.sum(tel.statModes*xStat,axis=2)
                phaseStat += phaseMap
                
        # FILTERING
        if not np.isscalar(spatialFilter):
            phaseStat = (np.dot(spatialFilter,phaseStat.reshape(-1))).reshape((nPup,nPup))
        
        # INSTRUMENTAL OTF
        otfStat = pupil2otf(tel.pupil * tel.apodizer,phaseStat,samp)
        if not otfStat is None and otfStat.shape!=nOtf:
            otfStat = interpolateSupport(otfStat,nOtf)
        otfStat/= otfStat.max()
        
        # DIFFRACTION-LIMITED OTF
        if np.all(phaseStat == 0):
            otfDL = otfStat
        else:
            otfDL = np.real(pupil2otf(tel.pupil * tel.apodizer,0*phaseStat,samp))
            if otfDL.shape !=nOtf:
                otfDL = interpolateSupport(otfDL,nOtf)
                otfDL/= otfDL.max()
                
        return otfStat, otfDL, phaseMap

def instantiateAngularFrequencies(nOtf,fact=2):
    # DEFINING THE DOMAIN ANGULAR FREQUENCIES
    U_,V_  = shift_array(nOtf,nOtf,fact = fact)     
    U2_    = U_**2
    V2_    = V_**2
    UV_    = U_*V_
    return U_, V_, U2_, V2_, UV_
          
def mcDonald(x):
        out = 3/5 * np.ones_like(x)
        idx  = x!=0
        if np.any(idx==False):
            out[idx] = x[idx] ** (5/6) * spc.kv(5/6,x[idx])/(2**(5/6) * spc.gamma(11/6))
        else:
            out = x ** (5/6) * spc.kv(5/6,x)/(2**(5/6) * spc.gamma(11/6))
        return out
        
def Ialpha(x,y):
    return mcDonald(np.hypot(x,y))

def otf2psf(otf):        
    nX,nY   = otf.shape
    u1d     = fft.fftshift(fft.fftfreq(nX))
    v1d     = fft.fftshift(fft.fftfreq(nY))
    u2d,v2d = np.meshgrid(u1d,v1d)
    
    if nX%2 == 0:
        fftPhasor = np.exp(1*complex(0,1)*np.pi*(u2d+v2d))
    else:
        fftPhasor = 1
    
    if nX%2 == 0:
        out = np.real(fft.fftshift(fft.ifft2(fft.fftshift(otf*fftPhasor))))
    else:
        out = np.real(fft.fftshift(fft.ifft2(fft.ifftshift(otf*fftPhasor))))
                
    return out/out.sum()

def otfShannon2psf(otf,nqSmpl,fovInPixel):
    if nqSmpl == 1:            
        # Interpolate the OTF to set the PSF FOV
        otf    = interpolateSupport(otf,fovInPixel)
        psf    = otf2psf(otf)
    elif nqSmpl >1:
        # Zero-pad the OTF to set the PSF pixel scale
        otf    = enlargeSupport(otf,nqSmpl)
        # Interpolate the OTF to set the PSF FOV
        otf    = interpolateSupport(otf,fovInPixel)
        psf    = otf2psf(otf)
    else:
        # Interpolate the OTF at high resolution to set the PSF FOV
        otf    = interpolateSupport(otf,int(nnp.round(fovInPixel/nqSmpl)))
        psf    = otf2psf(otf)
        # Interpolate the PSF to set the PSF pixel scale
        psf    = interpolateSupport(psf,fovInPixel)
    return psf
                        
def pistonFilter(D,f,fm=0,fn=0):    
    f[np.where(np.equal(f,0))] = 1e-10 
    if len(f.shape) ==1:
        Fx,Fy = np.meshgrid(f,f)            
        FX    = Fx -fm 
        FY    = Fy -fn    
        F     = np.pi*D*np.hypot(FX,FY)    
    else:
        F     = np.pi*D*f
        
    #out         = np.zeros_like(F)
    #idx         = F!=0
    #out[idx]    = spc.j1(F[idx])/F[idx]
    R         = sombrero(1,F)      
    pFilter   =  1 - 4 * R**2
    if np.min(pFilter)<0:
        pFilter[np.where(pFilter<0)] = 0

    return pFilter
             
def psd2cov(psd,pixelScale):
    nPts = np.array(psd.shape)
    psd  = fft.fftshift(psd)
    if len(nPts) ==1:
        out = fft.fft(psd)*pixelScale**2
    elif len(nPts) ==2:
        out = fft.fft2(psd)*pixelScale**2        
    return out

def psd2otf(psd,pixelScale):
    return sf2otf(cov2sf(psd2cov(psd,pixelScale)))

def psd2psf(psd,pixelScale):
    return otf2psf(fft.fftshift(psd2otf(psd,pixelScale)))
 
def psf2otf(psf):
    return fft.fft2(fft.fftshift(psf))/psf.sum()
       
def pupil2otf(pupil,phase,overSampling):   
    if np.all(phase == 0):
        E   = np.abs(pupil)
        E   = enlargeSupport(E,overSampling)
        otf = np.real(fft.fftshift(fftCorrel(E,E)))
    else:
        E   = pupil*np.exp(1*complex(0,1)*phase)    
        E   = enlargeSupport(E,overSampling)
        otf = fft.fftshift(fftCorrel(E,E))
        
    return otf

def pupil2psf(pupil,phase,overSampling):    
    otf = pupil2otf(pupil,phase,overSampling)
    return otf2psf(otf)
      
def sf2otf(sf):
    return np.exp(-0.5 * sf)

def shift_array(nX,nY,fact=2*np.pi*complex(0,1)):    
    X, Y = np.mgrid[0:nX,0:nY].astype(float)
    X = (X-nX/2) * fact/nX
    Y = (Y-nY/2) * fact/nY
    return X,Y
    
def sombrero(n,x):
    x = np.asarray(x)
    if n==0:
        return ssp.jv(0,x)/x
    else:
        if n>1:
            out = np.zeros(x.shape)
        else:
            out = 0.5*np.ones(x.shape)
            
        out = np.zeros_like(x)
        idx = x!=0
        out[idx] = ssp.j1(x[idx])/x[idx]
        return out
                 
def telescopeOtf(pupil,samp):    
    pup_pad  = enlargeSupport(pupil,samp)
    otf      = fft.fftshift(fft.ifft2(fft.fft2(fft.fftshift(pup_pad))**2))
    return otf/otf.max()
           
def telescopePsf(pupil,samp,kind='spline'):
    nSize = nnp.array(pupil.shape)
    
    if samp >=2:
        otf = telescopeOtf(pupil,samp)
        return otf2psf(interpolateSupport(otf,nSize,kind=kind))
    else:
        otf = interpolateSupport(telescopeOtf(pupil,2),nSize//samp,kind=kind)
        return interpolateSupport(otf2psf(otf),nSize,kind=kind)


def SF2PSF(sf,freq,ao,jitterX=0,jitterY=0,jitterXY=0,F=[[1.0]],dx=[[0.0]],dy=[[0.]],bkg=0,xStat=[],
           theta_ext=0,nPix=None,otfPixel=1,spatialFilter=1):
        """
          Computation of the PSF and the Strehl-ratio (from the OTF integral). The Phase structure function
          must be expressed in nm^2 and of the size nPx x nPx x nSrc
        """
        
        # INSTANTIATING THE OUTPUTS
        if nPix == None:
            nPix = int(freq.nOtf /freq.kRef_)
        
        PSF = np.zeros((nPix,nPix,ao.src.nSrc,freq.nWvl))
        SR  = np.zeros((ao.src.nSrc,freq.nWvl))

        # DEFINING THE RESIDUAL JITTER KERNEL
        if jitterX!=0 or jitterY!=0:        
            # Gaussian kernel
            # note 1 : Umax = self.samp*self.tel.D/self.wvlRef/(3600*180*1e3/np.pi) = 1/(2*psInMas)
            # note 2 : the 1.16 factor is needed to get FWHM=jitter for jitter-limited PSF; needed to be figured out
            Umax     = freq.samp*ao.tel.D/freq.wvl/(3600*180*1e3/np.pi)
            ff_jitter= 1.16
            normFact = ff_jitter*np.max(Umax)**2 *(2 * np.sqrt(2*np.log(2)))**2 #1.16
            Djitter  = normFact * (jitterX**2 * freq.U2_  + jitterY**2 * freq.V2_ + 2*jitterXY *freq.UV_)
            Kjitter  = np.exp(-0.5 * Djitter)
        else:
            Kjitter = 1
            
        # DEFINE THE FFT PHASOR AND MULTIPLY TO THE TELESCOPE OTF
        if np.any(dx!=0) or np.any(dy!=0):
            # shift by half a pixel
            fftPhasor = np.zeros((freq.nOtf,freq.nOtf,ao.src.nSrc,freq.nWvl),dtype=complex)
            if freq.kRef_ > 2: #[jWvl] >2:
                fact = freq.kRef_#[jWvl]
            else:
                fact = 1
            for iSrc in range(ao.src.nSrc):
                for jWvl in range(freq.nWvl):
                    # account for the binning
                    fftPhasor[:,:,iSrc,jWvl] = np.exp(-np.pi*complex(0,1)*fact*\
                             (dx[iSrc,jWvl]*freq.U_ + dy[iSrc,jWvl]*freq.V_))
        else:
            fftPhasor = np.ones((freq.nOtf,freq.nOtf,ao.src.nSrc,freq.nWvl),dtype=complex)

        # LOOP ON WAVELENGTHS   
        for jWvl in range(freq.nWvl):
            
            # UPDATE THE INSTRUMENTAL OTF
            if (not ao.tel.opdMap_on is None and freq.nWvl>1) or len(xStat)>0:
                freq.otfNCPA, freq.otfDL, freq.phaseMap = \
                getStaticOTF(ao.tel,int(freq.nOtf),freq.samp[jWvl],freq.wvl[jWvl],
                             xStat=xStat,theta_ext=theta_ext,spatialFilter=spatialFilter)
                
            # UPDATE THE RESIDUAL JITTER
            if freq.nyquistSampling == True and freq.nWvl > 1 and (jitterX!=0 or jitterY!=0):
                normFact2    = ff_jitter*(freq.samp[jWvl]*ao.tel.D/freq.wvl[jWvl]/(3600*180*1e3/np.pi))**2  * (2 * np.sqrt(2*np.log(2)))**2
                Kjitter = np.exp(-0.5 * Djitter * normFact2/normFact)    
                          
            # OTF MULTIPLICATION
            otfStat = freq.otfNCPA * Kjitter * otfPixel    
            otfStat = np.repeat(otfStat[:,:,np.newaxis],ao.src.nSrc,axis=2)      
            otfTurb = np.exp(-0.5*sf*(2*np.pi*1e-9/freq.wvl[jWvl])**2)
            otfTot  = fft.fftshift(otfTurb * otfStat * fftPhasor[:,:,:,jWvl],axes=(0,1))
                        
            # GET THE FINAL PSF - PIXEL SCALE IS NYQUIST - FOV DIFFERENT PER WVL
            psf_ = np.real(fft.fftshift(fft.ifftn(otfTot,axes=(0,1)),axes = (0,1)))
            
            # managing the undersampling
            psf = np.copy(psf_)
            
            if freq.k_[jWvl] >= 1: # binning the PSF
                psf = np.zeros((ao.cam.fovInPix,ao.cam.fovInPix,ao.src.nSrc))
                nC = freq.k_[jWvl]#psf_.shape[0]/ao.cam.fovInPix
                for iSrc in range(ao.src.nSrc):
                    if nC > 1:
                        tmp = binning(psf_[:,:,iSrc],int(nC))
                    else:
                        tmp = psf_[:,:,iSrc]
                        
                    psf[:,:,iSrc] = cropSupport(tmp,tmp.shape[0]/ao.cam.fovInPix)
                psf_ = psf
             
            # managing the field of view
            if nPix < ao.cam.fovInPix:
                psf = np.zeros((nPix,nPix,ao.src.nSrc))
                nC  = psf_.shape[0]/nPix
                for iSrc in range(ao.src.nSrc):
                    psf[:,:,iSrc] = cropSupport(np.squeeze(psf_[:,:,iSrc]),nC)                   

            # SCALING
            PSF[:,:,:,jWvl] = psf * F[:,jWvl]
            
            # STREHL-RATIO COMPUTATION
            SR[:,jWvl] = 1e2*np.abs(otfTot).sum(axis=(0,1))/np.real(freq.otfDL.sum())
        
        return PSF + bkg, SR
    
#%%  IMAGE PROCESSING
        
def binning(image, k):
    """Bin an image by a factor `k`
    
    Example
    -------
    >>> x,y = _np.mgrid[0:10,0:10]
    >>> data = (-1)**x * (-1)**y
    >>> data_bin = binning(data,2)
    
    """
    S = np.shape(image)
    S0 = int(S[0] / k)
    S1 = int(S[1] / k)
    out = np.zeros((S0, S1))
    for i in range(k):
        for j in range(k):
            out += image[i:k*S0:k, j:k*S1:k]
    return out

def cropSupport(im,n):    
    nx,ny = im.shape
    
    if np.isscalar(n) == 1:
        n = np.array([n,n])
    
    nx2     = int(nx/n[0])
    ny2     = int(ny/n[1])
    
    if np.any(np.iscomplex(im)):
        imNew = np.zeros((nx2,ny2)) + complex(0,1)*np.zeros((nx2,ny2))
    else:
        imNew = np.zeros((nx2,ny2))
        
    if nx2%2 ==0:
        xi = int(0.5*(nx-nx2))
        xf = int(0.5*(nx + nx2))
    else:
        xi = int(0.5*(nx-nx2))
        xf = int(0.5*(nx+nx2))
        
    if ny2%2 ==0:
        yi = int(0.5*(ny-ny2))
        yf = int(0.5*(ny + ny2))
    else:
        yi = int(0.5*(ny-ny2))
        yf = int(0.5*(ny+ny2))    
        
    imNew     = im[xi:xf,yi:yf]
    
    return imNew
            
            
def enlargeSupport(im,n):
    
    if len(im.shape) == 2:
        nx,ny  = im.shape
        return np.pad(im,[int((n-1)*nx/2),int((n-1)*ny/2)], mode='constant')
    elif len(im.shape) == 3:
        nx,ny,nz  = im.shape
        if (nz < nx) and (nz < ny):
            return np.pad(im,[int((n-1)*nx/2),int((n-1)*ny/2),(0,0)], mode='constant')
        else:
            return np.pad(im,[(0,0) , int((n-1)*ny/2),int((n-1)*nz/2)], mode='constant')
        
def inpolygon(xq, yq, xv, yv):
        shape = xq.shape
        xq = xq.reshape(-1)
        yq = yq.reshape(-1)
        xv = xv.reshape(-1)
        yv = yv.reshape(-1)
        q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
        p = Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
        return p.contains_points(q).reshape(shape)
    
def interpolateSupport(image,nRes,kind='spline'):
    
    nx,ny = image.shape
    # Define angular frequencies vectors
    if nnp.isscalar(nRes):
        mx = my = nRes
    else:        
        mx = nRes[0]
        my = nRes[1]
                   
            
    if kind == 'nearest':
        tmpReal = scnd.zoom(np.real(image),min([mx/nx,my/ny]),order=0)
        if np.any(np.iscomplex(image)):
            tmpImag = scnd.zoom(np.imag(image),min([mx/nx,my/ny]),order=0)
            return tmpReal + complex(0,1)*tmpImag
        else:
            return tmpReal
    else:        
        
        # Initial frequencies grid    
        if nx%2 == 0:
            uinit = nnp.linspace(-nx/2,nx/2-1,nx)*2/nx
        else:
            uinit = nnp.linspace(-np.floor(nx/2),np.floor(nx/2),nx)*2/nx
        if ny%2 == 0:
            vinit = nnp.linspace(-ny/2,ny/2-1,ny)*2/ny
        else:
            vinit = nnp.linspace(-np.floor(ny/2),np.floor(ny/2),ny)*2/ny    
             
        # Interpolated frequencies grid                  
        if mx%2 == 0:
            unew = nnp.linspace(-mx/2,mx/2-1,mx)*2/mx
        else:
            unew = nnp.linspace(-np.floor(mx/2),np.floor(mx/2),mx)*2/mx
        if my%2 == 0:
            vnew = nnp.linspace(-my/2,my/2-1,my)*2/my
        else:
            vnew = nnp.linspace(-np.floor(my/2),np.floor(my/2),my)*2/my
                   
        # Interpolation
    
        if kind == 'spline':
            # Surprinsingly v and u vectors must be shifted when using
            # RectBivariateSpline. See:https://github.com/scipy/scipy/issues/3164
            xin = np.real(image)
            if gpuEnabled:
                xin = xin.get()
            fun_real = interp.fitpack2.RectBivariateSpline(vinit, uinit, xin)
            if np.any(np.iscomplex(image)):
                xin = np.imag(image)
                if gpuEnabled:
                    xin = xin.get()
                fun_imag = interp.fitpack2.RectBivariateSpline(vinit, uinit, xin)
        else:
            xin = np.real(image)
            if gpuEnabled:
                xin = xin.get()
            fun_real = interp.interp2d(uinit, vinit, xin,kind=kind)
            if np.any(np.iscomplex(image)):
                xin = np.imag(image)
                if gpuEnabled:
                    xin = xin.get()
                fun_imag = interp.interp2d(uinit, vinit, xin,kind=kind)
    
        if np.any(np.iscomplex(image)):
            return np.asarray(fun_real(unew,vnew) + complex(0,1)*fun_imag(unew,vnew))
        else:
            return np.asarray(fun_real(unew,vnew))
            

def normalizeImage(im,normType=1,param=None):
    ''' Returns the normalized PSF :
        normtype = 0 : no normalization
        normType = 1 : Normalization by the sum of pixels
        normtype = 2 : min-max normalization
        normtype = 3 : Normalization by the flux estimates
        normtype = 4 : Normalization by the sum of positive pixels
        normtype > 4 : Normalization by the normType value
        
        If param is provided, the functions does unormalize
    '''

    if param == None:
        "NORMALIZATION"
        if normType == 0:
            param = 1
            im_n  = im
        elif normType == 1:
            param = im.sum()
            im_n  = np.copy(im)/param
        elif normType == 2:
            param = [im.min(),im.max()]
            im_n  = (np.copy(im)-param[0])/(param[1] - param[2])
        elif normType == 3:
            param = abs(getFlux(im))
            im_n  = np.copy(im)/param
        elif normType == 4:
            param = im[im>0].sum()
            im_n  = np.copy(im)/param
        else:
            param = normType
            im_n  = np.copy(im)/normType
        
        return im_n, param
    else:
        "UNORMALIZATION"
        if normType == 0:
            return im
        elif normType == 1:
            return np.copy(im) * param
        elif normType == 2:
            param = [im.min(),im.max()]
            return np.copy(im)*(param[1] - param[2]) + param[0]
        elif normType == 3:
            return np.copy(im) * param
        elif normType == 4:
            return np.copy(im) * param
        else:
            return np.copy(im) * param
    
#%%  IMAGE PROCESSING TOOLS

def addNoise(im,ron,darkBg,skyBg,DIT,nDIT):
    Texp      = nDIT*DIT
    nPts      = im.shape
    im[im<0]  = 0
    noise_det = np.sqrt(nDIT*ron**2 + darkBg*Texp)*np.random.randn(nPts[0],nPts[1])
    noise_ph  = np.random.poisson(im + skyBg*Texp) - skyBg*Texp
    return im + noise_det + noise_ph

def centerPsf(psf,rebin,nargout=1):
    flux        = psf.sum()
    npsfx,npsfy = psf.shape
    npsfx2      = npsfx*rebin
    npsfy2      = npsfy*rebin
            
    # Get the high-resolution PSF
    if rebin > 1:
        psf_hr = interpolateSupport(psf,npsfx2)
    else:
        psf_hr = psf
            
    # Get the max value
    idx,idy = np.unravel_index(psf_hr.argmax(), psf_hr.shape)
    dx      = npsfx2/2-idx
    dy      = npsfy2/2-idy
    # Get the OTF
    otf_hr  = fft.fftshift(psf2otf(psf_hr))
    # Apply the Phasor
    u       = fft.fftshift(fft.fftfreq(otf_hr.shape[0]))
    u,v     = np.meshgrid(u,u)
    fftPhasor = np.exp(-1*complex(0,1)*np.pi*(u*dy+v*dx))
    otf_hr    = otf_hr*fftPhasor
    # Get the PSF low-resolution
    imCor  = otf2psf(otf_hr)
    imCor  = interpolateSupport(imCor,npsfx)
    imCor  = flux*imCor/imCor.sum()
    otf_lr = fft.fftshift(psf2otf(imCor))
    
    if nargout == 1:
        return imCor
    else:
        return imCor,otf_lr

def correctFromDeadPixels(im,badPixFrame):
    # Correcting the bad pixels on the matrix im from the bad pixel frame
            
    npixd = badPixFrame.shape[1]
    imCor = im
    for i in np.arange(0,npixd,1):
        w =  np.sum(badPixFrame[i,2:npixd-1,1])
        if w!=0:
            imCor[badPixFrame[i,0,0]] = np.sum(im[badPixFrame[i,2:npixd,0]]*badPixFrame[i,2:npixd-1,1]) / w;
                            
    return imCor
        
def createDeadPixFrame(badPixelMap):
    # dpframe = createDeadPixFrame(badPixelMap)
    # badPixelMap is the map of dead pixels
    #frame  is the image to be corrected
            
    #The dead pixel is replaced by a weighted average of the neighbours,
    #1 2 1
    #2 X 2
    #1 2 1
    #when they are "available". "Available" means that the sum of the
    #weights of neighbouring pixels must exceeds 4.
            
    #If no neighbouring pixel is available, the dead pixel is not
    #corrected, but a new "dead pixel map" is created, and the function is
    #called once again (recursive calls).
            
    #Get the number of dead pixels
    sx,sy       = badPixelMap.shape
    npixnoncorr = 0
    nnx,nny     = np.where(badPixelMap)
    nn1D        = np.where(badPixelMap[:])
    nDeadPix    = len(nn1D)
    #Instantiation
    tmp          = badPixelMap*0
    frame        = np.zeros(nDeadPix,10,2) #2nd row: #pixel (one pixel + 8 neighbors)
    frame[:,:,0] = 1                    #3rd row: adresses
            
    #loop on Pixel
    for i in np.arange(0,nDeadPix,1):
        nb = 2
        frame[i,0,0] = nn1D[i]  # 1st row = bad pixel
        frame[i,1,0] = 0        # number of used neighbour pixel for correction
        x            = nnx[i]
        y            = nny[i]
        wcum         = 0
                
        # Edges neighbours
        if x>0 and x<=sx and y+1>0 and y+1<=sy:
            if badPixelMap[x,y+1] == 0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i] + sx
                frame[i,nb,1] = 2
                wcum          = wcum + 2
                    
            
                
        if x>0 and x<=sx and y-1>0 and y-1<=sy:
            if badPixelMap[x,y-1] ==0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]-sx
                frame[i,nb,1] = 2
                wcum          = wcum + 2
                    
                
        if x+1>0 and x+1<=sx and y>0 and y<=sy:
            if badPixelMap[x+1,y] ==0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]+1
                frame[i,nb,1] = 2
                wcum          = wcum + 2
                    
        if x-1>0 and x-1<=sx and y>0 and y<=sy:
            if badPixelMap[x-1,y] ==0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]-1
                frame[i,nb,1] = 2
                wcum          = wcum + 2
                                            
        #Diagonal neighbours
        if x+1>0 and x+1<=sx and y+1>0 and y+1<=sy:
            if badPixelMap(x+1,y+1) == 0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]+1+sx
                frame[i,nb,1] = 1
                wcum          = wcum + 1
                    
                
        if x-1>0 and x-1<=sx and y+1>0 and y+1<=sy:
            if badPixelMap[x-1,y+1] == 0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]-1+sx
                frame[i,nb,1] = 1
                wcum          = wcum + 1
                    
                
        if x+1>0 and x+1<=sx and y-1>0 and y-1<=sy:
            if badPixelMap[x+1,y-1]==0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]+1-sx
                frame[i,nb,1] = 1
                wcum          = wcum + 1
                
                
        if x-1>0 and x-1<=sx and y-1>0 and y-1<=sy:
            if badPixelMap[x-1,y-1] ==0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]-1-sx
                frame[i,nb,1] = 1
                wcum          = wcum + 1
                                                
        # Take decision regarding the number of avalaible neighbours
        if wcum<4:   #not enough neigbours
            npixnoncorr          = npixnoncorr + 1
            tmp[x,y]             = tmp(x,y) + 1
            frame[i,2:9,0]       = 1    # pixel adresses set to 1
            frame[i,:,1]         = 0    # weights set to 0
        else:
            frame[i,1,0]         = nb    #number of correcting pixels
                                        
    if npixnoncorr != 0:
        frame_suppl = createDeadPixFrame(tmp)
        nSup        = frame_suppl.shape[0]
        #Frame concatenation
        N                             = nDeadPix+nSup
        new_frame                     = np.zeros(N,10,2);
        new_frame[0:nDeadPix,:,:]     = frame
        new_frame[0+nDeadPix:N-1,:,:] = frame_suppl
        frame                         = new_frame
    
    return new_frame


#%% PSF METRICS
def getEnsquaredEnergy(psf):            
    
    S     = psf.sum()
    nY,nX = psf.shape
    y0,x0 = np.unravel_index(psf.argmax(), psf.shape)
    nEE   = min([nY-y0,nX-x0])
    
    EE = np.zeros(nEE+1)
    for n in range(nEE+1):
        EE[n] = psf[y0 - n:y0+n+1,x0-n:x0+n+1].sum()
    return EE/S

def getEncircledEnergy(psf,pixelscale=1,center=None,nargout=1):            
    
    rr, radialprofile2, ee = radial_profile(psf,ee=True, center=center, pixelscale=pixelscale)
    if nargout==1:
        return ee
    elif nargout == 2:
        return ee,rr


def radial_profile(image,ext=0,pixelscale=1,ee=False,center=None,stddev=False,binsize=None,maxradius=None,
                   normalize='None',pa_range=None,slice=0,nargout=2):
    """ Compute a radial profile of the image.

    This computes a discrete radial profile evaluated on the provided binsize. For a version
    interpolated onto a continuous curve, see measure_radial().

    Code taken pretty much directly from pydatatut.pdf

    Parameters
    ----------
    image : numpy array
    ext : int
        Extension in FITS file
    ee : bool
        Also return encircled energy (EE) curve in addition to radial profile?
    center : tuple of floats
        Coordinates (x,y) of PSF center, in pixel units. Default is image center.
    binsize : float
        size of step for profile. Default is pixel size.
    stddev : bool
        Compute standard deviation in each radial bin, not average?
    normalize : string
        set to 'peak' to normalize peak intensity =1, or to 'total' to normalize total flux=1.
        Default is no normalization (i.e. retain whatever normalization was used in computing the PSF itself)
    pa_range : list of floats, optional
        Optional specification for [min, max] position angles to be included in the radial profile.
        I.e. calculate that profile only for some wedge, not the full image. Specify the PA in degrees
        counterclockwise from +Y axis=0. Note that you can specify ranges across zero using negative numbers,
        such as pa_range=[-10,10].  The allowed PA range runs from -180 to 180 degrees.
    slice: integer, optional
        Slice into a datacube, for use on cubes computed by calc_datacube. Default 0 if a
        cube is provided with no slice specified.

    Returns
    --------
    results : tuple
        Tuple containing (radius, profile) or (radius, profile, EE) depending on what is requested.
        The radius gives the center radius of each bin, while the EE is given inside the whole bin
        so you should use (radius+binsize/2) for the radius of the EE curve if you want to be
        as precise as possible.
    """
        
    if normalize.lower() == 'peak':
        print("Calculating profile with PSF normalized to peak = 1")
        image /= image.max()
    elif normalize.lower() == 'total':
        print("Calculating profile with PSF normalized to total = 1")
        image /= image.sum()


    if binsize is None:
        binsize = pixelscale

    y, x = nnp.indices(image.shape, dtype=float)
    if center is None:
        # get exact center of image
        # center = (image.shape[1]/2, image.shape[0]/2)
        center = tuple((a - 1) / 2.0 for a in image.shape[::-1])

    x -= center[0]
    y -= center[1]

    r = nnp.sqrt(x ** 2 + y ** 2) * pixelscale / binsize  # radius in bin size steps

    if pa_range is None:
        # Use full image
        ind = nnp.argsort(r.flat)
        sr = r.flat[ind]  # sorted r
        sim = image.flat[ind]  # sorted image

    else:
        # Apply the PA range restriction
        pa = nnp.rad2deg(nnp.arctan2(-x, y))  # Note the (-x,y) convention is needed for astronomical PA convention
        mask = (pa >= pa_range[0]) & (pa <= pa_range[1])
        ind = nnp.argsort(r[mask].flat)
        sr = r[mask].flat[ind]
        sim = image[mask].flat[ind]

    ri = sr.astype(int)  # sorted r as int
    deltar = ri[1:] - ri[:-1]  # assume all radii represented (more work if not)
    rind = nnp.where(deltar)[0]
    nr = rind[1:] - rind[:-1]  # number in radius bin
    csim = nnp.nan_to_num(sim).cumsum(dtype=float)  # cumulative sum to figure out sums for each bin
    # nnp.nancumsum is implemented in >1.12
    tbin = csim[rind[1:]] - csim[rind[:-1]]  # sum for image values in radius bins
    radialprofile = tbin / nr

    # pre-pend the initial element that the above code misses.
    radialprofile2 = nnp.empty(len(radialprofile) + 1)
    if rind[0] != 0:
        radialprofile2[0] = csim[rind[0]] / (
                rind[0] + 1)  # if there are multiple elements in the center bin, average them
    else:
        radialprofile2[0] = csim[0]  # otherwise if there's just one then just take it.
    radialprofile2[1:] = radialprofile

    rr = nnp.arange(ri.min(), ri.min()+len(radialprofile2)) * binsize + binsize * 0.5  # these should be centered in the bins, so add a half.


    if maxradius is not None:
        crop = rr < maxradius
        rr = rr[crop]
        radialprofile2 = radialprofile2[crop]

    if stddev:
        stddevs = nnp.zeros_like(radialprofile2)
        r_pix = r * binsize
        for i, radius in enumerate(rr):
            if i == 0:
                wg = nnp.where(r < radius + binsize / 2)
            else:
                wg = nnp.where((r_pix >= (radius - binsize / 2)) & (r_pix < (radius + binsize / 2)))
                # wg = nnp.where( (r >= rr[i-1]) &  (r <rr[i] )))
            stddevs[i] = nnp.nanstd(image[wg])
        return rr, stddevs

    if nargout == 1:
        return radialprofile2
    
    if not ee:
        return rr, radialprofile2
    else:
        ee = csim[rind]
        return rr, radialprofile2, ee
    
            
def getFlux(psf,nargout=1):
    #Define the inner circle
    nx,ny    = psf.shape
    x        = np.linspace(-1,1,nx)
    y        = np.linspace(-1,1,ny)
    X,Y      = np.meshgrid(x,y)
    r        = np.hypot(X,Y)
    msk      = r>1
    #Computing the residual background
    psfNoise = psf*msk
    bg       = np.median(psfNoise)
    #Computing the read-out noise
    ron      = psfNoise.std()
    #Computing the normalized flux
    Flux     = np.sum(psf -bg)
    
    if nargout == 1:
        return Flux
    elif nargout == 2:
        return Flux,ron
    elif nargout == 3:
        return Flux,ron,bg

def getMSE(xtrue,xest,nbox=0,norm='L2'):
    if nbox != 0:
        n   = np.array(xtrue.shape)
        xest = cropSupport(xest,n/nbox)
        xtrue= cropSupport(xtrue,n/nbox)
        
    if norm == 'L2':
        return 1e2*np.sqrt(np.sum((xest-xtrue)**2))/xtrue.sum()
    elif norm == 'L1':
        return 1e2*np.sum(abs(xest-xtrue))/xtrue.sum()
    else:
        print('The input norm={:s} is not recognized, choose L1 or L2'.format(norm))
        return []

def getFWHM(psf,pixelScale,rebin=1,method='contour',nargout=2,center=None,std_guess=2):
            
    # Gaussian and Moffat fitting are not really efficient on
    # anisoplanatic PSF. Prefer the coutour function in such a
    # case. The cutting method is not compliant to PSF not oriented
    #along x or y-axis.

    #Interpolation            
    Ny,Nx = psf.shape
    if rebin > 1:
        im_hr = nnp.asarray(cpuArray(interpolateSupport(psf,rebin*nnp.array([Nx,Ny]))))
    else:
        im_hr = nnp.asarray(psf)
        
    ss = im_hr.shape
    sbx, sby = [[0,ss[1]],[0,ss[0]]]

    peak_val = nnp.max(im_hr)
    # initial guess of PSF center
    y_peak, x_peak = nnp.where(nnp.equal(im_hr, peak_val))
    x00 = sbx[0]
    y00 = sby[0]
    x_peak += x00
    y_peak += y00
    
    if method == 'cutting':
        # Brutal approach when the PSF is centered and aligned x-axis FWHM
        imx     = im_hr[:,y_peak]
        wx      = nnp.where(imx >= imx.max()/2)[0]
        FWHMx   = (wx.max() - wx.min())/rebin*pixelScale
        #y-axis FWHM
        imy     = im_hr[x_peak,:]
        wy      = nnp.where(imy >= imy.max()/2)[1]
        FWHMy   = (wy.max() - wy.min())/rebin*pixelScale
        theta   = 0

    elif method == 'contour':
        # Contour approach~: something wrong about the ellipse orientation
        mpl.interactive(False)
        try:
            plt.figure(666)
            C       = plt.contour(im_hr,levels=[im_hr.max()/2])
            plt.close(666)
            C       = C.collections[0].get_paths()[0]
            C       = C.vertices
            xC      = C[:,0]
            yC      = C[:,1]
            # centering the ellispe
            mx      = nnp.array([xC.max(),yC.max()])
            mn      = nnp.array([xC.min(),yC.min()])
            cent    = (mx+mn)/2
            wx      = xC - cent[0]
            wy      = yC - cent[1] 
            # Get the module
            wr      = nnp.hypot(wx,wy)/rebin*pixelScale                
            # Getting the FWHM
            FWHMx   = 2*wr.max()
            FWHMy   = 2*wr.min()
            #Getting the ellipse orientation
            xm      = wx[wr.argmax()]
            ym      = wy[wr.argmax()]
            theta   = nnp.mean(180*nnp.arctan2(ym,xm)/np.pi)
        except:
            FWHMx = -1 
            FWHMy = -1 
            aRatio= -1
            theta = -1
        mpl.interactive(True)   
    elif method == 'gaussian':
        # initial guess of the FWHM
        fwhm_guess = 2*nnp.sqrt((im_hr > peak_val/2).sum()/np.pi)
        
        stddev_guess = fwhm_guess/(2*nnp.sqrt(2*nnp.log(2)))
        p_init = models.Gaussian2D(amplitude=peak_val, x_mean=x_peak[0], y_mean=y_peak[0],
                                   x_stddev=stddev_guess, y_stddev=stddev_guess)
        
        fit_p = fitting.LevMarLSQFitter()
    
        y,x = nnp.mgrid[sby[0]:sby[1],sbx[0]:sbx[1]]
        g = fit_p(p_init, x, y, im_hr)
        
        FWHMx = 2 * nnp.sqrt(2 * nnp.log(2)) * nnp.abs(g.x_stddev.value*pixelScale)
        FWHMy = 2 * nnp.sqrt(2 * nnp.log(2)) * nnp.abs(g.y_stddev.value*pixelScale)
        theta = g.theta.value*180/np.pi
    elif method == 'moffat':      
        # initial guess of the FWHM
        fwhm_guess = 2*nnp.sqrt((im_hr > peak_val/2).sum()/np.pi)
        
        beta = 4.765    # expected beta value  for Seeing limited PSF
        p_init = models.Moffat2D(amplitude=peak_val, x_0=x_peak[0], y_0=y_peak[0],
                                 gamma=fwhm_guess/(2*nnp.sqrt(2**(1/beta)-1)))
        
        fit_p = fitting.LevMarLSQFitter()
    
        y,x = nnp.mgrid[sby[0]:sby[1],sbx[0]:sbx[1]]
        g = fit_p(p_init, x, y, im_hr)
        
        FWHMx = g.fwhm*pixelScale
        FWHMy = g.fwhm*pixelScale
        theta   = 0
        
    # Get Ellipticity
    aRatio      = nnp.max([FWHMx/FWHMy,FWHMy/FWHMx])
    
    if nargout == 1:
        return 0.5 * (FWHMx+FWHMy)
    elif nargout == 2:
        return FWHMx,FWHMy
    elif nargout == 3:
        return FWHMx,FWHMy,aRatio
    elif nargout == 4:
        return FWHMx,FWHMy,aRatio,theta

def getStrehl(psf0,pupil,samp,recentering=False,nR=5,method='otf'):
    if recentering:    
        psf = centerPsf(psf0,2)
    else:
        psf = psf0

    npsf   = nnp.array(psf.shape)

    # Get the OTF
    otfDL = telescopeOtf(pupil,samp)
        
    if method == 'otf':
        otf     = fft.fftshift(psf2otf(psf))
        otf     = otf/otf.max()
        notf    = nnp.array(otf.shape)
        otfDL   = interpolateSupport(otfDL,notf)
        # Get the Strehl
        SR      = np.real(otf.sum()/otfDL.sum())
    elif method == 'max':
        psfDL  = otf2psf(otfDL)
        psfDL[psfDL<0]  =0
        psfDL *= 1/psfDL.sum()
        psf[psf<0]  =0
        psf   *= 1/psf.sum()
        SR     = psf.max()/psfDL.max()
    else:
        raise ValueError("Method must be 'otf' or 'max'")
        
    return nnp.round(SR,nR)

#%% Data treatment
    
def eqLayers(Cn2, altitudes, nEqLayers, power=5/3):
    '''
             Cn2         ::  The input Cn2 profile (vector)
             altitudes   ::  The input altitudes (vector)
             nEqLayers   ::  The number of output equivalent layers (scalar)
             power       ::  the exponent of the turbulence (default 5/3)
             
             See: Saxenhuber17: Comparison of methods for the reduction of
             reconstructed layers in atmospheric tomography, App Op, Vol. 56, No. 10 / April 1 2017
    '''
    nCn2        = len(Cn2)
    if nEqLayers > nCn2:
        raise ValueError('nEqLayers is larger than the number of input layers')
    nAltitudes  = len(altitudes)
    nSlab       = np.floor(np.round(nCn2)/np.fix(nEqLayers))
             
    posSlab =  np.round((np.linspace(0, nEqLayers-1, num=nEqLayers))*nSlab)
    for iii in range(nEqLayers-1):
        if posSlab[iii] >= posSlab[iii+1]:
            posSlab[iii+1] = posSlab[iii]+1
                              
    posSlab1 = np.concatenate((posSlab, np.asarray([nAltitudes])))
    posSlab2 = cpuArray(posSlab1)
    posSlab = posSlab2.astype(int)

    Cn2eq = np.zeros(nEqLayers)
    altEq = np.zeros(nEqLayers)
    
    for ii in range(nEqLayers):
        Cn2eq[ii] = sum(Cn2[posSlab[ii].item():posSlab[ii+1].item()])
        altEq[ii] = (sum(altitudes[posSlab[ii].item():posSlab[ii+1].item()]**(power) * Cn2[posSlab[ii].item():posSlab[ii+1].item()])/Cn2eq[ii])**(1/power)
       
    return cpuArray(Cn2eq),cpuArray(altEq)

def toeplitz(matrix):
    n , m = matrix.shape
    #diags = [matrix[::-1,:].diagonal(i) for i in range(-n+1,m)]
    diags = [matrix.diagonal(i).sum() for i in range(0,m)]
    
    return diags
    #print( [j.tolist() for j in diags] )
            
#%% Analytical models and fitting facilities
def gaussian(x,xdata):                     
    # ------- Grabbing parameters ---------%
    I0 = x[0]          #Amplitude
    ax = x[1]          #x spreading
    ay = x[2]          #y-spreading
    th = x[3]*np.pi/180  #rotation
    x0 = x[4]          #x-shift
    y0 = x[5]          #y-shift
            
    # ------- Including shifts ---------
    X     = xdata[0]
    Y     = xdata[1]
    #Shifts
    X     = X - x0
    Y     = Y - y0
    #Rotation
    Xr    = X*nnp.cos(th) + Y*nnp.sin(th)
    Yr    = Y*nnp.cos(th) - X*nnp.sin(th)
    # Gaussian expression
    return I0*nnp.exp(-0.5*((Xr/ax)**2 + (Yr/ay)**2) )
