# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:17:43 2020

@author: omartin
"""
# Libraries
import numpy as np
import scipy.special as spc
import numpy.fft as fft
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import matplotlib as mpl
import scipy.interpolate as interp        
import scipy.ndimage as scnd

#%%  FOURIER TOOLS

def cov2sf(cov):
    return 2*cov.max() - cov - np.conjugate(cov)

def fftCorrel(x,y):
    nPts = x.shape
    
    if len(nPts) == 1:
        out =  fft.ifft(fft.fft(x)*np.conj(fft.fft(y)))/nPts
    elif len(nPts) == 2:
        out =  fft.ifft2(fft.fft2(x)*np.conj(fft.fft2(y)))/(nPts[0]*nPts[1])
    return out

def fftsym(x):
    if x.ndim ==2:
        nx,ny            = x.shape
        if np.any(np.iscomplex(x)):
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
        otf    = interpolateSupport(otf,int(np.round(fovInPixel/nqSmpl)))
        psf    = otf2psf(otf)
        # Interpolate the PSF to set the PSF pixel scale
        psf    = interpolateSupport(psf,fovInPixel)
    return psf
                        
def pistonFilter(D,f,fm=0,fn=0):    
    f[np.where(f==0)] = 1e-10 
    if len(f.shape) ==1:
        Fx,Fy = np.meshgrid(f,f)            
        FX    = Fx -fm 
        FY    = Fy -fn    
        F     = np.pi*D*np.hypot(FX,FY)    
    else:
        F     = np.pi*D*f
        
    out         = np.zeros_like(F)
    idx         = F!=0
    out[idx]    = spc.j1(F[idx])/F[idx]
            
    return 1 - 4 * out**2
             
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
    P    = enlargeSupport(pupil,overSampling)
    phi  = enlargeSupport(phase,overSampling)
    E    = P*np.exp(1*complex(0,1)*phi)    
    otf  = fft.fftshift(fftCorrel(E,E))
    return otf/otf.max()

def pupil2psf(pupil,phase,overSampling):    
    otf = pupil2otf(pupil,phase,overSampling)
    return otf2psf(otf)
      
def sf2otf(sf):
    return np.exp(-0.5 * sf)
                 
def telescopeOtf(pupil,samp):    
    pup_pad  = enlargeSupport(pupil,samp)
    otf      = fft.fftshift(fft.ifft2(fft.fft2(fft.fftshift(pup_pad))**2))
    return otf/otf.max()
           
def telescopePsf(pupil,samp,kind='spline'):
    nSize = np.array(pupil.shape)
    
    if samp >=2:
        otf = telescopeOtf(pupil,samp)
        return otf2psf(interpolateSupport(otf,nSize,kind=kind))
    else:
        otf = interpolateSupport(telescopeOtf(pupil,2),nSize//samp,kind=kind)
        return interpolateSupport(otf2psf(otf),nSize,kind=kind)

#%%  IMAGE PROCESSING
        
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
    # Otf sizes
    nx,ny  = im.shape
    nx2 = int(n*nx)
    ny2 = int(n*ny)
    
    if np.any(np.iscomplex(im)):
        imNew = np.zeros((nx2,ny2)) + complex(0,1)*np.zeros((nx2,ny2))
    else:
        imNew = np.zeros((nx2,ny2))
        
    #Zero-padding    
    if nx2%2 ==0:
        xi = int(0.5*(nx2-nx))
        xf = int(0.5*(nx2 + nx))
    else:
        xi = int(0.5*(nx2-nx))
        xf = int(0.5*(nx2+nx))
        
    if ny2%2 ==0:
        yi = int(0.5*(ny2-ny))
        yf = int(0.5*(ny2 + ny))
    else:
        yi = int(0.5*(ny2-ny))
        yf = int(0.5*(ny2+ny))        
        
            
    imNew[xi:xf,yi:yf] = im
    
    return imNew

def interpolateSupport(image,nRes,kind='spline'):
    
    # Define angular frequencies vectors
    nx,ny = image.shape
        
    if np.isscalar(nRes):
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
            uinit = np.linspace(-nx/2,nx/2-1,nx)*2/nx
        else:
            uinit = np.linspace(-np.floor(nx/2),np.floor(nx/2),nx)*2/nx
        if ny%2 == 0:
            vinit = np.linspace(-ny/2,ny/2-1,ny)*2/ny
        else:
            vinit = np.linspace(-np.floor(ny/2),np.floor(ny/2),ny)*2/ny    
             
        # Interpolated frequencies grid                  
        if mx%2 == 0:
            unew = np.linspace(-mx/2,mx/2-1,mx)*2/mx
        else:
            unew = np.linspace(-np.floor(mx/2),np.floor(mx/2),mx)*2/mx
        if my%2 == 0:
            vnew = np.linspace(-my/2,my/2-1,my)*2/my
        else:
            vnew = np.linspace(-np.floor(my/2),np.floor(my/2),my)*2/my
                   
        # Interpolation
    
        if kind == 'spline':
            # Surprinsingly v and u vectors must be shifted when using
            # RectBivariateSpline. See:https://github.com/scipy/scipy/issues/3164
            tmpReal = interp.fitpack2.RectBivariateSpline(vinit, uinit, np.real(image))
            if np.any(np.iscomplex(image)):
                tmpImag = interp.fitpack2.RectBivariateSpline(vinit, uinit, np.imag(image))
        else:
            tmpReal = interp.interp2d(uinit, vinit, np.real(image),kind=kind)
            if np.any(np.iscomplex(image)):
                tmpImag = interp.interp2d(uinit, vinit, np.imag(image),kind=kind)
    
        if np.any(np.iscomplex(image)):
            return tmpReal(unew,vnew) + complex(0,1)*tmpImag(unew,vnew)
        else:
            return tmpReal(unew,vnew)
            


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

def getEnsquaredEnergy(psf):            
    
    S     = psf.sum()
    nY,nX = psf.shape
    y0,x0 = np.unravel_index(psf.argmax(), psf.shape)
    nEE   = min([nY-y0,nX-x0])
    
    EE = np.zeros(nEE+1)
    for n in range(nEE+1):
        EE[n] = psf[y0 - n:y0+n+1,x0-n:x0+n+1].sum()
    return EE/S

def getEncircledEnergy(psf,pixelscale=1,nargout=1):            
    
    rr, radialprofile2, ee = radial_profile(psf,ee=True,pixelscale=pixelscale)
    if nargout==1:
        return ee
    elif nargout == 2:
        return ee,rr


def radial_profile(image, ext=0, pixelscale=1,ee=False, center=None, stddev=False, binsize=None, maxradius=None,
                   normalize='None', pa_range=None, slice=0,nargout=2):
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

    y, x = np.indices(image.shape, dtype=float)
    if center is None:
        # get exact center of image
        # center = (image.shape[1]/2, image.shape[0]/2)
        center = tuple((a - 1) / 2.0 for a in image.shape[::-1])

    x -= center[0]
    y -= center[1]

    r = np.sqrt(x ** 2 + y ** 2) * pixelscale / binsize  # radius in bin size steps

    if pa_range is None:
        # Use full image
        ind = np.argsort(r.flat)
        sr = r.flat[ind]  # sorted r
        sim = image.flat[ind]  # sorted image

    else:
        # Apply the PA range restriction
        pa = np.rad2deg(np.arctan2(-x, y))  # Note the (-x,y) convention is needed for astronomical PA convention
        mask = (pa >= pa_range[0]) & (pa <= pa_range[1])
        ind = np.argsort(r[mask].flat)
        sr = r[mask].flat[ind]
        sim = image[mask].flat[ind]

    ri = sr.astype(int)  # sorted r as int
    deltar = ri[1:] - ri[:-1]  # assume all radii represented (more work if not)
    rind = np.where(deltar)[0]
    nr = rind[1:] - rind[:-1]  # number in radius bin
    csim = np.nan_to_num(sim).cumsum(dtype=float)  # cumulative sum to figure out sums for each bin
    # np.nancumsum is implemented in >1.12
    tbin = csim[rind[1:]] - csim[rind[:-1]]  # sum for image values in radius bins
    radialprofile = tbin / nr

    # pre-pend the initial element that the above code misses.
    radialprofile2 = np.empty(len(radialprofile) + 1)
    if rind[0] != 0:
        radialprofile2[0] = csim[rind[0]] / (
                rind[0] + 1)  # if there are multiple elements in the center bin, average them
    else:
        radialprofile2[0] = csim[0]  # otherwise if there's just one then just take it.
    radialprofile2[1:] = radialprofile

    rr = np.arange(ri.min(), ri.min()+len(radialprofile2)) * binsize + binsize * 0.5  # these should be centered in the bins, so add a half.


    if maxradius is not None:
        crop = rr < maxradius
        rr = rr[crop]
        radialprofile2 = radialprofile2[crop]

    if stddev:
        stddevs = np.zeros_like(radialprofile2)
        r_pix = r * binsize
        for i, radius in enumerate(rr):
            if i == 0:
                wg = np.where(r < radius + binsize / 2)
            else:
                wg = np.where((r_pix >= (radius - binsize / 2)) & (r_pix < (radius + binsize / 2)))
                # wg = np.where( (r >= rr[i-1]) &  (r <rr[i] )))
            stddevs[i] = np.nanstd(image[wg])
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
        im_hr = interpolateSupport(psf,rebin*np.array([Nx,Ny]))
    else:
        im_hr = psf
        
    if method == 'cutting':
        # Brutal approach when the PSF is centered and aligned x-axis FWHM
        imy     = im_hr[:,int(Ny*rebin/2)]        
        w       = np.where(imy >= imy.max()/2)[0]
        FWHMy   = pixelScale*(w.max() - w.min())/rebin
        #y-axis FWHM
        imx     = im_hr[int(Nx*rebin/2),:]
        w       = np.where(imx >= imx.max()/2)[0]
        FWHMx   = (w.max() - w.min())/rebin*pixelScale
        theta   = 0
    elif method == 'contour':
        # Contour approach~: something wrong about the ellipse orientation
        mpl.interactive(False)
        fig     = plt.figure()
        C       = plt.contour(im_hr,levels=[im_hr.max()/2])
        plt.close(fig)
        C       = C.collections[0].get_paths()[0]
        C       = C.vertices
        xC      = C[:,0]
        yC      = C[:,1]
        # centering the ellispe
        mx      = np.array([xC.max(),yC.max()])
        mn      = np.array([xC.min(),yC.min()])
        cent    = (mx+mn)/2
        wx      = xC - cent[0]
        wy      = yC - cent[1] 
        # Get the module
        wr      = np.hypot(wx,wy)/rebin*pixelScale                
        # Getting the FWHM
        FWHMx   = 2*wr.max()
        FWHMy   = 2*wr.min()
        #Getting the ellipse orientation
        xm      = wx[wr.argmax()]
        ym      = wy[wr.argmax()]
        theta   = np.mean(180*np.arctan2(ym,xm)/np.pi)
        mpl.interactive(True)
    elif method == 'gaussian':
        # Prepare array r with radius in arcseconds
        y, x = np.indices(psf.shape, dtype=float)
        if center is None:
            # Normalize
            psf = psf/psf.max()
            # get exact center of image
            center = tuple((a - 1) / 2.0 for a in psf.shape[::-1])
            x -= center[0]
            y -= center[1]
            Y, X = np.mgrid[:Ny, :Nx]*pixelScale
            std_guess = std_guess*pixelScale
           # Define the model
            g_init = models.Gaussian2D(amplitude=1., x_mean=0, y_mean = 0,x_stddev=std_guess,y_stddev=std_guess)
            g_init.x_mean.fixed = True
            g_init.y_mean.fixed = True
            fit_g = fitting.LevMarLSQFitter()
            # fit x axis
            g = fit_g(g_init, X-center[0], Y-center[1], psf)
            FWHMx = 2 * np.sqrt(2 * np.log(2)) * np.abs(g.x_stddev)
            FWHMy = 2 * np.sqrt(2 * np.log(2)) * np.abs(g.y_stddev)
            
    # Get Ellipticity
    aRatio      = np.max([FWHMx/FWHMy,FWHMy/FWHMx])
    
    if nargout == 1:
        return 0.5 * (FWHMx+FWHMy)
    elif nargout == 2:
        return FWHMx,FWHMy
    elif nargout == 3:
        return FWHMx,FWHMy,aRatio
    elif nargout == 4:
        return FWHMx,FWHMy,aRatio,theta
                          
def getStrehl(psf0,pupil,samp,recentering=False):
    if recentering:    
        psf = centerPsf(psf0,2)
    else:
        psf = psf0
        
    #% Get the OTF
    otf     = abs(fft.fftshift(psf2otf(psf)))
    otf     = otf/otf.max()
    notf    = np.array(otf.shape)
    
    # Get the Diffraction-limit OTF
    nX,nY   = pupil.shape
    pup_pad = enlargeSupport(pupil,samp)
    otfDL   = fft.fftshift(abs(fft.ifft2(fft.fft2(fft.fftshift(pup_pad))**2)/pupil.sum()))
    otfDL   = interpolateSupport(otfDL,notf)
    otfDL   = otfDL/otfDL.max()
    
    # Get the Strehl
    return np.round(otf.sum()/otfDL.sum(),2)

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
    nAltitudes  = len(altitudes)
    nSlab       = np.floor(np.round(nCn2)/np.fix(nEqLayers))
             
    posSlab =  np.round((np.linspace(0, nEqLayers-1, num=nEqLayers))*nSlab)
    for iii in range(nEqLayers-1):
        if posSlab[iii] >= posSlab[iii+1]:
            posSlab[iii+1] = posSlab[iii]+1
                              
    posSlab = np.concatenate((posSlab, [nAltitudes]))
    posSlab = posSlab.astype('b')
    Cn2eq = np.zeros(nEqLayers)
    altEq = np.zeros(nEqLayers)
    
    for ii in range(nEqLayers):
        Cn2eq[ii] = sum(Cn2[posSlab[ii]:posSlab[ii+1]])
        altEq[ii] = (sum(altitudes[posSlab[ii]:posSlab[ii+1]]**(power) * Cn2[posSlab[ii]:posSlab[ii+1]])/Cn2eq[ii])**(1/power)
       
    return Cn2eq,altEq


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
    Xr    = X*np.cos(th) + Y*np.sin(th)
    Yr    = Y*np.cos(th) - X*np.sin(th)
    # Gaussian expression
    return I0*np.exp(-0.5*((Xr/ax)**2 + (Yr/ay)**2) )