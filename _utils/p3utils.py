#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:57:49 2018

@author: omartin
"""

# Libraries
import numpy as np
import scipy as sp
import scipy.special as spc
import matplotlib.pyplot as plt
import numpy.fft as fft
#%%  C2nh profiling
def compressProfile(Cn2, altitudes, nEqLayers):
    #Cn2         ::  The input Cn2 profile (vector)
    #altitudes   ::  The input altitudes (vector)
    #nEqLayers   ::  The number of output equivalent layers (scalar)
    #power       ::  the exponent of the turbulence (default 5/3)
            
    #See: Saxenhuber17: Comparison of methods for the reduction of
    #reconstructed layers in atmospheric tomography, App Op, Vol. 56, No. 10 / April 1 2017
                
    nCn2       = len(Cn2)
    nAltitudes = len(altitudes)
    power      = 5/3
    if nCn2 == nEqLayers:
        Cn2eq = Cn2
        altEq = altitudes                 
    else:
      
        # Get slab           
        nSlab   = round(nCn2)/np.fix(nEqLayers)
        posSlab = (np.linspace(0, nEqLayers-1, nEqLayers))*nSlab
        posSlab = posSlab.astype(int)
    
        for iS in np.arange(0,nEqLayers-1,1):
            if posSlab[iS] >= posSlab[iS+1]:
                posSlab[iS+1] = posSlab[iS]+1        
                
        posSlab = np.append(posSlab, nAltitudes+1)
        Cn2eq = np.zeros(nEqLayers)
        altEq = np.  zeros(nEqLayers)
    
        # Perform grouping    
        for iS in np.arange(0,nEqLayers,1):
            Cn2eq[iS] = np.sum(Cn2[posSlab[iS]:posSlab[iS+1]-1])
            altEq[iS] = (np.sum(altitudes[posSlab[iS]:posSlab[iS+1]-1]**power * Cn2[posSlab[iS]:posSlab[iS+1]-1])/Cn2eq[iS])**(1/power)
                    
    return Cn2eq , altEq

#%%  FOURIER TOOLS


def cov2sf(cov):
    return 2*cov.max() - cov - np.conjugate(cov)

def fftCorrel(x,y):
    nPts = x.shape
    
    if len(nPts) == 1:
        out =  fft.ifft(fft.fft(x)*np.conjugate(fft.fft(y)))/nPts
    elif len(nPts) == 2:
        out =  fft.ifft2(fft.fft2(x)*np.conjugate(fft.fft2(y)))/(nPts[0]*nPts[1])
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
                        
def pistonFilter(D,f,fm=0,fn=0):    
    f[np.where(f==0)] = 1e-10 
    if len(f.shape) ==1:
        Fx,Fy = np.meshgrid(f,f)            
        FX    = Fx -fm 
        FY    = Fy -fn    
        F     = np.pi*D*np.hypot(FX,FY)    
    else:
        F     = np.pi*D*f
    return 1-(2*spc.j1(F)/F)**2
             
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
    otf  = np.real(fft.fftshift(fftCorrel(E,E)))
    return otf/otf.max()

def pupil2psf(pupil,phase,overSampling):    
    otf = pupil2otf(pupil,phase,overSampling)
    return otf2psf(otf)
      
def sf2otf(sf):
    return np.exp(-0.5 * sf)
                 
def telescopeOtf(pupil,overSampling):    
    extendedPup  = enlargeSupport(pupil,2*overSampling)
    return fft.fftshift(fftCorrel(extendedPup,extendedPup))
           
def telescopePsf(pupil,overSampling,kind='spline'):
    nSize = np.array(pupil.shape)
    
    if overSampling >=1:
        otf = telescopeOtf(pupil,overSampling)
        return otf2psf(interpolateSupport(otf,nSize,kind=kind))
    else:
        otf = interpolateSupport(telescopeOtf(pupil,2),nSize/overSampling,kind=kind)
        return interpolateSupport(otf2psf(otf),nSize,kind=kind)
    

#%%  IMAGE STATISTICS
def getEnsquaredEnergy(psf,eeWidth,overSampling):            
    
    nRes  = np.array(psf.shape)
    nc    = 2*eeWidth*overSampling    
    if nc>1:
        psfEE = cropSupport(psf,nRes/nc)     
        return   np.trapz(np.trapz(psfEE))
    else:
        print('The value of input eeWidth is at a sub-pixel scale. Returning the max value instead')
        return psf.max()
            
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
    psfFlux  = psf*(r<=1)
    Flux     = np.sum(psfFlux -bg)
    
    if nargout == 1:
        return Flux
    elif nargout == 2:
        return Flux,ron
    elif nargout == 3:
        return Flux,ron,bg

def getFVU(xtrue,xest,nbox=0):
    if nbox != 0:
        n   = np.array(xtrue.shape)
        xest = cropSupport(xest,n/nbox)
        xtrue= cropSupport(xtrue,n/nbox)
        
    MSE = np.sum((xest-xtrue)**2)
    VarX= np.sum( (xtrue - xtrue.mean() )**2)
    
    return MSE/VarX     

def getFWHM(psf,pixelScale,rebin=4,method='cutting',nargout=2):
            
    # Gaussian and Moffat fitting are not really efficient on
    # anisoplanatic PSF. Prefer the coutour function in such a
    # case. The cutting method is not compliant to PSF not oriented
    #along x or y-axis.
            
           
    #Interpolation            
    Nx,Ny = psf.shape
    im_hr = interpolateSupport(psf,rebin*np.array([Nx,Ny]))
    
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
        theta   = np.mean(180*np.arctan(ym/xm)/np.pi)
        
        #Angle are counted positively in the reverse clockwise direction.                                 
        
    # Get Ellipticity
    aRatio      = np.max([FWHMx/FWHMy,FWHMy/FWHMx])
    
    if nargout == 1:
        return np.hypot(FWHMx,FWHMy)
    elif nargout == 2:
        return FWHMx,FWHMy
    elif nargout == 3:
        return FWHMx,FWHMy,aRatio
    elif nargout == 4:
        return FWHMx,FWHMy,aRatio,theta
                          
def getStrehl(psf0,pupil,overSampling):    
    psf     = centerPsf(psf0,2)
    #% Get the OTF
    otf     = abs(fft.fftshift(psf2otf(psf)))
    otf     = otf/otf.max()
    notf    = np.array(otf.shape)
    # Get the Diffraction-limit OTF
    otfDL   = abs(telescopeOtf(pupil,overSampling))
    otfDL   = interpolateSupport(otfDL,notf)
    otfDL   = otfDL/otfDL.max()
    # Get the Strehl
    return np.trapz(np.trapz(otf))/np.trapz(np.trapz(otfDL))
            
#%%  IMAGE PROCESSING FACILITIES
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

def gaussian(x,xdata):                     
    # ------- Grabbing parameters ---------%
    I0 = x[0]          #Amplitude
    ax = x[1]          #x spreading
    ay = x[2]          #y-spreading
    th = x[3]*np.pi/180  #rotation
    x0 = x[4]          #x-shift
    y0 = x[5]          #y-shift
            
    # ------- Including shifts ---------
    X     = xdata[0,:,:]
    Y     = xdata[1,:,:]
    #Shifts
    X     = X - x0
    Y     = Y - y0
    #Rotation
    Xr    = X*np.cos(th) + Y*np.sin(th)
    Yr    = Y*np.cos(th) - X*np.sin(th)
    # Gaussian expression
    return I0*np.exp(-0.5*((Xr/ax)**2 + (Yr/ay)**2) )

def interpolateSupport(otf,nRes,kind='spline'):
    # Define angular frequencies vectors
    nx,ny = otf.shape
    
    if np.isscalar(nRes):
        mx = my = nRes
    else:        
        mx = nRes[0]
        my = nRes[1]
               
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
    import scipy.interpolate as interp        

    if kind == 'spline':
        # Surprinsingly v and u vectors must be shifted when using
        # RectBivariateSpline. See:https://github.com/scipy/scipy/issues/3164
        tmpReal = interp.fitpack2.RectBivariateSpline(vinit, uinit, np.real(otf))
        tmpImag = interp.fitpack2.RectBivariateSpline(vinit, uinit, np.imag(otf))
    else:
        tmpReal = interp.interp2d(uinit, vinit, np.real(otf),kind=kind)
        tmpImag = interp.interp2d(uinit, vinit, np.imag(otf),kind=kind)
    
    if np.any(np.iscomplex(otf)):
        return tmpReal(unew,vnew) + complex(0,1)*tmpImag(unew,vnew)
    else:
        return tmpReal(unew,vnew)
            
def  moffat(x,xdata):            
    # ------- Grabbing parameters ---------%
    I0 = x[0]          #Amplitude
    ax = x[1]          #x spreading
    ay = x[2]          #y-spreading
    be = x[3]          #center slope
    th = x[4]*np.pi/180   #rotation
    x0 = x[5]          #x-shift
    y0 = x[6]          #y-shift
                
    # ------- Including shifts ---------
    X     = xdata[0,:,:]
    Y     = xdata[1,:,:]
    #Shifts
    X     = X - x0
    Y     = Y - y0
    #Rotation
    Xr    = X*np.cos(th) + Y*np.sin(th);
    Yr    = Y*np.cos(th) - X*np.sin(th);    
    # Moffat expression
    return I0*(1 + (Xr/ax)**2 + (Yr/ay)**2)**(-be)

def radial(data, center=[0,0]):
    
    if not np.any(center):
        m,n = data.shape       
        center[0] = int(m/2)
        center[1] = int(n/2)
               
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    
    return radialprofile 
            
#%%  PSF-R FACILITIES
def anisoplanatismTransferFunction(r0,L0,cnh,v,wd,nRes,npt,theta,wvlAtm,wvlSrc,P,D,zLgs=0):    
    '''
    r0  = 0.6
    L0  = 25
    cnh = np.array([[0,5e3,10e3],[0.7,0.2,0.1]])
    v   = np.array([5,10,15])
    wd  = np.array([0,np.pi/4,np.pi/2])
    nRes= 101
    npt = 21
    theta = np.array([10,0])/206264.8
    wvlAtm= 1.6e-6
    wvlSrc= 1.6e-6
    D     = 10    
    P     = np.ones((npt,npt))
    x     = np.linspace(-D/2,D/2,npt)
    X,Y   = np.meshgrid(x,x)
    R     = np.hypot(X,Y)
    P     = R<=5
    P     = np.ones((npt,npt))
    zLgs  = 0
    '''        
    # Grab atmospherical parameters
    nL      = cnh.shape[1]
    f0      = 2*np.pi/L0
    h       = cnh[0,:]
    fr0     = cnh[1,:]/sum(cnh[1,:])
    fracR0  = fr0*0.06*(wvlAtm)**2*r0**(-5/3)
    cte     = 0.12184*(2*np.pi/wvlSrc)**2
    thx     = theta[0]
    thy     = theta[1]
    # Phase sample locations in the pupil
    msk     = P.astype(bool)
    #npt2    = np.sum(msk)
    pitch   = D/(npt-1)
    x       = np.arange(-D/2,D/2+pitch,pitch)
    x1,y1   = np.meshgrid(x,x)
    X1      = np.ones((npt**2,1))*x1.T.reshape(-1)
    Y1      = np.tile(y1,[npt,npt])
    # Samples separation in the pupil
    rhoX    = np.transpose(X1.T - x1.T.reshape(-1))
    rhoY    = Y1 - y1.T.reshape(-1)
    # Instantiation   
    I0      = mcDonald(0)
    I1      = mcDonald(f0*np.hypot(rhoX,rhoY))
    def Ialpha(x,y):
        return mcDonald(f0*np.hypot(x,y))
    
    # Anisoplanatism Structure Function   
    Dani = np.zeros((npt**2,npt**2))
    for l in np.arange(0,nL,1):        
        zl    = h[l]
        if zl !=0:
            #Strech magnification factor
            if zLgs:
                g     = zl/zLgs                                 
                I2    = Ialpha(rhoX*(1-g),rhoY*(1-g))
                I3    = Ialpha(np.transpose((1-g)*X1.T - x1.T.reshape(-1))+zl*thx,(1-g)*Y1 - y1.T.reshape(-1)+zl*thy)
                I4    = Ialpha(g*X1-zl*thx,g*Y1-zl*thy)
                I5    = Ialpha(g*X1.T-zl*thx,g*Y1.T-zl*thy) 
                I6    = Ialpha(np.transpose(X1.T - (1-g)*x1.T.reshape(-1))-zl*thx,Y1 - (1-g)*y1.T.reshape(-1)-zl*thy)                 
                Dani  = Dani + cte*L0**(5/3)*fracR0[l]*(2.*I0 - I1 - I2 + I3 - I4 - I5 + I6)                      
            else:            
                I2    = Ialpha(rhoX+zl*thx,rhoY+zl*thy)
                I3    = Ialpha(zl*thx,zl*thy)
                I4    = Ialpha(rhoX-zl*thx,rhoY-zl*thy)
                Dani  = Dani + cte*L0**(5/3)*fracR0[l]*(2*I0 - 2*I1 + I2 - 2*I3  + I4)       
    
    # Tip-tilt filtering if any             

    # DM filtering
                   
    # ATF computation
    otfDL = zonalCovarianceToOtf(0*Dani,nRes,D,pitch,msk)
    ATF   = zonalCovarianceToOtf(-0.5*Dani,nRes,D,pitch,msk)/otfDL
    ATF   = ATF/ATF.max()
    
    return ATF    
    
def getNoiseVariance(s,nfit=0,nshift=1):
                       
    s       = s - s.mean()
    nS,nF   = s.shape
    out     = np.zeros(nS);
            
    if nfit !=0:
        # Polynomial fitting procedure
        delay   = np.linspace(0,1,nfit+1)
        for i in np.arange(0,nS,1):
            #pdb.set_trace()
            g      = fft.ifft(fft.fft(s[i,:])*np.conjugate(fft.fft(s[i,:])))/nF
            mx     = g.max()
            fit    = np.polyfit(delay[1:nfit+1],g[1:nfit+1],nfit)
            yfit   = 0*delay
                    
            for k in np.arange(0,nfit+1,1):
                yfit = yfit + fit[k]*delay**(nfit-k)
                    
            out[i] = mx - yfit[0]
                
        return np.diag(out)  
              
    elif nshift !=0:
        ds  = s - np.roll(s,(0,nshift))
        out = np.dot(s,np.transpose(ds))/nF
        return out
            
def getOLslopes(s,u,MI,dt):    
    return s + MI*(dt*np.roll(u,(-2,2)) + (1-dt)*np.roll(u,(-1,2)))

def getStructureFunction(phi,pupil,overSampling):
    # Create phi and pup an a 2x larger grid
    phi2    = enlargeSupport(phi,2*overSampling)
    pup2    = enlargeSupport(pupil,2*overSampling)
    corp2w  = fftCorrel(phi2**2,pup2)
    corpp   = fftCorrel(phi2,phi2)
    dphi    = corp2w + fftsym(corp2w) - 2*corpp
    corww   = fftCorrel(pup2,pup2)
    # Compute mask of locations where there is overlap between shifted pupils and normalized by the number of points used to compute dphi
    mask    = (corww > 1e-5)
    corww[corww <= 1] = 1
    
    return np.real(fft.fftshift(dphi * mask /corww))
            
def mcDonald(x):
    
    if np.isscalar(x):
        if x==0:
            out = 0.6
        else:
            out = x**(5/6)*spc.kv(5/6,x)/(2**(5/6)*spc.gamma(11/6))
    else:
        out = 0.6*np.ones(x.shape)
        idx = (x.nonzero())
        
        if len(idx):
            out[idx] = x[idx]**(5/6)*spc.kv(5/6,x[idx])/(2**(5/6)*spc.gamma(11/6))
       
        
    return out


def mkotf(indptsc,indptsc2,nU1d,ampl,dp,C_phi):            
    #Instantiation
    otf         = np.zeros((nU1d,nU1d))
    C_phip_diag = np.exp(np.diag(C_phi))
    C_phipi     = np.exp(-2*C_phi)
    C_phi_size2 = C_phi.shape[1]
            
    for iu in np.arange(0,nU1d,1):
        for ju in np.arange(0,nU1d,1):
            indpts  = indptsc[iu, ju]
            indpts2 = indptsc2[iu, ju]
            indpts  = indpts[0]
            indpts2 = indpts2[0]
            if len(indpts)== 0:
                otf[iu,ju] = 0
            else:
                #pdb.set_trace()
                msk        = np.unravel_index(C_phi_size2*indpts + indpts2, C_phipi.shape)
                myarg      = C_phip_diag[indpts2]*C_phip_diag[indpts]*C_phipi[msk]
                kernel     = np.dot(myarg,np.conjugate(ampl[indpts2])*ampl[indpts])
                otf[iu,ju] = kernel*dp**2
                
    dc  = np.sum(abs(ampl[:])**2) * dp**2
    otf = otf/dc;
    return otf    

def mkotf_indpts(nU1d,nPh,u1D,loc,dp):    
    # index pts in a 3x bigger array
    locInPitch = loc/dp
    minLoc     = np.array([locInPitch[0,:].min(),locInPitch[1,:].min()])
    ninloc     = loc.shape[1]
    nc         = 3*nPh
    n          = nc-nPh
    n1         = int((n-n%2)/2 + 1 + nPh%2)
    minLoc2    = minLoc -(n1-1)
    loc2       = np.round(locInPitch.T - np.ones((ninloc,2))*minLoc2+1)
    loc2       = loc2.T
    #embed the loc2 inside ncxnc array.
    indx_emb       = loc2[0,:] + (loc2[1,:]-1)*nc-1
    indx_emb       = indx_emb.astype(int)
    mask           = np.zeros((nc,nc))
    indx_emb       = np.unravel_index(indx_emb,mask.shape)
    mask[indx_emb] = 1
    indptsc        = np.zeros((nU1d,nU1d,), dtype=np.object)
    indptsc2       = np.zeros((nU1d,nU1d,), dtype=np.object)
            
    for iu in np.arange(0,nU1d,1):
        for ju in np.arange(0,nU1d,1):
            u2D        = np.array([u1D[iu],u1D[ju]])
            uInPitch   = u2D/dp
            iniloc_sh2 = np.round(loc2.T + np.ones((ninloc,2))*np.round(uInPitch))
            iniloc_sh2 = iniloc_sh2.T
            iniloc_sh2 = iniloc_sh2.astype(int)
            indxsh     = iniloc_sh2[0,:] + (iniloc_sh2[1,:]-1)*nc-1        
            indxsh     = np.unravel_index(indxsh,mask.shape)            
            #index of points in iniloc_sh2 that are intersect with iniloc_sh
            #indpts is the overlapping points in the shifted array
            indptsc[ju,iu]  = mask[indxsh].nonzero()
            mask2           = np.zeros((nc,nc),dtype=bool)
            mask2[indxsh]   = 1
            indptsc2[ju,iu] = mask2[indx_emb].nonzero()
            
    return indptsc,indptsc2

def modes2Otf(Cmm,modes,pupil,npsf,overSampling=1,method='Vii'):            
    #Autocorrelation of the pupil expressed in pupil
    nPx        = np.sqrt(modes.shape[0])
    pupExtended= enlargeSupport(pupil,2*overSampling)
    fftPup     = fft.fft2(pupExtended)
    conjPupFft = np.conjugate(fftPup)
    G          = fft.fftshift(np.real(fft.fft2(fftPup*conjPupFft)))
    #Defining the inverse
    den        = np.zeros(np.array(G.shape))
    msk        = G/G.max() > 1e-7
    den[msk]   = 1/G(msk)
            
            
    if np.any(Cmm[:]) & method == 'Vii':
        # Diagonalizing the Cvv matrix
        [U,S]   = np.linalgsvd(Cmm)
        s       = np.diag(S)
        nModes  = len(s)
        M       = modes * U
        #loop on actuators                
        buf     = np.zeros(np.array(pupExtended.shape))
                
        for k in sp.arange(1,nModes,1):
            Mk   = np.reshape(M[:,k],nPx,nPx)
            Mk   = enlargeSupport(Mk,2*overSampling)
            # Vii computation
            Vk   = np.real(fft.fft2(Mk**2*pupExtended)*conjPupFft) - abs(fft.fft2(Mk*pupExtended))**2
            # Summing modes into dm basis
            buf  = buf + s[k] * Vk
                        
        dphi     = den*fft.fftshift(np.real(fft.fft2(2*buf)))
        otf      = G*np.exp(-0.5*dphi)
                
                
    elif np.any(Cmm[:]) & method == 'Uij':
        nm   = modes.shape[1]
        dphi = 0*pupExtended
                
        #Double loops on modes
        for i in np.arange(1,nm,1):
            Mi = np.reshape(modes[:,i],nPx,nPx)
            Mi = enlargeSupport(Mi,2*overSampling)
            for j in np.arange(1,i,1):
                #Getting modes + interpolation to twice resolution
                Mj    = np.reshape(modes[:,j],nPx,nPx)
                Mj    = enlargeSupport(Mj,2*overSampling)
                term1 = np.real(fft.fft2(Mi*Mj*pupExtended)*conjPupFft)
                term2 = np.real(fft.fft2(Mi*pupExtended)*np.conjugate(fft.fft2(Mj*pupExtended)))
                # Uij computation
                Uij   = np.real(fft.ifft2(term1-term2))
                #Summing terms
                fact = np.double(i!=j) + 1
                dphi = dphi + fact*Cmm[i,j]*Uij                
                dphi = fft.fftshift(2*dphi)*den*msk
                otf  = G*np.exp(-0.5*dphi)
    else:
        #Diffraction-limit case
        G    = G/G.max()
        otf  = G;
        
            
    # Interpolation of the OTF => determination of the PSF fov
    otf = otf*(G>1e-5)
    otf = interpolateSupport(otf,npsf)
    otf = otf/otf.max()
    return otf

def pointWiseLocation(D,dp,idxValid):
    # Defining the point-wise locations
    xloc                 = np.arange(-D/2,D/2+dp,dp)
    actuLocY, actuLocX   = np.meshgrid(xloc,xloc)
    actuLocX             = actuLocX[idxValid]
    actuLocY             = actuLocY[idxValid]
    return np.array([actuLocX,actuLocY])


def sr2wfe(Strehl,wvl):
    return 1e9*np.sqrt(-np.log(Strehl))*wvl/2/np.pi
    
def wfe2sr(wfe,wvl):
    return np.exp(-(2*np.pi*wfe*1e-9/wvl)**2)

       
def zonalCovarianceToOtf(Cphi,npsf,D,dp,idxValid):            
    # Grabbing the valid actuators positions in meters
    loc  = pointWiseLocation(D,dp,idxValid)
    #OTF sampling
    nPh  = round(D/dp+1)
    #nU1d = 2*nPh;
    u1D  = np.arange(-nPh,nPh,1)*dp
    nU1d = len(u1D)
    # Determining couples of point with the same separation
    shiftX,shiftY = mkotf_indpts(nU1d,nPh,u1D,loc,dp)
    # WF Amplitude   
    amp0 = np.ones((np.count_nonzero(idxValid),1))
    # Long-exposure OTF
    otf = mkotf(shiftX,shiftY,nU1d,amp0,dp,-0.5*Cphi)
    #Interpolation
    otf = interpolateSupport(otf,npsf)
    otf = otf/otf.max()
    return otf

        

    
                
