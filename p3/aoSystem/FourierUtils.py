# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:17:43 2020

@author: omartin
"""

# Libraries
import numpy as nnp
from . import gpuEnabled, np, nnp, scnd, RectBivariateSpline, fft, spc, cpuArray
from . import gpuEnabled, np, nnp, scnd, RectBivariateSpline, fft, spc, cpuArray

import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import matplotlib as mpl
import scipy.ndimage as scnd
from matplotlib.path import Path
from scipy.interpolate import interp1d, CubicSpline
from scipy import ndimage

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

def freq_array(nX,L=1,offset=1e-10):
    k2D = np.mgrid[0:nX, 0:nX].astype(float)
    k2D[0] -= nX//2
    k2D[1] -= nX//2
    k2D     *= np.asarray(L)
    k2D     += offset
    return k2D[0],k2D[1]

def getStaticOTF(tel, nOtf, samp, wvl, xStat=[], theta_ext=0, spatialFilter=1):
    """
    Returns the instrumental OTF including the static aberration and the
    diffraction-limited OTF.
    """
    # DEFINING THE RESOLUTION/PUPIL
    nPup = tel.pupil.shape[0]

    # ADDING STATIC MAP
    phaseStat = np.zeros((nPup,nPup))
    if not tel.opdMap_on is None and nnp.any(tel.opdMap_on):
        if theta_ext:
            tel.opdMap_on = scnd.rotate(tel.opdMap_on,theta_ext,reshape=False)
        phaseStat = (2*np.pi*1e-9/wvl) * tel.opdMap_on

    # ADDING USER-SPECIFIED STATIC MODES
    phaseMap = 0
    xStat = np.asarray(xStat)
    if not tel.statModes is None and nnp.any(tel.statModes):
        if tel.statModes.shape[2]==xStat.size:
            phaseMap = 2*np.pi*1e-9/wvl * np.sum(tel.statModes*xStat,axis=2)
            phaseStat += phaseMap

    # FILTERING
    if not nnp.isscalar(spatialFilter):
        phaseStat = (np.dot(spatialFilter,phaseStat.reshape(-1))).reshape((nPup,nPup))

    # INSTRUMENTAL OTF
    otfStat = pupil2otf(tel.pupil * tel.apodizer, phaseStat, samp)
    if not otfStat is None and nnp.any(otfStat.shape!=nOtf):
        otfStat = interpolateSupport(otfStat,nOtf)
    otfStat /= otfStat.max()

    # DIFFRACTION-LIMITED OTF
    if nnp.all(phaseStat == 0):
        otfDL = otfStat
    else:
        otfDL = np.real(pupil2otf(tel.pupil * tel.apodizer, 0*phaseStat, samp))
        if nnp.any(otfDL.shape !=nOtf):
            otfDL = interpolateSupport(otfDL, nOtf)
            otfDL/= otfDL.max()

    return otfStat, otfDL, phaseMap

def instantiateAngularFrequencies(nOtf,fact=2):
    # DEFINING THE DOMAIN ANGULAR FREQUENCIES
    U_,V_  = shift_array(nOtf, nOtf, fact=fact) #from -1 to 1
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

def Ialpha(x, y):
    return mcDonald(np.hypot(x, y))

def _rfft2_to_fft2(im_shape, rfft):
    '''
        Returns the fft2 from the rfft2 array
    '''
    fcols = im_shape[-1]
    fft_cols = rfft.shape[-1]

    result = np.zeros(im_shape, dtype=rfft.dtype)

    result[:, :fft_cols] = rfft

    top = rfft[0, 1:]

    if fcols%2 == 0:
        result[0, fft_cols-1:] = top[::-1].conj()
        mid = rfft[1:, 1:]
        mid = np.hstack((mid, mid[::-1, ::-1][:, 1:].conj()))
    else:
        result[0, fft_cols:] = top[::-1].conj()
        mid = rfft[1:, 1:]
        mid = np.hstack((mid, mid[::-1, ::-1].conj()))

    result[1:, 1:] = mid

    return result


def otf2psf(otf,psfInOnePix=False):
    nX,nY   = otf.shape
    u1d     = fft.fftshift(fft.fftfreq(nX))
    v1d     = fft.fftshift(fft.fftfreq(nY))
    u2d,v2d = np.meshgrid(u1d,v1d)
    
    if nX%2 == 0 and psfInOnePix==False:
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
    R         = sombrero(1,F)
    pFilter   =  1 - 4 * R**2
    pFilter[np.where(pFilter<0)] = 0
    return pFilter

def phase_to_structure_function(phi, pupil, samp, threshold=1e-5):
    """
    Returns the N x N phase structure function from a N x N phase map
    """
    # Create phi and pup an a 2x larger grid
    phi2 = enlargeSupport(phi,samp)
    pup2 = enlargeSupport(pupil,samp)
    corp2w = fftCorrel(phi2**2,pup2)
    corpp = fftCorrel(phi2,phi2)
    dphi = corp2w + fftsym(corp2w) - 2*corpp
    corww = fftCorrel(pup2,pup2)
    # Compute mask of locations where there is overlap between shifted pupils and normalized by the number of points used to compute dphi
    mask = (corww > threshold)
    corww[corww <= 1] = 1

    return np.real(fft.fftshift(dphi * mask /corww))

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

def shift_array(nX, nY, fact=2*np.pi*complex(0,1)):
    X, Y = np.mgrid[0:nX,0:nY].astype(float)
    X = (X-nX/2) * fact/nX
    Y = (Y-nY/2) * fact/nY
    return X,Y

def sombrero(n,x):
    x = np.asarray(x)
    if n==0:
        return spc.jv(0,x)/x
    else:
        if n>1:
            out = np.zeros(x.shape)
        else:
            out = 0.5*np.ones(x.shape)

        out = np.zeros_like(x)
        idx = x!=0
        out[idx] = spc.j1(x[idx])/x[idx]
        return out


def sort_params_from_labels(psfModelInst, x0):
    '''
        Returns lists of parameters for the PSF model, static aberrations and the object
        wrt psfModelInst.param_labels and x0
    '''

    xall = x0

    # ---------- MANAGING THE CN2 PROFILE
    if psfModelInst.n_param_atm>0:
        # atmospheric parameters are included into the model
        nL = psfModelInst.ao.atm.nL
        if nL>1 and psfModelInst.ao.dms.nRecLayers>1:
            # N-LAYERS CASE : FIT OF CN2 in METERS**(-5/3)
            Cn2 = np.asarray(x0[:nL])
            r0 = np.sum(Cn2)**(-3/5)
        else:
            # 1-LAYER CASE : FIT OF r0 in METERS
            Cn2= None
            r0 = x0[0]
            nL = 1
    else:
        nL = 0
        r0 = None
        Cn2 = None

    # ---------- MANAGING THE PARAMETERS FOR DPHI
    n_dphi = nL + psfModelInst.n_param_dphi
    if n_dphi > 0:
        x0_dphi = list(xall[nL:n_dphi])
    else:
        x0_dphi = None

    # ---------- MANAGING THE JITTER
    if "jitterX" in psfModelInst.param_labels:
        if len(x0) > n_dphi:
            x0_jitter = list(xall[n_dphi:n_dphi+3])
        else:
            x0_jitter = psfModelInst.ao.cam.spotFWHM[0]
        n_tt = n_dphi+3
    else:
        x0_jitter = None
        n_tt = n_dphi

    # Astrometry/Photometry/Background
    n_star = psfModelInst.ao.src.nSrc
    n_wvl = psfModelInst.nwvl
    n_frame = len(psfModelInst.ao.src.wvl)
    n_src = n_star*n_wvl
    n_stellar = n_tt + n_src*3 + n_frame

    if len(x0) > n_tt:
        x0_stellar = np.array(xall[n_tt:n_stellar])
        # unpacking values
        if n_wvl == 1:
            F = x0_stellar[:n_star][:,np.newaxis] * np.array(psfModelInst.ao.cam.transmittance)
            dx = x0_stellar[n_star:2*n_star][:,np.newaxis] + np.array(psfModelInst.ao.cam.dispersion[0])
            dy = x0_stellar[2*n_star:3*n_star][:,np.newaxis] + np.array(psfModelInst.ao.cam.dispersion[1])
        else:
            F = x0_stellar[0:n_src].reshape((n_star, n_wvl))
            dx = x0_stellar[n_src:2*n_src].reshape((n_star, n_wvl))
            dy = x0_stellar[2*n_src:3*n_src].reshape((n_star, n_wvl))
        bkg = x0_stellar[3*n_src:]

    else:
        # intantiating objects properties : flux, 2D positions and image background
        vect_ones = np.ones(n_wvl)
        F = np.repeat(np.array(psfModelInst.ao.cam.transmittance)[np.newaxis,:]*vect_ones , n_star, axis=0)
        dx = np.repeat(np.array(psfModelInst.ao.cam.dispersion[0])[np.newaxis,:]*vect_ones, n_star, axis=0)
        dy = np.repeat(np.array(psfModelInst.ao.cam.dispersion[1])[np.newaxis,:]*vect_ones, n_star, axis=0)
        bkg = np.array(np.zeros(1)*vect_ones)

    x0_stellar = [F, dx, dy, bkg]

    # Static aberrations
    if len(x0) > n_stellar:
        x0_stat = list(x0[n_stellar:])
    else:
        x0_stat = None

    return (Cn2, r0, x0_dphi, x0_jitter, x0_stellar, x0_stat)

def telescopeOtf(pupil,samp):
    if samp >1:
        pup_pad = enlargeSupport(pupil,samp)
        otf = fft.fftshift(fft.ifft2(fft.fft2(fft.fftshift(pup_pad))**2))
    else:
        factor = nnp.ceil(1/samp)
        pup_pad = enlargeSupport(pupil,samp*factor)
        otf = fft.fftshift(fft.ifft2(fft.fft2(fft.fftshift(pup_pad))**2))
        otf = interpolateSupport(otf,float(otf.shape[0]/factor),kind='bilinear')
    return otf/otf.max()


def telescopePsf(pupil,samp,kind='spline'):
    nSize = nnp.array(pupil.shape)


    if samp >=2:
        otf = telescopeOtf(pupil,samp)
        return otf2psf(interpolateSupport(otf,nSize,kind=kind))
    else:
        otf = interpolateSupport(telescopeOtf(pupil,2),nSize//samp,kind=kind)
        return interpolateSupport(otf2psf(otf),nSize,kind=kind)

def sf_3D_to_psf_3D(sf, freq, ao, x_jitter=[0, 0, 0], x_stat=None,
                    x_stellar=[[1.0], [0.],[0.],[0]],
                    theta_ext=0, nPix=None, otfPixel=1):
        """
          Computation of the 3D PSF and the Strehl-ratio (from the OTF integral).
          The Phase structure function must be a nPx x nPx x nSrc array
          given in rad^2
        """

        # GETTING THE OBJECT PARAMETERS
        F = x_stellar[0]
        dx = x_stellar[1]
        dy = x_stellar[2]
        bkg = x_stellar[3]

        # INSTANTIATING THE OUTPUTS
        if nPix == None:
            nPix = int(freq.nOtf /freq.kRef_)

        PSF = np.zeros((nPix,nPix,ao.src.nSrc,freq.nWvl))
        SR  = np.zeros((ao.src.nSrc,freq.nWvl))

        # DEFINING THE RESIDUAL JITTER KERNEL
        Kjitter = 1
        if x_jitter is not None:
            u_max = freq.samp/2*ao.tel.D/freq.wvl/(3600*180*1e3/np.pi)
            norm_fact = u_max *(2*np.sqrt(2*np.log(2)))
            Djitter = norm_fact**2 * (x_jitter[0]*freq.U2_
                                    + x_jitter[1]*freq.V2_
                                    + 2*x_jitter[2]*freq.UV_)
            Kjitter = np.exp(-0.5 * Djitter)

        # DEFINE THE FFT PHASOR AND MULTIPLY TO THE TELESCOPE OTF
        fftPhasor = np.ones((freq.nOtf, freq.nOtf, ao.src.nSrc), dtype=complex)
        if np.any(dx) or np.any(dy):
            # accounting for the binning
            bin_fact = 1
            if freq.kRef_ > 1:
                bin_fact = freq.kRef_
            # computing the phasor
            dr = bin_fact*(freq.U_[:,:,np.newaxis]*dx + freq.V_[:,:,np.newaxis]*dy)
            fftPhasor = np.exp(-np.pi*complex(0, 1)*dr)

        # INSTRUMENTAL DEFECTS
        if x_stat is not None:
            freq.otfNCPA, _, freq.phaseMap = getStaticOTF(ao.tel, freq.nOtf,
                                                          freq.sampRef,
                                                          freq.wvlRef,
                                                          xStat=x_stat)
        # OTF MULTIPLICATION
        otfStat = freq.otfNCPA * Kjitter * otfPixel
        otfStat = np.repeat(otfStat[:, :, np.newaxis], ao.src.nSrc, axis=2)
        otfTurb = np.exp(-0.5*sf)
        otfTot = fft.fftshift(otfTurb * otfStat * fftPhasor, axes=(0, 1))

        # GET THE FINAL PSF - PIXEL SCALE IS NYQUIST - FOV DIFFERENT PER WVL
        psf_ = np.real(fft.fftshift(fft.ifftn(otfTot,axes=(0,1)),axes = (0,1)))

        # managing the undersampling
        psf = np.copy(psf_)

        if freq.kRef_ >= 1: # binning the PSF
            psf = np.zeros((ao.cam.fovInPix,ao.cam.fovInPix,ao.src.nSrc))
            nC = freq.kRef_
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
        PSF = psf/psf.sum(axis=(0,1)) * F
        SR = 1e2*np.abs(otfTot).sum(axis=(0,1))/np.real(freq.otfDL.sum())

        return PSF + bkg, SR

def sf_3D_to_psf_4D(sf, freq, ao, x_jitter=None, x_stat=None,
                    x_stellar = [[1.0], [0.],[0.],[0]],
                    theta_ext = 0, nPix=None, otfPixel=1):
        """
          Computation of the 4D PSF and the Strehl-ratio (from the OTF integral).
          The Phase structure function must be a nPx x nPx x nSrc array
          given in rad^2
        """

        # GETTING THE OBJECT PARAMETERS
        F = x_stellar[0]
        dx = x_stellar[1]
        dy = x_stellar[2]
        bkg = x_stellar[3]

        # INSTANTIATING THE OUTPUTS
        if nPix == None:
            nPix = int(freq.nOtf /freq.kRef_)

        PSF = np.zeros((nPix,nPix,ao.src.nSrc,freq.nWvl))
        SR  = np.zeros((ao.src.nSrc,freq.nWvl))

        # DEFINING THE RESIDUAL JITTER KERNEL
        Kjitter = 1
        if np.any(x_jitter[0:2]):
            u_max = freq.samp*ao.tel.D/freq.wvl/(3600*180*1e3/np.pi)
            norm_fact = np.max(u_max)**2 *(2 * np.sqrt(2*np.log(2)))**2
            Djitter = norm_fact * (x_jitter[0]**2 * freq.U2_
                                   + x_jitter[1]**2 * freq.V2_
                                   + 2*x_jitter[2] *freq.UV_)
            Kjitter = np.exp(-0.5 * Djitter)

        # DEFINE THE FFT PHASOR AND MULTIPLY TO THE TELESCOPE OTF
        if np.any(dx!=0) or np.any(dy!=0):
            # accounting for the binning
            bin_fact = 1
            if freq.kRef_ > 2:
                bin_fact = freq.kRef_

            # computing the phasor
            if freq.nWvl > 1:
                # instantiating the phasor
                fftPhasor = np.zeros((freq.nOtf,freq.nOtf,ao.src.nSrc,freq.nWvl),dtype=complex)
                for iSrc in range(ao.src.nSource):
                    for jWvl in range(freq.nWvl):
                        # account for the binning
                        dr = bin_fact*(dx[iSrc, jWvl]*freq.U_ + dy[iSrc, jWvl]*freq.V_)
                        fftPhasor[:, :, iSrc, jWvl] = np.exp(-np.pi*complex(0, 1)*dr)
            else:
                dr = bin_fact*(freq.U_[:,:,np.newaxis]*dx + freq.V_[:,:,np.newaxis]*dy)
                fftPhasor = np.exp(-np.pi*complex(0, 1)*dr)[:,:,:,np.newaxis]
        else:
            fftPhasor = np.ones((freq.nOtf,freq.nOtf,ao.src.nSrc,freq.nWvl),dtype=complex)

        # LOOP ON WAVELENGTHS
        for jWvl in range(freq.nWvl):

            # UPDATE THE INSTRUMENTAL OTF
            if freq.nWvl>1 or x_stat is not None:
                freq.otfNCPA, _, _ =  getStaticOTF(ao.tel, freq.nOtf,
                                                   freq.samp[jWvl],
                                                   freq.wvl[jWvl],
                                                   xStat=x_stat,
                                                   theta_ext=theta_ext)

            # UPDATE THE RESIDUAL JITTER
            if freq.nyquistSampling == True and freq.nWvl > 1 and x_jitter is not None:
                norm_fact2 = (freq.samp[jWvl]*ao.tel.D/freq.wvl[jWvl]/(3600*180*1e3/np.pi))**2
                norm_fact2 *= (2 * np.sqrt(2*np.log(2)))**2
                Kjitter = np.exp(-0.5 * Djitter * norm_fact2/norm_fact)

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
                nC = freq.k_[jWvl]
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
            PSF[:,:,:,jWvl] = psf/psf.sum(axis=(0,1))  * F[:,jWvl]

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

def create_wavelength_vector(ao):
    '''
        Returns the the vector containing all wavelengths from the  aoSystem
        object setup. The function accounts for the wavelength of the ao.src
        object as well as the science dector bandwidth.
        INPUTS:
            - ao, an aoSystem object
        OUTPUTS:
            - wvl, the vector of wavelengths. If wvl_cen is the vector of
            central wavelengths (sources) and bw the detector bandwidth, we have
            wvl = [wvl_cen[0]-bw/2, ... wvl_cen[0]+bw/2 ... wvl_cen[1]-bw/2 ---]
            - nwvl, the number of wavelengths
    '''
    # grabbing information about the detector
    n_bin = ao.cam.nWvl
    wvl_cen = np.unique(ao.src.wvl)
    n_cen = len(wvl_cen)

    # grabbing information about the wavelength of the point-sources
    cam_bw = ao.cam.bandwidth

    # getting the range of wavelengths
    w_min = wvl_cen - cam_bw/2
    w_max = wvl_cen + cam_bw/2

    # creating the vector of wavelengths
    nwvl = n_bin * n_cen
    wvl = np.zeros(nwvl)
    for j in range(n_cen):
        wvl[j:(j+1)*n_bin] = np.linspace(w_min[j], w_max[j], num=n_bin)

    return wvl, nwvl

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


def enlargeSupport(cube, n):

    if np.ndim(cube)==2:
        nx,ny = cube.shape
        return np.pad(cube, [int((n-1)*nx/2), int((n-1)*ny/2)])
    elif np.ndim(cube)==3:
        nx, ny, nz = cube.shape
        if nx==ny:
            n1 = int((n-1)*nx/2)
            n2 = int((n-1)*ny/2)
            return np.pad(cube, [(n1, n1), (n2, n2), (0,0)])
        elif ny==nz:
            n1 = int((n-1)*ny/2)
            n2 = int((n-1)*nz/2)
            return np.pad(cube,[(0,0) , (n1, n1), (n2, n2)])

def inpolygon(xq, yq, xv, yv):
        shape = xq.shape
        xq = xq.reshape(-1)
        yq = yq.reshape(-1)
        xv = xv.reshape(-1)
        yv = yv.reshape(-1)
        q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
        p = Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
        return p.contains_points(q).reshape(shape)
    
def interpolateSupport(image, n_out, kind='spline'):

    n_x,n_y = image.shape
    # Define angular frequencies vectors
    if nnp.isscalar(n_out):
        m_x = m_y = int(n_out)
    else:
        m_x = int(n_out[0])
        m_y = int(n_out[1])

    if n_x==m_x and n_y==m_y:
        return image

    if kind=="nearest":
        tmpReal = scnd.zoom(np.real(image),min([m_x/n_x,m_y/m_y]),order=0)
        if np.any(np.iscomplex(image)):
            tmpImag = scnd.zoom(np.imag(image),min([m_x/n_x,m_y/m_y]),order=0)
            return tmpReal + complex(0,1)*tmpImag
        else:
            return tmpReal
    else:
        # Initial frequencies grid
        if n_x%2 == 0:
            uinit = nnp.linspace(-n_x/2,n_x/2-1,n_x)*2/n_x
        else:
            uinit = nnp.linspace(-nnp.floor(n_x/2),nnp.floor(n_x/2),n_x)*2/n_x
        if n_y%2 == 0:
            vinit = nnp.linspace(-n_y/2,n_y/2-1,n_y)*2/n_y
        else:
            vinit = nnp.linspace(-nnp.floor(n_y/2),nnp.floor(n_y/2),n_y)*2/n_y

        # Interpolated frequencies grid
        if m_x%2 == 0:
            unew = nnp.linspace(-m_x/2,m_x/2-1,m_x)*2/m_x
        else:
            unew = nnp.linspace(-nnp.floor(m_x/2),nnp.floor(m_x/2),m_x)*2/m_x
        if m_y%2 == 0:
            vnew = nnp.linspace(-m_y/2,m_y/2-1,m_y)*2/m_y
        else:
            vnew = nnp.linspace(-nnp.floor(m_y/2),nnp.floor(m_y/2),m_y)*2/m_y

        # Interpolation

        if kind=="spline":
            # Surprinsingly v and u vectors must be shifted when using
            # RectBivariateSpline. See:https://github.com/scipy/scipy/issues/3164
            xin = np.real(image)
            fun_real = RectBivariateSpline(vinit, uinit, cpuArray(xin))
            if nnp.any(np.iscomplex(image)):
                xin = np.imag(image)
                fun_imag = RectBivariateSpline(vinit, uinit, cpuArray(xin))
        else:
            xin = np.real(image)
            fun_real = RectBivariateSpline(uinit, vinit, cpuArray(xin), kx=1, ky=1)
            if nnp.any(np.iscomplex(image)):
                xin = np.imag(image)
                fun_imag = RectBivariateSpline(uinit, vinit, cpuArray(xin), kx=1, ky=1)

        if np.any(np.iscomplex(image)):
            return np.asarray(fun_real(unew,vnew) + complex(0,1)*fun_imag(unew,vnew))
        else:
            return np.asarray(fun_real(unew,vnew))



def normalizeImage(im, normType=1, param=None):
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
            param = np.clip(im, 0, None).sum()
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
    
    rr, radialprofile2, ee = radial_profile(psf,ee=True,center=center,pixelscale=pixelscale)
    if nargout==1:
        return ee
    elif nargout == 2:
        return ee,rr


def radial_profile(image, ext=0, pixelscale=1,ee=False, center=None, stddev=False, binsize=None, maxradius=None,
                   normalize='None', pa_range=None, nargout=2, supersamp=None, slice=slice, 
                   polar_grid=None, r_vals=None, verbose=False):
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
    supersamp : [float, int], optional
        Step size and interpolation flag (1: 1D cubic, 2: 2D polar grid).

    Returns
    --------
    results : tuple
        Tuple containing (radius, profile) or (radius, profile, EE) depending on what is requested.
        The radius gives the center radius of each bin, while the EE is given inside the whole bin
        so you should use (radius+binsize/2) for the radius of the EE curve if you want to be
        as precise as possible.
    """

    # === Normalize image ===
    if normalize.lower() == 'peak':
        if verbose:
            print("Calculating profile with PSF normalized to peak = 1")
        image /= image.max()
    elif normalize.lower() == 'total':
        if verbose:
            print("Calculating profile with PSF normalized to total = 1")
        image /= image.sum()

    if binsize is None:
        binsize = pixelscale

    if center is None:
        # get exact center of image
        # center = (image.shape[1]/2, image.shape[0]/2)
        center = tuple((a - 1) / 2.0 for a in image.shape[::-1])

    # === Handle supersamp flag ===
    if supersamp is not None:
        try:
            step_interp, interp_flag = supersamp
        except ValueError:
            raise ValueError("supersamp must be a tuple of (step, interp_flag)")
        interp_flag = int(interp_flag)

        # ===== Supersampling - Polar 2D =====
        if interp_flag == 2: 
            if polar_grid is not None and r_vals is not None:
                r_vals, profile = interpolate_2d(image, polar_grid, r_vals, pixelscale, step_interp)
            else:
                raise ValueError("interp_flag=2 requires precomputed polar_grid and r_vals")
            if verbose:
                print(f"Radial profile:  2D polar interpolation to {step_interp:.2f} mas/pixel")  
            if nargout == 1: return profile
            if ee: 
                ee_curve = nnp.cumsum(profile * nnp.gradient(r_vals))
                ee_curve /= ee_curve[-1]
                return r_vals, profile, ee_curve
            return r_vals, profile

    # === Build radial coordinate array ===
    y, x = nnp.indices(image.shape, dtype=float)
    x -= center[0]
    y -= center[1]

    r = nnp.sqrt(x ** 2 + y ** 2) * pixelscale / binsize  # radius in bin size steps

    image = cpuArray(image)
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

    # DISCRETE BINNING: int conversion of radii
    ri = nnp.round(sr).astype(int)  # sorted r as int
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

    rr = nnp.arange(r.min(), r.min()+len(radialprofile2)) * binsize
    if maxradius is not None:
        crop = rr < maxradius
        rr = rr[crop]
        radialprofile2 = radialprofile2[crop]

    # ---- Stddev Option ----
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


    # === Supersampling 1D (after profile) ===
    if supersamp and interp_flag == 1:
        r_interp, p_interp = interpolate_1d(rr, radialprofile2, pixelscale, step_interp)
        if verbose:
            print(f"Radial profile: 1D interpolation to {step_interp:.2f} mas/pixel")
        if nargout == 1: return p_interp      
        if ee:
            ee_curve = nnp.cumsum(p_interp * nnp.gradient(r_interp))
            ee_curve /= ee_curve[-1]
            return r_interp, p_interp, ee_curve
        return r_interp, p_interp

    # === Return discrete version ===
    if nargout == 1:
        return radialprofile2
    if ee:
        ee = csim[rind]
        ee /= nnp.max(ee)
        return rr, radialprofile2, ee
    return rr, radialprofile2


def interpolate_1d(r,profile, pixelscale, step):
    """
    Interpolate a 1D radial profile using cubic splines.

    Parameters
    ----------
    r : ndarray
        Original radius values.
    profile : ndarray
        Corresponding profile values.
    step : float
        Desired output sampling step.

    Returns:
    r_interp : ndarray
        Interpolated radius values.
    p_interp : ndarray
        Interpolated profile.
    """
    r_interp = nnp.arange(r[0], r[-1], step)
    spline = CubicSpline(r, profile)
    p_interp = spline(r_interp)
    p_interp = nnp.clip(p_interp, 0, None)
    # === Scale to compensate for higher sampling resolution ===
    scaling = (step/pixelscale)**2
    p_interp *= scaling

    return r_interp, p_interp

def interpolate_2d(img, grid, r_vals, pixelscale, step):
    """
    Interpolate a 2D image on a precomputed polar grid to compute its radial profile.

    Parameters
    ----------
    img : ndarray
        2D image to interpolate.
    grid : list of (xp, yp)
        List of polar sampling coordinates (in pixels), one for each radius.
    r_vals : ndarray
        Corresponding radial positions
    pixelscale : float
        Physical size of one pixel (e.g., mas/pixel).
    step : float
        Desired output sampling step (same units as r_vals).

    Returns
    -------
    r_vals : ndarray
        Radii of the profile.
    profile : ndarray
        Interpolated radial profile (mean over theta at each radius).
    """
    ny, nx = img.shape
    y = nnp.arange(ny)
    x = nnp.arange(nx)
    # Interpolator in pixel space
    interp = RectBivariateSpline(y, x, cpuArray(img))

    profile = []
    for (xp, yp) in grid:
        values = interp.ev(yp, xp)
        profile.append(nnp.mean(values))

    profile = nnp.array(profile)
    profile = nnp.clip(profile, 0, None)
    # === Scale to compensate for higher sampling resolution ===
    scaling = (step/pixelscale)**2
    profile *= scaling

    return r_vals, profile

def precompute_polar_grid(step, pixelscale, maxradius, center, n_theta=180):
    """
    Precompute a polar coordinate grid for interpolation.

    This grid can be reused for multiple PSFs, saving computation time.

    Parameters
    ----------
    step : float
        Radial step in physical units (e.g., mas).
    pixelscale : float
        Physical size of one pixel (e.g., mas/pixel).
    maxradius : float
        Maximum radius to compute the grid (in physical units).
    center : tuple of float
        (x, y) pixel coordinates of the center.
    n_theta : int, optional
        Number of angular points per radius (default is 180).

    Returns
    -------
    r_vals : ndarray
        Array of radii in physical units.
    grid : list of tuples
        List of (xp, yp) arrays (pixel coordinates) for each radius.
    """
     
    r_vals = nnp.arange(0, maxradius, step)
    
    grid = []
    for r in r_vals:
        if r == 0:
            # For the center, just one point
            xp = nnp.array([center[0]])
            yp = nnp.array([center[1]])
        else:
            # Adaptive sampling: distance in azimuth = radial step
            # Circumference = 2 * pi * r, number of points = circumference / step
            n_theta_adaptive = max(3, int(nnp.ceil(2 * nnp.pi * r / step)))
            # Cap the number of points to avoid excessive sampling
            if n_theta is not None:
                n_theta = min(n_theta_adaptive, n_theta) # max n_theta points
            else:
                n_theta = n_theta_adaptive

            theta = nnp.linspace(0, 2 * nnp.pi, n_theta, endpoint=False)
            cos_theta = nnp.cos(theta)
            sin_theta = nnp.sin(theta)

            rpix = r / pixelscale
            xp = center[0] + rpix * cos_theta
            yp = center[1] + rpix * sin_theta

        grid.append((xp, yp))

    return r_vals, grid

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

def fwhm_1d(profile):
    """ FWHM of 1D profile """
    max_val = profile.max()
    min_val = profile.min()
    if min_val >= 0.5*max_val:
        return profile.size

    half_max = max_val / 2
    
    # Find indices where the profile exceeds half the height
    indices = nnp.where(profile >= half_max)[0]

    if len(indices) < 2:
        print('fwhm_1d: not enough points above half height')
        return 1  # Error: not enough points above half height
                  # FWHM is 1 pixel
    if len(indices) >= profile.size-1:
        print('fwhm_1d: not enough points below half height')
        return profile.size  # Error: not enough points below half height
                             # FWHM is the size of the profile
    
    # Points at the left and right edges of the region above the threshold
    left, right = indices[0], indices[-1]
    
    # Interpolation to obtain a more precise estimate
    f = interp1d(profile[left-1:left+2], [left-1, left, left+1], kind='linear', bounds_error=False, fill_value="extrapolate")
    x1 = f(half_max)

    f = interp1d(profile[right-1:right+2], [right-1, right, right+1], kind='linear', bounds_error=False, fill_value="extrapolate")
    x2 = f(half_max)
    
    return abs(x2 - x1)  # Distance between the two points FWHM

def find_contour_points(image, level):
    """
    Find contour points using radial interpolation from peak.
    """
    # Find the peak position
    peak_y, peak_x = nnp.unravel_index(nnp.argmax(image), image.shape)
    peak_val = image[peak_y, peak_x]

    # Find points above threshold
    above_threshold = image > level

    # Find boundary points
    kernel = nnp.array([[0,1,0], [1,1,1], [0,1,0]], dtype=bool)
    dilated = ndimage.binary_dilation(above_threshold, kernel)
    boundary_points = dilated & ~above_threshold

    # Get coordinates of boundary points
    boundary_y, boundary_x = nnp.where(boundary_points)

    # Calculate distance from peak for each boundary point
    distances = nnp.sqrt((boundary_y - peak_y)**2 + (boundary_x - peak_x)**2)

    contour_points = []

    # Get coordinates of above-threshold points
    above_y, above_x = nnp.where(above_threshold)

    # For every edge point
    for x, y, d in zip(boundary_x, boundary_y, distances):
        # Find closest points above threshold
        dists_to_above = nnp.sqrt((above_y - y)**2 + (above_x - x)**2)

        if len(dists_to_above) > 4:
            nearest_indices = nnp.argpartition(dists_to_above, 4)[:4]
        else:
            nearest_indices = nnp.argsort(dists_to_above)[:len(dists_to_above)]

        # Calculate distance from peak for these points
        nearest_points = nnp.column_stack((above_x[nearest_indices], above_y[nearest_indices]))
        dists_from_peak = nnp.sqrt(nnp.sum((nearest_points - [peak_x, peak_y])**2, axis=1))

        # Farthest point from peak
        farthest_idx = nnp.argmax(dists_from_peak)
        far_point = nearest_points[farthest_idx]
        far_val = float(image[int(far_point[1]), int(far_point[0])])

        # Current point (on the edge)
        curr_val = float(image[y, x])

        # Direction vector from peak to farthest point
        direction = nnp.array([far_point[0] - peak_x, far_point[1] - peak_y])
        direction_norm = nnp.sqrt(nnp.sum(direction**2))
        direction = direction / direction_norm if direction_norm > 0 else direction

        # Interpolation on the line
        if curr_val < level < far_val:
            # Calculate interpolation paramter
            t = (level - curr_val)/(far_val - curr_val)

            # distance between current point and peak
            curr_dist = nnp.sqrt((x - peak_x)**2 + (y - peak_y)**2)
            # distance between point above threshold and peak
            far_dist = nnp.sqrt(nnp.sum((far_point - [peak_x, peak_y])**2))

            # distance interpolation
            interp_dist = curr_dist + t * (far_dist - curr_dist)

            # contour point
            px = peak_x + direction[0] * interp_dist
            py = peak_y + direction[1] * interp_dist

            contour_points.append([px, py])

    if not contour_points:
        contour_points.append(0)
        return contour_points

    contour_points = nnp.array(contour_points)

    # Filter outliers
    distances_from_peak = nnp.sqrt(nnp.sum((contour_points - [peak_x, peak_y])**2, axis=1))
    med_distance = nnp.median(distances_from_peak)
    std_distance = nnp.std(distances_from_peak)
    mask = nnp.abs(distances_from_peak - med_distance) < 2 * std_distance
    contour_points = contour_points[mask]

    return contour_points


def getFWHM(psf,pixelScale,rebin=1,method='contour',nargout=2,center=None,std_guess=2):
            
    # Gaussian and Moffat fitting are not really efficient on
    # anisoplanatic PSF. Prefer the coutour function in such a
    # case. The cutting method is not compliant to PSF not oriented
    # along x or y-axis.

    # Interpolation            
    Ny,Nx = psf.shape
    if rebin > 1:
        im_hr = cpuArray(interpolateSupport(psf,rebin*nnp.array([Nx,Ny])))
    else:
        im_hr = cpuArray(psf)
        
    ss = im_hr.shape
    sbx, sby = [[0,ss[1]],[0,ss[0]]]

    peak_val = nnp.max(im_hr)
    min_val = nnp.min(im_hr)
    if min_val >= 0.5*peak_val:
        print('getFWHM: not enough points below half height')
        fwhmX = nnp.sqrt(2)*Nx*pixelScale
        fwhmY = nnp.sqrt(2)*Ny*pixelScale
        theta = 0
    else:
        # initial guess of PSF center
        y_max, x_max = nnp.unravel_index(nnp.argmax(im_hr), im_hr.shape)
        x00 = sbx[0]
        y00 = sby[0]
        x_peak = int(x_max + x00)
        y_peak = int(y_max + y00)

        # -----------------------------------------------------------------------
        # cutting method used first to check if FWHM is too large for other methods
        # X and Y profiles passing through the max
        profile_x = im_hr[y_max, :]
        profile_y = im_hr[:, x_max]
        # X and Y FWHM
        fwhmXcut = fwhm_1d(profile_x)
        fwhmYcut = fwhm_1d(profile_y)
        thetaCut = 0
        # -----------------------------------------------------------------------
        if fwhmXcut >= profile_x.size-1 or fwhmYcut >= profile_y.size-1:
            print("getFWHM: FWHM is too large, falling back to cutting method")
            method = 'cutting'
        if fwhmXcut == 1 and fwhmYcut == 1 and (method == 'contour' or method == 'oldContour'):
            print("getFWHM: FWHM is too small, falling back to cutting method")
            method = 'cutting'
        # -----------------------------------------------------------------------

        if method == 'oldContour':
            # Old contour approach~: something wrong about the ellipse orientation
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
                fwhmX   = 2*wr.max()
                fwhmY   = 2*wr.min()
                #Getting the ellipse orientation
                xm      = wx[wr.argmax()]
                ym      = wy[wr.argmax()]
                theta   = nnp.mean(180*nnp.arctan2(ym,xm)/np.pi)
            except:
                print("getFWHM: oldContour method failed, falling back to cutting method")
                method = 'cutting'
            mpl.interactive(True)
        elif method == 'contour':
            # Find contour points at half maximum
            contour_points = find_contour_points(im_hr, peak_val/2)

            if len(contour_points) < 3:  # Need at least 3 points for meaningful analysis
                print("getFWHM: not enough contour points found, falling back to cutting method")
                method = 'cutting'
            else:
                xC = contour_points[:, 0]
                yC = contour_points[:, 1]

                # Centering the ellipse
                mx = nnp.array([xC.max(), yC.max()])
                mn = nnp.array([xC.min(), yC.min()])
                cent = (mx + mn)/2
                wx = xC - cent[0]
                wy = yC - cent[1]

                # Get the module
                wr = nnp.hypot(wx, wy)/rebin*pixelScale

                # Getting the FWHM
                fwhmX = 2*wr.max()
                fwhmY = 2*wr.min()

                # Getting the ellipse orientation
                xm = wx[wr.argmax()]
                ym = wy[wr.argmax()]
                theta = nnp.mean(180*nnp.arctan2(ym, xm)/nnp.pi)
        elif method == 'gaussian':
            # initial guess of the FWHM
            fwhm_guess = 2*nnp.sqrt((im_hr > peak_val/2).sum()/np.pi)

            stddev_guess = fwhm_guess/(2*nnp.sqrt(2*nnp.log(2)))
            p_init = models.Gaussian2D(amplitude=peak_val, x_mean=x_peak, y_mean=y_peak,
                                       x_stddev=stddev_guess, y_stddev=stddev_guess)

            fit_p = fitting.LevMarLSQFitter()

            y,x = nnp.mgrid[sby[0]:sby[1],sbx[0]:sbx[1]]
            g = fit_p(p_init, x, y, im_hr)

            fwhmX = 2 * nnp.sqrt(2 * nnp.log(2)) * nnp.abs(g.x_stddev.value*pixelScale)
            fwhmY = 2 * nnp.sqrt(2 * nnp.log(2)) * nnp.abs(g.y_stddev.value*pixelScale)
            theta = g.theta.value*180/np.pi
        elif method == 'moffat':      
            # initial guess of the FWHM
            fwhm_guess = 2*nnp.sqrt((im_hr > peak_val/2).sum()/np.pi)

            beta = 4.765    # expected beta value  for Seeing limited PSF
            p_init = models.Moffat2D(amplitude=peak_val, x_0=x_peak, y_0=y_peak,
                                     gamma=fwhm_guess/(2*nnp.sqrt(2**(1/beta)-1)))

            fit_p = fitting.LevMarLSQFitter()

            y,x = nnp.mgrid[sby[0]:sby[1],sbx[0]:sbx[1]]
            g = fit_p(p_init, x, y, im_hr)
            
            fwhmX = g.fwhm*pixelScale
            fwhmY = g.fwhm*pixelScale
            theta   = 0

        # in a dedicated if clause as a fallback method when others have failed
        if method == 'cutting':
            fwhmX = fwhmXcut * pixelScale / rebin
            fwhmY = fwhmYcut * pixelScale / rebin
            theta = thetaCut

    # Get Ellipticity
    aRatio      = nnp.max([fwhmX/fwhmY,fwhmY/fwhmX])

    if nargout == 1:
        return 0.5 * (fwhmX+fwhmY)
    elif nargout == 2:
        return fwhmX,fwhmY
    elif nargout == 3:
        return fwhmX,fwhmY,aRatio
    elif nargout == 4:
        return fwhmX,fwhmY,aRatio,theta

def getStrehl(psf0,pupil,samp,recentering=False,nR=5,method='otf',psfInOnePix=False):
    if recentering:    
        psf = centerPsf(psf0,2)
    else:
        psf = psf0

    npsf   = nnp.array(psf.shape)
    
    
    # Get the OTF
    otfDL = telescopeOtf(pupil,samp)
        
    if method == 'otf':
        #% Get the OTF
        otf     = fft.fftshift(psf2otf(psf))
        otf     = otf/otf.max()
        notf    = nnp.array(otf.shape)
        otfDL   = interpolateSupport(otfDL,notf)
        # Get the Strehl
        SR      = np.real(otf.sum()/otfDL.sum())
    elif method == 'max':
        psfDL   = otf2psf(otfDL,psfInOnePix=psfInOnePix)
        psfDL.clip(0, None)
        psfDL  /= psfDL.sum()
        psf.clip(0, None)
        psf    /= psf.sum()
        # Get the Strehl
        SR      = psf.max()/psfDL.max()
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
    diags = [matrix.diagonal(i).sum() for i in range(0,m)]
    
    return diags

def matrix_to_map(mat, sep_x=None, sep_y=None):
    """
    Compute the NxN covariance map from a N**2 x N**2 covariance matrix by
    averaging the values of the spatial covariance corresponding to same baseline.
    INPUTS:
        - mat, the spatial covariance matrix of size n_src x n_layer x N**2 x N**2
        - sep_x and sep_y, the 2D arrays of zisze N**2 x N**2 containing the separations
        along x and y axes.
    OUTPUTS:
        map, the covariance map of size n_src x n_layer x N x N
    """
    # defining the geometry
    n_ph = int(np.sqrt(mat.shape[-1]))

    if sep_x is None or sep_y is None:
        x = np.array(range(n_ph))
        x1, y1 = np.meshgrid(x, x)
        X1 = np.ones((n_ph**2, 1))*x1.T.reshape(-1)
        Y1 = np.tile(y1,[n_ph,n_ph])
        sep_x = np.transpose(X1.T - x1.T.reshape(-1))
        sep_y = Y1 - y1.T.reshape(-1)

    # ----- allocating memory
    if np.ndim(mat)!=2:
        raise ValueError("The mat dimension is not supported")


    # ----- get the indexes for baselines
    y_ind = sep_y == (np.array(range(n_ph)) - n_ph+1)[:,np.newaxis,np.newaxis]
    x_ind = sep_x == (np.array(range(n_ph)) - n_ph+1)[:,np.newaxis,np.newaxis]
    id_x_y = np.logical_and(y_ind[:,np.newaxis,:,:], x_ind[np.newaxis,:,:,:])
    # ----- get one quadrant of the map
    cmap = [mat[id_x_y[jy, jx]].mean() for jy in range(n_ph) for jx in range(n_ph)]
    cmap = np.array(cmap).reshape(n_ph, n_ph)
    # ----- duplicating the quadrant to get the full map
    cmap_full = np.zeros((2*n_ph-1, 2*n_ph-1))
    cmap_full[:n_ph, :n_ph] = cmap
    tmp = cmap[:, :-1]
    cmap_full[:n_ph, n_ph:] = np.fliplr(tmp)
    cmap_full[n_ph:, :] = np.flipud(cmap_full[:n_ph-1, :])

    return cmap_full

def get_diags(matrix):
    n , m = matrix.shape
    diags = [matrix.diagonal(i).mean() for i in range(0,m)]

    return diags
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