#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:57:49 2018

@author: omartin
"""

#%% Libraries
import numpy as np
import numpy.fft as fft
import p3.aoSystem.FourierUtils as FourierUtils
import urllib.request
import os

#%% DATA TRANSFER
def get_data_file(path_sav,filename):
    if not os.path.isfile(path_sav + f"/{filename}"):
        print('Downloading of the file %s in the folder %s'%(filename,path_sav))
        urllib.request.urlretrieve(f"https://nuage.osupytheas.fr/s/WjeQ8BB3wp2mEyL/download?files={filename}",
                                   path_sav + f"{filename}")

#%%  PSF-R FACILITIES
def compute_psd_from_slopes(trs):
    """
    Compute the spatial PSD of the reconstructed phase and the DM influence function.
    """
    # get the residual wavefront in the actuator domain
    com_res = trs.rec.res
    # get the DM influence function
    modes = trs.mat.dmIF
    n_px = int(np.sqrt(modes.shape[0]))
    n_frames = com_res.shape[0]
    # get the cube of phase in randian
    phase = 2*np.pi/trs.wfs.wvl * np.dot(modes, com_res.T).reshape(n_px, n_px, n_frames)
    phase = FourierUtils.enlargeSupport(phase, 2)
    # get the psd
    psd = np.fft.fftshift(np.mean(abs(np.fft.fft2(phase, axes=(0,1)))**2, axis=2))
    return psd

def getOLslopes(s,u,MI,dt):
    return s + MI*(dt*np.roll(u,(-2,2)) + (1-dt)*np.roll(u,(-1,2)))

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

def modes2Otf(Cmm, modes, pupil, n_otf, samp=2, basis='Vii'):
    """
    Computes the OTF and phase structure function from the covariance matrix
    of given coeffcients and the corresponding modal basis.
    INPUTS:
        - Cmm, the n_m x n_M covariance matrix
        - modes, the n_pup**2 x n_m matrix concatenating 1D vector of modes
        - pupil, the n_pup x n_pup telescope pupil
        - nOtf, the output shape for the otf
        - samp, the oversampling factor; samp=2 means Nyquist sampling
        - basis, either Vii (Gendron+06) or Uij (Veran97)
    OUTPUTS:
        - otf, the nOtf x nOtf optical transfer function
        - dphi, the corresponding phase structure function in rad**2
    """
    #Autocorrelation of the pupil expressed in pupil
    n_pup = int(np.sqrt(modes.shape[0]))
    pup_pad = FourierUtils.enlargeSupport(pupil, samp)
    fft_pup = fft.fft2(pup_pad)
    conj_pup = np.conjugate(fft_pup)
    G = fft.fftshift(np.real(fft.ifft2(abs(fft_pup)**2)))

    #Defining the inverse
    den = np.zeros_like(G)
    msk = G/G.max() > 1e-7
    den[msk] = 1/G[msk]

    if np.any(Cmm) and basis=="Vii":
        # Diagonalizing the Cvv matrix
        U, ss ,V = np.linalg.svd(Cmm)
        nModes = len(ss)
        M_all = np.dot(modes, V)
        M_all = M_all.reshape(n_pup, n_pup, nModes)
        M_all = FourierUtils.enlargeSupport(M_all, samp)*pup_pad[:, :, np.newaxis]

        buf = np.zeros_like(G)
        for k in range(nModes):
            Mk = M_all[:, :, k]
            # Vii computation
            Vk = np.real(fft.fft2(Mk**2)*conj_pup) - abs(fft.fft2(Mk))**2
            # Summing modes into dm basis
            buf = buf + ss[k] * Vk

        dphi = den*fft.fftshift(np.real(fft.ifft2(2*buf)))
        otf = G*np.exp(-0.5*dphi)

    elif np.any(Cmm) and basis=="Uij":
        nm   = modes.shape[1]
        dphi = np.zeros_like(pup_pad)

        #Double loops on modes
        for i in np.arange(1,nm,1):
            Mi = np.reshape(modes[:,i],(n_pup, n_pup))
            Mi = FourierUtils.enlargeSupport(Mi,samp)
            for j in range(1,i):
                #Getting modes + interpolation to twice resolution
                Mj = np.reshape(modes[:,j],(n_pup, n_pup))
                Mj = FourierUtils.enlargeSupport(Mj,samp)
                term1 = np.real(fft.fft2(Mi*Mj*pup_pad)*conj_pup)
                term2 = np.real(fft.fft2(Mi*pup_pad)*np.conjugate(fft.fft2(Mj*pup_pad)))
                # Uij computation
                Uij = np.real(fft.ifft2(term1-term2))
                #Summing terms
                fact = np.double(i!=j) + 1
                dphi = dphi + fact*Cmm[i,j]*Uij
                dphi = fft.fftshift(2*dphi)*den*msk
        otf = G*np.exp(-0.5*dphi)
    else:
        #Diffraction-limit case
        G = G/G.max()
        otf = G

    # Interpolation of the OTF => determination of the PSF fov
    otf = otf*(G>1e-5) #3.6 ms
    otf = FourierUtils.interpolateSupport(otf, n_otf) #84ms
    dphi = FourierUtils.interpolateSupport(dphi, n_otf) #86ms
    otf = otf/otf.max()
    return otf, dphi

def pointWiseLocation(D, dp, idxValid=None):
    # Defining the point-wise locations
    xloc = np.arange(-D/2,D/2+dp,dp)
    actuLocY, actuLocX = np.meshgrid(xloc,xloc)
    if idxValid is not None:
        actuLocX = actuLocX[idxValid]
        actuLocY = actuLocY[idxValid]

    return np.array([actuLocX,actuLocY])

def sr2wfe(Strehl,wvl):
    return 1e9*np.sqrt(-np.log(Strehl))*wvl/2/np.pi

def wfe2sr(wfe,wvl):
    return np.exp(-(2*np.pi*wfe*1e-9/wvl)**2)

def zonalCovarianceToOtf(Cphi, npsf, D, dp, idxValid=None):
    # Grabbing the valid actuators positions in meters
    loc  = pointWiseLocation(D, dp, idxValid=idxValid)
    #OTF sampling
    nPh  = round(D/dp+1)
    #nU1d = 2*nPh;
    u1D  = np.arange(-nPh,nPh,1)*dp
    nU1d = len(u1D)
    # Determining couples of point with the same separation
    shiftX,shiftY = mkotf_indpts(nU1d,nPh,u1D,loc,dp)
    # WF Amplitude
    if idxValid is None:
        idxValid = np.ones(Cphi.shape[0])
    amp0 = np.ones((np.count_nonzero(idxValid),1))
    # Long-exposure OTF
    otf = mkotf(shiftX,shiftY,nU1d,amp0,dp,-0.5*Cphi)
    #Interpolation
    otf = FourierUtils.interpolateSupport(otf,npsf)
    otf = otf/otf.max()
    return otf