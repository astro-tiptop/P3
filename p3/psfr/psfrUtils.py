#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:57:49 2018

@author: omartin
"""

# Libraries
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
def getOLslopes(s,u,MI,dt):    
    return s + MI*(dt*np.roll(u,(-2,2)) + (1-dt)*np.roll(u,(-1,2)))

def getStructureFunction(phi,pupil,overSampling):
    # Create phi and pup an a 2x larger grid
    phi2    = FourierUtils.enlargeSupport(phi,2*overSampling)
    pup2    = FourierUtils.enlargeSupport(pupil,2*overSampling)
    corp2w  = FourierUtils.fftCorrel(phi2**2,pup2)
    corpp   = FourierUtils.fftCorrel(phi2,phi2)
    dphi    = corp2w + FourierUtils.fftsym(corp2w) - 2*corpp
    corww   = FourierUtils.fftCorrel(pup2,pup2)
    # Compute mask of locations where there is overlap between shifted pupils and normalized by the number of points used to compute dphi
    mask    = (corww > 1e-5)
    corww[corww <= 1] = 1
    
    return np.real(fft.fftshift(dphi * mask /corww))


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

def modes2Otf(Cmm,modes,pupil,nOtf,samp=2,basis='Vii'):            
    #Autocorrelation of the pupil expressed in pupil
    nPx        = int(np.sqrt(modes.shape[0]))
    pupExtended= FourierUtils.enlargeSupport(pupil,samp)
    fftPup     = fft.fft2(pupExtended)
    conjPupFft = np.conjugate(fftPup)
    G          = fft.fftshift(np.real(fft.fft2(fftPup*conjPupFft)))
    #Defining the inverse
    den        = np.zeros(np.array(G.shape))
    msk        = G/G.max() > 1e-7
    den[msk]   = 1/G[msk]
            
    if (np.any(Cmm)) & (basis == 'Vii'):
        # Diagonalizing the Cvv matrix
        U,ss,V   = np.linalg.svd(Cmm)
        nModes  = len(ss)
        M       = np.matmul(modes,U)
        #loop on actuators                
        buf     = np.zeros_like(pupExtended)
                
        for k in np.arange(1,nModes,1):
            Mk   = np.reshape(M[:,k],(nPx,nPx))
            Mk   = FourierUtils.enlargeSupport(Mk,samp)
            # Vii computation
            Vk   = np.real(fft.fft2(Mk**2 *pupExtended)*conjPupFft) - abs(fft.fft2(Mk*pupExtended))**2
            # Summing modes into dm basis
            buf  = buf +  ss[k] * Vk
                        
        dphi     = den*fft.fftshift(np.real(fft.fft2(2*buf)))
        otf      = G*np.exp(-0.5*dphi)
                
    elif (np.any(Cmm)) & (basis == 'Uij'):
        nm   = modes.shape[1]
        dphi = 0*pupExtended
                
        #Double loops on modes
        for i in np.arange(1,nm,1):
            Mi = np.reshape(modes[:,i],(nPx,nPx))
            Mi = FourierUtils.enlargeSupport(Mi,samp)
            for j in np.arange(1,i,1):
                #Getting modes + interpolation to twice resolution
                Mj    = np.reshape(modes[:,j],(nPx,nPx))
                Mj    = FourierUtils.enlargeSupport(Mj,samp)
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
        otf  = G
        
    # Interpolation of the OTF => determination of the PSF fov
    otf = otf*(G>1e-5)
    otf = FourierUtils.interpolateSupport(otf,nOtf)
    dphi= FourierUtils.interpolateSupport(dphi,nOtf)
    otf = otf/otf.max()
    return otf,dphi

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
    otf = FourierUtils.interpolateSupport(otf,npsf)
    otf = otf/otf.max()
    return otf