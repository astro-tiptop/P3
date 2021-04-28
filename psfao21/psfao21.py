#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:50:20 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
import numpy.fft as fft
import time
import sys as sys

import aoSystem.FourierUtils as FourierUtils
from aoSystem.aoSystem import aoSystem as aoSys
from aoSystem.frequencyDomain import frequencyDomain as frequencyDomain

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000

class psfao21:
    # INIT
    def __init__(self,path_ini,antiAlias=False,fitCn2=False):
        
        tstart = time.time()
        
        # PARSING INPUTS
        self.file      = path_ini
        self.antiAlias = antiAlias
        self.ao        = aoSys(path_ini)    
        self.isStatic = self.ao.tel.nModes > 0
        
        if self.ao.error==False:
            
            # DEFINING THE FREQUENCY DOMAIN
            self.freq = frequencyDomain(self.ao)
            
            # ONE OF SEVERAL FRAMES
            self.isCube = any(rad2mas * self.ao.src.direction[0]/self.ao.cam.psInMas > self.freq.nPix) \
            or all(rad2mas * self.ao.src.direction[1]/self.ao.cam.psInMas > self.freq.nPix)
                
            # DEFINING BOUNDS
            self.bounds = self.defineBounds()
        self.t_init = 1000*(time.time()  - tstart)
        
    def _repr__(self):
        return 'PSFAO21 model'
    
    def defineBounds(self):
          #Cn2/r0 , C , A , ax , p , theta , beta , sx , sy , sxy , F , dx , dy , bg , stat
          _EPSILON = np.sqrt(sys.float_info.epsilon)
          
          # Bounds on r0
          bounds_down = list(np.ones(self.ao.atm.nL)*_EPSILON)
          bounds_up   = list(np.inf * np.ones(self.ao.atm.nL))          
          # PSD Parameters
          bounds_down += [0,0,_EPSILON,_EPSILON,-np.pi,1+_EPSILON]
          bounds_up   += [np.inf,np.inf,np.inf,np.inf,np.pi,5]         
          # Jitter
          bounds_down += [0,0,-1]
          bounds_up   += [np.inf,np.inf,1]
          # Photometry
          bounds_down += list(np.zeros(self.ao.src.nSrc))
          bounds_up   += list(np.inf*np.ones(self.ao.src.nSrc))
          # Astrometry
          bounds_down += list(-self.freq.nPix//2 * np.ones(2*self.ao.src.nSrc))
          bounds_up   += list( self.freq.nPix//2 * np.ones(2*self.ao.src.nSrc))
          # Background
          bounds_down += [-np.inf]
          bounds_up   += [np.inf]
          # Static aberrations
          bounds_down += list(-self.freq.wvlRef/2*1e9 * np.ones(self.ao.tel.nModes))
          bounds_up   += list(self.freq.wvlRef/2 *1e9 * np.ones(self.ao.tel.nModes))
          return (bounds_down,bounds_up)
        
        
    def getPSD(self,x0):
        # Get the moffat PSD
        psd = self.moffat(self.freq.kx_,self.freq.ky_,list(x0[3:])+[0,0])
        # Piston filtering
        psd = self.freq.pistonFilter_ * psd
        # Combination
        A_emp = np.trapz(np.trapz(psd,self.freq.ky_[0]),self.freq.ky_[0])
        psd   = x0[0]**(-5/3) * self.freq.psdKolmo_ + self.freq.mskIn_ * (x0[1] + psd/A_emp * x0[2] )
        # Wavefront error
        self.wfe     = np.sqrt( np.trapz(np.trapz(psd,self.freq.ky_[0]),self.freq.kx_[0]) ) * self.freq.wvlRef*1e9/2/np.pi
        self.wfe_fit = np.sqrt(x0[0]**(-5/3)) * self.freq.wfe_fit_norm  * self.freq.wvlRef*1e9/2/np.pi
        return psd
    
    def getSF(self,Cn2=[]):
        #covariance map
        Bphi  = fft.fft2(fft.fftshift(self.psd)) / (self.ao.tel.D * self.freq.sampRef)**2       
        # On-axis phase structure function
        SF   = fft.fftshift(np.real(2*(Bphi.max() - Bphi)))
        # Anisoplanatism phase structure function
        if self.freq.isAniso and (len(Cn2) == self.freq.dani_ang.shape[1]):
            SF = SF[:,:,np.newaxis] + (self.freq.dphi_ani * Cn2).sum(axis=2)
        else:
            SF = np.repeat(SF[:,:,np.newaxis],self.ao.src.nSrc,axis=2)
        return SF /(2*np.pi*1e-9/self.freq.wvlRef)**2
        
    def __call__(self,x0,nPix=None):
        
        # INSTANTIATING
#        if nPix == None:
#            nPix = self.freq.nPix
#        psf = np.zeros((self.ao.src.nSrc,nPix,nPix))
        
        # ----------------- GETTING THE PARAMETERS
        xall = x0       
        # Cn2 profile
        nL   = self.ao.atm.nL
        if nL > 1: # fit the Cn2 profile
            Cn2  = np.asarray(x0[0:nL])
            r0   = np.sum(Cn2)**(-3/5)
        else: #fit the r0
            Cn2= []
            r0 = x0[0]
            
        # PSD
        x0_psd = list(xall[nL:nL+6])
        # Jitter
        x0_jitter = list(xall[nL+6:nL+9])
        # Astrometry/Photometry/Background
        x0_stellar = np.array(xall[nL+9:nL+10+3*self.ao.src.nSrc])
        if len(x0_stellar):
            F  = x0_stellar[0:self.ao.src.nSrc][:,np.newaxis] * np.array(self.ao.cam.transmittance)[np.newaxis,:]
            dx = x0_stellar[self.ao.src.nSrc:2*self.ao.src.nSrc][:,np.newaxis] + np.array(self.ao.cam.dispersion[0])[np.newaxis,:]
            dy = x0_stellar[2*self.ao.src.nSrc:3*self.ao.src.nSrc][:,np.newaxis] + np.array(self.ao.cam.dispersion[1])[np.newaxis,:]
            bkg= x0_stellar[3*self.ao.src.nSrc]
        else:
            F  = np.repeat(np.array(self.ao.cam.transmittance)[np.newaxis,:],self.ao.src.nSrc,axis=0)
            dx = np.repeat(np.array(self.ao.cam.dispersion[0])[np.newaxis,:],self.ao.src.nSrc,axis=0)
            dy = np.repeat(np.array(self.ao.cam.dispersion[1])[np.newaxis,:],self.ao.src.nSrc,axis=0)
            bkg= 0.0

        # Static aberrations
        if self.isStatic:
            x0_stat = list(xall[nL+10+3*self.ao.src.nSrc:])
        else:
            x0_stat = []    
        
        # ----------------- GETTING THE PHASE STRUCTURE FUNCTION
        self.psd = self.getPSD([r0]+ x0_psd)
        if self.antiAlias:
            	self.psd = np.pad(self.psd,(self.freq.nOtf//2,self.freq.nOtf//2)) 
#        #covariance map
#        Bphi  = fft.fft2(fft.fftshift(self.psd)) / (self.ao.tel.D * self.freq.sampRef)**2       
#        # phase structure function
#        self.SF   = fft.fftshift(np.real(2*(Bphi.max() - Bphi)))
        self.SF = self.getSF(Cn2=Cn2)
        if self.antiAlias:
            	self.SF   = FourierUtils.interpolateSupport(self.Dphi,self.freq.nOtf)    


        # ----------------- COMPUTING THE PSF
        PSF, self.SR = FourierUtils.SF2PSF(self.SF,self.freq,self.ao,\
                        jitterX=x0_jitter[0],jitterY=x0_jitter[1],jitterXY=x0_jitter[2],\
                        F=F,dx=dx,dy=dy,bkg=bkg,nPix=nPix,xStat=x0_stat)
        return np.squeeze(PSF)
#        # ADDITIONAL JITTER
#        if len(x0_jitter) and (x0_jitter[0]!=0 or x0_jitter[1]!=0): # note sure about the normalization
#            self.Djitter = (x0_jitter[0]**2 * self.freq.U2_ + x0_jitter[1]**2 * self.freq.V2_+ \
#                    2*x0_jitter[0] * x0_jitter[1] * x0_jitter[2] * self.freq.UV_)
#            Kjitter = np.exp(-0.5 * self.Djitter * (np.sqrt(2)/self.ao.cam.psInMas)**2)
#        else:
#            Kjitter = 1
#                               
#        for l in range(self.freq.nWvl): # LOOP ON WAVELENGTH
#            
#            # WAVELENGTH
#            wvl_l       = self.freq.wvl[l]
#            wvlRatio    = (self.freq.wvlRef/wvl_l) ** 2           
#            
#            # STATIC OTF
#            if len(x0_stat) or self.freq.nWvl > 1:
#                self.otfStat, self.otfDL, self.phaseMap = \
#                FourierUtils.getStaticOTF(self.ao.tel,int(self.freq.nPix*self.freq.k_[l]),self.freq.samp[l],wvl_l,xStat=x0_stat)
#            else:
#                self.otfStat = self.freq.otfNCPA
#              
#            # ON-AXIS OTF
#            otfOn = self.otfStat * np.exp(-0.5*self.Dphi * wvlRatio)
#            
#            for iSrc in range(self.ao.src.nSrc): # LOOP ON SOURCES
#                
#                # ANISOPLANATISM
#                if self.freq.isAniso and (len(Cn2) == self.freq.dani_ang.shape[1]):
#                    Dani = (self.Dani_l[iSrc].T * Cn2 * wvlRatio).sum(axis=2)
#                    Kaniso = np.exp(-0.5 * Dani)
#                else:
#                    Kaniso = 1.0#self.Kaniso[iSrc,l]
#                    
#                # Stellar parameters
#                if len(x0_stellar):
#                    F   = x0_stellar[iSrc]* self.ao.cam.transmittance[l]
#                    dx  = x0_stellar[iSrc + self.ao.src.nSrc] + self.ao.cam.dispersion[0][l]
#                    dy  = x0_stellar[iSrc + 2*self.ao.src.nSrc] + self.ao.cam.dispersion[1][l]
#                    bkg = x0_stellar[3*self.ao.src.nSrc]
#                else:
#                    F   = self.ao.cam.transmittance[l]
#                    dx  = self.ao.cam.dispersion[0][l]
#                    dy  = self.ao.cam.dispersion[1][l]
#                    bkg = 0
                
#                # Phasor
#                if dx !=0 or dy!=0:
#                    fftPhasor = np.exp(np.pi*complex(0,1)*(self.freq.U_*dx + self.freq.V_*dy))
#                else:
#                    fftPhasor = 1
#                    
#                # Get the total OTF
#                self.otfTot = otfOn * fftPhasor * Kaniso * Kjitter
#                
#                # Get the PSF
#                psf_i = np.real(fft.fftshift(fft.ifft2(fft.fftshift(self.otfTot))))
#                
#                # CROPPING
#                if self.freq.k_[l]> 1:
#                        psf_i = FourierUtils.interpolateSuport(psf_i,nPix);            
#                    
#                if nPix != self.freq.nPix:
#                    psf_i = FourierUtils.cropSupport(psf_i,nPix/self.freq.nPix)
#                
#                # SCALING
#                psf[iSrc] += F * psf_i/psf_i.sum()
#        
#        # DEFINING THE OUTPUT FORMAT
#        if self.isCube:
#            return np.squeeze(psf) + bkg
#        else:
#            return np.squeeze(psf.sum(axis=0)) + bkg
        
    def moffat(self,kx,ky,x0):
        
        # parsing inputs
        a       = x0[0]
        p       = x0[1]
        theta   = x0[2]
        beta    = x0[3]
        dx      = x0[4]
        dy      = x0[5]
        
        # updating the geometry
        ax      = a*p
        ay      = a/p
        c       = np.cos(theta)
        s       = np.sin(theta)
        s2      = np.sin(2.0 * theta)
           
        Rxx = (c/ax)**2 + (s/ay)**2
        Ryy = (c/ay)**2 + (s/ax)**2
        Rxy =  s2/ay**2 -  s2/ax**2
            
        u = Rxx * (kx - dx)**2 + Rxy * (kx - dx)* (ky - dy) + Ryy * (ky - dy)**2
        return (1.0 + u) ** (-beta)