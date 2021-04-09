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

import fourier.FourierUtils as FourierUtils
from aoSystem.atmosphere import atmosphere
from aoSystem.defineAoSystem import defineAoSystem

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000

class psfao21:
    
    # WAVELENGTH
    @property
    def wvl(self):
        return self.__wvl
    @wvl.setter
    def wvl(self,val):
        self.__wvl = val
        self.samp  = val* rad2mas/(self.psInMas*self.D)
    @property
    def wvlRef(self):
        return self.__wvlRef
    @wvlRef.setter
    def wvlRef(self,val):
        self.__wvlRef = val
        self.sampRef  = val* rad2mas/(self.psInMas*self.D)
    
    @property
    def wvlCen(self):
        return self.__wvlCen
    @wvlCen.setter
    def wvlCen(self,val):
        self.__wvlCen = val
        self.sampCen  = val* rad2mas/(self.psInMas*self.D)
        
    # SAMPLING
    @property
    def samp(self):
        return self.__samp
    @samp.setter
    def samp(self,val):
        self.k_      = np.ceil(2.0/val).astype('int') # works for oversampling
        self.__samp  = self.k_ * val     
    @property
    def sampCen(self):
        return self.__sampCen
    @sampCen.setter
    def sampCen(self,val):
        self.kCen_      = int(np.ceil(2.0/val))# works for oversampling
        self.__sampCen  = self.kCen_ * val  
    @property
    def sampRef(self):
        return self.__sampRef
    @sampRef.setter
    def sampRef(self,val):
        self.kRef_      = int(np.ceil(2.0/val)) # works for oversampling
        self.__sampRef  = self.kRef_ * val
        self.kxky_      = FourierUtils.freq_array(self.nPix*self.kRef_,self.__sampRef,self.D)
        self.k2_        = self.kxky_[0]**2 + self.kxky_[1]**2                   
        #piston filtering        
        self.pistonFilter_ = FourierUtils.pistonFilter(self.D,self.k2_)
        #self.pistonFilter_= 1 - 4*self.sombrero(1, np.pi*D*np.sqrt(self.k2_)) ** 2
        self.pistonFilter_[self.nPix*self.kRef_//2,self.nPix*self.kRef_//2] = 0
            
    # CUT-OFF FREQUENCY
    @property
    def nAct(self):
        return self.__nAct    
    @nAct.setter
    def nAct(self,val):
        self.__nAct = val
        # redefining the ao-corrected area
        self.kc_= (val-1)/(2.0*self.D)
        kc2     = self.kc_**2
        if self.aoFilter == 'circle':
            self.mskOut_   = (self.k2_ >= kc2)
            self.mskIn_    = (self.k2_ < kc2)
        else:
            self.mskOut_   = np.logical_or(abs(self.kxky_[0]) >= self.kc_, abs(self.kxky_[1]) >= self.kc_)
            self.mskIn_    = np.logical_and(abs(self.kxky_[0]) < self.kc_, abs(self.kxky_[1]) < self.kc_)
        self.psdKolmo_     = 0.0229 * self.mskOut_* ((1.0 /self.atm.L0**2) + self.k2_) ** (-11.0/6.0)
        self.wfe_fit_norm  = np.sqrt(np.trapz(np.trapz(self.psdKolmo_,self.kxky_[1][0]),self.kxky_[1][0]))
    # INIT
    def __init__(self,file,aoFilter='circle',antiAlias=False,pathStat=None,fitCn2=False):
        
        tstart = time.time()
        
        # PARSING INPUTS
        self.file      = file
        self.antiAlias = antiAlias
        self.pathStat  = pathStat
        self.status = defineAoSystem(self,file,aoFilter=aoFilter)
        
        if self.status:
            # DEFINING THE DOMAIN ANGULAR FREQUENCIES
            self.nOtf        = self.nPix * self.kRef_
            self.U_, self.V_, self.U2_, self.V2_, self.UV_=  FourierUtils.instantiateAngularFrequencies(self.nOtf,fact=2)
              
            # COMPUTING THE STATIC OTF IF A PHASE MAP IS GIVEN
            self.otfNCPA, _ = FourierUtils.getStaticOTF(self.tel,self.nOtf,self.sampRef,self.wvlRef, apodizer=self.apodizer,opdMap_ext=self.opdMap_ext)
            self.otfDL,_    = FourierUtils.getStaticOTF(self.tel,self.nOtf,self.sampRef,self.wvlRef, apodizer=self.apodizer)
    
    
            # ANISOPLANATISM
            if any(self.atm.heights) and any(self.src.zenith):
                self.Dani_l = self.instantiateAnisoplanatism(self.nOtf,self.sampRef)
                self.isAniso = True
                if fitCn2 == False: 
                    self.Kaniso = np.zeros((self.src.nSrc,self.nWvl,self.nOtf,self.nOtf))
                    Cn2  = self.atm.r0**(-5/3) * (self.atm.wvl/self.wvlRef)**2  * self.atm.weights
                    for l in range(self.nWvl):
                        wvl_l     = self.wvl[l]
                        wvlRatio  = (self.wvlRef/wvl_l) ** 2
                        for iSrc in range(self.src.nSrc):   
                            Dani                = (self.Dani_l[iSrc].T * Cn2 * wvlRatio).sum(axis=2)
                            self.Kaniso[iSrc,l] = np.exp(-0.5 * Dani)
                    self.atm = atmosphere(self.atm.wvl,self.atm.r0,[1],[0],self.atm.wSpeed.mean(),self.atm.wDir.mean(),self.atm.L0)   
                    self.isAniso = False
            else:
                self.Dani_l = None
                self.Kaniso = np.ones((self.src.nSrc,self.nWvl))
                self.isAniso = False
                if fitCn2 == False: 
                    self.atm = atmosphere(self.atm.wvl,self.atm.r0,[1],[0],self.atm.wSpeed.mean(),self.atm.wDir.mean(),self.atm.L0)  
            
            # ONE OF SEVERAL FRAMES
            self.isCube = any(rad2mas * self.src.direction[0]/self.psInMas > self.nPix) \
            or all(rad2mas * self.src.direction[1]/self.psInMas > self.nPix)
                
            # DEFINING BOUNDS
            self.bounds = self.defineBounds()
        self.t_init = 1000*(time.time()  - tstart)
        
    def _repr__(self):
        return 'PSFAO21 model'
    
    def defineBounds(self):
          #Cn2/r0 , C , A , ax , p , theta , beta , sx , sy , sxy , F , dx , dy , bg , stat
          _EPSILON = np.sqrt(sys.float_info.epsilon)
          
          # Bounds on r0
          bounds_down = list(np.ones(self.atm.nL)*_EPSILON)
          bounds_up   = list(np.inf * np.ones(self.atm.nL))          
          # PSD Parameters
          bounds_down += [0,0,_EPSILON,_EPSILON,-np.pi,1+_EPSILON]
          bounds_up   += [np.inf,np.inf,np.inf,np.inf,np.pi,5]         
          # Jitter
          bounds_down += [0,0,-1]
          bounds_up   += [np.inf,np.inf,1]
          # Photometry
          bounds_down += list(np.zeros(self.src.nSrc))
          bounds_up   += list(np.inf*np.ones(self.src.nSrc))
          # Astrometry
          bounds_down += list(-self.nPix//2 * np.ones(2*self.src.nSrc))
          bounds_up   += list( self.nPix//2 * np.ones(2*self.src.nSrc))
          # Background
          bounds_down += [-np.inf]
          bounds_up   += [np.inf]
          # Static aberrations
          bounds_down += list(-self.wvlRef/2*1e9 * np.ones(self.nModes))
          bounds_up   += list(self.wvlRef/2 *1e9 * np.ones(self.nModes))
          return (bounds_down,bounds_up)
        
        
    def getPSD(self,x0):
        # Get the moffat PSD
        pix2freq = 1/(self.tel.D * self.sampRef)
        psd = self.moffat(self.kxky_,list(x0[3:])+[0,0])
        # Piston filtering
        psd = self.pistonFilter_ * psd
        # Combination
        psd = x0[0]**(-5/3) * self.psdKolmo_ + self.mskIn_ * (x0[1] + psd/psd.sum() * x0[2]/pix2freq**2 )

        # Wavefront error
        self.wfe = np.sqrt( np.trapz(np.trapz(psd,self.kxky_[1][0]),self.kxky_[1][0]) ) * self.wvlRef*1e9/2/np.pi
        self.wfe_fit = np.sqrt(x0[0]**(-5/3)) * self.wfe_fit_norm  * self.wvlRef*1e9/2/np.pi
        return psd
    
    def __call__(self,x0,nPix=None):
        
        # INSTANTIATING
        if nPix == None:
            nPix = self.nPix
        psf = np.zeros((self.src.nSrc,nPix,nPix))
        
        # GETTING THE PARAMETERS
        xall = x0       
        # Cn2 profile
        nL   = self.atm.nL
        if nL > 1: # fit the Cn2 profile
            Cn2  = np.asarray(x0[0:nL])
            r0   = np.sum(Cn2)**(-3/5)
        else: #fit the r0
            r0 = x0[0]
            
        # PSD
        x0_psd = list(xall[nL:nL+6])
        # Jitter
        x0_jitter = list(xall[nL+6:nL+9])
        # Astrometry/Photometry/Background
        x0_stellar = list(xall[nL+9:nL+10+3*self.src.nSrc])
        # Static aberrations
        if self.isStatic:
            x0_stat = list(xall[nL+10+3*self.src.nSrc:])
        else:
            x0_stat = []    
        
        # GETTING THE PHASE STRUCTURE FUNCTION
        self.psd = self.getPSD([r0]+ x0_psd)
        if self.antiAlias:
            	self.psd = np.pad(self.psd,(self.nOtf//2,self.nOtf//2)) 
        #covariance map
        Bphi        = fft.fft2(fft.fftshift(self.psd)) / (self.tel.D * self.sampRef)**2       
        # phase structure function
        self.Dphi   = fft.fftshift(np.real(2*(Bphi.max() - Bphi)))
        if self.antiAlias:
            	self.Dphi   = FourierUtils.interpolateSupport(self.Dphi,self.nOtf)    

        # ADDITIONAL JITTER
        if len(x0_jitter) and (x0_jitter[0]!=0 or x0_jitter[1]!=0): # note sure about the normalization
            self.Djitter = (x0_jitter[0]**2 * self.U2_ + x0_jitter[1]**2 * self.V2_+ \
                    2*x0_jitter[0] * x0_jitter[1] * x0_jitter[2] * self.UV_)
            Kjitter = np.exp(-0.5 * self.Djitter * (np.sqrt(2)/self.psInMas)**2)
        else:
            	Kjitter = 1
                               
        for l in range(self.nWvl): # LOOP ON WAVELENGTH
            
            # WAVELENGTH
            wvl_l       = self.wvl[l]
            wvlRatio    = (self.wvlRef/wvl_l) ** 2           
            
            # STATIC OTF
            if len(x0_stat) or self.nWvl > 1:
                self.otfStat, self.phaseMap = FourierUtils.getStaticOTF(self.tel,int(self.nPix*self.k_[l]),self.samp[l],wvl_l,xStat=x0_stat,apodizer=self.apodizer,statModes=self.statModes,opdMap_ext=self.opdMap_ext)
            else:
                self.otfStat = self.otfNCPA
              
            # ON-AXIS OTF
            otfOn = self.otfStat * np.exp(-0.5*self.Dphi * wvlRatio)
            
            for iSrc in range(self.src.nSrc): # LOOP ON SOURCES
                
                # ANISOPLANATISM
                if self.isAniso and len(Cn2) == self.Dani_l.shape[1]:
                    Dani = (self.Dani_l[iSrc].T * Cn2 * wvlRatio).sum(axis=2)
                    Kaniso = np.exp(-0.5 * Dani)
                else:
                    Kaniso = self.Kaniso[iSrc,l]
                    
                # Stellar parameters
                if len(x0_stellar):
                    F   = x0_stellar[iSrc] * self.wvl_tr[l]
                    dx  = x0_stellar[iSrc + self.src.nSrc] + self.wvl_dpx[l]
                    dy  = x0_stellar[iSrc + 2*self.src.nSrc] + self.wvl_dpy[l]
                    bkg = x0_stellar[3*self.src.nSrc]
                else:
                    F   = self.wvl_tr[l]
                    dx  = self.wvl_dpx[l]
                    dy  = self.wvl_dpy[l]
                    bkg = 0
                
                # Phasor
                if dx !=0 or dy!=0:
                    fftPhasor = np.exp(np.pi*complex(0,1)*(self.U_*dx + self.V_*dy))
                else:
                    fftPhasor = 1
                    
                # Get the total OTF
                self.otfTot = otfOn * fftPhasor * Kaniso * Kjitter
                
                # Get the PSF
                psf_i = np.real(fft.fftshift(fft.ifft2(fft.fftshift(self.otfTot))))
                
                # CROPPING
                if self.k_[l]> 1:
                        psf_i = FourierUtils.interpolateSuport(psf_i,nPix);            
                    
                if nPix != self.nPix:
                    psf_i = FourierUtils.cropSupport(psf_i,nPix/self.nPix)
                
                # SCALING
                psf[iSrc] += F * psf_i/psf_i.sum()
        
        # DEFINING THE OUTPUT FORMAT
        if self.isCube:
            return np.squeeze(psf) + bkg
        else:
            return np.squeeze(psf.sum(axis=0)) + bkg
        
    def moffat(self,XY,x0):
        
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
            
        u = Rxx * (XY[0] - dx)**2 + Rxy * (XY[0] - dx)* (XY[1] - dy) + Ryy * (XY[1] - dy)**2
        return (1.0 + u) ** (-beta)