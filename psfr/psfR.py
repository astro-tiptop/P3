#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 22:20:28 2018

@author: omartin
"""
#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
import time
import sys as sys
import numpy.fft as fft

from fourier.fourierModel import fourierModel
from aoSystem.defineAoSystem import defineAoSystem
import fourier.FourierUtils as FourierUtils
import psfr.psfrUtils as psfrUtils

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000

class psfR:
    """
    """
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
        if hasattr(self,'Dcircle'):
            D  = self.Dcircle
        else:
            D = self.D          
        self.pistonFilter_ = FourierUtils.pistonFilter(D,self.k2_)
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
        if self.circularAOarea:
            self.mskOut_   = (self.k2_ >= kc2)
            self.mskIn_    = (self.k2_ < kc2)
        else:
            self.mskOut_   = np.logical_or(abs(self.kxky_[0]) >= self.kc_, abs(self.kxky_[1]) >= self.kc_)
            self.mskIn_    = np.logical_and(abs(self.kxky_[0]) < self.kc_, abs(self.kxky_[1]) < self.kc_)
        self.psdKolmo_     = 0.0229 * self.mskOut_* ((1.0 /self.atm.L0**2) + self.k2_) ** (-11.0/6.0)
        self.wfe_fit_norm  = np.sqrt(np.trapz(np.trapz(self.psdKolmo_,self.kxky_[1][0]),self.kxky_[1][0]))
    
    # INIT
    def __init__(self,file,trs):
        """
        """
        # READ PARFILE        
        tstart = time.time()
        
        # PARSING INPUTS
        self.file   = file
        self.status = defineAoSystem(self,file,circularAOarea='circle',Dcircle=None)
        self.trs    = trs
        
        if self.status:
            
            #INSTANTIATING THE ANGULAR FREQUENCIES
            self.nOtf = self.nPix * self.kRef_
            self.U_, self.V_, self.U2_, self.V2_, self.UV_, self.otfDL =\
            FourierUtils.instantiateAngularFrequencies(self.tel,self.nOtf,self.sampRef,self.wvlCen,fact=self.sampCen/2)
            
            # DEFINING BOUNDS
            self.bounds = self.defineBounds()
        
            # COMPUTING THE STATIC OTF IF A PHASE MAP IS GIVEN
            self.otfStat, _ = FourierUtils.getStaticOTF(self.tel,self.nOtf,self.sampRef,self.wvlRef, apodizer=self.apodizer,opdMap_ext=self.opdMap_ext)
            
            # INSTANTIATING THE FOURIER MODEL
            self.fao = fourierModel(file,calcPSF=False,display=False,aoFilter='circle')
            
            # INSTANTIATING THE FITTING PHASE STRUCTURE FUNCTION
            self.dphi_fit = self.fittingPhaseStructureFunction(1)
            
            # INSTANTIATING THE ALIASING PHASE STRUCTURE FUNCTION
            self.dphi_alias = self.aliasingPhaseStructureFunction(1)
            
            # INSTANTIATING THE AO RESIDUAL PHASE STRUCTURE FUNCTION 
            self.dphi_ao = self.aoResidualStructureFunction()
                
            # INSTANTIATING THE TT RESIDUAL PHASE STRUCTURE FUNCTION IN LGS MODE
            # IN NGS MODE, THE TIP-TILT CONTRIBUTION IS CONTAINED IN THE WFS MEASUREMENTS 
            if self.gs.height:
                self.dphi_tt = self.tipTiltPhaseStructureFunction()
            else:
                self.dphi_tt = 0
            
            # INSTANTIATING THE ANISOPLANATISM PHASE STRUCTURE FUNCTION IF ANY
            # ANISOKINETISM IS MISSING
            
            # COMPUTING THE DETECTOR PIXEL TRANSFER FUNCTION
            if self.trs.instrument != 'simulation':
                self.otfPixel = self.pixelOpticalTransferFunction()
            
        self.t_init = 1000*(time.time()  - tstart)
    
    def _repr__(self):
        return 'PSF-Reconstruction model'
    
    def defineBounds(self):
          #r0, gao, gtt, F , dx , dy , bg , stat
          _EPSILON = np.sqrt(sys.float_info.epsilon)
          
          # Bounds on r0
          bounds_down = list(np.ones(self.atm.nL)*_EPSILON)
          bounds_up   = list(np.inf * np.ones(self.atm.nL))            
          # optical gains 
          bounds_down += [0,0]
          bounds_up   += [np.inf,np.inf]         
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
      
    def fittingPhaseStructureFunction(self,r0):
        return r0**(-5/3) * np.real(fft.fftshift(FourierUtils.cov2sf(FourierUtils.psd2cov(self.psdKolmo_,2*self.fao.kc/self.fao.resAO))))
    
    def aliasingPhaseStructureFunction(self,r0):
        # computing the aliasing PSD over the AO-corrected area
        self.psdAlias_ = self.fao.aliasingPSD()
        # zero-padding the PSD
        self.psdAlias_ = FourierUtils.enlargeSupport(self.psdAlias_,self.nOtf/self.fao.resAO)
        # computing the aliasing phase structure function
        dphi_alias = r0**(-5/3) * np.real(fft.fftshift(FourierUtils.cov2sf(FourierUtils.psd2cov(self.psdAlias_,2*self.fao.kc/self.fao.resAO))))
        # interpolating the phase structure function if required
        if dphi_alias.shape[0] != self.nOtf:
            dphi_alias = FourierUtils.interpolateSupport(dphi_alias,self.nOtf,kind='spline')
        return dphi_alias
    
    def aoResidualStructureFunction(self,method='slopes-based',basis='Vii'):
        """
        """
        # computing the empirical covariance matrix of the AO-residual OPD in the DM actuators domain
        if method == 'slopes-based':
            du = self.trs.res
        elif method == 'dm-based':
            du = np.diff(self.trs.dm.com,axis=1)/self.fao.loopGain    
        Cao  =  du * du.T/du.shape[1]
        # Unbiasing noise and accounting for the wave number
        Cao = (2*np.pi/self.wvlCen)**2 * (Cao - self.trs.noise.Cn_ao)
        # Computing the phase structure function
        dphi_ao = psfrUtils.modes2Otf(Cao,self.trs.mat.dmIF,self.tel.pupil,self.nOtf,'basis',basis)
        
        return dphi_ao
            
    def tipTiltPhaseStructureFunction(self):
        """
        """
        # computing the empirical covariance matrix of the residual tip-tilt
        Ctt   =  self.trs.tipTilt.com * self.trs.tipTilt.com.T/self.trs.tipTilt.nExp
        # computing the coefficients of the Gaussian Kernel
        Guu = (2*np.pi/self.wvlCen)**2 *(Ctt - self.trs.noise.Cn_tt) 
        # rotating the axes
        ang = self.trs.wfs.pupilAngle * np.pi/180
        Ur  = self.U_*np.cos(ang) + self.V_*np.sin(ang)
        Vr  =-self.U_*np.sin(ang) + self.V_*np.cos(ang)  
        # computing the Gaussian-Kernel
        dphi_tt   = Guu[0,0]*Ur**2 + Guu[1,1]*Vr**2 + Guu[0,1]*Ur*Vr.T + Guu[1,0]*self.Vr*self.Ur.T
        
        return dphi_tt
        
    def pixelOpticalTransferFunction(self):
        """
        """
        otfPixel = np.sinc(self.U_*self.psInMas/rad2mas)* np.sinc(self.V_*self.psInMas/rad2mas);
        return otfPixel


    def __call__(self):
        otfTot = self.otfNCPA*self.otfParallel*self.otfTipTilt*self.otfPerpendicular
        
        if self.nqSmpl == 1:            
            # Interpolate the OTF to set the PSF FOV
            otfTot = FourierUtils.interpolateSupport(otfTot,self.fovInPixel)
            otfTot = otfTot/otfTot.max()
            psf    = FourierUtils.otf2psf(otfTot)
        elif self.nqSmpl >1:
            # Zero-pad the OTF to set the PSF pixel scale
            otfTot = FourierUtils.enlargeSupport(otfTot,self.nqSmpl)
            # Interpolate the OTF to set the PSF FOV
            otfTot = FourierUtils.interpolateSupport(otfTot,self.fovInPixel)
            otfTot = otfTot/otfTot.max()
            psf    = FourierUtils.otf2psf(otfTot)
        else:
            # Interpolate the OTF at high resolution to set the PSF FOV
            otfTot = FourierUtils.interpolateSupport(otfTot,int(np.round(self.fovInPixel/self.nqSmpl)))
            otfTot = otfTot/otfTot.max()
            psf    = FourierUtils.otf2psf(otfTot)
            # Interpolate the PSF to set the PSF pixel scale
            psf    = FourierUtils.interpolateSupport(psf,self.fovInPixel)
                   
        return psf