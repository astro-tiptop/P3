#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 22:20:28 2018

@author: omartin
"""
#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
import numpy.fft as fft
import time
import sys as sys

import fourier.fourierModel as fourierModel
import fourier.FourierUtils as FourierUtils
from aoSystem.defineAoSystem import defineAoSystem

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
            FourierUtils.instantiateAngularFrequencies(self.tel,self.nOtf,self.sampRef,self.wvlRef)
            
            # DEFINING BOUNDS
            self.bounds = self.defineBounds()
        
            # COMPUTING THE STATIC OTF IF A PHASE MAP IS GIVEN
            self.otfStat, _ =\
            FourierUtils.getStaticOTF(self.tel,self.nOtf,self.sampRef,self.wvlRef,\
                                      apodizer=self.apodizer,opdMap_ext=self.opdMap_ext)
            
            
            # INSTANTIATING THE FOURIER MODEL
            self.fao = fourierModel(file)
            
            # INSTANTIATING THE FITTING PHASE STRUCTURE FUNCTION
            self.dphi_fit = self.fittingPhaseStructureFunction(self.atm.r0)
            
            # INSTANTIATING THE ALIASING PHASE STRUCTURE FUNCTION
            self.dphi_alias = self.aliasingPhaseStructureFunction(self.atm.r0)
            
            # INSTANTIATING THE AO RESIDUAL PHASE STRUCTURE FUNCTION 
            
            # INSTANTIATING THE TT RESIDUAL PHASE STRUCTURE FUNCTION IN LGS MODE
            
            # INSTANTIATING THE ANISOPLANATISM PHASE STRUCTURE FUNCTION IF ANY
            
            # COMPUTING THE DETECTOR PIXEL TRANSFER FUNCTION
            
            
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
        return r0**(-5/3) * FourierUtils.cov2sf(FourierUtils.psd2cov(self.psdKolmo_))
    
    def aliasingPhaseStructureFunction(self,r0):
        self.psdAlias_ = self.fao.aliasingPSD()
        return r0**(-5/3) * FourierUtils.cov2sf(FourierUtils.psd2cov(self.psdAlias_))
    
#    def otfNCPA(self,iSrc=0):
#        if np.isscalar(self.ncpaMap):
#            self.otfNCPA =1
#        else:
#            ncpaMap     = self.ncpaMap*(self.wvlNcpa/self.src.wvl[iSrc])
#            E           = FourierUtils.enlargeSupport(self.pupLR*np.exp(complex(0,1)*ncpaMap),2)
#            otfNCPA     = fft.fftshift(FourierUtils.fftCorrel(E,E))
#            otfNCPA     = FourierUtils.interpolateSupport(otfNCPA,self.nOtf)
#            self.otfNCPA= otfNCPA/otfNCPA.max()    
#
#    def otfPerpendicular(self,iSrc=0):
#        """
#        """
#        self.fao.atm.wvl = self.src.wvl[iSrc]              
#        # Get PSDs
#        dk          = 2*self.fao.kc/self.fao.resAO        
#        pFit        = self.fao.fittingPSD(self.fao.kx,self.fao.ky,aoFilter='circle')
#        pAl         = self.fao.aliasingPSD(self.fao.kx,self.fao.ky,aoFilter='circle')
#        pAl         = FourierUtils.enlargeSupport(pAl,self.nTimes)        
#        self.psdPerp= pFit + pAl
#        # Get OTFs
#        otfPerp      = fft.fftshift(FourierUtils.psd2otf(self.psdPerp,dk))
#        otfPerp      = FourierUtils.interpolateSupport(otfPerp,self.nOtf)
#        self.otfPerp = otfPerp/otfPerp.max()
#        otfFit       = fft.fftshift(FourierUtils.psd2otf(pFit,dk))
#        otfFit       = FourierUtils.interpolateOtf(otfFit,self.nOtf)
#        self.otfFit  = otfFit/otfFit.max()
#        otfAl        = fft.fftshift(p3utils.psd2otf(pAl,dk))
#        otfAl        = FourierUtils.interpolateSupport(otfAl,self.nOtf)
#        self.otfAl   = otfAl/otfAl.max()
                
                        
    def otfParallel(self,iSrc=0):
        """
        """
        u     = self.waveFront*self.volt2meter
        Cnn   = FourierUtils.getNoiseVariance(u,nshift=1)
        Cuu   = u*u.T/u.shape[1] - Cnn
        Cphi  = Cuu*(2*np.pi/self.src.wvl[iSrc])**2
        Cphi  = self.modes*Cphi*self.modes
        otfDM = FourierUtils.zonalCovarianceToOtf(Cphi,self.nOtf,self.pitch,self.idxVal)
        self.otfDM = otfDM/self.otfDL
    
    def otfTipTilt(self,iSrc=0):
        """
        """
        utt   = self.tipTilt*self.tilt2meter
        utt   = utt - np.mean(utt,axis=1)
        Cnnt  = FourierUtils.getNoiseVariance(utt,nshift=1)
        Ctt   = utt*utt.T/utt.shape[1] - Cnnt
        Cuut  = Ctt*(2*np.pi/self.src.wvl[iSrc])**2
        # Get tip-tilt phase structure function
        x     = np.linspace(-self.tel.D/2,self.tel.D/2,self.nOtf)
        X,Y   = np.meshgrid(x,x)
        Dtt   = Cuut[0,0]*X**2 + Cuut[1,1]*Y**2 + Cuut[0,1]*X*Y + Cuut[1,0]*Y.T*X.T
        
        self.otfTT = np.exp(-Dtt*self.tel.D**2/16)
                                   
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