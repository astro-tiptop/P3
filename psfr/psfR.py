#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 22:20:28 2018

@author: omartin
"""
import numpy as np
import numpy.fft as fft
import p3utils
import spatialFrequency
from STORM.psfmodel import ParametricPSF

class psfModel(ParametricPSF):
    """
    """
    # CONSTRUCTOR
    def __init__(self,trs,tel,atm,src,psInMas,fovInArcsec,modes=0,ncpaMap=0,ncpaWvl=0):
        """
        """
        # READ PARFILE
        
        #PARSING INPUTS
        self.tel = tel
        self.atm = atm
        self.src = src
        self.fao = fao
        self.ncpaMap = ncpaMap
        self.ncpaWvl = ncpaWvl                         
        # Set the otf/psf resolution regarding the deisred PSF FOV/pixel scale
        fovInPixel  = int((np.ceil(2e3*fovInArcsec/psInMas))/2)
        fovInPixel  = max([fovInPixel,2*self.resAO])
        self.nTimes = int(np.round(fovInPixel/self.resAO))
        self.nPh    = 2*self.nActu+1
        self.nOtf   = self.nPh*self.nTimes
        self.pitch  = tel.D/(self.nPh-1)     
        # Define the influence function
        if np.isscalar(modes):
            modes = 1
        else:
            self.modes = modes
            
        self.pupLR   = p3utils.interpolateSupport(tel.pupil,self.nPh)
        self.idxVal  = self.pupLR.astype('bool')                         
        # Grab telemetry        
        self.wavefront = trs.wavefront
        self.volt2meter= trs.volt2meter
        self.tipTilt   = trs.tipTilt
        self.tilt2meter= trs.tilt2meter
        # Set up the spatialFrequency class
        self.fao = spatialFrequency(self.tel,self.atm,self.src,self.nActu,\
                               self.noiseVariance,self.loopGain,self.samplingTime\
                               ,self.latency,self.nPh)
        # Get OTFs
        self.otfDL(iSrc=0)
        self.otfNCPA(iSrc=0)
        self.otfPerpendicular(iSrc=0)
        self.otfParallel(iSrc=0)
        self.otfTipTilt(iSrc=0)
        
    def otfDL(self,iSrc=0):
        """
        """
        C0 = np.zeros((self.nPh**2,self.nPh**2)) 
        self.otfDL = p3utils.zonalCovarianceToOtf(C0,self.nOtf,self.tel.D,\
                                                  self.pitch,self.idxVal)

    def otfNCPA(self,iSrc=0):
        if np.isscalar(self.ncpaMap):
            self.otfNCPA =1
        else:
            ncpaMap     = self.ncpaMap*(self.wvlNcpa/self.src.wvl[iSrc])
            E           = p3utils.enlargeSupport(self.pupLR*np.exp(complex(0,1)*ncpaMap),2)
            otfNCPA     = fft.fftshift(p3utils.fftCorrel(E,E))
            otfNCPA     = p3utils.interpolateSupport(otfNCPA,self.nOtf)
            self.otfNCPA= otfNCPA/otfNCPA.max()    

    def otfPerpendicular(self,iSrc=0):
        """
        """
        self.fao.atm.wvl = self.src.wvl[iSrc]              
        # Get PSDs
        dk          = 2*self.fao.kc/self.fao.resAO        
        pFit        = self.fao.fittingPSD(self.fao.kx,self.fao.ky,aoFilter='circle')
        pAl         = self.fao.aliasingPSD(self.fao.kx,self.fao.ky,aoFilter='circle')
        pAl         = p3utils.enlargeSupport(pAl,self.nTimes)        
        self.psdPerp= pFit + pAl
        # Get OTFs
        otfPerp      = fft.fftshift(p3utils.psd2otf(self.psdPerp,dk))
        otfPerp      = p3utils.interpolateSupport(otfPerp,self.nOtf)
        self.otfPerp = otfPerp/otfPerp.max()
        otfFit       = fft.fftshift(p3utils.psd2otf(pFit,dk))
        otfFit       = p3utils.interpolateOtf(otfFit,self.nOtf)
        self.otfFit  = otfFit/otfFit.max()
        otfAl        = fft.fftshift(p3utils.psd2otf(pAl,dk))
        otfAl        = p3utils.interpolateSupport(otfAl,self.nOtf)
        self.otfAl   = otfAl/otfAl.max()
                
                        
    def otfParallel(self,iSrc=0):
        """
        """
        u     = self.waveFront*self.volt2meter
        Cnn   = p3utils.getNoiseVariance(u,nshift=1)
        Cuu   = u*u.T/u.shape[1] - Cnn
        Cphi  = Cuu*(2*np.pi/self.src.wvl[iSrc])**2
        Cphi  = self.modes*Cphi*self.modes
        otfDM = p3utils.zonalCovarianceToOtf(Cphi,self.nOtf,self.pitch,self.idxVal)
        self.otfDM = otfDM/self.otfDL
    
    def otfTipTilt(self,iSrc=0):
        """
        """
        utt   = self.tipTilt*self.tilt2meter
        utt   = utt - np.mean(utt,axis=1)
        Cnnt  = p3utils.getNoiseVariance(utt,nshift=1)
        Ctt   = utt*utt.T/utt.shape[1] - Cnnt
        Cuut  = Ctt*(2*np.pi/self.src.wvl[iSrc])**2
        # Get tip-tilt phase structure function
        x     = np.linspace(-self.tel.D/2,self.tel.D/2,self.nOtf)
        X,Y   = np.meshgrid(x,x)
        Dtt   = Cuut[0,0]*X**2 + Cuut[1,1]*Y**2 + Cuut[0,1]*X*Y + Cuut[1,0]*Y.T*X.T
        
        self.otfTT = np.exp(-Dtt*self.tel.D**2/16)
                                   
    def model(self):
        otfTot = self.otfNCPA*self.otfParallel*self.otfTipTilt*self.otfPerpendicular
        
        if self.nqSmpl == 1:            
            # Interpolate the OTF to set the PSF FOV
            otfTot = p3utils.interpolateSupport(otfTot,self.fovInPixel)
            otfTot = otfTot/otfTot.max()
            psf    = p3utils.otf2psf(otfTot)
        elif self.nqSmpl >1:
            # Zero-pad the OTF to set the PSF pixel scale
            otfTot = p3utils.enlargeSupport(otfTot,self.nqSmpl)
            # Interpolate the OTF to set the PSF FOV
            otfTot = p3utils.interpolateSupport(otfTot,self.fovInPixel)
            otfTot = otfTot/otfTot.max()
            psf    = p3utils.otf2psf(otfTot)
        else:
            # Interpolate the OTF at high resolution to set the PSF FOV
            otfTot = p3utils.interpolateSupport(otfTot,int(np.round(self.fovInPixel/self.nqSmpl)))
            otfTot = otfTot/otfTot.max()
            psf    = p3utils.otf2psf(otfTot)
            # Interpolate the PSF to set the PSF pixel scale
            psf    = p3utils.interpolateSupport(psf,self.fovInPixel)
                   
        return psf