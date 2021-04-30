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

from aoSystem.aoSystem import aoSystem
from aoSystem.fourierModel import fourierModel
from aoSystem.anisoplanatismModel import anisoplanatismStructureFunction
import aoSystem.FourierUtils as FourierUtils
import psfr.psfrUtils as psfrUtils
from aoSystem.frequencyDomain import frequencyDomain as frequencyDomain

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000

class psfR:
    """
    """
    # INIT
    def __init__(self,trs,nLayer=None,theta_ext=0):
        """
        """
        # READ PARFILE        
        tstart = time.time()
        
        # PARSING INPUTS
        if hasattr(trs,'path_ini') == False:
            print('ERROR : no .ini file attached with the telemetry object')
            return
        self.path_ini  = trs.path_ini
        self.trs       = trs
        self.theta_ext = theta_ext
        self.ao        = aoSystem(self.path_ini)
        self.tag       = 'PSF-R'
        if self.ao.error == False:
            
            # DEFINING THE FREQUENCY DOMAIN
            self.freq = frequencyDomain(self.ao)
            
            # DEFINING BOUNDS
            self.bounds = self.defineBounds()
        
            # INSTANTIATING THE FOURIER MODEL
            self.fao = fourierModel(self.path_ini,calcPSF=False,display=False)
            
            # INSTANTIATING THE FITTING PHASE STRUCTURE FUNCTION FOR r0=1m
            self.dphi_fit = self.fittingPhaseStructureFunction(1)
            
            # INSTANTIATING THE ALIASING PHASE STRUCTURE FUNCTION FOR r0=1m
            self.dphi_alias = self.aliasingPhaseStructureFunction(1)
            
            # INSTANTIATING THE AO RESIDUAL PHASE STRUCTURE FUNCTION 
            self.dphi_ao = self.aoResidualStructureFunction()
                
            # INSTANTIATING THE TT RESIDUAL PHASE STRUCTURE FUNCTION IN LGS MODE
            # IN NGS MODE, THE TIP-TILT CONTRIBUTION IS CONTAINED IN THE WFS MEASUREMENTS 
            if self.trs.aoMode == 'LGS':
                self.dphi_tt = self.tipTiltPhaseStructureFunction()
            else:
                self.dphi_tt = 0
            
            # INSTANTIATING THE ANISOPLANATISM PHASE STRUCTURE FUNCTION IF ANY
            self.dphi_ani = anisoplanatismStructureFunction(\
            self.ao.tel,self.ao.atm,self.ao.src,self.ao.lgs,self.ao.ngs,\
            self.freq.nOtf,self.freq.sampRef,Hfilter=self.trs.mat.Hdm)
            
            # COMPUTING THE DETECTOR PIXEL TRANSFER FUNCTION
            if self.trs.tel.name != 'simulation':
                self.otfPixel = self.pixelOpticalTransferFunction()
            else:
                self.otfPixel = 1.0
            
        self.t_init = 1000*(time.time()  - tstart)
    
    def _repr__(self):
        return 'PSF-Reconstruction model'
   
    def defineBounds(self):
          #r0, gao, gtt, F , dx , dy , bg , stat
          _EPSILON = np.sqrt(sys.float_info.epsilon)
          
          # Bounds on r0
          bounds_down = list(np.ones(self.ao.atm.nL)*_EPSILON)
          bounds_up   = list(np.inf * np.ones(self.ao.atm.nL))            
          # optical gains 
          bounds_down += [0,0]
          bounds_up   += [np.inf,np.inf]         
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
      
    def fittingPhaseStructureFunction(self,r0):
        return r0**(-5/3) * np.real(fft.fftshift(FourierUtils.cov2sf(FourierUtils.psd2cov(self.freq.psdKolmo_,2*self.freq.kc_/self.freq.resAO))))
    
    def aliasingPhaseStructureFunction(self,r0):
        # computing the aliasing PSD over the AO-corrected area
        self.psdAlias_ = self.fao.aliasingPSD()
        
        # zero-padding the PSD
        self.psdAlias_ = FourierUtils.enlargeSupport(self.psdAlias_,self.freq.nOtf/self.freq.resAO)
       
        # computing the aliasing phase structure function
        dphi_alias = r0**(-5/3) * np.real(fft.fftshift(FourierUtils.cov2sf(FourierUtils.psd2cov(self.psdAlias_,2*self.freq.kc_/self.freq.resAO))))
        
        # interpolating the phase structure function if required
        if dphi_alias.shape[0] != self.freq.nOtf:
            dphi_alias = FourierUtils.interpolateSupport(dphi_alias,self.freq.nOtf,kind='spline')
        return dphi_alias
    
    def aoResidualStructureFunction(self,method='slopes-based',basis='Vii'):
        """
        """
        # computing the empirical covariance matrix of the AO-residual OPD in the DM actuators domain
        if method == 'slopes-based':
            du = self.trs.rec.res
        elif method == 'dm-based':
            du = np.diff(self.trs.dm.com,axis=0)/self.ao.rtc.holoop['gain']    
        Cao  =  np.matmul(du.T,du)/du.shape[0]
       
        # Unbiasing noise and accounting for the wave number
        Cao = (2*np.pi/self.freq.wvlRef)**2 * (Cao - self.trs.wfs.Cn_ao)
        
        # Computing the phase structure function
        _,dphi_ao = psfrUtils.modes2Otf(Cao,self.trs.mat.dmIF,self.ao.tel.pupil,self.freq.nOtf,basis=basis,samp=self.freq.sampRef/2)
        
        return dphi_ao
            
    def tipTiltPhaseStructureFunction(self):
        """
        """
        # computing the empirical covariance matrix of the residual tip-tilt
        Ctt   =  np.matmul(self.trs.tipTilt.com.T,self.trs.tipTilt.com)/self.trs.tipTilt.nExp
        
        # computing the coefficients of the Gaussian Kernel in rad^2
        Guu = (2*np.pi/self.freq.wvlRef)**2 *(Ctt - self.trs.tipTilt.Cn_tt) 
        
        # rotating the axes
        ang = self.trs.tel.pupilAngle * np.pi/180
        Ur  = self.freq.U_*np.cos(ang) + self.freq.V_*np.sin(ang)
        Vr  =-self.freq.U_*np.sin(ang) + self.freq.V_*np.cos(ang)  
        
        # computing the Gaussian-Kernel
        dphi_tt   = Guu[0,0]*Ur**2 + Guu[1,1]*Vr**2 + Guu[0,1]*Ur*Vr.T + Guu[1,0]*Vr*Ur.T
        
        return dphi_tt
    
    def pixelOpticalTransferFunction(self):
        """
        """
        #note : self.U_/V_ ranges ar -1 to 1
        otfPixel = np.sinc(self.freq.U_/2)* np.sinc(self.freq.V_/2)
        return otfPixel

    def TotalPhaseStructureFunction(self,r0,gho,gtt,Cn2=[]):
        # On-axis phase structure function
        SF   = gho*self.dphi_ao + gtt*self.dphi_tt + r0**(-5/3) * (self.dphi_fit + self.dphi_alias)
        # Anisoplanatism phase structure function
        if self.freq.isAniso and (len(Cn2) == self.freq.dani_ang.shape[1]):
            SF = SF[:,:,np.newaxis] + (self.freq.dphi_ani * Cn2).sum(axis=2)
        else:
            SF = np.repeat(SF[:,:,np.newaxis],self.ao.src.nSrc,axis=2)
        return SF/(2*np.pi*1e-9/self.freq.wvlRef)**2
    
    def __call__(self,x0,nPix=None):
                
        # ----------------- GETTING THE PARAMETERS
        # Cn2 profile
        nL   = self.ao.atm.nL
        if nL > 1: # fit the Cn2 profile
            Cn2  = np.asarray(x0[0:nL])
            r0   = np.sum(Cn2)**(-3/5)
        else: #fit the r0
            Cn2= []
            r0 = x0[0]
            
        # PSD
        gho = x0[nL]
        gtt = x0[nL+1]
        
        # Astrometry/Photometry/Background
        x0_stellar = np.array(x0[nL+2:nL+4+3*self.ao.src.nSrc])
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
        if len(x0) > nL + 2 + 3*self.ao.src.nSrc + 1:
            x0_stat = list(x0[nL+3+3*self.ao.src.nSrc:])
        else:
            x0_stat = []   
         
        # ----------------- GETTING THE PHASE STRUCTURE FUNCTION    
        self.SF = self.TotalPhaseStructureFunction(r0,gho,gtt,Cn2=Cn2)
        
        # ----------------- COMPUTING THE PSF
        PSF, self.SR = FourierUtils.SF2PSF(self.SF,self.freq,self.ao,\
                        F=F,dx=dx,dy=dy,bkg=bkg,nPix=nPix,xStat=x0_stat,otfPixel=self.otfPixel)
        return PSF