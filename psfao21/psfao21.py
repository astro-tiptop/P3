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

from maoppy.psfmodel import Psfao
from maoppy.instrument import Instrument
import aoSystem.FourierUtils as FourierUtils
from aoSystem.aoSystem import aoSystem as aoSys
from aoSystem.frequencyDomain import frequencyDomain as frequencyDomain

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000

class psfao21:
    # INIT
    def __init__(self, path_ini, path_root='', 
                 fitCn2=False, otfPixel=1, coo_stars=None, 
                 param_labels = ['Cn2', 'C', 'A', 'ax', 'p', 'theta', 'beta',
                                 'jitterX', 'jitterY', 'jitterXY',
                                 'F', 'dx', 'dy', 'bkg', 'stat']):
        '''
            Instantiating the psfao21 model : 
                - creating the spatial frequency domain, 
                  including the computation of the static OTF and anisoplanatism
                - instantiating the psfao19 model
                - instantiating the bounds
        '''
        tstart = time.time()
        self.tag = 'PSFAO21 MODEL'
        self.param_labels = param_labels
        
        # INSTANTIATING THE AOSYSTEM CLASS
        self.file = path_ini
        self.ao = aoSys(path_ini,path_root=path_root,coo_stars=coo_stars)    
        
        # GETTING INPUTS
        self.isStatic = self.ao.tel.nModes > 0
        self.otfPixel = otfPixel
        
        if self.ao.error==False:
            
            # DEFINING THE FREQUENCY DOMAIN
            self.freq = frequencyDomain(self.ao)
            
            # INSTANTIATING THE MAOPPY MODEL
            npix = self.ao.cam.fovInPix
            system = Instrument(D=self.ao.tel.D, 
                                occ=self.ao.tel.obsRatio, 
                                res=self.ao.tel.resolution, 
                                Nact=self.ao.dms.nActu1D,
                                gain=self.ao.cam.gain,
                                ron=self.ao.cam.ron)
            
            samp_min = min(rad2mas * self.ao.src.wvl/(self.ao.cam.psInMas*self.ao.tel.D))
            self.psfao_19 = Psfao((npix, npix), system=system, samp=samp_min)


            # ONE OF SEVERAL FRAMES
            self.isCube = any(rad2mas * self.ao.src.direction[0]/self.ao.cam.psInMas > self.freq.nPix) \
            or all(rad2mas * self.ao.src.direction[1]/self.ao.cam.psInMas > self.freq.nPix)
            
            # DEFINING THE NUMBER OF PSF PARAMETERS
            self.n_param_atm = min(self.ao.atm.nL, self.ao.dms.nRecLayers)
            self.n_param_dphi = 6
            
            # DEFINING BOUNDS
            self.bounds = self.define_bounds()
                        
        self.t_init = 1000*(time.time()  - tstart)
        
    def __repr__(self):
        s = '---------------------------------------------' + self.tag  + '--------------------------------------------- \n\n'
        s+= self.ao.__repr__() + '\n'
        s+= self.freq.__repr__() + '\n'
        return s
    
    def define_bounds(self):
        """
        Defining bounds on the PSFAO21 parameters based physics-based a priori
        """
        #Cn2/r0 , C , A , ax , p , theta , beta , sx , sy , sxy , F , dx , dy , bg , stat
        _EPSILON = np.sqrt(sys.float_info.epsilon)
          
        # Bounds on r0
        bounds_down = list(np.ones(self.ao.atm.nL)*_EPSILON)
        bounds_up   = list(np.inf * np.ones(self.ao.atm.nL))          
        # PSD Parameters
        bounds_down += [0,0,_EPSILON,_EPSILON,-np.pi,1+_EPSILON]
        bounds_up   += [5e-2,np.inf,np.inf,np.inf,np.pi,10]         
        # Jitter
        bounds_down += [0,0,-1]
        bounds_up   += [np.inf,np.inf,1]
        # Photometry
        bounds_down += list(np.zeros(self.ao.src.nSrc * self.freq.nWvl))
        bounds_up   += list(np.inf*np.ones(self.ao.src.nSrc * self.freq.nWvl))
        # Astrometry
        bounds_down += list(-self.freq.nPix//2 * np.ones(2*self.ao.src.nSrc * self.freq.nWvl))
        bounds_up   += list( self.freq.nPix//2 * np.ones(2*self.ao.src.nSrc * self.freq.nWvl))
        # Background
        bounds_down += list(-np.inf * np.ones(self.freq.nWvl))
        bounds_up   += list(np.inf * np.ones(self.freq.nWvl))
        # Static aberrations
        bounds_down += list(-self.freq.wvlRef/2*1e9 * np.ones(self.ao.tel.nModes))
        bounds_up   += list(self.freq.wvlRef/2 *1e9 * np.ones(self.ao.tel.nModes))
        
        return (bounds_down,bounds_up)
        
    def update_bounds(self,xfinal,xerr,sig=5):
        '''
            Defining bounds on the PSFAO21 parameters based on the first step 
            of the split fitting
        '''
        
        # lower bounds
        bounds_low_psd = list(xfinal[0:7] - sig/3*xerr[0:7])
        bounds_low     = bounds_low_psd\
                        + [-np.inf,-np.inf,-np.inf,0,-np.inf,-np.inf,-np.inf]\
                        + [-self.freq.wvlRef*1e9/2,]*self.ao.tel.nModes
        
        #upper bounds
        bounds_up_psd  = list(xfinal[0:7] + sig/3*xerr[0:7])
        bounds_up      = bounds_up_psd\
                        + [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]\
                        + [self.freq.wvlRef*1e9/2,]*self.ao.tel.nModes
        
        return (bounds_low,bounds_up)
        
    def get_power_spectrum_density(self, x0, grad=False):
        '''
            Define the model of the AO-corrected power spectrum density of the 
            electric field phase in the pupil plan
        '''        
        if grad:
            psd, self.wfe, grad, integral_grad = self.psfao_19.psd(x0, grad=True)
        else:
            psd, self.wfe = self.psfao_19.psd(x0, grad=False)
                
        # wavefront errors
        cte = self.freq.wvlRef*1e9/2/np.pi
        self.wfe = np.sqrt(self.wfe) * cte
        self.wfe_fit = np.sqrt(x0[0]**(-5/3)) * self.freq.wfe_fit_norm  * cte
        
        return psd
    
    def get_opd_structure_function(self,Cn2=[]):
        '''
            Define the spatially-variable phase structure function from the
            psd model and the anisoplanatism model
        '''
        
        #covariance map        
        real_fft = fft.rfft2(fft.fftshift(self.psd)) / (self.ao.tel.D * self.freq.sampRef)**2       
        Bphi = fft.fftshift(FourierUtils._rfft2_to_fft2(self.psd.shape,real_fft))

        # On-axis phase structure function
        SF = np.real(2*(Bphi.max() - Bphi))
        
        # Anisoplanatism phase structure function
        if self.freq.isAniso and (len(Cn2) == self.freq.dani_ang.shape[1]):
            SF = SF[:, :, np.newaxis] + (self.freq.dphi_ani * Cn2).sum(axis=2)
        else:
            SF = np.repeat(SF[:, :, np.newaxis], self.ao.src.nSrc, axis=2)
        
        return SF/(2*np.pi*1e-9/self.freq.wvlRef)**2
     
        
    def __call__(self, x0, nPix=None):
        '''
            Returns the 4D array containing 2D psf for each wavelength 
            and field direction
        '''
        
        # ----------------- GETTING THE PARAMETERS
        (Cn2, r0, x0_dphi, x0_jitter, x0_stellar, x0_stat) = FourierUtils.sort_params_from_labels(self,x0)
        
        # ----------------- GETTING THE PHASE STRUCTURE FUNCTION
        self.psd = self.get_power_spectrum_density([r0]+ x0_dphi)
        self.SF = self.get_opd_structure_function(Cn2=Cn2)
        
        # ----------------- COMPUTING THE PSF
        PSF, self.SR = FourierUtils.sf_3D_to_psf_4D(self.SF, 
                                                    self.freq, 
                                                    self.ao,
                                                    x_jitter = x0_jitter, 
                                                    x_stat = x0_stat,
                                                    x_stellar = x0_stellar, 
                                                    nPix = nPix,
                                                    otfPixel = self.otfPixel)
        
        return PSF
    