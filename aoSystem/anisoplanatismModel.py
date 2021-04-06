#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:54:57 2021

@author: omartin
"""
import numpy as np
import scipy.special as ssp
import numpy.fft as fft


def instantiateAnisoplanatism(src,gs,atm,tel,nPix,samp,method='psd'):
        
        def mcDonald(x):
            out = 3/5 * np.ones_like(x)
            idx  = x!=0
            if np.any(idx==False):
                out[idx] = x[idx] ** (5/6) * ssp.kv(5/6,x[idx])/(2**(5/6) * ssp.gamma(11/6))
            else:
                out = x ** (5/6) * ssp.kv(5/6,x)/(2**(5/6) * ssp.gamma(11/6))
            return out
        
        def Ialpha(x,y):
            return mcDonald(np.hypot(x,y))
        
        nSrc    = src.nSrc
        nLayer  = atm.nL
        Hs      = atm.heights * tel.airmass
        
        if method == 'psd':   
            # PSD METHOD : FASTER PSD METHOD WORKING FOR ANGULAR ANISOPLANATISM        
            Dani_l = np.zeros((nSrc,nLayer,nPix,nPix))
            L  = (tel.D * samp)**2
            
            # DEFINING THE FREQUENCY DOMAIN
            
            
            cte  = (24*ssp.gamma(6/5)/5)**(5/6)*(ssp.gamma(11/6)**2./(2.*np.pi**(11/3)))
            kern = cte * ((1.0 /atm.L0**2) + self.k2_) ** (-11/6)
            kern = self.pistonFilter_ * kern
            kern[self.nOtf//2,self.nOtf//2] = 0
            
            # LOOP ON SOURCES
            for s in range(nSrc):
                ax = src.direction[0] - gs.direction[0]
                ay = src.direction[1] - gs.direction[1]
                for iSrc in range(nSrc):
                    thx = ax[iSrc]
                    thy = ay[iSrc]            
                    if thx !=0 or thy !=0:
                        for l  in range(nLayer):
                            zl = Hs[l]
                            if zl !=0:
                                psd = 2*self.mskIn_*kern*(1 - np.cos(2*np.pi*zl*(self.kxky_[0]*thx + self.kxky_[1]*thy)))    
                                cov = fft.fftshift(fft.fft2(fft.fftshift(psd)))/L
                                Dani_l[iSrc,l] = np.real(2*(cov.max()-cov))
        else:        
            # FLICKER'S MODEL - need to be generalize to focal anisoplanatism + anisokitenism
            #1\ Defining the spatial filters
            D       = tel.D
            f0      = 2*np.pi/atm.L0

            #2\ SF Calculation
            Dani_l = np.zeros((nSrc,nLayer,nPix,nPix))

            # Angular frequencies
            U,V = self.shift_array(nPix,nPix,fact=D*samp)
       
            # Instantiation
            I0      = 3/5
            I1      = Ialpha(f0*U,f0*V)
            cte     = 0.12184*0.06*(2*np.pi)**2*atm.L0**(5/3)

            # Anisoplanatism Structure Function
            ax = src.direction[0] - gs.direction[0]
            ay = src.direction[1] - gs.direction[1]
            for iSrc in range(nSrc):
                thx = ax[iSrc]
                thy = ay[iSrc]            
                if thx !=0 or thy !=0:
                    for l  in range(nLayer):
                        zl    = Hs[l]
                        if zl !=0: 
                            I2    = Ialpha(f0*zl*thx,f0*zl*thy)
                            I3    = Ialpha(f0 * (U + zl*thx) , f0 * (V + zl*thy))
                            I4    = Ialpha(f0 * (U - zl*thx) , f0 * (V - zl*thy))
                            Dani_l[iSrc,l]  = cte*(2*I0 - 2*I1 - 2*I2  + I3  + I4)
            
        return Dani_l