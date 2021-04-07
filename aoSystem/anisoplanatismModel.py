#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:54:57 2021

@author: omartin
"""
import numpy as np
import fourier.FourierUtils as FourierUtils

def angularFocalAnisoplanatism(src,tel,atm,gs,nOtf,samp):
    """
    """
    
    #1\ Defining the spatial filters
    D       = tel.D
    f0      = 2*np.pi/atm.L0
    nLayer  = atm.nL
    Hs      = atm.heights * tel.airmass
    ax      = src.direction[0] - gs.direction[0]
    ay      = src.direction[1] - gs.direction[1]
    nSrc    = len(ax)
    Hfilter = 1
    
    #2\ SF Calculation
    Dani_l = np.zeros((nSrc,nLayer,nOtf,nOtf))

    # Angular frequencies
    pitch   = D/(nOtf-1)
    x       = np.arange(-D/2,D/2+pitch,pitch)
    x1,y1   = np.meshgrid(x,x)
    X1      = np.ones((nOtf**2,1))*x1.T.reshape(-1)
    Y1      = np.tile(y1,[nOtf,nOtf])
    # Samples separation in the pupil
    rhoX    = np.transpose(X1.T - x1.T.reshape(-1))
    rhoY    = Y1 - y1.T.reshape(-1)
    
    # Instantiation
    I0      = 3/5
    I1      = FourierUtils.Ialpha(f0*rhoX,f0*rhoY)
    cte     = 0.12184*0.06*(2*np.pi)**2*atm.L0**(5/3)

    # Anisoplanatism Structure Function      
    for iSrc in range(nSrc):
        thx = ax[iSrc]
        thy = ay[iSrc]            
        if thx !=0 or thy !=0:
            for l in range(nLayer):
                zl    = Hs[l]
                if zl !=0: 
                    if gs.height: # focal-angular case
                        g     = zl/gs.height
                        I2    = FourierUtils.Ialpha(rhoX*(1-g) , rhoY*(1-g))
                        I3    = FourierUtils.Ialpha(rhoX -g*X1 + zl*thx , rhoY - g*Y1 + zl*thy)
                        I4    = FourierUtils.Ialpha(g*X1 - zl*thx , g*Y1 - zl*thy)
                        I5    = FourierUtils.Ialpha(g*(rhoX-X1) -zl*thx , g*(rhoY-Y1) - zl*thy)
                        I6    = FourierUtils.Ialpha((1-g)*rhoX + g*X1 - zl*thx , (1-g)*rhoY + g*Y1 - zl*thy)
                        Dani_l[iSrc,l] = Hfilter*(2*I0 - I1 - I2 + I3 - I4 - I5 + I6)*Hfilter.T
                    else: #angular case
                       I2    = FourierUtils.Ialpha(rhoX+zl*thx,rhoY+zl*thy)
                       I3    = FourierUtils.Ialpha(zl*thx,zl*thy)
                       I4    = FourierUtils.Ialpha(rhoX-zl*thx,rhoY-zl*thy)
                       Dani_l[iSrc,l] = Hfilter*(2*I0 - 2*I1 + I2 - 2*I3  + I4)*Hfilter.T  
        
    return cte*Dani_l

def anisokinetism(src,tel,atm,gs,nOtf,samp):
    """
    """
    
    #1\ Defining the spatial filters
    nLayer  = atm.nL
    Hs      = atm.heights * tel.airmass
    ax      = src.direction[0] - gs.direction[0]
    ay      = src.direction[1] - gs.direction[1]
    nSrc    = len(ax)
    
    #2\ defining tip-tilt modes 
    from aoSystem.zernike import zernike
    zern = zernike([2,3],'resolution',nOtf)
    X    = np.reshape(zern.modes[0],nOtf,nOtf)
    Y    = np.reshape(zern.modes[1],nOtf,nOtf)
    X2   = X**2
    Y2   = Y**2
    XY   = X*Y.T
    YX   = Y*X.T
    # instantiating the phase structure function
    Dani_l = np.zeros(nSrc,nLayer,nOtf,nOtf)
    
    # computing the phase structure function for each layers and src
    for iSrc in range(nSrc):
        thx = ax[iSrc]
        thy = ay[iSrc]
        if thx !=0 or thy !=0:
            for l in range(nLayer):
                zl    = Hs[l]
                if zl !=0: 
                    # update the atmosphere
                    atm_l = atm.slab(l)
                    fr0   = atm_l.r0**(-5/3)
                    # Get the 2x2 covariance matrices of the tip-tilt
                    covTT = zern.angularCovariance(atm_l,src[iSrc],gs)
                    # get the 2x2 anisokinetism covariance matrix
                    covAniso = covTT[0,0] + covTT[1,1] - covTT[0,1] - covTT[1,0]
                    # defining the Gaussian kernel
                    Dani_l[iSrc,l] = (covAniso[0,0] * X2 + covAniso[1,1]*Y2 + covAniso[0,1]*XY + covAniso[1,0]*YX)/fr0
    return Dani_l