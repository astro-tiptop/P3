#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:54:57 2021

@author: omartin
"""
import numpy as np
import aoSystem.FourierUtils as FourierUtils

def anisoplanatismStructureFunction(tel,atm,src,lgs,ngs,nOtf,samp,Hfilter=1):
        
    if (lgs is None) or (lgs.height == 0):
        # NGS mode, angular anisoplanatism 
        dani_ang = AngularFocalAnisoplanatismPhaseStructureFunction(tel,atm,src,ngs,nOtf,samp,Hfilter=Hfilter)
        return dani_ang
    else:
        # LGS mode, focal-angular anisoplanatism + anisokinetism
        # angular + focal anisoplanatism
        H = lgs.heights
        dani_focang = AngularFocalAnisoplanatismPhaseStructureFunction(tel,atm,src,lgs,nOtf,samp,Hfilter=Hfilter)
        # angular anisoplanatism only
        lgs.heights = 0
        dani_ang = AngularFocalAnisoplanatismPhaseStructureFunction(tel,atm,src,lgs,nOtf,samp,Hfilter=Hfilter)
        lgs.heights = H
        # anisokinetism
        dani_tt = AnisokinetismPhaseStructureFunction(tel,atm,src,ngs,nOtf,samp)
        return dani_focang, dani_ang, dani_tt
    
def AngularFocalAnisoplanatismPhaseStructureFunction(tel,atm,src,gs,nOtf,samp,Hfilter=1):
    """
    """
    
    #1\ Defining the spatial filters
    umax    = tel.D * max([1,samp/2])
    f0      = 2*np.pi/atm.L0
    nLayer  = atm.nL
    Hs      = atm.heights * tel.airmass
    ax      = src.direction[0] - gs.direction[0]
    ay      = src.direction[1] - gs.direction[1]
    nSrc    = len(ax)
    
    #2\ SF Calculation
    Dani_l = np.zeros((nSrc,nLayer,nOtf,nOtf))

    # Angular frequencies
    if np.mod(nOtf,2):
        x = np.linspace(-umax/2,umax/2-umax/nOtf,num=nOtf)
    else:
        x = np.linspace(-umax/2,umax/2,num=nOtf)
                
    if gs.height:
        # the matrix is not Toeplitz -> to optimize
        x1,y1   = np.meshgrid(x,x)
        X1      = np.ones((nOtf**2,1))*x1.T.reshape(-1)
        Y1      = np.tile(y1,[nOtf,nOtf])
        # Samples separation in the pupil
        rhoX    = np.transpose(X1.T - x1.T.reshape(-1))
        rhoY    = Y1 - y1.T.reshape(-1)
    else:
        # NGS case - we assume the matrix is Toeplitz
        rhoX, rhoY = np.meshgrid(x,x)
        
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
                    if gs.height: # focal-angular case -> computing the covariance matrix
                        g     = zl/gs.height
                        I2    = FourierUtils.Ialpha(f0*(rhoX*(1-g)) , f0*(rhoY*(1-g)) )
                        I3    = FourierUtils.Ialpha(f0*(rhoX -g*X1 + zl*thx) , f0*(rhoY - g*Y1 + zl*thy) )
                        I4    = FourierUtils.Ialpha(f0*( g*X1 - zl*thx) , f0*(g*Y1 - zl*thy))
                        I5    = FourierUtils.Ialpha(f0*(g*(rhoX-X1) -zl*thx) , f0*(g*(rhoY-Y1) - zl*thy) )
                        I6    = FourierUtils.Ialpha(f0*((1-g)*rhoX + g*X1 - zl*thx) , f0*((1-g)*rhoY + g*Y1 - zl*thy))
                        Dani_l[iSrc,l] = Hfilter*(2*I0 - I1 - I2 + I3 - I4 - I5 + I6)*Hfilter.T
                    else: #angular case -> computing the covariance map
                        I2    = FourierUtils.Ialpha( f0*(rhoX+zl*thx) , f0*(rhoY+zl*thy) )
                        I3    = FourierUtils.Ialpha( f0*(zl*thx) , f0*(zl*thy) )
                        I4    = FourierUtils.Ialpha( f0*(rhoX-zl*thx) , f0*(rhoY-zl*thy) )
                        Dani_l[iSrc,l] = (2*I0 - 2*I1 + I2 - 2*I3  + I4)  
        
    return cte*Dani_l

def AnisokinetismPhaseStructureFunction(tel,atm,src,gs,nOtf,samp):
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

    zern = zernike([2,3],nOtf)
    X    = zern.modes[0]/4
    Y    = zern.modes[1]/4
    X2   = X**2
    Y2   = Y**2
    XY   = X*Y.T
    YX   = Y*X.T
    # instantiating the phase structure function
    Dani_l = np.zeros((nSrc,nLayer,nOtf,nOtf))
    
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
                    # get the 2x2 anisokinetism covariance matrix
                    covAniso = zern.anisokinetism(tel,atm_l,src,gs)
                    # defining the Gaussian kernel
                    Dani_l[iSrc,l] = (covAniso[0,0] * X2 + covAniso[1,1]*Y2 + covAniso[0,1]*XY + covAniso[1,0]*YX)/fr0
    return Dani_l
