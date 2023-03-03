#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:54:57 2021

@author: omartin
"""
import numpy as np
import p3.aoSystem.FourierUtils as FourierUtils



def focal_anisoplanatism_variance(tel,atm,lgs):
    '''
        Compute the variance of the focal anisoplanatism due to the finite altitude of the LGS in SLAO mode
        INPUTS:
            - tel, atm and lgs objects
        OUTPUTS
            - wfe, the focal anisoplanatism error in nm
    '''
    
    var = 0
    zLgs = float(lgs.height[0])
    for k in range(atm.nL):
        if atm.heights[k] > 0:
            var1 = 0.5*(atm.heights[k]/zLgs)**(5/3)
            var2 = 0.425*(atm.heights[k]/zLgs)**2
            var  += atm.weights[k] * (var1 - var2)
              
    wfe = np.sqrt(var * (tel.D/atm.r0)**(5/3)) * (atm.wvl*1e9/2/np.pi)
    return wfe
    
def anisoplanatism_structure_function(tel,atm,src,lgs,ngs,nOtf,samp,nActu,Hfilter=1):
        
    if (lgs==None) or (lgs.height == 0):
        # NGS mode, angular anisoplanatism 
        dani_ang = angular_focal_anisoplanatism_phase_structure_function(tel,atm,src,ngs,nOtf,samp,nActu,Hfilter=Hfilter)
        return dani_ang
    else:
        # LGS mode, focal-angular anisoplanatism + anisokinetism
        # angular + focal anisoplanatism
        H = lgs.height
        dani_focang = angular_focal_anisoplanatism_phase_structure_function(tel,atm,src,lgs,nOtf,samp,nActu,Hfilter=Hfilter)
        # angular anisoplanatism only
        lgs.height = 0
        dani_ang = angular_focal_anisoplanatism_phase_structure_function(tel,atm,src,lgs,nOtf,samp,nActu,Hfilter=Hfilter)
        #np.zeros((src.nSrc,atm.nL,nOtf,nOtf))#
        lgs.height = H
        # anisokinetism
        dani_tt = anisokinetism_phase_structure_function(tel,atm,src,ngs,nOtf,samp)
        return dani_focang, dani_ang, dani_tt

def angular_focal_anisoplanatism_phase_structure_function(tel,atm,src,gs,nOtf,samp,nActu,Hfilter=1):
    """
    """
    
    #1\ Defining the spatial filters
    umax    = tel.D * max([1,samp])
    f0      = 2*np.pi/atm.L0
    nLayer  = atm.nL
    Hs      = atm.heights * tel.airmass
    ax      = src.direction[0] - gs.direction[0]
    ay      = src.direction[1] - gs.direction[1]
    nSrc    = len(ax)
    
    #2\ SF Calculation
    
    nOtf_hr = nOtf
    if gs.height:
        # Must reduce nOtf to decrease the computationnal power and memory; causes Python crash otherwise
        nOtf    = int(nActu * 2+1)#samp/2)
        
    Dani_l = np.zeros((nSrc,nLayer,nOtf_hr,nOtf_hr))

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
                    if np.isscalar(Hfilter):
                        tmp   = (2*I0 - I1 - I2 + I3 - I4 - I5 + I6)
                    else:
                        tmp   = Hfilter*(2*I0 - I1 - I2 + I3 - I4 - I5 + I6)*Hfilter.T
                    
                    # need to compute the covariance map !!
                    
                    # interpolating
                    if nOtf != nOtf_hr:
                        Dani_l[iSrc,l] = FourierUtils.interpolateSupport(tmp,nOtf_hr)
                    else:
                        Dani_l[iSrc,l] = tmp
                        
                else: #angular case -> computing the covariance map
                    if thx !=0 or thy !=0:
                        I2    = FourierUtils.Ialpha( f0*(rhoX+zl*thx) , f0*(rhoY+zl*thy) )
                        I3    = FourierUtils.Ialpha( f0*(zl*thx) , f0*(zl*thy) )
                        I4    = FourierUtils.Ialpha( f0*(rhoX-zl*thx) , f0*(rhoY-zl*thy) )
                        Dani_l[iSrc,l] = (2*I0 - 2*I1 + I2 - 2*I3  + I4)  
        
    return cte*Dani_l

def anisokinetism_phase_structure_function(tel,atm,src,gs,nOtf,samp):
    """
    """
    
    #1\ Defining the spatial filters
    nLayer  = atm.nL
    Hs      = atm.heights * tel.airmass
    ax      = src.direction[0] - gs.direction[0]
    ay      = src.direction[1] - gs.direction[1]
    nSrc    = len(ax)
    
    #2\ defining tip-tilt modes 
    from p3.aoSystem.zernike import zernike

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
                zl = Hs[l]
                if zl !=0: 
                    # update the atmosphere
                    atm_l = atm.slab(l)
                    fr0   = atm_l.r0**(-5/3)
                    # get the 2x2 anisokinetism covariance matrix
                    covAniso = zern.anisokinetism(tel,atm_l,src,gs)
                    # defining the Gaussian kernel
                    Dani_l[iSrc,l] = (covAniso[iSrc,0,0]*X2 +
                                      covAniso[iSrc,1,1]*Y2 +
                                      covAniso[iSrc,0,1]*XY +
                                      covAniso[iSrc,1,0]*YX)/fr0
    return Dani_l
