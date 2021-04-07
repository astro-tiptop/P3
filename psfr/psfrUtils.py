#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:57:49 2018

@author: omartin
"""

# Libraries
import numpy as np
import scipy as sp
import numpy.fft as fft
import scipy.special as ssp
import fourier.FourierUtils as FourierUtils

#%%  PSF-R FACILITIES
def anisoplanatismTransferFunction(r0,L0,cnh,v,wd,nRes,npt,theta,wvlAtm,wvlSrc,P,D,zLgs=0):    
    '''
    r0  = 0.6
    L0  = 25
    cnh = np.array([[0,5e3,10e3],[0.7,0.2,0.1]])
    v   = np.array([5,10,15])
    wd  = np.array([0,np.pi/4,np.pi/2])
    nRes= 101
    npt = 21
    theta = np.array([10,0])/206264.8
    wvlAtm= 1.6e-6
    wvlSrc= 1.6e-6
    D     = 10    
    P     = np.ones((npt,npt))
    x     = np.linspace(-D/2,D/2,npt)
    X,Y   = np.meshgrid(x,x)
    R     = np.hypot(X,Y)
    P     = R<=5
    P     = np.ones((npt,npt))
    zLgs  = 0
    '''        
    # Grab atmospherical parameters
    nL      = cnh.shape[1]
    f0      = 2*np.pi/L0
    h       = cnh[0,:]
    fr0     = cnh[1,:]/sum(cnh[1,:])
    fracR0  = fr0*0.06*(wvlAtm)**2*r0**(-5/3)
    cte     = 0.12184*(2*np.pi/wvlSrc)**2
    thx     = theta[0]
    thy     = theta[1]
    # Phase sample locations in the pupil
    msk     = P.astype(bool)
    #npt2    = np.sum(msk)
    pitch   = D/(npt-1)
    x       = np.arange(-D/2,D/2+pitch,pitch)
    x1,y1   = np.meshgrid(x,x)
    X1      = np.ones((npt**2,1))*x1.T.reshape(-1)
    Y1      = np.tile(y1,[npt,npt])
    # Samples separation in the pupil
    rhoX    = np.transpose(X1.T - x1.T.reshape(-1))
    rhoY    = Y1 - y1.T.reshape(-1)
    # Instantiation   
    I0      = mcDonald(0)
    I1      = mcDonald(f0*np.hypot(rhoX,rhoY))
    def Ialpha(x,y):
        return mcDonald(f0*np.hypot(x,y))
    
    # Anisoplanatism Structure Function   
    Dani = np.zeros((npt**2,npt**2))
    for l in np.arange(0,nL,1):        
        zl    = h[l]
        if zl !=0:
            #Strech magnification factor
            if zLgs:
                g     = zl/zLgs                                 
                I2    = Ialpha(rhoX*(1-g),rhoY*(1-g))
                I3    = Ialpha(np.transpose((1-g)*X1.T - x1.T.reshape(-1))+zl*thx,(1-g)*Y1 - y1.T.reshape(-1)+zl*thy)
                I4    = Ialpha(g*X1-zl*thx,g*Y1-zl*thy)
                I5    = Ialpha(g*X1.T-zl*thx,g*Y1.T-zl*thy) 
                I6    = Ialpha(np.transpose(X1.T - (1-g)*x1.T.reshape(-1))-zl*thx,Y1 - (1-g)*y1.T.reshape(-1)-zl*thy)                 
                Dani  = Dani + cte*L0**(5/3)*fracR0[l]*(2.*I0 - I1 - I2 + I3 - I4 - I5 + I6)                      
            else:            
                I2    = Ialpha(rhoX+zl*thx,rhoY+zl*thy)
                I3    = Ialpha(zl*thx,zl*thy)
                I4    = Ialpha(rhoX-zl*thx,rhoY-zl*thy)
                Dani  = Dani + cte*L0**(5/3)*fracR0[l]*(2*I0 - 2*I1 + I2 - 2*I3  + I4)       
    
    # Tip-tilt filtering if any             

    # DM filtering
                   
    # ATF computation
    otfDL = zonalCovarianceToOtf(0*Dani,nRes,D,pitch,msk)
    ATF   = zonalCovarianceToOtf(-0.5*Dani,nRes,D,pitch,msk)/otfDL
    ATF   = ATF/ATF.max()
    
    return ATF    
    
def getNoiseVariance(s,nfit=0,nshift=1):
                       
    s       = s - s.mean()
    nS,nF   = s.shape
    out     = np.zeros(nS);
            
    if nfit !=0:
        # Polynomial fitting procedure
        delay   = np.linspace(0,1,nfit+1)
        for i in np.arange(0,nS,1):
            #pdb.set_trace()
            g      = fft.ifft(fft.fft(s[i,:])*np.conjugate(fft.fft(s[i,:])))/nF
            mx     = g.max()
            fit    = np.polyfit(delay[1:nfit+1],g[1:nfit+1],nfit)
            yfit   = 0*delay
                    
            for k in np.arange(0,nfit+1,1):
                yfit = yfit + fit[k]*delay**(nfit-k)
                    
            out[i] = mx - yfit[0]
                
        return np.diag(out)  
              
    elif nshift !=0:
        ds  = s - np.roll(s,(0,nshift))
        out = np.dot(s,np.transpose(ds))/nF
        return out
            
def getOLslopes(s,u,MI,dt):    
    return s + MI*(dt*np.roll(u,(-2,2)) + (1-dt)*np.roll(u,(-1,2)))

def getStructureFunction(phi,pupil,overSampling):
    # Create phi and pup an a 2x larger grid
    phi2    = FourierUtils.enlargeSupport(phi,2*overSampling)
    pup2    = FourierUtils.enlargeSupport(pupil,2*overSampling)
    corp2w  = FourierUtils.fftCorrel(phi2**2,pup2)
    corpp   = FourierUtils.fftCorrel(phi2,phi2)
    dphi    = corp2w + FourierUtils.fftsym(corp2w) - 2*corpp
    corww   = FourierUtils.fftCorrel(pup2,pup2)
    # Compute mask of locations where there is overlap between shifted pupils and normalized by the number of points used to compute dphi
    mask    = (corww > 1e-5)
    corww[corww <= 1] = 1
    
    return np.real(fft.fftshift(dphi * mask /corww))
          

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
        
#def mcDonald(x):
#    
#    if np.isscalar(x):
#        if x==0:
#            out = 0.6
#        else:
#            out = x**(5/6)*spc.kv(5/6,x)/(2**(5/6)*spc.gamma(11/6))
#    else:
#        out = 0.6*np.ones(x.shape)
#        idx = (x.nonzero())
#        
#        if len(idx):
#            out[idx] = x[idx]**(5/6)*spc.kv(5/6,x[idx])/(2**(5/6)*spc.gamma(11/6))
#       
#    return out


def mkotf(indptsc,indptsc2,nU1d,ampl,dp,C_phi):            
    #Instantiation
    otf         = np.zeros((nU1d,nU1d))
    C_phip_diag = np.exp(np.diag(C_phi))
    C_phipi     = np.exp(-2*C_phi)
    C_phi_size2 = C_phi.shape[1]
            
    for iu in np.arange(0,nU1d,1):
        for ju in np.arange(0,nU1d,1):
            indpts  = indptsc[iu, ju]
            indpts2 = indptsc2[iu, ju]
            indpts  = indpts[0]
            indpts2 = indpts2[0]
            if len(indpts)== 0:
                otf[iu,ju] = 0
            else:
                #pdb.set_trace()
                msk        = np.unravel_index(C_phi_size2*indpts + indpts2, C_phipi.shape)
                myarg      = C_phip_diag[indpts2]*C_phip_diag[indpts]*C_phipi[msk]
                kernel     = np.dot(myarg,np.conjugate(ampl[indpts2])*ampl[indpts])
                otf[iu,ju] = kernel*dp**2
                
    dc  = np.sum(abs(ampl[:])**2) * dp**2
    otf = otf/dc;
    return otf    

def mkotf_indpts(nU1d,nPh,u1D,loc,dp):    
    # index pts in a 3x bigger array
    locInPitch = loc/dp
    minLoc     = np.array([locInPitch[0,:].min(),locInPitch[1,:].min()])
    ninloc     = loc.shape[1]
    nc         = 3*nPh
    n          = nc-nPh
    n1         = int((n-n%2)/2 + 1 + nPh%2)
    minLoc2    = minLoc -(n1-1)
    loc2       = np.round(locInPitch.T - np.ones((ninloc,2))*minLoc2+1)
    loc2       = loc2.T
    #embed the loc2 inside ncxnc array.
    indx_emb       = loc2[0,:] + (loc2[1,:]-1)*nc-1
    indx_emb       = indx_emb.astype(int)
    mask           = np.zeros((nc,nc))
    indx_emb       = np.unravel_index(indx_emb,mask.shape)
    mask[indx_emb] = 1
    indptsc        = np.zeros((nU1d,nU1d,), dtype=np.object)
    indptsc2       = np.zeros((nU1d,nU1d,), dtype=np.object)
            
    for iu in np.arange(0,nU1d,1):
        for ju in np.arange(0,nU1d,1):
            u2D        = np.array([u1D[iu],u1D[ju]])
            uInPitch   = u2D/dp
            iniloc_sh2 = np.round(loc2.T + np.ones((ninloc,2))*np.round(uInPitch))
            iniloc_sh2 = iniloc_sh2.T
            iniloc_sh2 = iniloc_sh2.astype(int)
            indxsh     = iniloc_sh2[0,:] + (iniloc_sh2[1,:]-1)*nc-1        
            indxsh     = np.unravel_index(indxsh,mask.shape)            
            #index of points in iniloc_sh2 that are intersect with iniloc_sh
            #indpts is the overlapping points in the shifted array
            indptsc[ju,iu]  = mask[indxsh].nonzero()
            mask2           = np.zeros((nc,nc),dtype=bool)
            mask2[indxsh]   = 1
            indptsc2[ju,iu] = mask2[indx_emb].nonzero()
            
    return indptsc,indptsc2

def modes2Otf(Cmm,modes,pupil,npsf,overSampling=1,basis='Vii'):            
    #Autocorrelation of the pupil expressed in pupil
    nPx        = np.sqrt(modes.shape[0])
    pupExtended= FourierUtils.enlargeSupport(pupil,2*overSampling)
    fftPup     = fft.fft2(pupExtended)
    conjPupFft = np.conjugate(fftPup)
    G          = fft.fftshift(np.real(fft.fft2(fftPup*conjPupFft)))
    #Defining the inverse
    den        = np.zeros(np.array(G.shape))
    msk        = G/G.max() > 1e-7
    den[msk]   = 1/G(msk)
            
            
    if np.any(Cmm[:]) & basis == 'Vii':
        # Diagonalizing the Cvv matrix
        [U,S]   = np.linalgsvd(Cmm)
        s       = np.diag(S)
        nModes  = len(s)
        M       = modes * U
        #loop on actuators                
        buf     = np.zeros(np.array(pupExtended.shape))
                
        for k in sp.arange(1,nModes,1):
            Mk   = np.reshape(M[:,k],nPx,nPx)
            Mk   = FourierUtils.enlargeSupport(Mk,2*overSampling)
            # Vii computation
            Vk   = np.real(fft.fft2(Mk**2*pupExtended)*conjPupFft) - abs(fft.fft2(Mk*pupExtended))**2
            # Summing modes into dm basis
            buf  = buf + s[k] * Vk
                        
        dphi     = den*fft.fftshift(np.real(fft.fft2(2*buf)))
        otf      = G*np.exp(-0.5*dphi)
                
                
    elif np.any(Cmm[:]) & basis == 'Uij':
        nm   = modes.shape[1]
        dphi = 0*pupExtended
                
        #Double loops on modes
        for i in np.arange(1,nm,1):
            Mi = np.reshape(modes[:,i],nPx,nPx)
            Mi = FourierUtils.enlargeSupport(Mi,2*overSampling)
            for j in np.arange(1,i,1):
                #Getting modes + interpolation to twice resolution
                Mj    = np.reshape(modes[:,j],nPx,nPx)
                Mj    = FourierUtils.enlargeSupport(Mj,2*overSampling)
                term1 = np.real(fft.fft2(Mi*Mj*pupExtended)*conjPupFft)
                term2 = np.real(fft.fft2(Mi*pupExtended)*np.conjugate(fft.fft2(Mj*pupExtended)))
                # Uij computation
                Uij   = np.real(fft.ifft2(term1-term2))
                #Summing terms
                fact = np.double(i!=j) + 1
                dphi = dphi + fact*Cmm[i,j]*Uij                
                dphi = fft.fftshift(2*dphi)*den*msk
                otf  = G*np.exp(-0.5*dphi)
    else:
        #Diffraction-limit case
        G    = G/G.max()
        otf  = G;
        
            
    # Interpolation of the OTF => determination of the PSF fov
    otf = otf*(G>1e-5)
    otf = FourierUtils.interpolateSupport(otf,npsf)
    otf = otf/otf.max()
    return otf

def pointWiseLocation(D,dp,idxValid):
    # Defining the point-wise locations
    xloc                 = np.arange(-D/2,D/2+dp,dp)
    actuLocY, actuLocX   = np.meshgrid(xloc,xloc)
    actuLocX             = actuLocX[idxValid]
    actuLocY             = actuLocY[idxValid]
    return np.array([actuLocX,actuLocY])


def sr2wfe(Strehl,wvl):
    return 1e9*np.sqrt(-np.log(Strehl))*wvl/2/np.pi
    
def wfe2sr(wfe,wvl):
    return np.exp(-(2*np.pi*wfe*1e-9/wvl)**2)

       
def zonalCovarianceToOtf(Cphi,npsf,D,dp,idxValid):            
    # Grabbing the valid actuators positions in meters
    loc  = pointWiseLocation(D,dp,idxValid)
    #OTF sampling
    nPh  = round(D/dp+1)
    #nU1d = 2*nPh;
    u1D  = np.arange(-nPh,nPh,1)*dp
    nU1d = len(u1D)
    # Determining couples of point with the same separation
    shiftX,shiftY = mkotf_indpts(nU1d,nPh,u1D,loc,dp)
    # WF Amplitude   
    amp0 = np.ones((np.count_nonzero(idxValid),1))
    # Long-exposure OTF
    otf = mkotf(shiftX,shiftY,nU1d,amp0,dp,-0.5*Cphi)
    #Interpolation
    otf = FourierUtils.interpolateSupport(otf,npsf)
    otf = otf/otf.max()
    return otf

        

    
                
