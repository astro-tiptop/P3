#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:54:57 2021

@author: omartin
"""
import numpy as np
import scipy.special as spc
import aoSystem.FourierUtils as FourierUtils
from aoSystem.zernike import zernike

#%% VARIANCE
def anisoplanatism_wfe(tel, atm, src, ngs):
    """
    Returns the anisoplanatism wavefront error.
    """

    nLayer = atm.nL
    Hs = atm.heights * tel.airmass
    ax = src.direction[0] - ngs.direction[0]
    ay = src.direction[1] - ngs.direction[1]
    nSrc = len(ax)
    wfe = np.zeros(nSrc)

    # definition of the spatial frequency domain
    PSDstep = 1/(tel.D * 2)
    nOtf = tel.resolution*2
    ky, kx = FourierUtils.freq_array(nOtf, offset=1e-10, L=PSDstep)
    k2 = kx**2 + ky**2

    # PSD for the atmospheric turbulence
    cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
    Watm = cte*(1/atm.L0**2 + k2)**(-11/6)
    Watm[nOtf//2, nOtf//2] = 0

    for iSrc in range(nSrc):
        thx = ax[iSrc]
        thy = ay[iSrc]
        psd = np.zeros((nOtf, nOtf))
        for l in range(nLayer):
            zl = Hs[l]
            if zl and (thx or thy):
                psd += 2*(1 - np.cos(2*np.pi*zl*(kx*thx + ky*thy))) * Watm

        wfe[iSrc] = np.sqrt(np.trapz(np.trapz(psd, kx[0]), kx[0]))* atm.wvl*1e9/2/np.pi

    return wfe

def focal_anisoplanatism_wfe(tel, atm, lgs, method="Sasiela"):
    """
        Computes the wavefront error of the focal anisoplanatism due to the finite altitude of the LGS in SLAO mode
        INPUTS:
            - tel, atm and lgs objects
            - method : Sasiela or Tyler (this latter not validated yet)
        OUTPUTS
            - wfe, the focal anisoplanatism error in nm
    """

    var = 0
    zLgs = float(lgs.height[0])

    if method == "Sasiela":
        for k in range(atm.nL):
            if atm.heights[k] > 0:
                var1 = 0.5*(atm.heights[k]/zLgs)**(5/3)
                var2 = 0.452*(atm.heights[k]/zLgs)**2
                var += atm.weights[k] * (var1 - var2)/0.423
        wfe = np.sqrt(var * (tel.D/atm.r0)**(5/3)) * (atm.wvl*1e9/2/np.pi)

    elif method == "Tyler":
        alpha_factor = 54 * spc.gamma(14/3)/(391 * spc.gamma(17/6)**2)
        beta_factor = 2**(1/3) * spc.gamma(1/6)**2/(55 * 4**(5/6) * spc.gamma(1/3))

        for k in range(atm.nL):
            if atm.heights[k] > 0:
                z_ratio = atm.heights[k]/zLgs
                aa =-11/6
                bb = -5/6
                cc = 2
                z = (1- z_ratio)**(5/3)
                f1 =0
                tmp = 0
                ss = 0
                ee = 1
                while ee <= 1e-6:
                    tmp = np.copy(f1)
                    f1 += spc.gamma(aa+ss) * spc.gamma(bb+ss)/\
                          (spc.gamma(cc+ss) * spc.factorial(ss))*z**ss
                    ss += 1
                    ee = abs((f1-tmp)/f1)
                f1 *= spc.gamma(cc)/spc.gamma(aa)/spc.gamma(bb)

                aa = -11/6
                bb = 1/6
                cc = 3
                z = (1 - z_ratio)**2
                f2 = 0
                tmp = 0
                ss = 0
                ee = 1
                while ee<1e-6:
                    tmp=np.copy(f2)
                    f2 += spc.gamma(aa+ss) * spc.gamma(bb+ss)/\
                         (spc.gamma(cc+ss) * spc.factorial(ss))*z**ss
                    ss+= 1
                    ee = abs((f2-tmp)/f2)
                f2 *= spc.gamma(cc)/spc.gamma(aa)/spc.gamma(bb)

                tmp = atm.weights[k]*(alpha_factor*(1 + (1-z_ratio)**(5/3))\
                                     -6*(f1-z_ratio**(5/3))\
                                     +10*(1-z_ratio)*f2)

                var += beta_factor*tmp
        wfe = np.sqrt(var * (tel.D/atm.r0)**(5/3)) * (atm.wvl*1e9/2/np.pi)
    else:
        raise ValueError("The method " + method + "is not supported." )

    return wfe

def anisokinetism_wfe(tel, atm, src, ngs, method="Sasiela"):
    """
    Computes the wavefront error of the anisokinetism.
    """

    if method == "Sasiela":
        z_l = atm.heights
        cn2 = atm.r0**(-5/3) * atm.weights
        D = tel.D
        cx = np.array([2.67*3, -3.68*5, 2.35*(17/3), 0.304*7, 0.306*(23/3)])
        cy = np.array([2.67, -3.68, 2.35, 0.304, 0.306])
        thx = abs(src.direction[0] - ngs.direction[0])
        thy = abs(src.direction[1] - ngs.direction[1])

        mu002 = np.sum(cn2*z_l**2)
        mu004 = np.sum(cn2*z_l**4)
        mu006 = np.sum(cn2*z_l**6)
        mu143 = np.sum(cn2*z_l**(14/3))
        mu203 = np.sum(cn2*z_l**(20/3))

        var_x = cx[0]*mu002/D**(1/3)*(thx/D)**2\
               +cx[1]*mu004/D**(1/3)*(thx/D)**4\
               +cx[2]*mu143/D**(1/3)*(thx/D)**(14/3)\
               +cx[3]*mu006/D**(1/3)*(thx/D)**6\
               +cx[4]*mu203/D**(1/3)*(thx/D)**(20/3)

        var_y = cy[0]*mu002/D**(1/3)*(thy/D)**2\
               +cy[1]*mu004/D**(1/3)*(thy/D)**4\
               +cy[2]*mu143/D**(1/3)*(thy/D)**(14/3)\
               +cy[3]*mu006/D**(1/3)*(thy/D)**6\
               +cy[4]*mu203/D**(1/3)*(thy/D)**(20/3)

        return np.sqrt(var_x + var_y)* atm.wvl*1e9/2/np.pi

    elif method == "covariance":
        # get the covariance matrix of Zernike
        zern = zernike([2,3], tel.resolution)
        cov_zer = zern.anisokinetism(tel, atm, src, ngs)

        # get the variance
        return np.sqrt(cov_zer.trace()) * atm.wvl*1e9/2/np.pi

    else:
        raise ValueError("The method " + method + "is not supported." )

#%% PHASE STRUCTURE FUNCTION
def anisoplanatism_structure_function(tel, atm, src, lgs, ngs,
                                      nOtf, samp, nActu, Hfilter=1):
    """
    Computes the anisoplanatic phase structure functions
    INPUTS:
        - a telescope and atmosphere object
        - src, a source object that defines the target positions
        - lgs, ngs, laser guide star and natural guide star source objects
        - nOtf, the size of the angular frequency domain in pixels
        - samp, the sampling factor (2=Nyquist)
        - nActu, the 1D number DM actuators
        - Hfilter, a matrix to filter out the high spatial frequencies
    OUTPUTS:
        - NGS case : dani_ang, the angular anisoplanatism function
        - LGS case :
            - dfocani_ang, the focal-angular function
            - dani_ang
            - dani_tt, the tip-tilt anisoplanatism function
    """

    if lgs is None or lgs.height==0:
        # NGS mode, angular anisoplanatism
        dani_ang = angular_function(tel, atm, src, ngs, nOtf, samp)
        return dani_ang

    else:
        # LGS mode, focal-angular anisoplanatism + anisokinetism

        # angular + focal anisoplanatism
#        H = lgs.height
        #dani_focang = focal_angular_function(tel, atm, src, lgs,
        #                                     samp, nActu, Hfilter=Hfilter)
        dani_foc = focal_function(tel, atm, lgs, nOtf, samp)
        dani_ang = angular_function(tel, atm, src, lgs, nOtf, samp)
        dani_focang = dani_foc[np.newaxis,:,:,:] + dani_ang
        # angular anisoplanatism only
        #lgs.height = 0
        #dani_ang = focal_angular_function(tel, atm, src, lgs, nActu, samp, Hfilter=Hfilter)
        #lgs.height = H

        # anisokinetism
        dani_tt = anisokinetism_function(tel, atm, src, ngs, nOtf, samp)

        return dani_focang, dani_ang, dani_tt

def angular_function(tel, atm, src, gs, nOtf, samp, method="Fourier"):
    """
    Returns the nOtf x nOtf phase structure function for the angular anisoplanatism
    by following Flicker's 2008 report or the Fourier approach (Rigaut+98)
    """

    nLayer = atm.nL
    Hs = atm.heights * tel.airmass
    ax = src.direction[0] - gs.direction[0]
    ay = src.direction[1] - gs.direction[1]
    nSrc = len(ax)
    Dani_l = np.zeros((nSrc,nLayer,nOtf,nOtf))

    if method == "Flicker":
        umax = tel.D * samp
        f0 = 2*np.pi/atm.L0
        # Angular frequencies
        if not np.mod(nOtf, 2):
            x = np.linspace(-umax/2, umax/2-umax/nOtf, num=nOtf)
        else:
            x = np.linspace(-umax/2, umax/2, num=nOtf)
        rhoX, rhoY = np.meshgrid(x, x)

        # Instantiation
        I0 = 3.0/5.0
        I1 = FourierUtils.Ialpha(f0*rhoX, f0*rhoY)
        cte = 0.12184*0.06*(2*np.pi)**2*atm.L0**(5/3)

        # Anisoplanatism Structure Function
        for iSrc in range(nSrc):
            thx = ax[iSrc]
            thy = ay[iSrc]
            for l in range(nLayer):
                zl = Hs[l]
                if zl !=0:
                    if thx or thy:
                        I2 = FourierUtils.Ialpha(f0*(zl*thx),
                                                 f0*(zl*thy) )
                        I3 = FourierUtils.Ialpha(f0*(rhoX+zl*thx),
                                                 f0*(rhoY+zl*thy) )
                        I4 = FourierUtils.Ialpha(f0*(rhoX-zl*thx),
                                                 f0*(rhoY-zl*thy) )
                        Dani_l[iSrc, l] = (2*I0 - 2*I1 - 2*I2 + I3  + I4)
        Dani_l *= cte

    elif method == "Fourier":
        # definition of the spatial frequency domain
        PSDstep = 1/(tel.D * samp)
        ky, kx = FourierUtils.freq_array(nOtf, offset=1e-10, L=PSDstep)
        k2 = kx**2 + ky**2
        # PSD for the atmospheric turbulence
        cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
        Watm = cte*(1/atm.L0**2 + k2)**(-11/6)
        Watm[nOtf//2, nOtf//2] = 0

        for iSrc in range(nSrc):
            thx = ax[iSrc]
            thy = ay[iSrc]
            for l in range(nLayer):
                zl = Hs[l]
                if zl and (thx or thy):
                    psd = 2*(1 - np.cos(2*np.pi*zl*(kx*thx + ky*thy))) * Watm
                    Bphi = np.fft.fft2(np.fft.fftshift(psd)) * PSDstep**2
                    Dani_l[iSrc, l] = np.fft.fftshift(np.real(2*(Bphi.max() - Bphi)))

    else:
        raise ValueError("The  method " + method + " is not supported")

    return Dani_l

def focal_function(tel, atm, lgs, nOtf, samp):
    """
    Computes the focal anisoplanatism phase structure function
    """

    # instantiating a focus mode
    zern = zernike([4], nOtf)
    foc = zern.modes[0]
    pix2freq = 1/(tel.D*samp)
    Dfoc_l = np.zeros((atm.nL, nOtf, nOtf))
    for l in range(atm.nL):
        atm_l = atm.slab(l)
        if atm.heights[l] and atm.heights[l]<lgs.height[0] :
            # Get the normalized wavefront error from Sasiela+94
            wfe = focal_anisoplanatism_wfe(tel, atm_l, lgs, method="Sasiela")
            wfe *= atm_l.r0**(5/6) # normalize from the r0
            sig2 = (2*np.pi*wfe*1e-9/atm.wvl) **2

            # get the phase structure function
            n_pix = 2*int( nOtf*(1 - atm.heights[l]/lgs.height[0])/2/samp)
            foc = zernike([4], n_pix).modes[0]
            foc = np.pad(foc,[(nOtf - n_pix)//2, (nOtf-n_pix)//2])

            # get the phase structure function
            psd = abs(np.fft.fft2(foc))**2
            psd *= sig2/psd.sum() / pix2freq**2
            cov = FourierUtils.psd2cov(np.fft.fftshift(psd), pix2freq)
            #cov *= sig2/cov.max()
            Dfoc_l[l] = np.fft.fftshift(np.real(FourierUtils.cov2sf(cov)))

    return Dfoc_l

def focal_angular_function(tel, atm, src, gs, nActu, samp, Hfilter=1):
    """
    Returns the nActu**2 x nActu**2 point wise phase structure function
    for the focal-angular anisoplanatism  by following Flicker's 2008 report.
    """

    if gs.height == 0 or gs.height==np.inf:
        return angular_function(tel, atm, src, gs, nActu, samp, method="Fourier")

    #1\ Defining the spatial filters
    umax = tel.D * max([1,samp])
    f0 = 2*np.pi/atm.L0
    nLayer = atm.nL
    Hs = atm.heights * tel.airmass
    ax = src.direction[0] - gs.direction[0]
    ay = src.direction[1] - gs.direction[1]
    nSrc = len(ax)

    #2\ SF Calculation
    nOtf = int(nActu * 2+1)
    Dani_l = np.zeros((nSrc,nLayer,nOtf**2,nOtf**2))

    # Angular frequencies
    if np.mod(nOtf,2):
        x = np.linspace(-umax/2, umax/2-umax/nOtf, num=nOtf)
    else:
        x = np.linspace(-umax/2, umax/2, num=nOtf)

    x1,y1 = np.meshgrid(x,x)
    X1 = np.ones((nOtf**2,1))*x1.T.reshape(-1)
    Y1 = np.tile(y1,[nOtf,nOtf])
    # Samples separation in the pupil
    rhoX = np.transpose(X1.T - x1.T.reshape(-1))
    rhoY = Y1 - y1.T.reshape(-1)

    # Instantiation
    I0 = 3/5
    I1 = FourierUtils.Ialpha(f0*rhoX, f0*rhoY)
    cte = 0.12184*0.06*(2*np.pi)**2*atm.L0**(5/3)

    # Anisoplanatism Structure Function
    for iSrc in range(nSrc):
        thx = ax[iSrc]
        thy = ay[iSrc]
        for l in range(nLayer):
            zl    = Hs[l]
            if zl:
                g = zl/gs.height
                I2 = FourierUtils.Ialpha(f0*(rhoX*(1-g)), f0*(rhoY*(1-g)) )
                I3 = FourierUtils.Ialpha(f0*(rhoX -g*X1 + zl*thx), f0*(rhoY - g*Y1 + zl*thy) )
                I4 = FourierUtils.Ialpha(f0*( g*X1 - zl*thx), f0*(g*Y1 - zl*thy))
                I5 = FourierUtils.Ialpha(f0*(g*(rhoX-X1) -zl*thx), f0*(g*(rhoY-Y1) - zl*thy) )
                I6 = FourierUtils.Ialpha(f0*((1-g)*rhoX + g*X1 - zl*thx), f0*((1-g)*rhoY + g*Y1 - zl*thy))
                if np.isscalar(Hfilter):
                    Dani_l[iSrc, l] = (2*I0 - I1 - I2 + I3 - I4 - I5 + I6)
                else:
                    Dani_l[iSrc, l] = Hfilter*(2*I0 - I1 - I2 + I3 - I4 - I5 + I6)*Hfilter.T

    return cte*Dani_l

def anisokinetism_function(tel, atm, src, gs, nOtf, samp):
    """
    Returns the nOtf x nOtf anisokinetism phase structure function as a Gaussian
    Kernel.
    """

    #1\ Defining the spatial filters
    nLayer = atm.nL
    Hs = atm.heights * tel.airmass
    ax = src.direction[0] - gs.direction[0]
    ay = src.direction[1] - gs.direction[1]
    nSrc = len(ax)

    #2\ defining tip-tilt modes
    zern = zernike([2,3], nOtf)
    X = zern.modes[0]
    Y = zern.modes[1]
    X2 = X**2
    Y2 = Y**2
    XY = X*Y.T
    YX = Y*X.T

    # instantiating the phase structure function
    Dani_l = np.zeros((nSrc, nLayer, nOtf, nOtf))

    # computing the phase structure function for each layers and src
    for iSrc in range(nSrc):
        thx = ax[iSrc]
        thy = ay[iSrc]
        if thx or thy:
            for l in range(nLayer):
                zl = Hs[l]
                if zl !=0:
                    # update the atmosphere
                    atm_l = atm.slab(l)
                    fr0   = atm_l.r0**(-5/3)
                    # get the 2x2 anisokinetism covariance matrix
                    covAniso = zern.anisokinetism(tel, atm_l, src, gs) * tel.D**2/4
                    # defining the Gaussian kernel
                    Dani_l[iSrc,l] = (covAniso[0, 0]*X2\
                                    + covAniso[1,1]*Y2\
                                    + covAniso[0,1]*XY\
                                    + covAniso[1,0]*YX)/fr0

    return Dani_l
