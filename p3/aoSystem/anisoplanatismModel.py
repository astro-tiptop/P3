#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:54:57 2021

@author: omartin
"""
import numpy as np
import scipy.special as spc
import p3.aoSystem.FourierUtils as FourierUtils
from p3.aoSystem.zernike import zernike

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

def focal_anisoplanatism_wfe(tel, atm, lgs):
    """
        Computes the wavefront error of the focal anisoplanatism due to the finite altitude
        of the LGS in SLAO mode by using Parenti and Sassiela+04
        INPUTS:
            - tel, atm and lgs objects
            - method : Sasiela or Tyler (this latter not validated yet)
        OUTPUTS
            - wfe, the focal anisoplanatism error in nm
    """

    var = 0
    zLgs = float(lgs.height[0])

    for k in range(atm.nL):
        if atm.heights[k] > 0:
            var1 = 0.5*(atm.heights[k]/zLgs)**(5/3)
            var2 = 0.452*(atm.heights[k]/zLgs)**2
            var += atm.weights[k] * (var1 - var2)/0.423
    wfe = np.sqrt(var * (tel.D/atm.r0)**(5/3)) * (atm.wvl*1e9/2/np.pi)

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

#%% PHASE STRUCTURE FUNCTION
def anisoplanatism_structure_function(tel, atm, src, lgs, ngs,
                                      nOtf, samp, nActu, msk_in=1, Hfilter=1):
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
        dani_ang = angular_function(tel, atm, src, ngs,
                                    nOtf, samp, msk_in=msk_in)
        return dani_ang, 0, 0

    else:
        # angular + focal anisoplanatism
        lgs_dir = [lgs.zenith, lgs.azimuth]
        lgs.azimuth = 0
        lgs.zenith = 0
        dani_foc = focal_angular_function(tel, atm, src, lgs,
                                          nActu, nOtf, samp,
                                          h_filt=Hfilter)
        # angular anisoplanatism only
        lgs.zenith = lgs_dir[0]
        lgs.azimuth = lgs_dir[1]
        dani_ang = angular_function(tel, atm, src, lgs,
                                    nOtf, samp, msk_in=msk_in)

        dani_focang = dani_foc + dani_ang
        # anisokinetism
        dani_tt = anisokinetism_function(tel, atm, src, ngs, nOtf, samp)

        return dani_focang, dani_ang, dani_tt

def angular_function(tel, atm, src, gs, nOtf, samp, method="Fourier", msk_in=1):
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
        x = umax/nOtf * (np.array(range(nOtf)) - nOtf//2)
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
                    psd = 2*(1 - np.cos(2*np.pi*zl*(kx*thx + ky*thy))) * Watm * msk_in
                    Bphi = np.fft.fft2(np.fft.fftshift(psd)) * PSDstep**2
                    Dani_l[iSrc, l] = np.fft.fftshift(np.real(2*(Bphi.max() - Bphi)))

    else:
        raise ValueError("The  method " + method + " is not supported")

    return Dani_l

def focal_angular_function(tel, atm, src, gs, nActu, nOtf, samp, h_filt=1):
    """
    Returns the nActu**2 x nActu**2 point wise phase structure function
    for the focal-angular anisoplanatism  by following Flicker's 2008 report.
    """

    if gs.height == 0 or gs.height==np.inf:
        return angular_function(tel, atm, src, gs, nOtf, samp,
                                method="Fourier", msk_in=h_filt)


    #1\ Grabbing inputs
    umax = tel.D/2 * max([1,samp])/2
    f0 = 2*np.pi/atm.L0
    dist = [ src.direction[0] - gs.direction[0],
              src.direction[1] - gs.direction[1]]

    if np.isscalar(h_filt):
        h_filt = 1
    else:
        h_filt_t = h_filt.T

    def vector_to_separation(u):
        x, y = np.meshgrid(u, u)
        X = np.ones((n_ph**2, 1))*x.T.reshape(-1)
        Y = np.tile(y,[n_ph, n_ph])
        # Samples separation in the pupil
        rho_x = np.transpose(X.T - x.T.reshape(-1))
        rho_y = Y - y.T.reshape(-1)
        return X, Y, rho_x, rho_y

    #2\ SF Calculation
    Dani_l = np.zeros((src.nSrc, atm.nL, nOtf, nOtf))
    n_ph = int(nActu*2 + 1)

    # Angular frequencies
    X1, Y1, rhoX, rhoY = vector_to_separation(umax/n_ph * (np.array(range(n_ph)) - n_ph//2))
    _, _, sep_x, sep_y = vector_to_separation(np.array(range(n_ph)))

    # Instantiation
    I0 = 3.0/5.0
    I1 = FourierUtils.Ialpha(f0*rhoX, f0*rhoY)
    cte = 0.12184*0.06*(2*np.pi)**2 * atm.L0**(5/3)

    # Anisoplanatism Structure Function
    for iSrc in range(src.nSrc):
        thx = dist[0][iSrc]
        thy = dist[1][iSrc]
        for l, zl in enumerate(atm.heights):
            if zl:
                g = zl/gs.height
                I2 = FourierUtils.Ialpha(f0*(rhoX*(1-g)),
                                         f0*(rhoY*(1-g)))
                I3 = FourierUtils.Ialpha(f0*(rhoX - g*X1 + zl*thx),
                                         f0*(rhoY - g*Y1 + zl*thy))
                I4 = FourierUtils.Ialpha(f0*(g*X1 - zl*thx),
                                         f0*(g*Y1 - zl*thy))
                I5 = FourierUtils.Ialpha(f0*(g*(rhoX-X1) - zl*thx),
                                         f0*(g*(rhoY-Y1) - zl*thy))
                I6 = FourierUtils.Ialpha(f0*((1-g)*rhoX + g*X1 - zl*thx),
                                         f0*((1-g)*rhoY + g*Y1 - zl*thy))
                tmp = 2*I0 - I1 - I2 + I3 - I4 - I5 + I6
                if not np.isscalar(h_filt):
                    tmp = np.dot(h_filt, np.dot(tmp, h_filt_t))

                # ---- get the maps
                cmap = FourierUtils.matrix_to_map(tmp, sep_x=sep_x, sep_y=sep_y)

                # ---- interpolation
                Dani_l[iSrc, l] = FourierUtils.interpolateSupport(cmap, nOtf)

    return cte*Dani_l

def anisokinetism_function(tel, atm, src, gs, nOtf, samp):
    """
    Returns the nOtf x nOtf anisokinetism phase structure function as a Gaussian
    Kernel.
    """

    #1\ Defining the spatial filters
    dist_x = src.direction[0] - gs.direction[0]
    dist_y = src.direction[1] - gs.direction[1]
    nSrc = len(dist_x)

    #2\ defining tip-tilt modes
    zern = zernike([2,3], nOtf)
    X = zern.modes[0]/2
    Y = zern.modes[1]/2
    X2 = X**2
    Y2 = Y**2
    XY = X*Y.T
    YX = Y*X.T

    # instantiating the phase structure function
    Dani_l = np.zeros((nSrc, atm.nL, nOtf, nOtf))

    # computing the phase structure function for each layers and src
    for iSrc in range(nSrc):
        thx = dist_x[iSrc]
        thy = dist_y[iSrc]
        if thx or thy:
            for l, zl in enumerate(atm.heights):
                if zl !=0:
                    # update the atmosphere
                    atm_l = atm.slab(l)
                    fr0 = atm_l.r0**(-5/3)
                    # get the 2x2 anisokinetism covariance matrix
                    covAniso = zern.anisokinetism(tel, atm_l, src, gs)
                    # defining the Gaussian kernel
                    Dani_l[iSrc,l] = (covAniso[0, 0]*X2\
                                    + covAniso[1,1]*Y2\
                                    + covAniso[0,1]*XY\
                                    + covAniso[1,0]*YX)/fr0

    return Dani_l

