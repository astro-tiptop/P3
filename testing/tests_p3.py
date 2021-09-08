#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:17:49 2021

@author: omartin
"""

#%% IMPORTING LIBRARIES
import os
import sys
import time

import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from distutils.spawn import find_executable

from aoSystem.fourierModel import fourierModel
import aoSystem.FourierUtils as FourierUtils

import psfao21 as psfao21Main
from telemetry.telemetryKeck import telemetryKeck
from telemetry.telemetryKASP import telemetryKASP
from telemetry.systemDiagnosis import systemDiagnosis
from telemetry.configFile import configFile
from psfr.psfR import psfR
import psfr.psfrUtils as psfrUtils

from psfao21.psfao21 import psfao21
from psfFitting.psfFitting import psfFitting
from psfFitting.psfFitting import displayResults


#%% DISPLAY FEATURES
mpl.rcParams['font.size'] = 16

if find_executable('tex'): 
    usetex = True
else:
    usetex = False

plt.rcParams.update({
    "text.usetex": usetex,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
    
rad2mas = 3600 * 180 * 1e3/np.pi
    
#%% MANAGING PATHS

if sys.platform[0:3] == 'win':
    path_root  = '\\'.join(psfao21Main.__file__.split('\\')[0:-3]) + '\\'
    path_p3    = '\\'.join(psfao21Main.__file__.split('\\')[0:-2])
    path_ini   = path_p3 + '\aoSystem\parFiles\KECKII_NIRC2_20130801_12_00_19.254.ini'
    path_img   = path_p3 + '\data\20130801_n0004.fits'
    path_calib = path_p3 + '\aoSystem\data\KECK_CALIBRATION\\'
else:
    path_root  = '/'.join(psfao21Main.__file__.split('/')[0:-3]) + '/'
    path_p3    = '/'.join(psfao21Main.__file__.split('/')[0:-2])
    path_ini   = path_p3 + '/aoSystem/parFiles/KECKII_NIRC2_20130801_12_00_19.254.ini'
    #path_ini   = path_p3 + '/aoSystem/parFiles/nirc2.ini'
    path_img   = path_p3 + '/data/20130801_n0004.fits'
    path_calib = path_p3 + '/aoSystem/data/KECK_CALIBRATION/'

im_nirc2 = fits.getdata(path_img)    
filename   = 'n0004_fullNGS_trs.sav'

#%% TEST THE SPATIAL FREQUENCY ADAPTIVE OPTICS MODEL

def test_Fourier_fitting():
    '''
       Test the fitting of object parameters on a NIRC2 frame with a PSF model
       derived from the Fourier model.
    '''
      
    #instantiating the model
    fao = fourierModel(path_ini, path_root=path_p3, calcPSF=False, display=False)    
    
    # fitting the residual jitter + astrometry/photometry
    x0 = fao.ao.cam.spotFWHM[0] + [1,0,0,0]
    fixed = (True,)*3 + (False,)*4
    res_Fourier = psfFitting(im_nirc2, fao, x0, verbose=2, fixed=fixed)
    
    # display
    displayResults(fao, res_Fourier, nBox=90, scale='log10abs')
    
    return res_Fourier, fao

#%% TEST THE PSFAO21 MODEL

def test_psfao21_instantiation():
    '''
        Test the instantiation of the PSFAO21 model
    '''
    
    t0 = time.time()
    # instantiating the model
    psfao = psfao21(path_ini,path_root=path_p3)
    plt.close('all')
    kx  = psfao.freq.kx_[psfao.freq.nOtf//2+1:,psfao.freq.nOtf//2]/psfao.freq.kc_
    nCases = 5

    # -----------  Visual inspection of the PSD/PSF WRT r0
    fig, axs = plt.subplots(1,2,constrained_layout=True)
    axs[0].set_xlabel('Spatial frequency [$k_c$ units]')
    axs[0].set_ylabel('PSD profile [$nm^2.m$]')    
    axs[1].set_xlabel('Angular separation [$\lambda k_c$ units]')
    axs[1].set_ylabel('PSF profile')
    fig.suptitle('PSFAO21 PSD AND PSD VERSUS $r_0$ FOR A NIRC2 CASE')
    
    for j in range(nCases):
        if j == 0:
            x0  = [0.2,4e-2,0.5,1e-2,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==1:
            x0  = [0.3,4e-2,0.5,1e-2,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==2:
            x0  = [0.4,4e-2,0.5,1e-2,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==3:
            x0  = [0.5,4e-2,0.5,1e-2,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==4:
            x0  = [0.8,4e-2,0.5,1e-2,1,0,1.5,0,0,0,1.0,0,0,0]
            
        idlab = '$r_0 = $' + str(x0[0]) + ' m'
        psf = np.squeeze(psfao(x0))
        psd = psfao.psd * (psfao.freq.wvlRef*1e9/2/np.pi)**2
        kx  = psfao.freq.kx_[psfao.freq.nOtf//2+1:,psfao.freq.nOtf//2]/psfao.freq.kc_
        
        # Display
        axs[0].semilogy(kx,psd[psfao.freq.nOtf//2+1:,psfao.freq.nOtf//2],label=idlab)        
        axs[1].semilogy(kx,psf[psfao.freq.nOtf//2+1:,psfao.freq.nOtf//2],label=idlab)
    axs[1].legend()
    
    # -----------  Visual inspection of the PSD/PSF WRT sigma^2
    fig, axs = plt.subplots(1,2,constrained_layout=True)
    axs[0].set_xlabel('Spatial frequency [$k_c$ units]')
    axs[0].set_ylabel('PSD profile [$nm^2.m$]')    
    axs[1].set_xlabel('Angular separation [$\lambda k_c$ units]')
    axs[1].set_ylabel('PSF profile')
    fig.suptitle('PSFAO21 PSD AND PSD VERSUS $\sigma^2$ FOR A NIRC2 CASE')
    
    for j in range(nCases):
        if j == 0:
            x0  = [0.4,4e-2,0.05,1e-2,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==1:
            x0  = [0.4,4e-2,0.1,1e-2,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==2:
            x0  = [0.4,4e-2,0.25,1e-2,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==3:
            x0  = [0.4,4e-2,0.5,1e-2,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==4:
            x0  = [0.4,4e-2,1,1e-2,1,0,1.5,0,0,0,1.0,0,0,0]
            
        idlab = '$\sigma^2 = $' + str(x0[2]) + ' rad$^2$'
        psf = np.squeeze(psfao(x0))
        psd = psfao.psd * (psfao.freq.wvlRef*1e9/2/np.pi)**2
        kx  = psfao.freq.kx_[psfao.freq.nOtf//2+1:,psfao.freq.nOtf//2]/psfao.freq.kc_
        
        # Display
        axs[0].semilogy(kx,psd[psfao.freq.nOtf//2+1:,psfao.freq.nOtf//2],label=idlab)        
        axs[1].semilogy(kx,psf[psfao.freq.nOtf//2+1:,psfao.freq.nOtf//2],label=idlab)
    axs[1].legend()
    
    # -----------  Visual inspection of the PSD/PSF WRT ALPHA
    fig, axs = plt.subplots(1,2,constrained_layout=True)
    axs[0].set_xlabel('Spatial frequency [$k_c$ units]')
    axs[0].set_ylabel('PSD profile [$nm^2.m$]')    
    axs[1].set_xlabel('Angular separation [$\lambda k_c$ units]')
    axs[1].set_ylabel('PSF profile')
    fig.suptitle(r'PSFAO21 PSD AND PSD VERSUS $\alpha$ FOR A NIRC2 CASE')
    
    for j in range(nCases):
        if j == 0:
            x0  = [0.4,4e-2,0.5,1e-2,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==1:
            x0  = [0.4,4e-2,0.5,1e-1,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==2:
            x0  = [0.4,4e-2,0.5,2.5e-1,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==3:
            x0  = [0.4,4e-2,0.5,5e-1,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==4:
            x0  = [0.4,4e-2,0.5,1e0,1,0,1.5,0,0,0,1.0,0,0,0]
            
        idlab = r'$\alpha = $' + str(x0[3]) + ' m$^{-1}$'
        psf = np.squeeze(psfao(x0))
        psd = psfao.psd * (psfao.freq.wvlRef*1e9/2/np.pi)**2
        kx  = psfao.freq.kx_[psfao.freq.nOtf//2+1:,psfao.freq.nOtf//2]/psfao.freq.kc_
        
        # Display
        axs[0].semilogy(kx,psd[psfao.freq.nOtf//2+1:,psfao.freq.nOtf//2],label=idlab)        
        axs[1].semilogy(kx,psf[psfao.freq.nOtf//2+1:,psfao.freq.nOtf//2],label=idlab)
    axs[1].legend()
    
    # -----------  Visual inspection of the PSD/PSF WRT BETA
    fig, axs = plt.subplots(1,2,constrained_layout=True)
    axs[0].set_xlabel('Spatial frequency [$k_c$ units]')
    axs[0].set_ylabel('PSD profile [$nm^2.m$]')    
    axs[1].set_xlabel('Angular separation [$\lambda k_c$ units]')
    axs[1].set_ylabel('PSF profile')
    fig.suptitle(r'PSFAO21 PSD AND PSD VERSUS $\beta$ FOR A NIRC2 CASE')
    
    for j in range(nCases):
        if j == 0:
            x0  = [0.4,4e-2,0.5,1e-2,1,0,1.1,0,0,0,1.0,0,0,0]
        elif j ==1:
            x0  = [0.4,4e-2,0.5,1e-2,1,0,1.5,0,0,0,1.0,0,0,0]
        elif j ==2:
            x0  = [0.4,4e-2,0.5,1e-2,1,0,1.8,0,0,0,1.0,0,0,0]
        elif j ==3:
            x0  = [0.4,4e-2,0.5,1e-2,1,0,2.8,0,0,0,1.0,0,0,0]
        elif j ==4:
            x0  = [0.4,4e-2,0.5,1e-2,1,0,3.8,0,0,0,1.0,0,0,0]
            
        idlab = r'$\beta = $' + str(x0[6])
        psf = np.squeeze(psfao(x0))
        psd = psfao.psd * (psfao.freq.wvlRef*1e9/2/np.pi)**2
        
        # Display
        axs[0].semilogy(kx,psd[psfao.freq.nOtf//2+1:,psfao.freq.nOtf//2],label=idlab)        
        axs[1].semilogy(kx,psf[psfao.freq.nOtf//2+1:,psfao.freq.nOtf//2],label=idlab)
        
    axs[1].legend()
    print('%d psf computations done in %.2f s'%(nCases*4, time.time() - t0))
    return psfao
    
def test_psfao21_fitting(tol=1e-5):
    '''
        Test the fitting of a NIRC2 image with the PSFAO21 model with two strategies:
            - joint fitting of PSD parameters and static aberrations
            - split fitting : PSD parameters first, redefinition of the bounds as xf+-5/3xerr and fit of the static aberrations
    '''
    # instantiating the model
    psfao = psfao21(path_ini, path_root=path_p3)
    
    # -------- Joint fitting the 7 PSD parameters + static aberrations
    n_modes = psfao.ao.tel.nModes
    x0 = [0.7, 4e-2, 0.5, 1e-2, 1, 0, 1.8, 0, 0, 0, 1.0, 0, 0, 0] + [0,]*n_modes
    fixed = (False,)*7 +(True,)*3 + (False,)*4 + (False,)*n_modes
    res_psfao21_joint = psfFitting(im_nirc2, psfao, x0, verbose=2,fixed=fixed,
                                   ftol=tol, gtol=tol, xtol=tol)
    
    displayResults(psfao, res_psfao21_joint, nBox=90, scale='log10abs')
    psd_stat = psfao.psd * (psfao.freq.wvlRef*1e9/2/np.pi)**2
    
    # -------- Split fitting
    # fitting - 7 PSD parameters + no static aberrations
    fixed = (False,)*7 +(True,)*3 + (False,)*4 + (True,)*n_modes
    res_psfao21 = psfFitting(im_nirc2, psfao, x0, verbose=2, fixed=fixed, 
                             ftol=tol, gtol=tol, xtol=tol)
    displayResults(psfao, res_psfao21, nBox=90, scale='log10abs')
    psd_nostat = psfao.psd * (psfao.freq.wvlRef*1e9/2/np.pi)**2
    
    # fitting - no PSD parameters + static aberrations
    x0 = list(res_psfao21.x[:7]) + [0,]*3 + [1.0,0,0,0] + [0,]*n_modes
    fixed = (False, )*7 +(True,)*3 + (False,)*4 + (False,)*n_modes
    # redefining bounds
    bounds= psfao.updateBounds(res_psfao21.x,res_psfao21.xerr,sig=5)
    res_psfao21_split = psfFitting(im_nirc2, psfao, x0, verbose=2, fixed=fixed,
                                   ftol=tol, gtol=tol, xtol=tol, bounds=bounds)
    
    displayResults(psfao, res_psfao21_split, nBox=90, scale='log10abs')
    psd_stat2 = psfao.psd * (psfao.freq.wvlRef*1e9/2/np.pi)**2
    
    # PSD static aberrations
    psd = np.abs(np.fft.fftshift(np.fft.fft2(res_psfao21_joint.opd)))**2 
    psd = FourierUtils.interpolateSupport(psd,psfao.freq.nOtf)/np.size(res_psfao21_joint.opd)/(psfao.ao.tel.D * psfao.freq.sampRef)    
    psd2 = np.abs(np.fft.fftshift(np.fft.fft2(res_psfao21_split.opd)))**2 
    psd2 = FourierUtils.interpolateSupport(psd2,psfao.freq.nOtf)/np.size(res_psfao21_split.opd)/(psfao.ao.tel.D * psfao.freq.sampRef)
    
    
    plt.figure()
    kx,_  = FourierUtils.radial_profile(psd_nostat) 
    kx *= np.mean(np.diff(psfao.freq.kx_[:,0]))/psfao.freq.kc_[0]
    plt.semilogy(kx,FourierUtils.radial_profile(psd_nostat,nargout=1),label='No static aberrations')        
    plt.semilogy(kx,FourierUtils.radial_profile(psd_stat,nargout=1),label='Joint fit')
    plt.semilogy(kx,FourierUtils.radial_profile(psd_stat2,nargout=1),label='Split fit')
    plt.semilogy(kx,FourierUtils.radial_profile(psd,nargout=1),label='Static aberrations - Joint fit')
    plt.semilogy(kx,FourierUtils.radial_profile(psd2,nargout=1),label='Static aberrations  - Split fit')
    plt.legend()        
    plt.xlabel('Spatial frequency [$k_c$ units]')
    plt.ylabel('PSD azimuthal profile [$nm^2.m$]')    
    
    plt.figure()
    plt.semilogy(kx,FourierUtils.radial_profile(res_psfao21.psf,nargout=1),label='No static aberrations')        
    plt.semilogy(kx,FourierUtils.radial_profile(res_psfao21_joint.psf,nargout=1),label='Joint fit')
    plt.semilogy(kx,FourierUtils.radial_profile(res_psfao21_split.psf,nargout=1),label='Split fit')
    plt.legend()        
    plt.xlabel('Angular separation [$\lambda k_c$ units]')
    plt.ylabel('PSF azimuthal profile')    
    
    fig, axs = plt.subplots(1,3,constrained_layout=True)
    vmin = np.min([res_psfao21_joint.opd.min(), res_psfao21_split.opd.min()])
    vmax = np.max([res_psfao21_joint.opd.max(), res_psfao21_split.opd.max()])
    axs[0].imshow(res_psfao21_joint.opd,vmin=vmin,vmax=vmax)
    axs[0].set_title('Joint fit')
    axs[0].axis('off')
    axs[1].imshow(res_psfao21_split.opd,vmin=vmin,vmax=vmax)
    axs[1].set_title('Split fit')
    axs[1].axis('off')
    imm = axs[2].imshow(res_psfao21_split.opd - res_psfao21_joint.opd,vmin=vmin,vmax=vmax)
    axs[2].set_title('Diff')
    axs[2].axis('off')
    fig.colorbar(imm, ax=axs.ravel().tolist(), shrink=0.6)
    return res_psfao21, res_psfao21_split, res_psfao21_joint


#%% PSF RECONSTRUCTION

def test_kasp_psfr(path_mat,path_save,true_r0 = True,tol=1e-5):
    ''' Test the interface with the KASP .mat output file.
    '''
    
    # instantiating/processing the telemetry and creating the .ini file
    trs = telemetryKASP(path_mat,path_save=path_save)
    r0_true = trs.atm.r0
    L0_true = trs.atm.L0
    sd  = systemDiagnosis(trs,noiseMethod='nonoise')        
    if true_r0:
        trs.atm.r0_tel = r0_true
        trs.atm.r0     = r0_true
        trs.atm.L0_tel = L0_true
        trs.atm.L0     = L0_true
        trs.atm.seeing = 3600*180/np.pi*0.98*trs.atm.wvl/r0_true
    configFile(sd)
    
    # get the psf
    psfr = psfR(trs)
    if psfr.ao.atm.nL > 1:
        Cn2 = list(np.array(trs.atm.Cn2) * (trs.atm.wvl/trs.cam.wvl[0])**2)
    else:
        Cn2 = [trs.atm.r0 * (trs.cam.wvl/trs.atm.wvl)**1.2]
    x0  = Cn2 + [1,1] + [1,0,0,0]
    
    fixed = (True,)*(psfr.ao.atm.nL + 2)+(False,)*3 + (True,)
    
    #adjust the flux and the position
    res   = psfFitting(psfr.trs.cam.image,psfr,x0,verbose=2,fixed=fixed,\
                       ftol=tol,xtol=tol,gtol=tol)
    displayResults(psfr,res,scale='log10abs')
    
    return psfr , res
    
def test_psfr(path_trs):
    ''' Test the PSF reconstruction
    '''
    
    # Get the telemetry
    trs = test_telemetry(path_trs)
    sd  = test_systemDiagnosis(trs)
    test_configFile(sd)
    
    # Get the PSF
    psfr = psfR(sd.trs)
    x0 = [0.7, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    psf = np.squeeze(psfr(x0))
   
    plt.figure()
    plt.imshow(np.log10(psf))
    
    return psfr

def test_prime(psfr, fixed_stat = True, tol=1e-5):
    ''' Test the PRIME approach
    '''
    # defining the initial guess and fixed parameters
    n_modes = psfr.ao.tel.nModes
    x0 = [0.7, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0] + [0,]*n_modes
    fixed = (False,)*2 + (True,) + (False,)*4 + (fixed_stat,)*n_modes
    
    # fitting
    res = psfFitting(psfr.trs.cam.image, psfr, x0, verbose=2,
                     fixed=fixed, xtol=tol, ftol=tol, gtol=tol)
    
    # Display
    displayResults(psfr,res,nBox=90,scale='log10abs')
    
    return res
       
def test_telemetry(path_trs):
    ''' Test the instantiation of the telemetryKeck object
    '''
    
    #load the telemetry
    if sys.platform[0:3] == 'win':
        path_sav = path_trs + '\\' + filename
    else:
        path_sav = path_trs + '/' + filename
    
    if not os.path.isfile(path_sav):
        psfrUtils.get_data_file(path_trs,filename)
        
    # path image/calibration
    trs = telemetryKeck(path_sav,path_img,path_calib,path_save=path_trs,nLayer=1)
    
    return trs

def test_systemDiagnosis(trs):
    ''' Test the instantiation of the telemetryKeck and systemDiagnosis objects
    '''
    
    sd  = systemDiagnosis(trs)
    
    return sd

def test_configFile(sd):
    ''' Test the instantiation of the telemetryKeck, systemDiagnosis and configFile objects
    '''

    cfg = configFile(sd)
    
    return cfg

#%% RUN FUNCTIONS
test_Fourier_fitting()
test_psfao21_instantiation()
test_psfao21_fitting()
psfr = test_psfr(path_root)
res  = test_prime(psfr)