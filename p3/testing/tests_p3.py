#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:17:49 2021

@author: omartin
"""

#%% IMPORTING LIBRARIES
import os
import pathlib
import sys
import time

import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from distutils.spawn import find_executable

from p3.aoSystem.fourierModel import fourierModel
import p3.aoSystem.FourierUtils as FourierUtils

import p3.psfao21 as psfao21Main
from p3.telemetry.telemetryKeck import telemetryKeck
from p3.telemetry.telemetryKASP import telemetryKASP
from p3.telemetry.systemDiagnosis import systemDiagnosis
from p3.telemetry.configFile import configFile
from p3.psfr.psfR import psfR
import p3.psfr.psfrUtils as psfrUtils

from p3.psfao21.psfao21 import psfao21
from p3.psfFitting.psfFitting import psfFitting
from p3.psfFitting.psfFitting import displayResults


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
path_root = str(pathlib.Path(psfao21Main.__file__).parent.parent.parent.absolute()) +"/"
path_p3 = str(pathlib.Path(psfao21Main.__file__).parent.parent.absolute())
path_ini = path_p3 + '/aoSystem/parFiles/KECKII_NIRC2_20130801_12_00_19.254.ini'
path_img = path_p3 + '/aoSystem/data/20130801_n0004.fits'
path_calib = path_p3 + '/aoSystem/data/KECK_CALIBRATION/'

im_nirc2 = fits.getdata(path_img, ignore_missing_simple=True)
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

def test_psfao21_fitting(tol=1e-8):
    '''
        Test the fitting of a NIRC2 image with the PSFAO21 model with two strategies:
            - joint fitting of PSD parameters and static aberrations
            - split fitting : PSD parameters first, redefinition of the bounds as xf+-5/3xerr and fit of the static aberrations
    '''
    # instantiating the model
    psfao = psfao21(path_ini, path_root=path_p3)

    # initial guess
    n_modes = psfao.ao.tel.nModes
    x0 = [0.7, 4e-2, 0.5, 1e-2, 1, 0, 1.8, 0, 0, 0, 1.0, 0, 0, 0] + [0,]*n_modes
    fixed = (False,)*7 +(True,)*3 + (False,)*4 + (True,)*n_modes

    # fit
    res = psfFitting(im_nirc2, psfao, x0, verbose=2,fixed=fixed,
                                   ftol=tol, gtol=tol, xtol=tol,
                                   weights=None, normType=1)

    # display
    displayResults(psfao, res, nBox=90, scale='log10abs')

    return res


#%% PSF RECONSTRUCTION

def test_kasp_psfr(path_mat, path_save, true_r0=True, tol=1e-5):
    """
    Test the interface with the KASP .mat output file.
    """

    # instantiating/processing the telemetry and creating the .ini file
    trs = telemetryKASP(path_mat, path_save=path_save)
    r0_true = trs.atm.r0
    L0_true = trs.atm.L0
    sd = systemDiagnosis(trs, noiseMethod='nonoise')
    if true_r0:
        trs.atm.r0_tel = r0_true
        trs.atm.r0 = r0_true
        trs.atm.L0_tel = L0_true
        trs.atm.L0 = L0_true
        trs.atm.seeing = 3600*180/np.pi*0.978*trs.atm.wvl/r0_true
    configFile(sd)

    # get the psf
    psfr = psfR(trs)
    if psfr.ao.atm.nL>1:
        Cn2 = list(psfr.ao.atm.weights * psfr.ao.atm.r0**(-5/3))
    else:
        Cn2 = [psfr.ao.atm.r0]
    x0  = Cn2 + [1, 1] + [1, 0, 0, 0]

    fixed = (True,)*(psfr.ao.atm.nL + 2) + (False,)*3 + (True,)

    #adjust the flux and the position
    res = psfFitting(psfr.trs.cam.image, psfr, x0, verbose=2, fixed=fixed,
                     ftol=tol, xtol=tol, gtol=tol)

    psfr.wfe['PSF SR'] = psfr.SR[0]
    displayResults(psfr, res, scale='arcsinh')

    return psfr, res

def test_psfr(path_trs, tol=1e-5):
    """
    Test the PSF reconstruction
    """

    # Get the telemetry
    trs = test_telemetry(path_trs)
    sd  = test_systemDiagnosis(trs)
    test_configFile(sd)

    # Get the PSF
    psfr = psfR(sd.trs)
    r0 = psfr.ao.atm.r0
    x0 = [r0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    fixed = (True,)*3 + (False,)*4
    #adjust the flux and the position
    res = psfFitting(im_nirc2, psfr, x0, verbose=2, fixed=fixed,
                     ftol=tol, xtol=tol, gtol=tol)
    psfr.wfe['PSF SR'] = psfr.SR[0]
    displayResults(psfr, res, nBox=100, scale='arcsinh')

    return psfr, res

def test_prime(psfr, fixed_stat = True, tol=1e-5):
    """
    Test the PRIME approach, i.e. fit a few parameters of the reconstructed
    PSF model, namely the r0 and the optical gain
    """
    # defining the initial guess and fixed parameters
    n_modes = psfr.ao.tel.nModes
    x0 = [0.7, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0] + [0,]*n_modes
    fixed = (False,)*2 + (True,) + (False,)*4 + (fixed_stat,)*n_modes

    # fitting
    res = psfFitting(im_nirc2, psfr, x0, verbose=2,
                     fixed=fixed, xtol=tol, ftol=tol, gtol=tol)
    psfr.wfe['PSF SR'] = psfr.SR[0]
    # Display
    displayResults(psfr, res, nBox=100, scale='log10abs')

    return res

def test_telemetry(path_trs):
    """
    Test the instantiation of the telemetryKeck object
    """

    #load the telemetry
    if sys.platform[0:3] == 'win':
        path_sav = path_trs + '\\' + filename
    else:
        path_sav = path_trs + '/' + filename

    if not os.path.isfile(path_sav):
        psfrUtils.get_data_file(path_trs,filename)

    # path image/calibration
    trs = telemetryKeck(path_sav, path_img, path_calib, path_save=path_trs, nLayer=1)

    return trs

def test_systemDiagnosis(trs):
    """
    Test the instantiation of the telemetryKeck and systemDiagnosis objects
    """

    sd  = systemDiagnosis(trs)

    return sd

def test_configFile(sd):
    """
    Test the instantiation of the telemetryKeck, systemDiagnosis and configFile objects
    """

    cfg = configFile(sd)

    return cfg

#%% RUN FUNCTIONS
test_Fourier_fitting()
test_psfao21_instantiation()
test_psfao21_fitting()
psfr, res = test_psfr(path_root)
res  = test_prime(psfr)
