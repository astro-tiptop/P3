#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 14:57:57 2021

@author: omartin
"""

#%% IMPORTING LIBRAIRIES
import os
import time
import numpy as np
from numpy.random import uniform
import random
import numpy.fft as fft

from astropy.io import fits
from p3.psfao21.psfao21 import psfao21
import p3.aoSystem.FourierUtils as FourierUtils
np.random.seed(1)

def generate_psf(path_ini, n_inter=10, n_psf_folder=3500, add_static=0, nmodes=9,
                 mag=0, zp=25.44, dit=0.5, ndit=50, sky_mag=13.6, ron=60, norm=1,
                 save_path=None, nround=7, full_name=False, n_test=0,
                 bounds=[[0.05, 1e-3, 100, 1e-3, 0.5, 1.1, -0.1],
                         [0.4, 5e-2, 390, 1e-1, 2, 3.0, 0.1]]):

    '''
        Generate sub-folders of psfao21 PSFs from the .ini file pointed from path_ini
        and distributed over n_inter**2 sub-folders. Each sub-folder contains nPSFperFolder PSFs.

        INPUTS:
            -addStatic : If 1, static aberrations are included
            from the modal basis refered in the .ini file. Takes the nModes first modes
            -mag, zp, dit, ndit, sky_mag, ron : photometry, detector and sky characteristics
            - norm : normalization of the data; see fittingUtils.py
            - save_path : path where to save data
            - nround : number of digits to round the value in the labels
            - bounds : bounds of the unform distribution for each parameter
    '''

    tstart = time.time()
    #%% MODEL INSTANTIATION
    psfao = psfao21(path_ini)
    wvl = psfao.freq.wvlRef
    wvl_fact = 2*np.pi*1e-9/wvl

    #%% CHECKING THE NUMBER OF MODES
    if add_static and nmodes > psfao.ao.tel.nModes:
        print('WARNING : the inputs number of modes is too high and then cropped')
        nmodes = psfao.ao.tel.nModes

    #%% DEFINITION OF PARAMETERS DOMAIN

    # ------------------ DEFINING FOLDERS
    n_main = 2 # r0 and A are the primary parameters
    n_sub_folder = n_inter ** n_main

    # DEFINING THE DATA FOLDER
    id_noise = 'NO'
    id_static = 'NO'
    if add_static:
        id_static = ''
    if mag != 0:
        flux = 10 ** (-0.4*(mag - zp))*dit*ndit
        sky_flux = 10 ** (-0.4*(sky_mag - zp)) * ndit * dit * (psfao.ao.cam.psInMas/1e3) ** 2
        ron_stack = ron * np.sqrt(ndit)
        id_noise = ''

    # creating tha main data folder
    if save_path:
        # defining prefix
        instru = path_ini.split('/')[-1].split('.')[0]
        if full_name:
            psf_prefix = 'psfao21_' + instru + '_wvl_' + str(round(wvl*1e6, 3)) + 'µm'
        else:
            psf_prefix = 'psf'
        
        save_main_folder = 'PSFAO21_' + id_noise + 'NOISE_' + id_static + 'STATIC/'
        save_path = os.path.join(save_path, save_main_folder)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)


    # ------------------- DEFINING BOUNDS
    if (len(bounds[0]) == 0) or (len(bounds[1]) == 0):
        raise ValueError('Bounds can not be empty')
    if len(bounds[0]) != len(bounds[1]):
        raise ValueError('Bounds must be of same size')
    if (not add_static) and ((len(bounds[0]) < 6) or (len(bounds[1]) < 6)):
        raise ValueError('You must provide bounds values of the 6 parameters')
    if add_static and ((len(bounds[0]) < 7) or (len(bounds[1]) < 7)):
        raise ValueError('Bounds on static aberrations must be provided as well')

    r0_lb_500nm = bounds[0][0]
    r0_ub_500nm = bounds[1][0]
    A_lb_nm = bounds[0][2]
    A_ub_nm = bounds[1][2]
    C_lb = bounds[0][1]
    C_ub = bounds[1][1]
    ax_lb = bounds[0][3]
    ax_ub = bounds[1][3]
    p_lb = bounds[0][4]
    p_ub = bounds[1][4]
    beta_lb = bounds[0][5]
    beta_ub = bounds[1][5]
    if add_static:
        stat_lb = bounds[0][6] * wvl*1e9
        stat_ub = bounds[1][6] * wvl*1e9

    # ------------------- DEFINING R0 AND A DISTRIBUTIONS OVER THE WHOLE DATA SETS
    # bounds
    r0_lb = r0_lb_500nm*(wvl/500e-9)**1.2
    r0_ub = r0_ub_500nm*(wvl/500e-9)**1.2
    A_lb = (wvl_fact*A_lb_nm)**2
    A_ub = (wvl_fact*A_ub_nm)**2
    # intervals
    r0_int = np.round(np.linspace(r0_lb, r0_ub, n_inter+1), nround)
    A_int = np.round(np.linspace(A_lb, A_ub, n_inter+1), nround)
    # uniform distributions
    t = -1
    r0 = np.zeros((n_psf_folder, n_sub_folder))
    A = np.zeros((n_psf_folder, n_sub_folder))
    idr0 = np.zeros((n_psf_folder, n_sub_folder), dtype='int')
    idA = np.zeros((n_psf_folder, n_sub_folder), dtype='int')

    for k in range(n_inter):
        for j in range(n_inter):
            t = t+1
            # uniform distribution for each interval
            r0[:, t] = r0_int[k] + (r0_int[k+1] - r0_int[k])*uniform(size=n_psf_folder)
            A[:, t] = A_int[j] + (A_int[j+1] - A_int[j])*uniform(size=n_psf_folder)
            idr0[:, t] = k*np.ones(n_psf_folder)
            idA[:, t] = j*np.ones(n_psf_folder)
            # CREATING SUBFOLDERS
            if save_path:
                # define subfolders name
                idsub = 'r0_' + str(r0_int[k]) + '_sig2_' + str(A_int[j])
                # create subdirectories
                path_folder = save_path + idsub
                if not os.path.isdir(path_folder):
                    os.mkdir(path_folder)

    r053 = r0 ** (-5.0/3.0)

    # -------------------  DEFINING SECONDARY PARAMETERS
    C = uniform(low=C_lb, high=C_ub, size=n_psf_folder)
    ax = uniform(low=ax_lb, high=ax_ub, size=n_psf_folder)
    pe = np.linspace(p_lb, p_ub, n_psf_folder)
    be = uniform(low=beta_lb, high=beta_ub, size=n_psf_folder)
    if add_static:
        stat = uniform(low=stat_lb, high=stat_ub, size=(nmodes, n_psf_folder))

    #%% LOOPS
    pix2freq = 1/(psfao.ao.tel.D * psfao.freq.sampRef)**2
    otf_dl = fft.fftshift(psfao.freq.otfDL)
    id_stat = ''
    id_noise = ''
    # PRE-CALCULATE THE PSD
    for j in range(n_psf_folder):
        # DERIVING THE AO-CORRECTED PSD
        psd_in = psfao.freq.pistonFilter_ * psfao.moffat(psfao.freq.kx_,
                                                        psfao.freq.ky_,
                                                        [ax[j], pe[j],
                                                         0, be[j], 0, 0])

        Aemp = np.trapz(np.trapz(psd_in, psfao.freq.ky_[0]), psfao.freq.ky_[0])
        psd_in = psd_in/Aemp

        # DERIVING THE TELESCOPE OTF
        if add_static:
            otf_stat, _, _ = FourierUtils.getStaticOTF(psfao.ao.tel,
                                                      psfao.freq.nOtf,
                                                      psfao.freq.sampRef,
                                                      psfao.freq.wvlRef,
                                                      xStat=stat[:, j])
            otf_stat = fft.fftshift(otf_stat)
        else:
            otf_stat = otf_dl

        for k in range(n_sub_folder):
            # Total PSD
            psd = r053[j, k] *(psfao.freq.psdKolmo_) + \
                  psfao.freq.mskIn_ *(C[j] + psd_in * A[j, k])
            # COMPUTING THE PSF
            Bphi = fft.fft2(fft.fftshift(psd)) * pix2freq
            Dphi = np.real(2*(Bphi.max() - Bphi))
            psf_i = np.real(fft.fftshift(fft.ifft2(otf_stat * np.exp(-0.5*Dphi))))

            # ADDING THE NOISE
            if mag != 0:
                #noise_sky = np.random.poisson(skyFlux*np.ones_like(psf_i)) - skyFlux
                noise_sky = np.random.poisson(sky_flux*np.ones_like(psf_i))
                noise_dec = ron_stack*np.random.randn(psf_i.shape[0], psf_i.shape[1])
                noise_dec -= noise_dec.min()
                psf_i = np.random.poisson(flux*psf_i) + noise_sky  + noise_dec

            # NORMALIZING PSF
            psf_i, _ = FourierUtils.normalizeImage(psf_i, normType=norm)

            # SAVING
            if save_path:
                # psf name
                if add_static:
                    s = ['_m'+str(nn+1)+'_' + str(round(stat[nn, j], nround))
                         for nn in range(nmodes)]
                    id_stat = ''.join(s)
                if mag != 0:
                    id_noise = '_mag_' + str(mag)

                idpsf = psf_prefix + '_r0_' \
                            + str(round(r0[j, k], nround))\
                            + '_bg_' + str(round(C[j], nround))\
                            + '_s2_' + str(round(A[j, k], nround))\
                            + '_ax_' + str(round(ax[j], nround))\
                            + '_pe_' + str(round(pe[j], nround))\
                            + '_be_' + str(round(be[j], nround))\
                            + id_stat + id_noise  +  '_norm_'  + str(norm)

                # corresponding subfolders
                idsub = 'r0_' + str(r0_int[idr0[j, k]]) + '_sig2_' + str(A_int[idA[j, k]])
                # save
                hdu = fits.PrimaryHDU(psf_i)
                hdu.writeto(save_path + idsub + '/' + idpsf + '.fits', overwrite=True)

    print('Simulation of %d PSF done in %.2fs'%(n_psf_folder*n_sub_folder, time.time()-tstart))

    #%% ADDING THE TEST FOLDER
    if n_test:
        # getting the list of sub-folders
        list_folders = os.listdir(save_path)
        list_folders = [x for x in list_folders if x != 'test_data']
        
        # creating the sub-folder within the main saving folder
        test_path = os.path.join(save_path, 'test_data')
        if not os.path.isdir(test_path):
            os.mkdir(test_path)
        else:
            for filename in os.listdir(test_path):
                os.remove(test_path+'/'+filename)
            
        # selecting the test data randomly
        if n_test < 1: # convert fraction to absolute value
            n_test = int(n_test * n_psf_folder * n_sub_folder)
            
        n_test_folder = int(n_test/n_sub_folder) # test data per sub-folders

        for folder in list_folders:
            # list files within the sub-folder
            sub_path = os.path.join(save_path, folder)
            list_data = os.listdir(sub_path)
            # randomly define the indexes of test data
            ran_idx = random.sample(range(len(list_data)), n_test_folder)
            sub_list = [list_data[x] for x in ran_idx]
            
            for file_name in sub_list:
                path_file = os.path.join(sub_path, file_name)
                #if os.path.isfile(path_file):
                os.rename(path_file, os.path.join(test_path, file_name))
        