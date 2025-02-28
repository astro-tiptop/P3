#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 14:57:57 2021

@author: omartin
"""

#%% IMPORTING LIBRAIRIES
import os
import time
from pathlib import Path
import numpy as np
from numpy.random import uniform
import random
import numpy.fft as fft

from astropy.io import fits
from p3.psfao21.psfao21 import psfao21
import p3.aoSystem.FourierUtils as FourierUtils
np.random.seed(1)

def generate_psf(path_ini, n_inter=10, main_param=["r0", "sig2"], n_psf_folder=3500,
                 add_static=0, nmodes=None, mag=0, zp=25.44, dit=0.5, ndit=50,
                 sky_mag=13.6, ron=60, norm=1, save_path=None, nround=5,
                 full_name=False, n_test=0, n_split=10,
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
    if main_param[0]=="stat":
        add_static=True
    #%% CHECKING THE NUMBER OF MODES
    if add_static:
        if nmodes is None:
            nmodes = psfao.ao.tel.nModes
        else:
            if nmodes > psfao.ao.tel.nModes:
                print('WARNING : the inputs number of modes is too high and then cropped')
                nmodes = psfao.ao.tel.nModes

    #%% DEFINITION OF PARAMETERS DOMAIN

    # ------------------ DEFINING FOLDERS
    if main_param!=["r0", "sig2"] and main_param!=["stat", "sig2"]:
        raise ValueError("The first main parameter is either r0 or stat")
    n_main = 2 # r0 and A are the primary parameters
    n_sub_folder = n_inter ** n_main

    # DEFINING THE DATA FOLDER
    id_noise = 'NO'
    id_static = 'NO'
    id_mag = ""
    id_instru = "_" + psfao.ao.cam.tag
    if add_static:
        id_static = ''
    if mag:
        flux = 10 ** (-0.4*(mag - zp))*dit*ndit
        sky_flux = 10 ** (-0.4*(sky_mag - zp)) * ndit * dit * (psfao.ao.cam.psInMas/1e3) ** 2
        ron_stack = ron * np.sqrt(ndit)
        id_noise = ""
        id_mag = "_MAG" + str(mag)

    # creating tha main data folder
    if save_path:
        # defining prefix
        instru = path_ini.split('/')[-1].split('.')[0]
        if full_name:
            psf_prefix = 'psfao21_' + instru + '_wvl_' + str(round(wvl*1e6, 3)) + 'µm'
        else:
            psf_prefix = 'psf'

        save_main_folder = 'PSFAO21_' + id_noise + 'NOISE_' + id_static + "STATIC" + id_mag + id_instru + "/"
        save_path = os.path.join(save_path, save_main_folder)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    # ------------------- DEFINING BOUNDS
    val, inter, idxs = generate_parameters_values(bounds, wvl_fact, n_inter,
                                                  n_psf_folder, n_sub_folder,
                                                  nround, nmodes=nmodes,
                                                  add_static=add_static,
                                                  save_path=save_path,
                                                  main_param=main_param)
    r0, C, sig2, ax, pe, be, stat = val
    inter_1, inter_2 = inter
    id_1, id_2 = idxs
    r053 = r0**(-5.0/3.0)

    #%% LOOPS
    pix2freq = 1/(psfao.ao.tel.D * psfao.freq.sampRef)**2
    otf_dl = fft.fftshift(psfao.freq.otfDL)
    str_stat = ""
    str_mag = ""
    # PRE-CALCULATE THE PSD
    for j in range(n_psf_folder):
        # DERIVING THE AO-CORRECTED PSD
        psd_in, _ = psfao.psfao_19.psd([1, 0, 1, ax[j], pe[j], 0, be[j]])
        psd_in -= psfao.psfao_19._vk
        if main_param[0]=="stat":
            psd_fit = r053[j]*psfao.freq.psdKolmo_

        # DERIVING THE TELESCOPE OTF
        if add_static and main_param[0]=="r0":
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
            psd = psfao.freq.mskIn_ *(C[j] + psd_in * sig2[j, k])
            if main_param[0]=="stat":
                psd += psd_fit

            # COMPUTING THE PSF
            Bphi = fft.fft2(fft.fftshift(psd)) * pix2freq
            Dphi = np.real(2*(Bphi.max() - Bphi))
            if add_static and main_param[0]=="stat":
                otf_stat, _, _ = FourierUtils.getStaticOTF(psfao.ao.tel,
                                                          psfao.freq.nOtf,
                                                          psfao.freq.sampRef,
                                                          psfao.freq.wvlRef,
                                                          xStat=stat[:, j, k])
                otf_stat = fft.fftshift(otf_stat)
            psf_i = np.real(fft.fftshift(fft.ifft2(otf_stat * np.exp(-0.5*Dphi))))

            # ADDING THE NOISE
            if mag:
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
                if mag:
                    str_mag = '_mag_' + str(mag)

                idpsf = psf_prefix
                if main_param[0]=="r0":
                    idpsf+=  '_r0_' + str(round(r0[j, k], nround))
                    if add_static:
                        s = ['_m'+str(nn+1)+'_' + str(round(stat[nn, j], nround))
                             for nn in range(nmodes)]
                        str_stat = ''.join(s)
                else:
                    idpsf+=  ('_r0_' + str(round(r0[j], nround)))*(len(np.unique(r0))!=1)
                    s = ['_m'+str(nn+1)+'_' + str(round(stat[nn, j, k], nround))
                             for nn in range(nmodes)]
                    str_stat = ''.join(s)

                idpsf += ('_bg_' + str(round(C[j], nround)))*(len(np.unique(C))!=1)\
                       + '_s2_' + str(round(sig2[j, k], nround))\
                       + ('_ax_' + str(round(ax[j], nround)))*(len(np.unique(ax))!=1)\
                       + ('_pe_' + str(round(pe[j], nround)))*(len(np.unique(pe))!=1)\
                       + ('_be_' + str(round(be[j], nround)))*(len(np.unique(be))!=1)
                idpsf += str_stat + str_mag  +  '_norm_'  + str(norm)

                # corresponding subfolders
                idsub = main_param[0]+'_' + str(inter_1[id_1[j, k]]) + '_sig2_' + str(inter_2[id_2[j, k]])
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

        # split in multiple subfolders
        split_test_data(save_path, n_sub_folder=n_split)
        # put the test data in the main folder
        path = Path(save_path)
        path_parent = str(path.parent.absolute())
        new_name = "/PSFAO21_TESTDATA_" + id_noise + 'NOISE_' + id_static + "STATIC" + id_mag + id_instru + "/"
        os.rename(test_path, path_parent + new_name)


def split_test_data(path_folder, n_sub_folder=10, mode=511):
    """
    Randomly distribute the data into the testdata directory into n_sub_folder
    sub-folders to ease the data transfer management
    """

    # check the path
    if not os.path.isdir(path_folder):
        raise ValueError("The path does not exist")

    # checking the presence of a test_data folder
    path_test = os.path.join(path_folder, "test_data")
    if not os.path.isdir(path_test):
        raise ValueError("The test_data folder does not exist")

    # selecting the .fits files
    tmp = os.listdir(path_test)
    list_file = [tmp[n] for n in range(len(tmp)) if ".fits" in tmp[n]]
    n_files = len(list_file)

    # checking that the number of files can be divided into n_sub_folder folders
    #of same size
    if np.mod(n_files, n_sub_folder):
        n_values = np.array(range(1, n_sub_folder*2))
        idx = np.argwhere(n_values == n_sub_folder)[0]
        id_0 = np.argwhere(np.mod(n_files, n_values) ==0)
        good_val = n_values[id_0]
        n_sub_folder = good_val[np.argmin(abs(id_0-idx))][0]
        print("The test data will be distributed over "+ str(n_sub_folder)\
              +" sub_folders to have the same number of data per sub-folder")

    if n_sub_folder==1:
        raise ValueError("The files can not be equally distributed in an integer number of sub-folders")

    # creating the list of list of files
    n_files_per_sub = int(n_files/n_sub_folder)
    arr_of_index = np.array(random.sample(range(n_files), n_files))
    arr_of_index = arr_of_index.reshape(n_sub_folder, n_files_per_sub)

    # creating anf filling sub_folders
    for k in range(n_sub_folder):
        # creating the sub_folder
        path_test_k = os.path.join(path_test, "sub_folder_" + str(k))
        if not os.path.isdir(path_test_k):
            os.mkdir(path_test_k, mode=mode)

        # moving the files
        print(n_files_per_sub)
        for j in range(n_files_per_sub):
            file_name = list_file[arr_of_index[k][j]]
            path_file_old = os.path.join(path_test, file_name)
            path_file_new = os.path.join(path_test_k, file_name)
            os.rename(path_file_old, path_file_new)
            print("done")

def generate_parameters_values(bounds, wvl_fact, n_inter, n_psf_folder, n_sub_folder,
                               nround, nmodes=None, add_static=False, save_path=None,
                               main_param=["r0", "sig2"]):
    """
    Generate lists of random values for the parameters
    """

    if (len(bounds[0]) == 0) or (len(bounds[1]) == 0):
        raise ValueError('Bounds can not be empty')
    if len(bounds[0]) != len(bounds[1]):
        raise ValueError('Bounds must be of same size')
    if (not add_static) and ((len(bounds[0]) < 6) or (len(bounds[1]) < 6)):
        raise ValueError('You must provide bounds values of the 6 parameters')
    if add_static and ((len(bounds[0]) < 7) or (len(bounds[1]) < 7)):
        raise ValueError('Bounds on static aberrations must be provided as well')

    def check_bounds(low_bound, high_bound, fact=1):
        return low_bound*fact, high_bound*fact

    def generate_distribution(low_bound, high_bound, size, type_distribution="uniform"):
        if type_distribution=="uniform":
            return uniform(low=low_bound, high=high_bound, size=size)
        elif type_distribution=="linspace":
            return np.linspace(low_bound, high_bound, size)

    wvl = 2*np.pi*1e-9/wvl_fact

    r0_lb, r0_ub = check_bounds(bounds[0][0], bounds[1][0], fact=(wvl/500e-9)**1.2)
    C_lb, C_ub = check_bounds(bounds[0][1], bounds[1][1])
    sig2_lb_nm, sig2_ub_nm = check_bounds(bounds[0][2], bounds[1][2])
    ax_lb, ax_ub = check_bounds(bounds[0][3], bounds[1][3])
    p_lb, p_ub = check_bounds(bounds[0][4], bounds[1][4])
    beta_lb, beta_ub = check_bounds(bounds[0][5], bounds[1][5])
    if add_static:
        stat_lb, stat_ub = check_bounds(bounds[0][6], bounds[1][6], fact=wvl*1e9)

    # ------------------- DEFINING R0 AND A DISTRIBUTIONS OVER THE WHOLE DATA SETS
    # bounds
    if main_param[0]=="r0":
        x1_lb = r0_lb
        x1_ub = r0_ub
        x_1 = np.zeros((n_psf_folder, n_sub_folder))
    elif main_param[0]=="stat":
        x1_lb = stat_lb
        x1_ub = stat_ub
        x_1 = np.zeros((nmodes, n_psf_folder, n_sub_folder))
    else:
        raise ValueError("The first main parameter must be r0 or stat")
    sig2_lb = (wvl_fact*sig2_lb_nm)**2
    sig2_ub = (wvl_fact*sig2_ub_nm)**2

    # intervals
    x_int = np.round(np.linspace(x1_lb, x1_ub, n_inter+1), nround)
    sig2_int = np.round(np.linspace(sig2_lb, sig2_ub, n_inter+1), nround)
    # uniform distributions
    t = -1
    sig2 = np.zeros((n_psf_folder, n_sub_folder))
    id_x = np.zeros((n_psf_folder, n_sub_folder), dtype='int')
    id_sig2 = np.zeros((n_psf_folder, n_sub_folder), dtype='int')

    for k in range(n_inter):
        for j in range(n_inter):
            t = t+1
            # uniform distribution for each interval
            if main_param[0]=="r0":
                x_1[:, t] = x_int[k] + (x_int[k+1] - x_int[k])*uniform(size=n_psf_folder)
            else:
                x_1[:, :, t] = x_int[k] + (x_int[k+1] - x_int[k])*uniform(size=(nmodes, n_psf_folder))
            sig2[:, t] = sig2_int[j] + (sig2_int[j+1] - sig2_int[j])*uniform(size=n_psf_folder)
            id_x[:, t] = k*np.ones(n_psf_folder)
            id_sig2[:, t] = j*np.ones(n_psf_folder)
            # CREATING SUBFOLDERS
            if save_path:
                # define subfolders name
                idsub = main_param[0] + '_' + str(x_int[k]) + '_sig2_' + str(sig2_int[j])
                # create subdirectories
                path_folder = save_path + idsub
                if not os.path.isdir(path_folder):
                    os.mkdir(path_folder)

    # -------------------  DEFINING SECONDARY PARAMETERS
    stat = None
    C = generate_distribution(C_lb, C_ub, n_psf_folder)
    ax = generate_distribution(ax_lb, ax_ub, n_psf_folder)
    pe = generate_distribution(p_lb, p_ub, n_psf_folder, type_distribution="linspace")
    be = generate_distribution(beta_lb, beta_ub, n_psf_folder)

    if main_param[0]=="r0":
        r0 = x_1
        if add_static:
            stat = generate_distribution(stat_lb, stat_ub, (nmodes, n_psf_folder))
    else:
        stat = x_1
        r0 = generate_distribution(r0_lb, r0_ub, n_psf_folder)

    val = (r0, C, sig2, ax, pe, be, stat)
    inter = (x_int, sig2_int)
    idxs = (id_x, id_sig2)

    return val, inter, idxs