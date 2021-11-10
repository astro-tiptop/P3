#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:37:55 2021

@author: omartin
"""
import os
import shutil
from pathlib import Path
from astropy.time import Time
from configparser import ConfigParser
import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
# Julien Milli's
from query_eso_archive import query_simbad
from plot_sparta_data import plot_sparta_data
# Arthur Vigan's
from sphere.transmission import irdis_nd
# P3 librairies
from psfao21.psfao21 import psfao21
from psfFitting.psfFitting import psfFitting
import aoSystem.FourierUtils as FourierUtils
from aoSystem.zernike import zernike

#%% IMAGE PROCESSING
def remove_corrupted_data(path_data, path_out):
    """
        Read all data folders and move the data with corrupted PSF to the outliers folder.
        Corruption is tracked by bad centering of the PSF.
    """

    # list the data sets
    listData = os.listdir(path_data)

    # loop on all folders
    list_corr = []
    for kData, file in enumerate(tqdm.tqdm(listData)): # data folder
        path_data_k = path_data + file +'/'
        if os.path.isfile(path_data_k + 'ird_convert_recenter_dc5-SPH_SPARTA_PSFDATA-psf_sparta_data.fits'):
            path_fits = path_data_k + 'ird_convert_recenter_dc5-IRD_SCIENCE_PSF_MASTER_CUBE-median_unsat.fits'
            # read the image
            im   = fits.getdata(path_fits) #first dim : left/right, 2nd dim : nacquisition
            if im.ndim == 3:
                im = im[0]
            elif im.ndim == 4:
                im = im[-1,0]

            # getting metrics
            hdr = fits.getheader(path_fits)
            SRMEAN, SRMIN, SRMAX = read_strehl_value(hdr)
            fwhm, _, _, _, _, _ = get_ambi_parameters(hdr)

            F, ron = FourierUtils.getFlux(im, nargout=2)

            # getting through metrics
            is_corrupted = im[32,32]!=im.max() + F/ron<1e4\
                         + fwhm>2 + SRMEAN<0.35 + (SRMAX-SRMIN) >0.2

            if is_corrupted:
                list_corr.append(listData[kData])

        else:
            list_corr.append(listData[kData])

    print('Detection of %d corrupted files'%(len(list_corr)))
    for n in tqdm.tqdm(range(len(list_corr))):
        print(list_corr[n])
        shutil.move(path_data+list_corr[n], path_out)

def process_clean_data(path_data, nPix):
    """
        Process pre-cleaned SPHERE cubes of coadds to recenter the frame,
        stack them and normalize them
    """
    # read data
    im = fits.getdata(path_data)
    hdr= fits.getheader(path_data)
    # processing
    nF = im.shape[0]
    out = np.zeros((nF, nPix, nPix))
    for kf in range(nF):
        tmp = im[kf]
        x0, y0 = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)
        out[kf] = tmp[x0-nPix//2:x0+nPix//2, y0-nPix//2:y0+nPix//2]
    # stacking
    out = np.mean(out, axis=0)
    # normalizing
    DIT = hdr['ESO DET SEQ1 DIT']
    ND  = irdis_nd(hdr['HIERARCH ESO INS COMB IFLT'], ['HIERARCH ESO INS4 FILT2 NAME'])
    out = out/DIT/ND
    return out

def sort_telemetry_folder(path_telemetry):
    """
    Gather the SPHER.* .fits file in the path_dtts folder into sub-folder for
    each observation date.
    """
    if not os.path.isdir(path_telemetry):
        raise ValueError("The folder does not exist")
    path_telemetry = Path(path_telemetry)

    years = os.listdir(path_telemetry)
    for year in years:
        path_year = Path(os.path.join(path_telemetry, year))
        # get the list of .fits file
        list_fits = os.listdir(path_year)
        list_fits = [file for file in list_fits if ".fits" in file]
        list_date = [file.split(".")[1].split("T")[0] for file in list_fits]
        list_unique = np.unique(list_date)
        for date in list_unique:
            path_date = os.path.join(path_year, date)
            # create the folder
            if not os.path.isdir(path_date):
                os.mkdir(path_date)
            # move the corresponding file into the folder
            idx = [i for i, x in enumerate(list_date) if x == date]
            sub_list = np.take(list_fits, idx)
            for file in sub_list:
                path_old = os.path.join(path_year, file)
                path_new = os.path.join(path_date, file)
                os.rename(path_old, path_new)

def query_sparta_data(path_telemetry, debug=False, redo=False):
    """
    Run the plot_sparat_data script to query SPARTA data (acquired every 30s)
    as well as additionnal parameters from the ASM, SLODAR, ECMWF,...
    """
    if not os.path.isdir(path_telemetry):
        raise ValueError("The folder does not exist")
    path_telemetry = Path(path_telemetry)

    years = os.listdir(path_telemetry)
    for year in years:
        path_year = Path(os.path.join(path_telemetry, year))
        list_sub = os.listdir(path_year)
        # run
        for folder in list_sub:
            path_folder = os.path.join(path_year, folder)
            print("Process folder " + folder)
            if os.path.isdir(path_folder):
                list_file = os.listdir(path_folder)
                wfs_is_there = np.any(["sparta_visible_WFS" in file for file in list_file])
                if redo or not wfs_is_there:
                    plot_sparta_data(path_raw=path_folder,
                                     path_output=path_folder,
                                     plot=False,
                                     debug=debug)
#%% IMAGE FITTING
def reading_ini_file(path_ini, path_to_modes=None):
    """
    Return the parser to read and modify the .ini file.
    We can change the modal basis with path_to_modes. By default, the modes will
    be defined from the path given in the .ini file
    """
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(path_ini)
    if path_to_modes is not None:
        parser.set('telescope','PathStatModes', '\'' + path_to_modes + '\'')
        with open(path_ini, 'w') as configfile:
            parser.write(configfile)

    return parser

def get_number_modes(parser, path_p3="/home/omartin/Projects/P3"):
    """
    Read the .ini file to get the number of modes
    """
    path_stat = parser.get('telescope','PathStatModes')
    mat_mod = fits.getdata(path_p3 + path_stat[1:-1])
    return mat_mod.shape[0]

def instantiate_psfao_model(parser, path_ini, seeing_los, wvl, n_px,
                            path_p3="/home/omartin/Projects/P3"):
    """
    Return an instance of the PSFAO21 model.
    """

    # updating the .ini file
    if seeing_los < 0:
        seeing_los = 0.7
    parser.set('atmosphere','Seeing',str(seeing_los))
    parser.set('sources_science','Wavelength',str([wvl]))
    parser.set('sensor_science','FieldOfView',str(n_px))

    with open(path_ini, 'w') as configfile:
        parser.write(configfile)

    # instantiating the model
    return psfao21(path_ini, path_root=path_p3)

def define_initial_guess(im, wvl, r0=None, SR=None,  n_modes=0, normType=0):
    """
    Define the initial guess for the image fitting with PSFAO21
    """
    if r0 is None:
        r0 = 0.16 * (wvl/500e-9)**2
    if SR is None:
        sig2 = (2*np.pi*120e-9/wvl)**2
    else:
        sig2 = -np.log(SR)
    im_nnz = np.copy(im)
    if im_nnz.min()<0:
        im_nnz -= im_nnz.min()
    F = FourierUtils.normalizeImage(im_nnz, normType=normType)[0].sum()
    x0 =[r0, 0, sig2, 1e-2, 1, 0, 1.3, # psfao21 model
         0, 0, 0, # jitter x-y-xy
         F, 0, 0, 0] # flux, dx, dy, background
    x0+= [0,]*n_modes
    return x0

def define_fixed_parameters(n_px, D, wvl, n_actu, pixel_scale_mas,
                            fix_C=True, fix_shift=False, n_modes=0):
    """
    Define the tuple of fixed/free parameters for the fitting:
    INPUTS:
        - n_px, the size of the image in pixels
        - D, the telescope diameter in meters
        - n_actu, the number of DM actuators along one dimension
        - pixel_scale, the pixel scale of the scientific camera in mas
        - fix_C, to fix the constant PSD in the PSFAO model
        - fix_shift, to fit the astrometry
        - n_modes, the number of instrumental modes to fit
    """
    # determining if the image contains the PSF wings
    dm_cutoff = 3600*1e3*180/np.pi*n_actu * wvl/D/pixel_scale_mas
    if dm_cutoff > n_px:
        fix_r0 = True
    else:
        fix_r0 = False

    # defining the tuple of xied parameters
    fixed = (fix_r0, fix_C, False, False, False, False, False,)
    fixed += (True,)*3 # jitter
    fixed += (False, fix_shift, fix_shift, False) # photometry, astrometry, background
    fixed += (False,)*n_modes
    return fixed

def fitting_image(im, psfao, r0=None, SR=None, fit_stat=True, weights=None, normType=0,
                  tol=1e-8, max_nfev=100, verbose=-1, fix_C=True):
    """
    Fitting a 2D image of a star by considering the instance psfao as a model
    for the PSF. If fit_lwe is True, the fit jointly estimates the atmospheric
    parameters and the static aberrations modes.
    Returns the res dictionnary from the psffitting function.
    """

    if not fit_stat or SR<0.4:
        x0 = define_initial_guess(im, psfao.wvl[0], r0=r0, SR=SR,
                                  normType=normType, n_modes=0)
        fixed = define_fixed_parameters(im.shape[0], psfao.ao.tel.D, psfao.wvl[0],
                                        psfao.ao.dms.nActu1D, psfao.ao.cam.psInMas,
                                        fix_C=fix_C, fix_shift=False, n_modes=0)
        # fit
        res = psfFitting(im, psfao, x0, fixed=fixed, weights=weights, normType=normType,
                         verbose=verbose, ftol=tol, gtol=tol, xtol=tol, max_nfev=max_nfev)

    else:
        x0 = define_initial_guess(im, psfao.wvl[0], r0=r0, SR=SR,
                                  normType=normType, n_modes=psfao.ao.tel.nModes)
        fixed = define_fixed_parameters(im.shape[0], psfao.ao.tel.D, psfao.wvl[0],
                                        psfao.ao.dms.nActu1D, psfao.ao.cam.psInMas,
                                        fix_C=fix_C, fix_shift=True,
                                        n_modes=psfao.ao.tel.nModes)


    # fit them all
    res = psfFitting(im, psfao, x0, fixed=fixed, weights=weights, normType=normType,
                     verbose=verbose, ftol=tol, gtol=tol, xtol=tol, max_nfev=max_nfev)

    if not fit_stat or SR<0.4:
        res.x = np.concatenate((res.x, -np.ones(psfao.ao.tel.nModes)))
        res.xerr = np.concatenate((res.xerr, -np.ones(psfao.ao.tel.nModes)))

    # getting the parameters
    SR = [res.SR_sky, res.SR_fit]
    FWHM = [res.FWHMx_sky, res.FWHMy_sky, res.FWHMx_fit, res.FWHMx_fit]
    err = [res.mse, res.mae, res.fvu]

    return (list(res.x), list(res.xerr), SR, FWHM, err)

def update_data_struct(data, out):
    """
    Udate the data structure with the results of the fit
    """
    # unpacking the parameters from the out tuple
    res_x, res_xerr, SR, FWHM, err = out
    data += res_x[:7] + res_x[10:]\
          + res_xerr[:7] + res_xerr[10:] + SR + FWHM + err
    return data

def define_database_colnames(fit=False, n_modes=0):
    """
    Define the name for the columns of the Dataframe.
    """
    names=['TARGET', 'OBS ID', 'DATE', 'EXP TIME', "RA", "DEC", 'AIRMASS', 'MEAN WAVELENGTH [µm]',
           'BANDWIDTH [µm]', 'V MAG', 'R MAG', 'G MAG', 'J MAG', 'H MAG', 'K MAG',
           "WFS NPH [#photons/aperture/frame]", "WFS frame rate [Hz]", "WFS FILTER", "WFS CENTRAL WAVELENGTH [nm]",
           "WFS SPECTRAL BANDWIDTH [nm]", "WFS COG WINDOW RADIUS [PX]", "LOOP DELAY [FRAME]", "LOOP GAIN",
           'FWHMLINOBS [as]', 'tau0 [s]', 'wSpeed [m/s]', 'wDir [deg]', 'RHUM [%]','PRESSURE',
           'SRMIN', 'SRMAX', 'SRMEAN', 'FWHM [mas]', 'MIN SEEING SPARTA [as]',
           'MAX SEEING SPARTA [as]', 'MEAN SEEING SPARTA [as]', 'MIN WSPEED SPARTA [m/s]',
           'MAX WSPEED SPARTA [m/s]', 'MEAN WSPEED SPARTA [m/s]', 'MIN SR SPARTA',
           'MAX SR SPARTA', 'MEAN SR SPARTA', 'PIXEL SCALE [mas]', 'gain [e-/ADU]',
           'ron [e-]', 'DIT [s]', 'NDIT']

    if fit:
        names += ["FIT R0", "FIT C" , "FIT A", "FIT AX", "FIT P", "FIT THETA", "FIT BETA",
                  "FIT F", "FIT DX", "FIT DY", "FIT BKG"]
        names += ["FIT MODE" + str(n) for n in range(n_modes)]
        names += ["ERR R0", "ERR C" , "ERR A", "ERR AX", "ERR P", "ERR THETA", "ERR BETA",
                  "ERR F", "ERR DX", "ERR DY", "ERR BKG"]
        names += ["ERR MODE" + str(n) for n in range(n_modes)]
        names += ["SR SKY", "SR FIT", "FWHMX SKY", "FWHMY SKY", "FWHMX FIT", "FWHMY FIT"]
        names += ["MSE", "MAE", "FVU"]

    return names

#%% DATA ANALYSIS
def compute_tiptilt_filtering_matrix(pupil, mat_modes=None):
    """
    Compute the matrix to filter the tip-tilt compnent from ` phase map.
    """
    # create tip-tilt modes
    n_pup = pupil.shape[0]
    z = zernike([2,3], n_pup, pupil=pupil.astype(bool))
    M = z.modes.reshape((2, n_pup**2)).T
    # compute the filtering matrix
    M_inv = np.linalg.pinv(M)
    H_filter = np.eye(n_pup**2) - np.dot(M, M_inv)
    # compute the phase to coeffs matrix
    phase_to_coeffs = None
    if mat_modes is not None:
        if mat_modes.shape[0]!=n_pup**2:
            raise ValueError("The modal_basis must be shaped as a n_pup**2 x n_modes array")
        phase_to_coeffs = np.linalg.pinv(mat_modes)
    return H_filter, phase_to_coeffs, M_inv

def remove_tiptilt_from_phase(phase, H_filter, phase_to_coeffs=None):
    """
    Remove the tip-tilt component from the phase and reproject over the modes.
    """
    # create tip-tilt modes
    n_pup = int(np.sqrt(H_filter.shape[0]))
    if phase.shape[0]!=n_pup:
        raise ValueError("The phase must have the same resolution than the filter")
    #filtering the phase
    phase_filtered = np.dot(H_filter, phase.reshape(n_pup**2))
    coeffs_filtered = None
    # projecting over the tip-tilt free basis
    if phase_to_coeffs is not None:
        coeffs_filtered = np.dot(phase_to_coeffs, phase_filtered)

    return phase_filtered.reshape(n_pup, n_pup),  coeffs_filtered

def compute_lwe_wavefront_error(df, path_ini, path_root=None, remove_tiptilt=True):
    """
    Computes the LWE wavefront error in nm from dataframe df.
    """
    if "FIT MODE0" not in df:
        raise ValueError("There is no LWE data in the data frame")

    # subsetting the data frame to get the static coefficients only
    mode_col =  [col for col in df.columns if "FIT MODE" in col]
    df_sub = df[mode_col]

    # grab the modal basis as a n_px**2 x n_modes matrix
    psfao = psfao21(path_ini, path_root=path_root)
    n_pup = psfao.ao.tel.resolution
    mat_modes = psfao.ao.tel.statModes.reshape(n_pup**2, psfao.ao.tel.nModes)
    pupil = psfao.ao.tel.pupil
    mat_modes = mat_modes*pupil.reshape(n_pup**2)[:, np.newaxis]

    # Computting the filtering matrix to remove the tip-tilt component
    lwe_col = []
    if remove_tiptilt:
        coeffs_col = []
        H_filter, P2C, _ = compute_tiptilt_filtering_matrix(pupil, mat_modes=mat_modes)

    # looping on rows to get the tip-tilt-free coefficients
    for n in tqdm.tqdm(range(df_sub.shape[0])):
        tmp = np.array(df_sub.loc[n])
        if tmp.any():
            opd = np.reshape(np.sum(mat_modes*tmp, axis=1), (n_pup, n_pup))
            if remove_tiptilt:
                opd, coeffs = remove_tiptilt_from_phase(opd, H_filter, phase_to_coeffs=P2C)
                coeffs_col.append(coeffs)
            # compute the LWE wavefront error
            lwe_err = np.std(opd[opd!=0])
        else:
            coeffs_col.append([-1,]*psfao.ao.tel.nModes)
            lwe_err = -1
        lwe_col.append(lwe_err)

    # update the data frame
    df["LWE [nm]"] = lwe_col
    if remove_tiptilt:
        coeffs_col = np.array(coeffs_col)
        for n in range(psfao.ao.tel.nModes):
            df["FIT MODE" + str(n) + " TT REMOVED"] = coeffs_col[:, n]
    return df

def plot_lwe_histograms(df_in, bins=20, alpha=0.5, thres=200):
    """
    Plot histograms of LWE unsigned amplitude.
    """
    #remove cases where the fit did not occur
    df = df_in.copy()
    mode_col =  [col for col in df.columns if "FIT MODE" in col]
    df[mode_col][df[mode_col]>300] = np.nan
    df.dropna(subset=mode_col, inplace=True)


    # plots histograms
    fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    m = abs(df["FIT MODE0"])[abs(df["FIT MODE0"])<thres]
    weights = np.ones(m.count())/m.count()
    axs[0].hist(m, bins=bins, label="East", weights=weights, alpha=alpha)
    m = abs(df["FIT MODE1"])[abs(df["FIT MODE1"])<thres]
    weights = np.ones(m.count())/m.count()
    axs[0].hist(m, bins=bins, label="South", weights=weights, alpha=alpha)
    m = abs(df["FIT MODE2"])[abs(df["FIT MODE2"])<thres]
    weights = np.ones(m.count())/m.count()
    axs[0].hist(m, bins=bins, label="West", weights=weights, alpha=alpha)
    axs[0].set_xlabel("Differential piston [nm]")
    axs[0].set_ylabel("Probability")
    axs[0].legend()

    weights = np.ones(df[mode_col[0]].count())/df[mode_col[0]].count()
    axs[1].hist(df["FIT MODE3"], bins=bins, label="North", weights=weights, alpha=alpha)
    axs[1].hist(df["FIT MODE4"], bins=bins, label="East", weights=weights, alpha=alpha)
    axs[1].hist(df["FIT MODE5"], bins=bins, label="South", weights=weights, alpha=alpha)
    axs[1].hist(df["FIT MODE6"], bins=bins, label="West", weights=weights, alpha=alpha)
    axs[1].set_xlabel("Tip [nm]")
    axs[1].legend()

    axs[2].hist(df["FIT MODE7"], bins=bins, label="North", weights=weights, alpha=alpha)
    axs[2].hist(df["FIT MODE8"], bins=bins, label="East", weights=weights, alpha=alpha)
    axs[2].hist(df["FIT MODE9"], bins=bins, label="South", weights=weights, alpha=alpha)
    axs[2].hist(df["FIT MODE10"], bins=bins, label="West", weights=weights, alpha=alpha)
    axs[2].set_xlabel("Tilt [nm]")
    axs[2].legend()

def plot_wfe_versus_ngs_mag(df_in, band="G", step_mag=0.5):
    """
    Plot the SPHERE wavefront error/H-band SR versus the NGS magnitude.
    """
    pd.options.mode.chained_assignment = None

    df = df_in.copy()
    # Dropping missing mG values or meaningless SRMEAN values
    df['SRMEAN'][df['SRMEAN']>=0.99]=np.nan
    df['SRMEAN'][df['SRMEAN']==0]=np.nan
    df[band+' MAG'][df[band+' MAG']==-1]=np.nan
    df.dropna(subset=["SRMEAN", band+" MAG"], inplace=True)

    # Computing the WFE
    df["WFE"] = np.sqrt(-np.log(df["SRMEAN"]))*1.65e3/2/np.pi
    # grouping
    cat = np.arange(2, 15, step_mag)
    bins = pd.cut(df[band+' MAG'], cat)
    agg_df_SR = df.groupby(bins)['SRMEAN'].agg(['count', np.nanmedian, np.nanstd])
    agg_df_WFE = df.groupby(bins)['WFE'].agg(['count', np.nanmedian, np.nanstd])

    #plot H-band SR v G mag
    plt.figure()
    plt.errorbar(cat[:-1] +np.diff(cat)/2,
                 agg_df_SR['nanmedian'],
                 xerr=step_mag/2,
                 yerr=agg_df_SR['nanstd'],
                 fmt='o', capthick=2)
    plt.ylim([0,1])
    plt.grid()
    plt.xlabel(band+' magnitude')
    plt.ylabel('Measured Strehl at 1.65µm')

    #plot WFE v G mag
    plt.figure()
    plt.errorbar(cat[:-1] +np.diff(cat)/2,
                 agg_df_WFE['nanmedian'],
                 xerr=step_mag/2,
                 yerr=agg_df_WFE['nanstd'],
                 fmt='o', capthick=2)
    plt.grid()
    plt.xlabel(band+' magnitude')
    plt.ylabel('Measured wavefront error [nm]')

def plot_mag_versus_flux(df_in, band="G", y_unit="nph/s/nm", step_mag=0.5):
    """
    Plot the SPHERE wavefront error/H-band SR versus the NGS magnitude.
    """
    pd.options.mode.chained_assignment = None

    df = df_in.copy()
    # Dropping missing mG values or meaningless SRMEAN values
    df[band+' MAG'][df[band+' MAG']==-1]=np.nan
    df.dropna(subset=["WFS NPH [#photons/aperture/frame]",
                      "WFS frame rate [Hz]",
                      band+" MAG"], inplace=True)

    if y_unit=="nph/subaperture/frame":
        flux = df["WFS NPH [#photons/aperture/frame]"]
    else:
        flux = df["WFS NPH [#photons/aperture/frame]"]\
                * 1240 * df["WFS frame rate [Hz]"]/df["WFS SPECTRAL BANDWIDTH [nm]"]
    df["TOTAL FLUX"] = np.log10(flux)

    # grouping
    plt.figure()
    if step_mag:
        cat = np.arange(2, 15, step_mag)
        bins = pd.cut(df[band+' MAG'], cat)
        agg_df = df.groupby(bins)['TOTAL FLUX'].agg(['count', np.nanmedian, np.nanstd])
        plt.errorbar(agg_df['nanmedian'],
                     cat[:-1] +np.diff(cat)/2,
                     yerr=step_mag/2,
                     xerr=agg_df['nanstd'],
                     fmt='o', capthick=2)
    else:
        plt.scatter(df["TOTAL FLUX"], df[band+' MAG'])
    plt.ylim([2,15])
    plt.grid()
    plt.ylabel(band+' magnitude')
    if y_unit=="nph/subaperture/frame":
        plt.xlabel('log10(#photons/subaperture/frame)')
    else:
        plt.xlabel('log10(#photons/s/nm)')
    #plt.xscale('log', nonposx='clip')


#%% READING THE HEADER
def grab_wfs_parameters(hdr, g_mag, frame_rate):
    """
    Grab the information related to the WFS and loop.
    """
    filter_name, wvl, bw = define_wfs_wavelength(hdr)
    win = define_wfs_window(g_mag)
    delay = define_loop_delay(frame_rate)
    gain = define_loop_gain(g_mag)

    return (filter_name, wvl, bw, win, delay, gain)

def define_wfs_wavelength(hdr):
    """
    Return the central wavelength and the bandwidth of the spectral filter in front of the WFS.
    """
    filter_name = hdr["ESO INS4 FILT3 NAME"]
    if filter_name=="OPEN":
        return filter_name, 700, 450
    elif filter_name=="LP_780":
        return filter_name, 840, 120
    elif filter_name=="LP_475":
        return filter_name, 700, 400

def define_wfs_window(g_mag):
    """
    Return the radius in pixels of the Gaussian windowing function for the cog
    """
    if g_mag<8:
        return 3.0
    elif g_mag<10:
        return 2.0
    elif g_mag<10.5:
        1.5
    return 1.0

def define_loop_delay(frame_rate):
    """
    Return the AO loop delay in number of frame as a function of the frame rate in Hz.
    """
    if frame_rate<333:
        return 1
    elif frame_rate<667:
        return 2
    else:
        return 3

def define_loop_gain(g_mag):
    """
     Return the AO loop gain as a function of the star magnitude
    """
    if g_mag <=6.0:
        return 0.4
    elif g_mag <8.0:
        return 0.35
    elif g_mag<10.0:
        return 0.25
    elif g_mag<10.5:
        return 0.2
    elif g_mag<11.3:
        return 0.6

def get_obs_id(file_name, hdr):
    """
    Retrieve the name of the target
    """
    obj_name = hdr["ESO OBS TARG NAME"]
    obs_id = file_name.split("_")[-1]

    return obj_name, obs_id
def get_wavelength(path_data):
    """
        Read the file IRD_SCIENCE_LAMBDA_INFO to extract the filter information
    """
    wvl = 1e-6*fits.getdata(path_data + '/ird_convert_recenter_dc5-IRD_SCIENCE_LAMBDA_INFO-lam.fits')
    bw = wvl.max() - wvl.min()
    wvl = wvl.mean()

    return wvl, bw

def read_sparta_data(path_data, date_obs=None, path_dtts=None, which='last', n_subap=1240):
    """
        Read the file SPH_SPARTA_PSFDATA-psf_sparta_data to get the atmospheric parameters.
        if the files exists, this is a [3, npsf+1,4] array
        ○    dim 1 : r0, wind, strehl ratio, seeing
        ○    dim 2 : 0: npsf-1 : for each PSF file.   Last line = stat combining all frames of allfiles
        ○    dim 3 : min, max, avg
        reminder:  The SPARTA Strehl ratio is provided at 1.6 micron and the turbulence parameters (r0, seeing) at 500nm.
    """

    #"/ird_convert_recenter_dc5-SPH_SPARTA_SAMPLEDDATA-sampled_sparta_data.fits" contains time series
    # dim 0 : timestamp - dim 1: r0 500nm los - dim 2 :windspeed - dim 3 : SR - dim4 : seeing
    #dim5: ?, dim6: ? related to the WFS cog ?
    path_sparta = path_data + '/ird_convert_recenter_dc5-SPH_SPARTA_PSFDATA-psf_sparta_data.fits'
    if os.path.isfile(path_sparta):
        sparta = fits.getdata(path_sparta)
        # number of acquisitions during the observation
        nPSF = sparta.shape[1]-1
        # note : nPSF must == nFiles/2 with raw images or im.shape[0]//2 with processed images
        r0 = sparta[0, :, :]
        vspeed = sparta[1, :, :]
        SR = sparta[2, :, :]
        seeing = sparta[3, :, :]

        if which == 'last':
            r0 = r0[nPSF, :]
            vspeed = vspeed[nPSF, :]
            SR = SR[nPSF, :]
            seeing = seeing[nPSF, :]

    # grab the number of photons
    def find_closest(df, date, Texp):
        df["date"] = pd.to_datetime(df["date"])
        id_closest = np.argmin(abs(df["date"] - date))
        date_min = df["date"][id_closest] - pd.DateOffset(seconds=Texp/2)
        date_max = df["date"][id_closest] + pd.DateOffset(seconds=Texp/2)
        id_min = np.argmin(abs(df["date"] - date_min))
        id_max = np.argmin(abs(df["date"] - date_max))
        return id_min, id_max

    n_ph = np.nan
    rate = np.nan
    if path_dtts is not None and date_obs is not None:
        year = date_obs[:4]
        path_sub = path_dtts + year + "/" + date_obs
        if os.path.isdir(path_sub):
            # grab the exposure time and the exact date
            hdr = fits.getheader(path_sparta)
            Texp  = hdr['ESO DET SEQ1 DIT'] * hdr['ESO DET NDIT']
            date = pd.to_datetime(hdr["DATE-OBS"])
            # Get the number of photons
            tmp = [file for file in os.listdir(path_sub) if "sparta_visible_WFS" in file]
            if len(tmp)>0:
                file_name = tmp[0]
                df = pd.read_csv(path_sub + "/" + file_name)
                if 'flux_VisLoop[#photons/aperture/frame]' in df:
                    df_flux = df['flux_VisLoop[#photons/aperture/frame]']
                    id_min, id_max = find_closest(df, date, Texp)
                    n_ph = df_flux[id_min:id_max+1].median()/n_subap
                if 'Frame rate [Hz]' in df:
                    df_rate = df['Frame rate [Hz]']
                    rate = df_rate[id_min:id_max+1].median()

    return r0, vspeed, SR, seeing, n_ph, rate

def get_star_coordinates(hdr):
    """
        Returns the name of the object as well as its RA/DEC coordinates
    """
    # get the star coordinates and name
    return hdr['ESO OBS NAME'], float(hdr['RA']), float(hdr['DEC'])

def get_star_magnitudes(hdr):
    """
        Query simbad to get the star magnitudes.
    """

    VMAG = RMAG = GMAG = JMAG = HMAG = KMAG = RA = DEC = -1

    OB_NAME, RA, DEC = get_star_coordinates(hdr)

    # query simbad
    DICT_SIMBAD= query_simbad(Time(hdr['DATE-OBS']),
                              SkyCoord(RA*u.degree,DEC*u.degree),
                              name=OB_NAME)

    if DICT_SIMBAD is None:
        return VMAG, RMAG, GMAG, JMAG, HMAG, KMAG, RA, DEC

    # get magnitudes
    if 'simbad_FLUX_V' in DICT_SIMBAD:
        VMAG = DICT_SIMBAD['simbad_FLUX_V']
    if 'simbad_FLUX_R' in DICT_SIMBAD:
        RMAG = DICT_SIMBAD['simbad_FLUX_R']
    if 'simbad_FLUX_G' in DICT_SIMBAD:
        GMAG = DICT_SIMBAD['simbad_FLUX_G']
    if 'simbad_FLUX_J' in DICT_SIMBAD:
        JMAG = DICT_SIMBAD['simbad_FLUX_J']
    if 'simbad_FLUX_H' in DICT_SIMBAD:
        HMAG = DICT_SIMBAD['simbad_FLUX_H']
    if 'simbad_FLUX_K' in DICT_SIMBAD:
        KMAG = DICT_SIMBAD['simbad_FLUX_K']

    return VMAG, RMAG, GMAG, JMAG, HMAG, KMAG, RA, DEC

def get_detector_config(hdr):
    """
        Read the header to get the detector configuration.
    """
    if 'PIXSCAL' in hdr:
        psInMas = float(hdr['PIXSCAL'])
    else:
        psInMas = 12.25
    gain = float(hdr['ESO DET CHIP1 GAIN'])
    ron  = float(hdr['ESO DET CHIP1 RON'])
    if ron==0:
        ron = 4.0
    # http://www.eso.org/observing/dfo/quality/SPHERE/reports/HEALTH/trend_report_IRDIS_DARK_ron_HC.html
    DIT  = hdr['ESO DET SEQ1 DIT']
    NDIT = hdr['ESO DET NDIT']
    NDSKIP = hdr['ESO DET NDSKIP']

    return psInMas, gain, ron, DIT, NDIT, NDSKIP

def read_strehl_value(hdr):
    """
    Read the Strehl-ratio values in the header
    """
    if 'SRMEAN' in hdr:
        SRMEAN = float(hdr['SRMEAN'])
        SRMIN = float(hdr['SRMIN'])
        SRMAX = float(hdr['SRMAX'])
        return SRMEAN, SRMIN, SRMAX
    else:
        print('No Strehl-ratio values in the header')
        return -1, -1, -1

def get_date(hdr):
    """
    Get the date from the header
    """
    DATE_OBS = hdr['DATE-OBS']
    date = DATE_OBS[:10]
    exptime = DATE_OBS[11:]

    return date, exptime

def get_telescope_pointing(hdr):
    """
    Get the telescope pointing from the header.
    """
    TELAZ = float(hdr['ESO TEL AZ'])
    TELALT = float(hdr['ESO TEL ALT'])
    airmass = 0.5*(float(hdr['ESO TEL AIRM END']) + float(hdr['ESO TEL AIRM START']))

    return TELAZ, TELALT, airmass

def get_ambi_parameters(hdr):
    """
    Get atmosphere parameters from the header.
    """
    tau0 = float(hdr['ESO TEL AMBI TAU0'])
    wDir = float(hdr['ESO TEL AMBI WINDDIR'])
    wSpeed = float(hdr['ESO TEL AMBI WINDSP'])
    RHUM  = float(hdr['ESO TEL AMBI RHUM'])
    pressure = 0.5*(float(hdr['ESO TEL AMBI PRES START']) + float(hdr['ESO TEL AMBI PRES END']))
    fwhm_linobs = float(hdr['ESO TEL IA FWHMLINOBS'])
    # note: corrected active optics FWHM of the SH spots (arcsec)
    #(along the line of sight, PSF-based algorithm as described in
    #Martinez et al. 2011)

    return fwhm_linobs, tau0, wSpeed, wDir, RHUM, pressure

def get_temperature(hdr):
    """
        Get values from temperature sensors.
    """

    temp_col = [col for col in hdr.keys() if "TEMP" in col]
    temp_val = [hdr[tt] for tt in temp_col]
    idx = [type(tmp) == float for tmp in temp_val]
    NAME = [temp_col[n] for n in range(len(idx)) if idx[n]==True]
    TEMP = [temp_val[n] for n in range(len(idx)) if idx[n]==True]

    return TEMP, NAME

def get_moon_position(hdr):
    """
        Returns the RA/DEC of the moon during the observation.
    """
    return hdr['ESO TEL MOON RA'], hdr['ESO TEL MOON DEC']



