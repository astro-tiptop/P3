#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:37:55 2021

@author: omartin
"""
import os
import shutil
import numpy as np
from astropy.io import fits
from query_eso_archive import query_simbad
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from sphere.transmission import irdis_nd
from psfao21.psfao21 import psfao21
from psfFitting.psfFitting import psfFitting
import aoSystem.FourierUtils as FourierUtils
import tqdm
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

#%% IMAGE FITTING
def reading_ini_file(path_ini, path_to_modes=None):
    """
    Return the parser to read and modify the .ini file.
    We can change the modal basis with path_to_modes. By default, the modes will
    be defined from the path given in the .ini file
    """
    from configparser import ConfigParser

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
    F = FourierUtils.normalizeImage(im, normType=normType)[0].sum()
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

def fitting_image(im, psfao, r0=None, SR=None, fit_stat=True, weights=None, normType=1,
                  tol=1e-8, max_nfev=100, sig=5, verbose=-1, fix_C=True):
    """
    Fitting a 2D image of a star by considering the instance psfao as a model
    for the PSF. If fit_lwe is True, the fit jointly estimates the atmospheric
    parameters and the static aberrations modes.
    Returns the res dictionnary from the psffitting function.
    """

    # STEP 1 :  FIT THE ATMOSPHERIC PARAMETERS ONLY
    x0 = define_initial_guess(im, psfao.wvl[0], r0=r0, SR=SR,
                              normType=normType, n_modes=0)
    fixed = define_fixed_parameters(im.shape[0], psfao.ao.tel.D, psfao.wvl[0],
                                    psfao.ao.dms.nActu1D, psfao.ao.cam.psInMas,
                                    fix_C=fix_C, fix_shift=False, n_modes=0)
    # fit
    res = psfFitting(im, psfao, x0, fixed=fixed, weights=weights, normType=normType,
                     verbose=verbose, ftol=tol, gtol=tol, xtol=tol, max_nfev=max_nfev)

    # STEP 2: JOINT FIT OF THE ATMOSPHERIC AND INSTRUMENTAL PARAMETERS
    if fit_stat:
        # defining the new initial guess
        #x0 = list(res.x[:6]) + [min(x0[6], res.x[6])] + list(res.x[7:11]) + [0,]*3 + [0,]*psfao.ao.tel.nModes
        x0 = list(res.x[:11]) + [0,]*3 + [0,]*psfao.ao.tel.nModes
        print(x0)
        # narrowing the bounds of the atmospheric parameters
        psfao.bounds = psfao.update_bounds(res.x, res.xerr, sig=5)
        # disable the fit of astrometry (conflict with  with tip-tilt modes)
        fixed = define_fixed_parameters(im.shape[0], psfao.ao.tel.D, psfao.wvl[0],
                                        psfao.ao.dms.nActu1D, psfao.ao.cam.psInMas,
                                        fix_C=fix_C, fix_shift=True,
                                        n_modes=psfao.ao.tel.nModes)

        # storing the astrometry values from the previous fit
        dx_fit = res.x[11]*psfao.ao.cam.psInMas
        dy_fit = res.x[12]*psfao.ao.cam.psInMas
        # fit them all
        res = psfFitting(im, psfao, x0, fixed=fixed, weights=weights, normType=normType,
                         verbose=verbose, ftol=tol, gtol=tol, xtol=tol, max_nfev=max_nfev)
        res.x[11] = dx_fit
        res.x[12] = dy_fit


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
    names=['YYMMDDHHMMSS', 'AIRMASS', 'MEAN WAVELENGTH [µm]', 'BANDWIDTH [µm]',
           'V MAG', 'R MAG', 'G MAG', 'J MAG', 'H MAG', 'K MAG',
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
#%% READING THE HEADER

def get_wavelength(path_data):
    """
        Read the file IRD_SCIENCE_LAMBDA_INFO to extract the filter information
    """
    wvl = 1e-6*fits.getdata(path_data + '/ird_convert_recenter_dc5-IRD_SCIENCE_LAMBDA_INFO-lam.fits')
    bw = wvl.max() - wvl.min()
    wvl = wvl.mean()

    return wvl, bw

def read_sparta_data(path_data, which='last'):
    """
        Read the file SPH_SPARTA_PSFDATA-psf_sparta_data to get the atmospheric parameters.
        if the files exists, this is a [3, npsf+1,4] array
        ○    dim 1 : r0, wind, strehl ratio, seeing
        ○    dim 2 : 0: npsf-1 : for each PSF file.   Last line = stat combining all frames of allfiles
        ○    dim 3 : min, max, avg
        reminder:  The SPARTA Strehl ratio is provided at 1.6 micron and the turbulence parameters (r0, seeing) at 500nm.
    """

    path_sparta = path_data + '/ird_convert_recenter_dc5-SPH_SPARTA_PSFDATA-psf_sparta_data.fits'
    if os.path.isfile(path_sparta):
        sparta = fits.getdata(path_sparta)
        # number of acquisitions during the observation
        nPSF          = sparta.shape[1]-1
        # note : nPSF must == nFiles/2 with raw images or im.shape[0]//2 with processed images

        r0 = np.array(sparta[0, :, :])
        vspeed = np.array(sparta[1, :, :])
        SR = np.array(sparta[2, :, :])
        seeing = np.array(sparta[3, :, :])

        if which == 'last':
            r0 = r0[nPSF, :]
            vspeed = vspeed[nPSF, :]
            SR = SR[nPSF, :]
            seeing = seeing[nPSF, :]

        return r0, vspeed, SR, seeing

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

    VMAG = RMAG = GMAG = JMAG = HMAG = KMAG = -1

    OB_NAME, RA, DEC = get_star_coordinates(hdr)

    # query simbad
    DICT_SIMBAD= query_simbad(Time(hdr['DATE-OBS']),
                              SkyCoord(RA*u.degree,DEC*u.degree),
                              name=OB_NAME)

    if DICT_SIMBAD is None:
        return VMAG, RMAG, GMAG, JMAG, HMAG, KMAG

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

    return VMAG, RMAG, GMAG, JMAG, HMAG, KMAG

def get_detector_config(hdr):
    """
        Read the header to get the detector configuration.
    """
    if 'PIXSCAL' in hdr:
        psInMas = float(hdr['PIXSCAL'])
    else:
        psInMas = -1
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
    return DATE_OBS[0:4] + DATE_OBS[5:7] + DATE_OBS[8:10]\
          +DATE_OBS[11:13] + DATE_OBS[14:16] + DATE_OBS[17:19]

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

    # temperature sensors
    TEMP = np.zeros(60)
    TEMP[0] = float(hdr['ESO TEL AMBI TEMP'])
    TEMP[1] = float(hdr['ESO TEL TH M1 TEMP'])
    for t in range(1,6):
        TEMP[t+2]   = float(hdr['ESO INS1 TEMP10'+str(t)+' VAL'])
    INS = 0
    for iii in range(4):
        if 'ESO INS'+str(iii+1)+' TEMP401 ID' in hdr:
            INS = int(iii+1)
    if INS>0:
        for t in range(3,53):
            if t<12 or t>51:
                dd = -273 # from Kelvin to degrees Celsius
            else:
                dd = 0
            TEMP[t+7]   = float(hdr['ESO INS'+str(INS)+' TEMP4'+str(t+3).zfill(2)+' VAL']) + dd

    return TEMP

def get_moon_position(hdr):
    """
        Returns the RA/DEC of the moon during the observation.
    """
    return hdr['ESO TEL MOON RA'], hdr['ESO TEL MOON DEC']



