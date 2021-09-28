#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:13:40 2021

@author: omartin
"""
#%% importing librairies
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import aoSystem.FourierUtils as FourierUtils
from psfr.psfR import psfR
from psfFitting.psfFitting import psfFitting
from psfFitting.psfFitting import displayResults
from telemetry.telemetryKeck import telemetryKeck
from telemetry.systemDiagnosis import systemDiagnosis
from telemetry.configFile import configFile
import tqdm
import warnings
warnings.simplefilter("ignore")

#%% PSFR
def psfr_on_multiple_databases(path_save,
                               year=["20130203", "20130801", "20130914"],
                               merge_df=True, fit=False):
    """
    Run the PSFR over multiple Keck telemetry databases"
    """
    # loop on data bases
    for date in year:
        psfr_on_database(path_save, year=date, fit=fit)

    if merge_df:
        merge_dataframe(path_save, tocsv=True)

def psfr_on_database(path_save, year="20130801", fit=False, display=False, tol=1e-5):
    """
    Run PSFR over a large data base. Must be copied/pasted to another file
    not within the P3 repo before changing the paths
    """

    # defining paths
    path_p3 = "/home/omartin/Projects/P3"
    path_data = "/run/media/omartin/OlivierMartinHDD/DATA"
    path_fits = path_data+"/APPLY/PSF_LIBRARY/KECK_SCAO/"
    path_sav = path_data + "/KECK_DATA/"+year+"/TRS/"

    # grabbing the list of data
    list_sav = os.listdir(path_sav)
    n_sav = len(list_sav)
    list_sav = [list_sav[n] for n in range(n_sav) if ".sav" in list_sav[n]]
    list_sav.sort()
    n_sav = len(list_sav)

    list_fits = os.listdir(path_fits)
    n_fits = len(list_fits)
    list_fits = [list_fits[n] for n in range(n_fits) if ".fits" in list_fits[n]]
    n_fits = len(list_fits)
    list_fits = [list_fits[n] for n in range(n_fits) if year in list_fits[n]]
    list_fits.sort()
    n_fits = len(list_fits)

    # match catalog
    id_file_fits = [list_fits[n].split('_')[-1].split('.')[0] for n in range(n_fits)]
    id_file_sav = [list_sav[n].split('_')[0] for n in range(n_sav)]
    id_match = list(set(id_file_fits) & set(id_file_sav))
    id_match.sort()

    # creating the DataFrame
    names = define_database_colnames()
    df_name = "keck_psfr_" + year
    data = []
    # running PSFR
    for id_file in tqdm.tqdm(id_match):
        # get files
        file_sav = path_sav + list_sav[id_file_sav.index(id_file)]
        file_fits = path_fits + list_fits[id_file_fits.index(id_file)]
        # run psfr
        psfr, res = run_keck_psfr(file_sav, file_fits, path_p3, path_save=path_save,
                                  display=display, tol=tol, verbose=-1, fit=fit)
        # grabbing results
        data.append(unpack_results(psfr, res))

    # saving the data frame
    df_psfr = pd.DataFrame(data, columns=names)
    df_psfr.set_index('ID')
    df_psfr.to_csv(path_save + df_name + ".csv", index=False)

def run_keck_psfr(path_sav, path_fits, path_p3, path_save="./",
             display=False, tol=1e-5, verbose=-1, fit=False):
    """
    Run the P3 PSFR by using the telemetry data in the .sav files and compare
    with the NIRC2 frame given in the path_fits file.
    """

    # find out the AO case
    # ------ Get the telemetry
    path_calib = path_p3 + "/aoSystem/data/KECK_CALIBRATION/"
    trs = telemetryKeck(path_sav, path_fits, path_calib, path_save=path_save,
                        nLayer=1)
    # ---- Process the telemetry to get the r0 and the noise
    sd  = systemDiagnosis(trs)

    # ---- create the .ini file
    configFile(sd)

    # ---- instantiate the PSF mode
    psfr = psfR(sd.trs)

    # ---- get the PSF
    r0 = psfr.ao.atm.r0
    #if psfr.trs.atm.seeing_dimm!=-1:
    #    r0 = (0.978*206264.8*500e-9)/psfr.trs.atm.seeing_dimm * psfr.trs.tel.airmass**(-3/5)
    #    r0 *= (psfr.ao.atm.wvl/500e-9)**1.2
    x0 = [r0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    psf = np.squeeze(psfr(x0))
    psfr.wfe['PSF SR'] = psfr.SR[0]

    if fit:
        # ---- adjust the flux and the position
        fixed = (True,)*3 + (False, False, False, True)
        res = psfFitting(psfr.trs.cam.image, psfr, x0, verbose=verbose, fixed=fixed,
                         ftol=tol, xtol=tol, gtol=tol)
        psfr.wfe['FIT PSF SR'] = psfr.SR[0]

        if display:
            displayResults(psfr, res, nBox=100, scale='arcsinh')
    else:
        class res_obj(object):
            def __init__(self, psfr, psf):
                self.im_sky = psfr.trs.cam.image
                self.SR_sky = FourierUtils.getStrehl(self.im_sky,
                                                     psfr.ao.tel.pupil,
                                                     psfr.freq.sampRef)
                self.SR_fit = psfr.SR[0]/1e2
                self.FWHMx_sky, self.FWHMy_sky = FourierUtils.getFWHM(self.im_sky,
                                                                      psfr.ao.cam.psInMas,
                                                                      nargout=2)
                self.FWHMx_fit, self.FWHMy_fit = FourierUtils.getFWHM(psf,
                                                                      psfr.ao.cam.psInMas,
                                                                      nargout=2)
                self.mse = 0
                self.x = x0
        res = res_obj(psfr, psf)

    return psfr, res


def define_database_colnames():
    """
    Defines the name of the recorded data"
    """
    name = ["DATE", "ID", "AOMODE", "AIRMASS", "WVL [µm]", "BANDWIDTH [µm]",
            "PUP ANG [DEG]", "PUP MASK", "PUP RES [PX]", "CAM PS [MAS]", "CAM FOV [PX]",
            "CAM SR", "CAM FWHMX [MAS]", "CAM FWHMY [MAS]", "CAM FIT BKG",
            "SEEING DIMM [AS]", "SEEING MASS", "CN2 0KM", "CN2 0.5KM", "CN2 1KM", "CN2 2KM", "CN2 4KM",
            "CN2 8KM", "CN2 16KM", "HO GAIN", "HO LAT [ms]", "HO RATE [Hz]", "HO NPH",
            "HO WFE [NM]", "TT GAIN", "TT LAT [ms]", "TT RATE [Hz]", "TT NPH","TT WFE [NM]",
            "JITTERX [MAS]", "JITTERY [MAS]", "JITTERXY [MAS]",
            "TEL R0 [CM]", "TEL L0 [M]", "TEL TAU0 [MS]", "TEL V [M/S]", "TEL THETA0 [AS]",
            "HO NOISE [RAD]", "TT NOISE [RAD]", "WFE NCPA [NM]", "WFE FITTING [NM]",
            "WFE ALIASING [NM]", "WFE HO NOISE [NM]", "WFE TT NOISE [NM]",
            "WFE SERVO-LAG [NM]", "WFE JITTER [NM]", "WFE ANGULAR ANISO [NM]",
            "WFE FOCAL ANISO [NM]", "WFE TT ANISO [NM]", "WFE TOTAL [NM]",
            "PSF PIXEL [NM]", "PSFR SR", "FIT PSFR SR",
            "PSFR FWHMX", "PSFR FWHMY",  "PSFR MSE"]

    return name

def unpack_results(psfr, res):
    """
    Return a list of parameters from the psfR instance and the output of the
    psffitting function.
    """

    data = []
    # OBSERVATIONS
    DATE = psfr.trs.path_img.split("/")[-1].split("_")[0]
    ID = psfr.trs.path_img.split("/")[-1].split("_")[1].split('.')[0].upper()
    AOMODE = psfr.ao.aoMode.upper()
    AIRMASS = psfr.ao.tel.airmass
    WVL = psfr.wvl[0]*1e6
    BW = psfr.ao.cam.bandwidth*1e6
    PUPANG = psfr.ao.tel.pupilAngle
    PUPMASK = psfr.trs.tel.pupilMaskName
    PUPRES = psfr.ao.tel.resolution
    CAMPS = psfr.ao.cam.psInMas
    CAMFOV = psfr.trs.cam.image.shape[0]
    CAMSR = res.SR_sky
    CAMFWHMX = res.FWHMx_sky
    CAMFWHMY = res.FWHMy_sky
    CAMFITBKG = res.x[-1]
    SEEINGDIMM = float(psfr.trs.atm.seeing_dimm)
    SEEINGMASS = float(psfr.trs.atm.seeing_mass)
    CN20, CN205, CN21, CN22, CN24, CN28, CN216 = psfr.trs.atm.Cn2
    HOGAIN =psfr.ao.rtc.holoop["gain"]
    HOLAT = psfr.ao.rtc.holoop["delay"]
    HORATE = psfr.ao.rtc.holoop["rate"]
    HONPH = psfr.trs.wfs.nph
    HOWFE = psfr.ao.rtc.holoop["wfe"]
    TTGAIN = psfr.ao.rtc.ttloop["gain"]
    TTLAT = psfr.ao.rtc.ttloop["delay"]
    TTRATE = psfr.ao.rtc.ttloop["rate"]
    TTNPH = psfr.trs.wfs.nph
    TTWFE = psfr.ao.rtc.holoop["wfe"]
    JX, JY, JXY = psfr.ao.cam.spotFWHM[0]
    TELR0 = psfr.trs.atm.r0_tel*1e2
    TELL0 = psfr.trs.atm.L0_tel
    TELTAU0 = psfr.trs.atm.tau0_tel*1e3
    TELV = psfr.trs.atm.v0_tel
    psfr.ao.atm.wvl = 500e-9
    TELTHETA0 = psfr.ao.atm.theta0
    HONOISE = psfr.trs.wfs.noiseVar[0]
    TTNOISE = psfr.trs.tipTilt.noiseVar[0]
    WFENCPA = psfr.wfe["NCPA"]
    WFEFIT = psfr.wfe["FITTING"]
    WFEALIAS = psfr.wfe["ALIASING"]
    WFEHONOISE = psfr.wfe["HO NOISE"]
    WFETTNOISE = psfr.wfe["TT NOISE"]
    WFELAG = psfr.wfe["SERVO-LAG"]
    WFETT = psfr.wfe["TIP-TILT-WFE"]
    WFEANISO = psfr.wfe["ANGULAR ANISOPLANATISM"]
    WFEFOC = psfr.wfe["ANISOKINETISM"]
    WFETTANISO = psfr.wfe["FOCAL ANISOPLANATISM"]
    WFETOTAL = psfr.wfe["TOTAL WFE"]
    PSFPIX = psfr.wfe["PIXEL TF"]
    PSFSR = psfr.wfe["PSF SR"]/1e2
    FITPSFSRBKG = res.SR_fit
    PSFWHMX = res.FWHMx_fit
    PSFWHMY = res.FWHMy_fit
    MSE = res.mse
    # PACKING THE DATA
    data = [DATE, ID, AOMODE, AIRMASS, WVL, BW, PUPANG, PUPMASK, PUPRES, CAMPS,
            CAMFOV, CAMSR, CAMFWHMX, CAMFWHMY, CAMFITBKG, SEEINGDIMM, SEEINGMASS,
            CN20, CN205, CN21, CN22, CN24, CN28, CN216, HOGAIN, HOLAT, HORATE,
            HONPH, HOWFE, TTGAIN, TTLAT, TTRATE, TTNPH, TTWFE, JX, JY, JXY, TELR0, TELL0,
            TELTAU0, TELV, TELTHETA0, HONOISE, TTNOISE, WFENCPA, WFEFIT,
            WFEALIAS, WFEHONOISE, WFETTNOISE, WFELAG, WFETT, WFEANISO, WFEFOC,
            WFETTANISO, WFETOTAL, PSFPIX, PSFSR, FITPSFSRBKG,
            PSFWHMX, PSFWHMY, MSE]

    return data
#%% PROCESSING
def merge_dataframe(save_folder, tocsv=False):
    """
    Merge .csv file inside the save_folder
    """
    # grab .csv data that were not merged already
    list_csv = os.listdir(save_folder)
    n_file = len(list_csv)
    list_csv = [list_csv[n] for n in range(n_file) if ".csv" in list_csv[n]]
    n_file = len(list_csv)
    list_csv = [list_csv[n] for n in range(n_file) if "merged" not in list_csv[n]]

    # creating data frames
    df_merged = pd.DataFrame()
    for file in list_csv:
        path_file = save_folder + file
        df_tmp = pd.read_csv(path_file)
        df_merged = pd.concat([df_merged, df_tmp])

    # sort
    df_merged = df_merged.sort_values("DATE")

    # save
    if tocsv:
        pref = list_csv[0].split('_')[0]
        df_merged.to_csv(save_folder + pref + "_psfr_merged.csv")

    return df_merged

def plot_metrics(df, val_x, val_y=None, sort_by=None, n_bins=10, grid_on=True,
                 line_xy=False, aspect='equal'):
    """
    Plot values of val_y versus val_x taken from the dataframe df.
    If val_y is None, plot the histograms
    """

    # checking inputs
    if val_x not in df:
        raise ValueError(val_x + " is not a column of the data frame")
    if val_y is not None and val_y not in df:
        raise ValueError(val_y + " is not a column of the data frame")
    if sort_by is not None and sort_by not in df:
        raise ValueError(sort_by + " is not a column of the data frame")
    if val_y is None and n_bins is None or n_bins < 0:
        raise ValueError("the number of bins must be a postive integer")

    # finding unique values for the reference column
    sort_unique = None
    n_sort = 1
    alpha = 1 # transparency
    if sort_by is not None:
        sort_unique = pd.unique(df[sort_by])
        n_sort = len(sort_unique)
        alpha = 0.5

    # histogram
    if val_y is None: # histogram
        plt.figure()
        for k in range(n_sort):
            if n_sort == 1:
                n_val = df[val_x].count()
                weights = np.ones(n_val)/n_val
                plt.hist(df[val_x], bins=n_bins, alpha=alpha, weights=weights)
            else:
                df_tmp = df[df[sort_by]==sort_unique[k]]
                n_val = df_tmp[val_x].count()
                weights = np.ones(n_val)/n_val
                plt.hist(df_tmp[val_x], bins=n_bins, alpha=alpha,
                         label=sort_unique[k], weights=weights)
        plt.ylabel("Probability")


    # versus plot
    else:
        plt.figure()
        for k in range(n_sort):
            if n_sort == 1:
                plt.scatter(df[val_x], df[val_y])
            else:
                df_tmp = df[df[sort_by]==sort_unique[k]]
                plt.scatter(df_tmp[val_x], df_tmp[val_y], label=sort_unique[k])
        plt.ylabel(val_y)
        if line_xy:
            plt.plot(plt.xlim(), plt.ylim(), 'k--')

    plt.xlabel(val_x)
    if n_sort>1:
        plt.legend()
    if grid_on:
        plt.grid('on')
    if aspect and line_xy:
        ax = plt.gca()
        ax.set_aspect('equal')
