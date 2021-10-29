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
from telemetry.telemetryKASP import telemetryKASP
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
        df_merged = merge_dataframe(path_save, tocsv=True)
        return df_merged

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
             display=False, tol=1e-5, verbose=-1, fit=False,
             decimation=1):
    """
    Run the P3 PSFR by using the telemetry data in the .sav files and compare
    with the NIRC2 frame given in the path_fits file.
    """

    # find out the AO case
    # ------ Get the telemetry
    path_calib = path_p3 + "/aoSystem/data/KECK_CALIBRATION/"
    trs = telemetryKeck(path_sav, path_fits, path_calib, path_save=path_save,
                        nLayer=1, decimation=decimation)
    # ---- Process the telemetry to get the r0 and the noise
    sd = systemDiagnosis(trs)#, Dout=9, Din=2.65, noise=0)

    # ---- create the .ini file
    configFile(sd)

    # ---- instantiate the PSF model and get the PSF
    psfr = psfR(sd.trs)

    if fit:
        # ---- adjust the flux and the position
        x0 = [psfr.ao.atm.r0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        fixed = (True,)*3 + (False, False, False, True)
        res = psfFitting(psfr.trs.cam.image, psfr, x0, verbose=verbose, fixed=fixed,
                         ftol=tol, xtol=tol, gtol=tol)

        if display:
            displayResults(psfr, res, nBox=100, scale='arcsinh')
    else:
        class res_obj(object):
            def __init__(self, psfr):
                self.im_sky = psfr.trs.cam.image
                self.SR_sky = psfr.wfe[psfr.ao.cam.tag + " SR"]
                self.SR_fit = psfr.wfe["PSFR SR PEAK"]
                self.FWHMx_sky, self.FWHMy_sky = FourierUtils.getFWHM(self.im_sky,
                                                                      psfr.ao.cam.psInMas,
                                                                      nargout=2)
                self.FWHMx_fit, self.FWHMy_fit = FourierUtils.getFWHM(np.squeeze(psfr.psf),
                                                                      psfr.ao.cam.psInMas,
                                                                      nargout=2)
                self.mse = 0
                self.x = [psfr.ao.atm.r0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        res = res_obj(psfr)

    return psfr, res


def define_database_colnames():
    """
    Defines the name of the recorded data"
    """
    name = ["DATE", "ID", "AOMODE", "AIRMASS", "WVL [µm]", "BANDWIDTH [µm]",
            "PUP ANG [DEG]", "PUP MASK", "PUP RES [PX]", "CAM PS [MAS]", "CAM FOV [PX]",
            "CAM SR", "CAM FWHMX [MAS]", "CAM FWHMY [MAS]", "CAM FIT BKG", "CAM SATURATION",
            "SEEING DIMM [AS]", "SEEING MASS", "CN2 0KM", "CN2 0.5KM", "CN2 1KM", "CN2 2KM", "CN2 4KM",
            "CN2 8KM", "CN2 16KM", "HO GAIN", "HO LAT [ms]", "HO RATE [Hz]", "HO NPH",
            "HO WFE [NM]", "TT GAIN", "TT LAT [ms]", "TT RATE [Hz]", "TT NPH","TT WFE [NM]",
            "JITTERX [MAS]", "JITTERY [MAS]", "JITTERXY [MAS]",
            "TEL R0 [CM]", "TEL L0 [M]", "TEL TAU0 [MS]", "TEL V [M/S]", "TEL THETA0 [AS]",
            "HO NOISE [RAD]", "TT NOISE [RAD]", "WFE NCPA [NM]", "WFE FITTING [NM]",
            "WFE ALIASING [NM]", "WFE HO NOISE [NM]", "WFE TT NOISE [NM]",
            "WFE SERVO-LAG [NM]", "WFE JITTER [NM]", "WFE ANGULAR ANISO [NM]",
            "WFE FOCAL ANISO [NM]", "WFE TT ANISO [NM]", "WFE TOTAL [NM]",
            "PSF PIXEL [NM]", "PSFR SR PEAK", "PSFR SR OTF",
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
    CAMSR = res.SR_sky/1e2
    CAMFWHMX = res.FWHMx_sky
    CAMFWHMY = res.FWHMy_sky
    CAMFITBKG = res.x[-1]
    CAMSAT = psfr.trs.cam.is_saturated
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
    WFETT = psfr.wfe["TIP-TILT"]
    WFEANISO = psfr.wfe["ANGULAR ANISOPLANATISM"]
    WFEFOC = psfr.wfe["ANISOKINETISM"]
    WFETTANISO = psfr.wfe["FOCAL ANISOPLANATISM"]
    WFETOTAL = psfr.wfe["TOTAL WFE"]
    PSFPIX = psfr.wfe["PIXEL TF"]
    PSFSRPEAK = psfr.wfe["PSFR SR PEAK"]/1e2
    PSFSROTF = psfr.wfe["PSFR SR OTF"]/1e2
    PSFWHMX = res.FWHMx_fit
    PSFWHMY = res.FWHMy_fit
    MSE = res.mse
    # PACKING THE DATA
    data = [DATE, ID, AOMODE, AIRMASS, WVL, BW, PUPANG, PUPMASK, PUPRES, CAMPS,
            CAMFOV, CAMSR, CAMFWHMX, CAMFWHMY, CAMFITBKG, CAMSAT, SEEINGDIMM, SEEINGMASS,
            CN20, CN205, CN21, CN22, CN24, CN28, CN216, HOGAIN, HOLAT, HORATE,
            HONPH, HOWFE, TTGAIN, TTLAT, TTRATE, TTNPH, TTWFE, JX, JY, JXY, TELR0, TELL0,
            TELTAU0, TELV, TELTHETA0, HONOISE, TTNOISE, WFENCPA, WFEFIT,
            WFEALIAS, WFEHONOISE, WFETTNOISE, WFELAG, WFETT, WFEANISO, WFEFOC,
            WFETTANISO, WFETOTAL, PSFPIX, PSFSRPEAK, PSFSROTF,
            PSFWHMX, PSFWHMY, MSE]

    return data
#%% PSFR ON SIMULATED DATA
def psfr_kasp_simulations(path_save, patterns=None, true_r0=True,
                          compute_dphi_jitter=False):
    """
    Run the psfr algorithm on all .mat file whose name contains the strings pattern
    and returns PSF metrics.
    """
    # grabbing and filtering the files
    if not os.path.isdir(path_save):
        raise ValueError("the folder does not exist")

    list_files = os.listdir(path_save)
    if patterns is not None and type(patterns)==list:
        for pattern in patterns:
            list_files = [file for file in list_files if pattern in file]

    list_files.sort()

    # run psfr
    def run_psfr_from_kasp(path_mat, path_save, true_r0=True,
                           compute_dphi_jitter=False):
        # grab and transform the data
        trs = telemetryKASP(path_mat, path_save=path_save)
        # data processing : r0/noise estimation
        r0_true = trs.atm.r0
        L0_true = trs.atm.L0
        sd = systemDiagnosis(trs, noiseMethod='nonoise', noise=0)
        r0_tel = trs.atm.r0_tel
        L0_tel = trs.atm.L0_tel
        if true_r0:
            trs.atm.r0_tel = r0_true
            trs.atm.r0 = r0_true
            trs.atm.L0_tel = L0_true
            trs.atm.L0 = L0_true
            trs.atm.seeing = 3600*180/np.pi*0.978*trs.atm.wvl/r0_true
        # write the .ini file
        configFile(sd)
        # get the psf
        psfr = psfR(trs, compute_dphi_jitter=compute_dphi_jitter)
        trs.atm.r0_tel = r0_tel
        trs.atm.L0_tel = L0_tel
        return psfr

    metrics=[]
    for file in tqdm.tqdm(list_files):
        # run psfr
        psfr = run_psfr_from_kasp(path_save+file, path_save, true_r0=true_r0,
                                  compute_dphi_jitter=compute_dphi_jitter)
        # append metrics
        metrics.append([psfr.trs.cam.strehl, psfr.SR[0]/1e2, psfr.wfe["PSFR SR PEAK"]/1e2,
                       psfr.trs.atm.r0_tel, psfr.trs.atm.L0_tel, psfr.trs.atm.tau0_tel, psfr.trs.atm.v0_tel,
                       psfr.trs.atm.r0, psfr.trs.atm.L0, psfr.ao.atm.tau0*1e3, psfr.ao.atm.meanWind])

    # get the dataframe
    df = pd.DataFrame(metrics)
    df = df.rename(columns={0:'SR CAMERA',1:'SR PSFR OTF', 2:'SR PSFR PEAK',
                       3:'R0', 4:'L0', 5:'TAU0', 6:'V0',
                       7:'TRUE R0', 8:'TRUE L0', 9:'TRUE TAU0', 10:'TRUE V0'})

    return df

def plot_kasp_psfr_results(df):
    """
    Plot results from PSFR on KASP data
    """
    plt.close('all')

    # SR VERSUS TRUE r0
    plt.figure()
    plt.scatter(df["TRUE R0"], df["SR CAMERA"], label="CAMERA")
    plt.scatter(df["TRUE R0"], df["SR PSFR OTF"], label="PSFR")
    plt.xlabel('Simulation r0 [m]')
    plt.ylabel('Strehl-ratio')
    plt.legend()
    plt.grid('on')


    # telemetry 0 VERSUS TRUE r0
    plt.figure()
    plt.scatter(df["TRUE R0"], df["R0"])
    plt.xlabel('Simulation r0 [m]')
    plt.ylabel('Estimated r0 [m]')
    r0_min = df[["TRUE R0", "R0"]].min().min()
    r0_max = df[["TRUE R0", "R0"]].max().max()
    plt.plot([r0_min, r0_max], [r0_min, r0_max], 'k--')
    plt.grid('on')
    ax = plt.gca()
    ax.set_aspect('equal')

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
                 line_xy=False, line_10=False, aspect='equal'):
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
            xx = np.linspace(plt.xlim()[0], plt.xlim()[1], num=50)
            plt.plot(xx, xx, 'k--', linewidth=1.2)
            if line_10:
                plt.plot(xx, xx*1.1, 'k-.', linewidth=1)
                plt.plot(xx, xx*0.9, 'k-.', linewidth=1)

    plt.xlabel(val_x)
    if n_sort>1:
        plt.legend()
    if grid_on:
        plt.grid('on')
    if aspect and line_xy:
        ax = plt.gca()
        ax.set_aspect('equal')
