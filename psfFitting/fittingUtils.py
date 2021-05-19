#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 3 13:30:49 2021

@author: mdumont
"""

# IMPORTING LIBRAIRIES
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math as math
from random import random
from configparser import ConfigParser
import os

from astropy.io import fits
from astropy.table import QTable

from psfao21.psfao21 import psfao21
from psfFitting.psfFitting import psfFitting
import aoSystem.FourierUtils as FourierUtils
import telemetry.keckUtils as keckUtils

mpl.rcParams['font.size'] = 12
usetex = False
plt.rcParams.update({
    "text.usetex": usetex,
    "font.family": "serif",
    "font.serif": ["Palatino"],
}) 


def choose_stat_modes(mode, path_ini):
    if mode == 'Piston':
        pathMode = 'aoSystem/data/KECK_CALIBRATION/KECK_STAT/keck_piston_modes_200px.fits'
    elif mode == 'Zernike':
        pathMode = 'aoSystem/data/KECK_CALIBRATION/KECK_STAT/keck_piston_modes_200px_all_lr.fits'
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(path_ini)
    parser.set('telescope', 'path_statModes', '\'' + pathMode + '\'')
    with open(path_ini, 'w') as configfile:
        parser.write(configfile)

def get_split_date(hdr): 
    date = hdr['DATE-OBS']
    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    return year, month, day

def get_PSF_number(psfName):
    return float(psfName[10:14])
    
def adjust_init_file(path_file, path_ini):
    path_filter = 'aoSystem/data/KECK_CALIBRATION/NIRC2_FILTERS/'
    im = fits.getdata(path_file)
    hdr = fits.getheader(path_file)

    PA = keckUtils.getPA(hdr)
    NCPAname = keckUtils.getNCPAstr(hdr)
    wvl,bw,tr,disp = keckUtils.samplingFilterProfile(path_filter,hdr)

    nPix = im.shape[0]

    wvl = float(wvl[0])
    
    year, month, day = get_split_date(hdr)

    pmsName = keckUtils.getPupilMask(hdr)
    if pmsName == 'OPEN':
        pathPup = 'aoSystem/data/KECK_CALIBRATION/NIRC2_MASK/keck_pupil_open_240px.fits'
    elif  pmsName == 'LARGEHEX':
        pathPup = 'aoSystem/data/KECK_CALIBRATION/NIRC2_MASK/keck_pupil_largeHex_272px.fits'

    parser = ConfigParser()
    parser.optionxform = str
    parser.read(path_ini)
    parser.set('telescope','path_pupil','\'' + pathPup + '\'')
    parser.set('telescope','Path_static','\'' + NCPAname + '\'')
    parser.set('telescope', 'pupilAngle', str(PA))
    
    parser.set('sources_HO','Wavelength',str(wvl))
    parser.set('sensor_science','FiedOfView',str(nPix))
    
    parser.set('sensor_HO','SpectralBandwidth',str(bw))
    parser.set('sensor_HO','Dispersion',str([disp[0], disp[1]]))
    parser.set('sensor_HO','Transmittance',str(tr))
    
    with open(path_ini, 'w') as configfile:
        parser.write(configfile)


def split_fitting(im, psfao, x0, fixed, weights) : 
    # fitting - 7 PSD parameters + no static aberrations

    res_psfao21  = psfFitting(im,psfao,x0+list(np.zeros((psfao.ao.tel.nModes))),verbose=2,fixed=fixed,ftol=1e-5,gtol=1e-5,xtol=1e-5)
    
    # fitting - no PSD parameters + static aberrations
    x0    = list(res_psfao21.x[0:7]) + [0,0,0,1.0,0,0,0] + list(np.zeros((psfao.ao.tel.nModes))) 
    fixed = (False, False, False, False, False, False, False) +(True,)*3 + (False,False,False,False) + (False,)*36
    # redefining bounds
    bounds= psfao.updateBounds(res_psfao21.x,res_psfao21.xerr,sig=5)
    res_psfao21_split  = psfFitting(im,psfao,x0,verbose=2,\
                        fixed=fixed,bounds=bounds)
    
    return res_psfao21_split

def joint_fit(im, psfao, x0, fixed, weights):
    x0    = [0.7,4e-2,0.5,1e-2,1,0,1.8,0,0,0,1.0,0,0,0] + list(np.zeros((psfao.ao.tel.nModes)))
    fixed = (False, False, False, False, False, False, False) +(True,)*3 + (False,False,False,False) + (False,)*36
    res_psfao21_joint  = psfFitting(im,psfao,x0,verbose=2,fixed=fixed,ftol=1e-5,gtol=1e-5,xtol=1e-5)
    return res_psfao21_joint

    
def fit_all_PSFs(pathData, x0, fixed, weights, statMode, path_ini, pathFilter, outputName, NbPSF = None):
    
    files = os.listdir(pathData)
    names  = ['psfNumber', 'Day', 'Month', 'Year',  'Wvl', 'SRfit', 'SRsky', 'FWHMxsky', 'FWHMysky', 'FWHMxfit', 'FWHMyfit']
    TEMP   = ['Res_x'+str(s) for s in range(50)]
    TEMP  += ['Res_xerr'+str(s) for s in range(50)]
    names += TEMP
    Tab    = QTable(names=names)
    choose_stat_modes('Piston', path_ini)
    if NbPSF is None:
        NbPSF = len(files)

    for i in range (NbPSF):
        if NbPSF is not None:
            choosenFile = math.floor( random()*len(files) )
        else:
            choosenFile = i

        f        = files[choosenFile]
        pathFile = pathData + f
        im       = fits.getdata(pathFile)

        adjust_init_file(pathFile, path_ini)
        choose_stat_modes(statMode, path_ini)
        psfao = psfao21(path_ini)
        FWHM = FourierUtils.getFWHM(im,psfao.ao.cam.psInMas,nargout=1)
        SR = FourierUtils.getStrehl(im,psfao.ao.tel.pupil, psfao.freq.sampRef)
        
        if SR > 0.35 and FWHM > 35 : 

            res = split_fitting(im, psfao, x0, fixed, weights)

            hdr = fits.getheader(pathFile)
            year, month, day = get_split_date(hdr)
            wvl,bw,tr,disp = keckUtils.samplingFilterProfile(pathFilter,hdr)
            psfNumber = get_PSF_number(f)

            data  = (psfNumber ,day, month, year, wvl, res.SR_fit, res.SR_sky, res.FWHMx_sky, res.FWHMy_sky, res.FWHMx_fit, res.FWHMy_fit)
            data += tuple(res.x)
            data += tuple(res.xerr)
            Tab.add_row(data)
    Tab.write(outputName, overwrite = True)


def fits_2_dict(pathFile):
    data = fits.getdata(pathFile)
    nbPSF = len(data)
    names = ['psfNumber', 'Day', 'Month', 'Year',  'Wvl', 'SRfit', 'SRsky', 'FWHMxsky', 'FWHMysky', 'FWHMxfit', 'FWHMyfit']

    names += ['r0', 'C', 'A', 'ax', 'p', 'Theta', 'Beta']
    names += ['jitterx', 'jittery', 'jitterz']
    names += ['flux', 'posx', 'posy', 'bckgrnd']
    names += ['StatMode_x'+str(s) for s in range(36)]

    names += ['r0err', 'Cerr', 'Aerr', 'axerr', 'perr', 'Thetaerr', 'Betaerr']
    names += ['jitterxerr', 'jitteryerr', 'jitterzerr']
    names += ['fluxerr', 'posxerr', 'posyerr', 'bckgrnderr']
    names += ['StatMode_xerr'+str(s) for s in range(36)]
    
    res = dict()

    for num,featureName in enumerate(names):
        res[featureName] = [data[i][num] for i in range(nbPSF)]
    return res

def display_results(path_fits):
    dicti = fits_2_dict(path_fits)
    data  = fits.getdata(path_fits)
    nbPSF = len(data)
    nSeg  = 36
    plt.figure()
    plt.plot(dicti['SRsky'], dicti['SRfit'], '*k')
    plt.plot([ np.min(dicti['SRfit']),np.max(dicti['SRsky']) ], [ np.min(dicti['SRfit']),np.max(dicti['SRsky']) ], 'r')
    plt.xlabel('SRsky')
    plt.ylabel('SRfit')
    plt.title('Strehl')

    plt.figure()
    plt.plot(dicti['StatMode_x0'], dicti['StatMode_xerr0'], '*b')
    plt.xlabel('x0')
    plt.ylabel('xerr0')
    plt.title('Mode TIP-TILP xerr0(x0)')

    meanXList = []
    meanXerrList = []
    for i in range (nSeg):
        currentName = 'StatMode_x' + str(i)
        currentMean =  np.mean(dicti[currentName])
        meanXList.append(currentMean)

        currentName = 'StatMode_xerr' + str(i)
        currentMean =  np.mean(dicti[currentName])
        meanXerrList.append(currentMean)

    plt.figure()
    plt.plot(meanXList, meanXerrList, '*b')
    plt.xlabel('mean x0')
    plt.ylabel('mean xerr0')
    plt.title('Mean')

    paramsList = ['r0', 'C', 'A', 'ax', 'p', 'Theta', 'Beta']
    for num, paramName in enumerate(paramsList):
        ratio = np.array(dicti[paramName]) / np.array(dicti[paramName +'err'])
        plt.figure()
        plt.plot(ratio, 'b')
        plt.title(paramName + ' ratio')
        plt.xlabel('PSFs')
        plt.ylabel('ratio')

    #r0 Histogram
    plt.figure()
    plt.hist(dicti['r0'])
    plt.title('Histogram r0')
    plt.xlabel('r0')
    plt.ylabel('Nb PSFs')

    meanXList = []
    meanXerrList = []
    Tab = QTable(names = ['StatMode_x' + str(s) for s in range(36)])
    Tableauresult = []
    for i in range (nbPSF):
        data = []
        for j in range(nSeg):
            currentName = 'StatMode_x' + str(j)
            currentStatMode =  dicti[currentName][i]
            data.append(currentStatMode)
        Tab.add_row(data)
        Tableauresult.append(data)

    meanList = []
    stdList  = []
    for j in range (nSeg):
        statModeList = []
        for i in range (nbPSF):
            statModeList.append( np.abs(Tableauresult[i][j]) )
        currentMean = np.median(statModeList)
        meanList.append(currentMean)
        currentStd = np.std(statModeList)/np.sqrt(nbPSF)
        stdList.append(currentStd)

    #median values with error bars and weighted mean
    plt.figure()
    plt.errorbar(range(len(stdList)), np.abs(meanList), yerr = stdList, ecolor = 'red', capthick = 3, capsize = 3)
    plt.title('Amplitude fonction du nombre de modes')
    plt.xlabel('Modes')
    plt.ylabel('Amplitude')

    Wlist = []
    for i in range(nbPSF):
        W = dicti['SRsky'][i] / np.abs(dicti['SRfit'][i] - dicti['SRsky'][i])
        Wlist.append(W)
    Wlist = Wlist/np.sum(Wlist)

    sumList  = []
    for i in range(nSeg):
        currentSum = 0
        for num,row in enumerate(Tableauresult):
            currentSum += np.abs(row[i]) * Wlist[num]
        sumList.append(currentSum)
    meanWeightedList = [currentSum / 170 for currentSum in sumList]

    #Weighted mean
    plt.figure()
    plt.plot(np.sort(meanWeightedList), 'o')
    #plt.errorbar(range(len(stdList)), np.abs(meanWeightedList), yerr = stdList, ecolor = 'red', capthick = 3, capsize = 3)
    plt.title('Amplitude fonction du nombre de modes')
    plt.xlabel('Modes')
    plt.ylabel('Amplitude')
    plt.show()
