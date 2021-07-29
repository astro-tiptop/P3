#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:37:55 2021

@author: omartin
"""

import numpy as np
from astropy.io import fits
from query_eso_archive import query_simbad
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from sphere.transmission import irdis_nd
import os
import shutil

#%% IMAGE PROCESSING
def RemoveCorruptedData(path_data,path_out):
    """
        Read all data folders and move the data with corrupted PSF to the outliers folder. Corruption is tracked by bad centering of the PSF.
    """
    
    # list the data sets
    listData   = os.listdir(path_data)
    nData      = len(listData)
    
    # loop on all folders
    list_corr = []
    for kData in range(nData): # data folder
        path_data_k = path_data + listData[kData] +'/'        
        if os.path.isfile(path_data_k + 'ird_convert_recenter_dc5-SPH_SPARTA_PSFDATA-psf_sparta_data.fits'):
            path_fits = path_data_k + 'ird_convert_recenter_dc5-IRD_SCIENCE_PSF_MASTER_CUBE-median_unsat.fits'
            im   = fits.getdata(path_fits) #first dim : left/right, 2nd dim : nacquisition
            if im.ndim == 3:
                im = im[0]
            elif im.ndim == 4:
                im = im[-1,0]
            if im[32,32] != im.max():
                list_corr.append(listData[kData])
        else:
            list_corr.append(listData[kData])
            
    print('Detection of %d corrupted files'%(len(list_corr)))
    for n in range(len(list_corr)):
        print(list_corr[n])
        shutil.move(path_data+list_corr[n], path_out)
        
def ProcessCleanData(path_data,nPix):
    """
        Process pre-cleaned SPHERE cubes of coadds to recenter the frame, stack them and normalize them
    """
    # read data
    im = fits.getdata(path_data)
    hdr= fits.getheader(path_data)
    # processing
    nF    = im.shape[0]
    out   = np.zeros((nF,nPix,nPix))
    for kf in range(nF):
        tmp     = im[kf]
        x0,y0   = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)
        out[kf] = tmp[x0-nPix//2:x0+nPix//2,y0-nPix//2:y0+nPix//2]
    # stacking 
    out = np.mean(out,axis=0)
    # normalizing
    DIT = hdr['ESO DET SEQ1 DIT']
    ND  = irdis_nd(hdr['HIERARCH ESO INS COMB IFLT'],['HIERARCH ESO INS4 FILT2 NAME'])
    out = out/DIT/ND
    return out

#%% READING THE HEADER
    
def GetWavelength(path_data):
    """
        Read the file IRD_SCIENCE_LAMBDA_INFO to extract the filter information
    """
    wvl = 1e-6*fits.getdata(path_data + '/ird_convert_recenter_dc5-IRD_SCIENCE_LAMBDA_INFO-lam.fits')
    bw  = wvl.max() - wvl.min()
    wvl = wvl.mean()
    
    return wvl, bw

def ReadSpartaData(path_data,which='last'):
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
                
        r0     = np.array(sparta[0,:,:])    
        vspeed = np.array(sparta[1,:,:])
        SR     = np.array(sparta[2,:,:])
        seeing = np.array(sparta[3,:,:])
        
        if which == 'last':
            r0     = r0[nPSF,:]
            vspeed = vspeed[nPSF,:]
            SR     = SR[nPSF,:]
            seeing = seeing[nPSF,:]
            
        return r0, vspeed, SR, seeing

def GetStarCoordinates(hdr):
    """
        Returns the name of the object as well as its RA/DEC coordinates
    """
    # get the star coordinates and name
    return hdr['ESO OBS NAME'], float(hdr['RA']), float(hdr['DEC'])
    
def GetStarMagnitudes(hdr):
    """
        Query simbad to get the star magnitudes.
    """
    
    OB_NAME, RA, DEC = GetStarCoordinates(hdr)

    # query simbad
    DICT_SIMBAD= query_simbad(Time(hdr['DATE-OBS']),SkyCoord(RA*u.degree,DEC*u.degree),name=OB_NAME)
    
    # get magnitudes
    VMAG = RMAG = GMAG = JMAG = HMAG = KMAG = -1
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

def GetDetectorConfig(hdr):
    """
        Read the header to get the detector configuration.
    """
    if 'PIXSCAL' in hdr:
        psInMas   = float(hdr['PIXSCAL'])
    else:
        psInMas = -1
    gain      = float(hdr['ESO DET CHIP1 GAIN'])
    ron       = float(hdr['ESO DET CHIP1 RON'])
    DIT       = hdr['ESO DET SEQ1 DIT']
    NDIT      = hdr['ESO DET NDIT']
    NDSKIP    = hdr['ESO DET NDSKIP']  
    
    return psInMas, gain, ron, DIT, NDIT, NDSKIP

def ReadStrehlValue(hdr):
    """
    Read the Strehl-ratio values in the header
    """
    if 'SRMEAN' in hdr:
        SRMEAN  = float(hdr['SRMEAN'])         
        SRMIN   = float(hdr['SRMIN'])                                
        SRMAX   = float(hdr['SRMAX'])
        return SRMEAN, SRMIN, SRMAX
    else:
        print('No Strehl-ratio values in the header')
        return -1, -1, -1
    
def GetDate(hdr):
    """
    Get the date from the header
    """
    DATE_OBS  = hdr['DATE-OBS']
    return DATE_OBS[0:4]+DATE_OBS[5:7]+DATE_OBS[8:10]+DATE_OBS[11:13]+DATE_OBS[14:16]+DATE_OBS[17:19]

def GetTelescopePointing(hdr):
    """
    Get the telescope pointing from the header.
    """
    TELAZ     = float(hdr['ESO TEL AZ'])
    TELALT    = float(hdr['ESO TEL ALT'])
    airmass   = 0.5*(float(hdr['ESO TEL AIRM END']) + float(hdr['ESO TEL AIRM START']))     
    
    return TELAZ, TELALT, airmass

def GetAmbiParameters(hdr):
    """
    Get atmosphere parameters from the header.
    """
    tau0      = float(hdr['ESO TEL AMBI TAU0'])
    wDir      = float(hdr['ESO TEL AMBI WINDDIR'])
    wSpeed    = float(hdr['ESO TEL AMBI WINDSP'])
    RHUM      = float(hdr['ESO TEL AMBI RHUM'])
    pressure  = 0.5*(float(hdr['ESO TEL AMBI PRES START']) + float(hdr['ESO TEL AMBI PRES END']))
    fwhm_linobs= float(hdr['ESO TEL IA FWHMLINOBS']) 
    # note: corrected active optics FWHM of the SH spots (arcsec) 
    #(along the line of sight, PSF-based algorithm as described in 
    #Martinez et al. 2011)
    
    return fwhm_linobs, tau0, wSpeed, wDir, RHUM, pressure

def GetTemperature(hdr):
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

def GetMoonPosition(hdr):
    """
        Returns the RA/DEC of the moon during the observation.
    """
    return hdr['ESO TEL MOON RA'], hdr['ESO TEL MOON DEC']

        
        
