#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:03:05 2020

@author: omartin
"""

from STORM.psfmodel import Psfao, psffit
from astropy.io import fits
from STORM.opticalsystem import Instrument
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

#%% Define the instrument
D = 8.22#telescope diameter in m
obs = 0.111#obstrucration ratio in D units
psInRad = 15*1e-3/206264.8 #pixel scale in rad
gCam = 2# Camera gain e-/ADU
ron = 10 #read out noise in e-
nAct = 29# linear number of DM actuators
LBT = Instrument(D=D,occ=obs,res=psInRad,gain=gCam,ron=ron,Nact=nAct)
LBT.name = "LBT SOUL/LUCI"

wvl = 1.646e-6
samp = wvl/D/psInRad #PSF sampling in lambda/D units

#%% READ fits file

path_fits = '/run/media/omartin/OlivierMartinHDD/DATA/LBT_DATA/SOUL_PsfAndTelemetry-20200228T122058Z-001/SOUL_PsfAndTelemetry/PSFs_woTelemetry/'
path_save = '/home/omartin/Projects/PSFR/PSF_FIT/_results/'
date = ('Data_20190406_045229','Data_20190406_045345','Data_20190406_061225',
        'Data_20190406_062632','Data_20190406_070025','Data_20190406_072319',
        'Data_20190406_075830','Data_20190406_080409')
nDate = len(date)

for j in np.arange(nDate):
    path_data = path_fits+date[j]
    obj = fits.open(path_data+'/luci.fits')
    im = obj[0].data
    hdr = obj[0].header
    obj.close()



#%% Fitting
    nIm = im.shape[0]
    nPx = im.shape[1]
    eqm = np.zeros(nIm+1)
    seeing = np.zeros((2,nIm+1))
    xf = np.zeros((11,nIm+1))
    im_mod = np.zeros((nIm+1,nPx,nPx))

    for k in np.arange(nIm):       
        x0 = [0.4,1e-2,2,0.05,0.05,0,2.5]
        dxdy = (-4,10)
        
        
 #x[0] - Fried parameter r0 [m] science wavelength line of sight
 #x[1] - PSD corrected area background [rad² m²]
 #x[2] - PSD corrected area sigma² [rad²]
 #x[3] - PSD alpha X [1/m]
 #x[4] - PSD alpha Y [1/m]   (not defined in symmetric case)
 #x[5] - PSD theta   [rad]   (not defined in symmetric case)
 #x[6] - PSD beta power law  (becomes x[4] in symmetric case)
            
        Wmat = 1/(np.abs(im[k,:,:]*gCam) + ron**2)
        res = psffit(im[k,:,:], Psfao, x0, symmetric=False, weights=Wmat, 
             samp = samp,system=LBT,dxdy=dxdy)

        F,bg = res.flux_bck
        im_mod[k,:,:] = F*res.psf + bg

        xf[0:7,k] = res.x
        xf[7,k] = F
        xf[8:10,k] = res.dxdy
        xf[10,k] = bg    
        eqm[k] = 1e2*np.sqrt(np.sum((im_mod[k,:,:] - im[k,:,:])**2))/np.sum(im[k,:,:])
        seeing[0,k] = hdr['DIMMFWHM']
        seeing[1,k] = 0.100/(res.x[0]*hdr['AIRMASS']**(3/5)*(0.5e-6/wvl)**1.2)    
   
   
#%% DISPLAY
 
    
    x0 = [0.3,1e-2,2,0.05,0.05,0,1.5]
    imLE = im.sum(axis=0)
    Wmat = 1/(np.abs(imLE*gCam) + ron**2)
    Wmat[np.where(Wmat<0)] = 1/ron**2    
    res = psffit(imLE, Psfao, x0, symmetric=False, weights=Wmat, 
             samp = samp,system=LBT,dxdy=dxdy)
    F,bg = res.flux_bck
    im_mod[k+1,:,:] = F*res.psf + bg
    xf[0:7,k+1] = res.x
    xf[7,k+1] = F
    xf[8:10,k+1] = res.dxdy
    xf[10,k+1] = bg
    eqm[k+1] = 1e2*np.sqrt(np.sum((im_mod[k+1,:,:] - imLE)**2))/np.sum(imLE)
    seeing[0,k+1] = np.mean(seeing[0,0:nIm])
    seeing[1,k+1] = 0.100/(res.x[0]*hdr['AIRMASS']**(3/5)*(0.5e-6/wvl)**1.2)    
    
    plt.clf()
    plt.imshow(np.log10(np.abs(im_mod[k+1,:,:] - imLE)))
    # Write imagesc
    hdu = fits.PrimaryHDU(im_mod)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path_data+'/psf_fit.fits',overwrite=True)
    1
    # Write eqm
    hdu = fits.PrimaryHDU(eqm)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path_data+'/eqm_lbt.fits',overwrite=True)

    # Write seeing
    hdu = fits.PrimaryHDU(seeing)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path_data+'/seeing_lbt.fits',overwrite=True) 

    # Write fitted parameters
    hdu = fits.PrimaryHDU(xf)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path_data+'/param_lbt.fits',overwrite=True) 