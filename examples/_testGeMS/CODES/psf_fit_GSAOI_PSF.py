#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:20:02 2020

@author: omartin
"""

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
import scipy.signal as sp
import os

#%% Define the instrument
D = 8.1#telescope diameter in m
obs = 0.12#obstrucration ratio in D units
gCam = np.mean([2.434,2.01,2.411,2.644])# Camera gain e-/ADU
ron = np.mean([23.63,9.85,14.22,16.78]) #read out noise in e-
nAct = 27# linear number of DM actuators

#%% READ fits file
path_fits_root = '/home/omartin/Projects/PSFR/GEMS_PSFR/DOCUMENTS/'
name_K = 'psf_gems_K_r0_54.914cm_sr_2_Samp_1.3687_rgS20190416S0115_proj.fits'
name_H = 'psf_gems_H_r0_40.7014cm_sr_3_Samp_1.0409_rgS20190418S0042_proj.fits'
name_J = 'psf_gems_J_r0_43.4147cm_sr_8_Samp_0.79577_rgS20190417S0096_proj.fits'

band = ['K','H','J']
nPx = 128
r0_500 = 0.103/0.8
samp = [1.3687,1.0409,0.79577]*2
wvl = [2.15*1e-6,1.635*1e-6,1.25*1e-6]
r0 = [0.54,0.4,0.43]

for k in [0,1,2]:    
    
    # read the images
    obj = fits.open(path_fits_root+name_K)
    im = obj[0].data
    obj.close()
    nIm = im.shape[1]
    
    # Define the instrument
    psInRad = 20*1e-3/206264.8
    GSAOI = Instrument(D=D,occ=obs,res=psInRad,gain=gCam,ron=ron,Nact=nAct)

    #%% Fitting 
    xf = np.zeros((7,nIm))
    im_mod = np.zeros((nIm,nPx,nPx))
    xf = np.zeros((7,nIm))
    for j in np.arange(nIm):
        if j==0:
            x0 = [r0[k],1e-2,10,4e-3,4e-3,0,1.5]
        else:
            x0 = res.x
            
        res = psffit(im[:,j,:], Psfao, x0, symmetric=False, weights=im[:,j,:]>0,samp = samp[k],system=GSAOI)
        F,bg = res.flux_bck
        im_mod[j,:,:] = F*res.psf + bg
        xf[:,j] = res.x

    #%% WRITE   
    plt.close('all')
    plt.figure()
    plt.imshow(np.arcsinh((im_mod[j,:,:] - im[:,j,:])))        
    
    eqm = 1e2*np.sqrt(np.sum((im_mod[j,:,:] - im[:,j,:])**2))/np.sum(im[:,j,:])
    print(eqm)
    hdu = fits.PrimaryHDU(im_mod)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path_fits_root+'psf_fit_'+band[k]+'.fits',overwrite=True)
    hdu = fits.PrimaryHDU(xf)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path_fits_root+'param_fit_'+band[k]+'.fits',overwrite=True)    
    #%%
    nPx = 128    
    plt.figure()
    plt.clf()
    plt.semilogy(im[int(nPx/2),j,:]/im[:,j,:].max(),label='GSAOI PSF')
    plt.semilogy(im_mod[j,int(nPx/2),:]/im[:,j,:].max(),label='Adjusted model')
    im_dif = np.abs(im_mod[j,:,:] - im[:,j,:])/im[:,j,:].max()
    plt.semilogy(im_dif[int(nPx/2),:],label='Residual')
    plt.legend(fontsize=16)
    
    

