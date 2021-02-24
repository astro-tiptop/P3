#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:40:03 2020

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
import scipy as sp
import os
import re
from numpy import unravel_index

#%% Define the instrument
D = 8.#telescope diameter in m
obs = 0.14#obstrucration ratio in D units
psInRad = 25*1e-3/206264.8 #pixel scale in rad
gCam = 2# Camera gain e-/ADU
ron = 75 #read out noise in e-
nAct = 39# linear number of DM actuators
MUSE = Instrument(D=D,occ=obs,res=psInRad,gain=gCam,ron=ron,Nact=nAct)
MUSE.name = "VLT GALACSI/MUSE"

#%% READ fits file

path_fits = '/run/media/omartin/OlivierMartinHDD/DATA/APPLY/PSF_LIBRARY/MUSE_LTAO/'
list_file = os.listdir(path_fits)
list_file = [x for x in list_file if re.search('.fits', x)] 
nIm = len(list_file)

nWvl      = 186;
wvl_min   = 471.2e-9
wvl_slice = 2.5e-9
wvl = np.linspace(wvl_min,wvl_min+wvl_slice*(nWvl-1),nWvl)
#remove wavelength corresponding to the notch filter
idGood = np.concatenate([np.linspace(0,44,45),np.linspace(51,nWvl-1,nWvl-51)]).astype('int')
nG = len(idGood)



#%% Fitting
for j in [3]:#np.arange(nIm):
    j=5
    path_data = path_fits+list_file[j]
    obj = fits.open(path_data)
    im = obj[0].data
    hdr = obj[0].header
    obj.close()
    nWvl = im.shape[0]    
    im_mod = np.zeros((nWvl,128,128))
    xf = np.zeros((11,nWvl))

    k =185
    # k in idGood[nG-1]:#np.arange(nIm):       
    im_tmp = im[k,:,:]
    [x,y] = unravel_index(im_tmp.argmax(), im_tmp.shape)
    im_tmp = im_tmp[x-64:x+64,y-64:y+64]        
    x0 = [0.4,1e-2,1,0.05,0.05,0,1.7]
    dxdy = (0,0)
    samp = wvl[k]/D/psInRad
    plt.clf()
    plt.imshow(np.arcsinh(im_tmp))
 #x[0] - Fried parameter r0 [m] science wavelength line of sight
 #x[1] - PSD corrected area background [rad² m²]
 #x[2] - PSD corrected area sigma² [rad²]
 #x[3] - PSD alpha X [1/m]
 #x[4] - PSD alpha Y [1/m]   (not defined in symmetric case)
 #x[5] - PSD theta   [rad]   (not defined in symmetric case)
 #x[6] - PSD beta power law  (becomes x[4] in symmetric case)
            
        Wmat = 1/(im_tmp*(im_tmp>0) + ron**2)
        res = psffit(im_tmp, Psfao, x0, symmetric=False, weights=Wmat, 
             samp = samp,system=MUSE,dxdy=dxdy)

        F,bg = res.flux_bck
        im_mod[k,:,:] = F*res.psf + bg

        xf[0:7,k] = res.x
        xf[7,k] = F
        xf[8:10,k] = res.dxdy
        xf[10,k] = bg    
      
     
        1e2*np.sqrt(np.sum((im_mod[k,:,:] - im_tmp)**2))/np.sum(im_tmp)
        
        plt.figure()
        plt.semilogy(im_tmp[64,:])
        plt.semilogy(im_mod[k,64,:])
        plt.figure()
        plt.semilogy(im_tmp[:,64])
        plt.semilogy(im_mod[k,:,64])
        1e2*np.sqrt(np.sum((im_mod[k,:,:] - im_tmp)**2))/np.sum(im_tmp)
        
#%% DISPLAY
 
    
   
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