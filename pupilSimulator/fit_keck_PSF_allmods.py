#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 15:36:44 2021

@author: dumontmaxime
"""
# IMPORTING LIBRAIRIES
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from configparser import ConfigParser
import os
from scipy.ndimage import rotate

from astropy.io import fits
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture

path_filter = '/home/mumont/Desktop/LAM_Stage/Data/filters/'

#update the ini file
path_ini = '/home/mumont/Desktop/LAM_Stage/Data/fit_all_modes.ini'
parser = ConfigParser()
parser.optionxform = str
parser.read(path_ini)

def getScale(hdr):
    # Setup NIRC2 plate scales
    scales = {"narrow": 9.942,
              "medium": 19.829,
              "wide": 39.686}

    return scales[hdr['CAMNAME']]  

def getPA(hdr):
    theta = float(hdr['EL']) - float(hdr['ROTPOSN']) - float(hdr['INSTANGL']) 
    return -theta



def get_filter_profile(path_filter,filter):
    """
    Returns the wavelength (in microns) and the transmission for 
    the specified NIRC2 filter.

    Example: 
    (wave, trans) = nirc2.photometry.get_filter_profile('Kp')
    py.clf()
    py.plot(wave, trans)
    py.xlabel('Wavelength (microns)')
    py.ylabel('Transmission')
    """
    filters = ['J', 'H', 'K', 'Kcont', 'Kp', 'Ks', 'Lp', 'Ms',
               'Hcont', 'Brgamma', 'FeII']

    if filter not in filters:
        print( 'Could not find profile for filter %s.' % filter)
        print( 'Choices are: ', filters)
        return

    table = Table.read(path_filter + filter + '.dat', format='ascii')

    wavelength = table[table.colnames[0]]
    transmission = table[table.colnames[1]]

    # Lets fix wavelength array for duplicate values
    diff = np.diff(wavelength)
    idx = np.where(diff <= 0)[0]
    wavelength[idx+1] += 1.0e-7

    # Get rid of all entries with negative transmission
    idx = np.where(transmission > 1)[0]
    wavelength = wavelength[idx]
    transmission = transmission[idx] / 100.0 # convert from % to ratio

    return (wavelength, transmission)


with open(path_ini, 'w') as configfile:
    parser.write(configfile)
   

#IMPORTING DATA
path_data = '/home/mumont/Desktop/LAM_Stage/Data/IMAGE_KECK_Kp_c0155.fits'
im = fits.getdata(path_data)
hdr = fits.getheader(path_data)


#GRABBING PARAMETERS FROM HEADER
#telescope status
airmass = float(hdr['AIRMASS'])
nirc2PA = getPA(hdr)
psInMas = getScale(hdr)   


# EXTRACTING THE IMAGE
mean, median, std = sigma_clipped_stats(im, sigma=5.0)  
daofind = DAOStarFinder(fwhm=2.0, threshold=5.*std)  
sources = daofind(im - median)  

positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=SqrtStretch())

plt.figure()
x0 = sources['xcentroid'][sources['flux'].argmax()]
y0 = sources['ycentroid'][sources['flux'].argmax()]
    
    
# Number of pixels to crop the PSF
nPix = 150
im_c    = im[int(y0)-nPix:int(y0)+nPix,int(x0)-nPix:int(x0)+nPix]
(yM,xM) = np.unravel_index(im_c.argmax(),im_c.shape)
im_c    = im_c[int(yM)-nPix//2:int(yM)+nPix//2,int(xM)-nPix//2:int(xM)+nPix//2]
plt.imshow(np.log10(im_c))
    
    
    
# DATA FITTING
# init guess
F     = abs(im_c.sum())
im_cn = im_c/F
r0    = 0.16 * airmass**(-3/5) * (wvl_c.mean()/500e-9)**1.2
x0    = [r0**(-5/3),1e-4,2,1e-5,1,0,2.8,0,0,0]+ [0.5,0.5] + list(xSrc) + list(ySrc) + [0] + list(np.zeros(37))
fixed = (False,)*7 + (True,)*3 + (False,)*(44)
# weights
weights = np.zeros_like(im_cn)
weights[im_cn>0] = im_cn[im_cn>0]
weights = 1.0/(weights + 10**2)

weights = np.ones_like(im_cn)
weights[im_cn<0] = 0
# fitting
res   = psfFitting(im_cn,psfao,x0,fixed=fixed,method='trf',verbose=2,weights=weights)