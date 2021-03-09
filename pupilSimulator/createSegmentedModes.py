#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:28:53 2021

@author: omartin
"""

#%% IMPORTING LIBRAIRIES
from pupilSimulator.pupil import pupil
from pupilSimulator.segment import segment
from aoSystem.zernike import zernike
import matplotlib.pyplot as plt
import numpy as np
import fourier.FourierUtils as FourierUtils
from astropy.io import fits

#%% CREATING THE PUPIL CLASS 
nPx     = 200
path_txt = '/home/omartin/Projects/P3/pupilSimulator/_txtFile/Keck_segmentVertices.txt'
pupSeg   = pupil(segClass=segment(6,0.9,nPx),segCoord=path_txt,D=11.25)
nSeg     = pupSeg.nSegments

#%% CREATING THE SEGMENTED MODE
mode = np.zeros((nSeg,nPx,nPx))
for j in range(nSeg):
    pupSeg.reset()
    pupSeg.applyPhaseErrorSegment([j],[1],[1])
    mode[j] = FourierUtils.interpolateSupport(np.angle(pupSeg.matrix),nPx,kind='nearest')
    
#%% SAVING
path_save = '/home/omartin/Projects/APPLY/DATA/STATIC'
hdu = fits.PrimaryHDU(mode)
hdu.writeto(path_save + '/keck_piston_modes_'+str(nPx)+'px.fits',overwrite=True)

# get waffle
waffle = fits.getdata(path_save+'/keck_waffle_200px.fits')
waffle = waffle/waffle[waffle!=0].std()
#merge
hdu = fits.PrimaryHDU(np.concatenate((mode,waffle[np.newaxis,:,:]),axis=0))
hdu.writeto(path_save + '/keck_piston_waffle_modes_'+str(nPx)+'px.fits',overwrite=True)

