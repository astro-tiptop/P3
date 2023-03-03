#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 17:28:53 2021

@author: omartin, mdumont
"""

#%% IMPORTING LIBRAIRIES
import pathlib
import numpy as np
from astropy.io import fits

import p3.aoSystem as aoSystemMain
from p3.aoSystem.pupil import pupil
from p3.aoSystem.segment import segment
from p3.aoSystem.zernike import zernike
from p3.aoSystem.spiders import spiders
import matplotlib.pyplot as plt
import p3.aoSystem.FourierUtils as FourierUtils
path_ao = str(pathlib.Path(aoSystemMain.__file__).parent.absolute())

#%% CREATING ELT PETALLING MODES 
nPx = 50
path_txt = path_ao + '/_txtFile/ELT_segmentVertices.txt'
spiRef = spiders([0, 60, 120], 0.5, symetric=True, D=39) 
eltPup = pupil(segClass=segment(6, 1.42/2, nPx), segCoord=path_txt, D=39,
                cobs=0.2375, spiderClass=spiRef, getPetal=True, fill_gap=True)
modes = eltPup.computeModes([1,2,3], area="petal")
plt.figure();plt.imshow(modes[:,0].sum(axis=0))
plt.figure();plt.imshow(modes[:,1].sum(axis=0))
plt.figure();plt.imshow(modes[:,2].sum(axis=0))

#%% CREATING KECK SEGMENTDED MODES
nPx = 200
path_txt = path_ao + '/_txtFile/Keck_segmentVertices.txt'
spiRef = spiders([0,60,120], 0.0254, symetric=True, D=10.5) 
pupSeg = pupil(segClass=segment(6,0.9,nPx),segCoord=path_txt, spiderClass=spiRef)
nSeg  = pupSeg.nSegments
# CREATING ZERNIKE MODES
modNumbers = list(range(1,3))
zerMod_list = zernike(modNumbers, pupSeg.matrix.shape[0])
nMod = len(modNumbers)

# CREATING THE SEGMENTED MODE
mode_lr = np.zeros((nMod,nPx,nPx))
mode_hr = np.zeros((nMod,pupSeg.matrix.shape[0],pupSeg.matrix.shape[0]))

for i in range(nMod):
    for j in range(nSeg):
        pupSeg.reset()
        pupSeg.applyPhaseErrorSegment([j], [1], [1])
        segShape = np.angle(pupSeg.matrix)
        segShape[segShape > 0] = 1
        intersec = segShape * zerMod_list.modes[i]

        val = np.mean(intersec[(segShape > 0)])
        pupSeg.applyPhaseErrorSegment([j],[1],[val])
        
        mode_hr[i] += np.angle(pupSeg.matrix)
        mode_lr[i] += FourierUtils.interpolateSupport(np.angle(pupSeg.matrix),nPx,kind='linear')
        
        pupSeg.reset()
    plt.figure()
    plt.imshow( mode_hr[i] )

# #%% SAVING
# path_save = '/Users/dumontmaxime/Desktop/LAM_stage/Results'
# hdu = fits.PrimaryHDU(mode_lr)
# hdu.writeto(path_save + '/keck_piston_modes_'+str(nPx)+'px_all_lr.fits',overwrite=True)

# hdu = fits.PrimaryHDU(mode_hr)
# hdu.writeto(path_save + '/keck_piston_modes_'+str(pupSeg.matrix.shape[0])+'px_all_hr.fits',overwrite=True)

