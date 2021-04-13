#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:13:03 2021

@author: omartin
"""

import numpy as np

class detector:
    """
    """
    
    def __init__(self,pixel_scale,fov,bandwidth=0.0,transmittance=[1.0],dispersion=[[0.0],[0.0]],\
                 nph=np.inf,ron=0.0,dark=0.0,sky=0.0,excess=1.0, spotFWHM=[0.0,0.0],\
                 nL=20, loopGain=0.5, delay=2, SensorFrameRate=500.0, noise=None,\
                 SlopeAlgorithm='window',wcog=[5.0,0.0,0.0],tech_fov=120, tag='SENSOR'):
        
        # PARSING INPUTS
        # scales
        self.psInMas = pixel_scale
        self.fovInPix    = fov
        self.fovInArcsec = pixel_scale * fov/1e3
        self.spotFWHM    = spotFWHM
        
        # spectral width
        self.bandwidth     = bandwidth
        self.transmittance = transmittance
        self.dispersion    = dispersion
        # verification
        if len(self.dispersion[0]) != len(self.transmittance):
            print('Error, the number of wavelengths is not consistent in the input lists')
            return
        self.nWvl = len(transmittance)
        
        # photometry and detector config
        self.nph  = nph
        self.ron  = ron
        self.dark = dark
        self.sky  = sky
        
        # lenselts and centroiding
        self.nLenslets = nL
        self.loopGain  = loopGain
        self.delay     = delay
        self.noise     = None
        self.SensorFrameRate = SensorFrameRate
        self.SlopeAlgorithm = 'window'
        self.WindowRadiusWCoG = wcog[0]
        self.ThresholdWCoG = wcog[1]
        self.NewValueThrPix = wcog[2]
        self.tech_fov = tech_fov
        self.tag = tag

    def __repr__(self):
        s = self.tag + ' CLASS'
        return s