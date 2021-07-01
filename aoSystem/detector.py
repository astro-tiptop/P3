#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:13:03 2021

@author: omartin
"""

import numpy as np

class detector:
    """
        Detector class that gathers characteristics of the detector
    """
    
    def __init__(self,pixel_scale,fov,binning=1,spotFWHM=[[0.0,0.0,0.0]],saturation=np.inf,\
                 nph=np.inf,bandwidth=0.0,transmittance=[1.0],dispersion=[[0.0],[0.0]],\
                 ron=0.0,gain=1.0,dark=0.0,sky=0.0,excess=1.0, tag='DETECTOR'):
        
        # PARSING INPUTS
        self.tag = tag
        # scales
        self.psInMas     = pixel_scale
        self.fovInPix    = fov
        self.fovInArcsec = pixel_scale * fov/1e3
        self.spotFWHM    = spotFWHM
        self.saturation  = saturation
        
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
        self.excess = excess
        self.gain = gain
        self.binning  = binning
        self.tag = tag

    def __repr__(self):
        s = '__ ' + self.tag + ' __\n'
        s += '. Pixel scale : %.2f mas\n'%(self.psInMas)
        s += '. Field of view : %d pixels\n'%(self.fovInPix)
        s += '. Field of view : %.2f arcsecs\n'%(self.fovInArcsec)
        
        return s