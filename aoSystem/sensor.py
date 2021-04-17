#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 13:57:21 2021

@author: omartin
"""

import numpy as np
from aoSystem.optics import optics
from aoSystem.detector import detector
from aoSystem.processing import processing

class sensor:
    """
    Wavefront sensor class. This class is instantiated through three sub-classes:\
    optics, detector and processing.
    """
    
    def __init__(self,pixel_scale,fov,binning=1,spotFWHM=[0.0,0.0],\
                 nph=np.inf,bandwidth=0.0,transmittance=[1.0],dispersion=[[0.0],[0.0]],\
                 ron=0.0,gain=1.0,dark=0.0,sky=0.0,excess=1.0, \
                 wfstype='Shack-Hartmann',nL=[1],dsub=[1],nSides=None,modulation=None,\
                 algorithm='wcog', algo_param=[5,0,0], noiseVar=0, tag='SENSOR'):
        
                
        # optics class
        self.optics = optics(nL=nL,dsub=dsub,nSides=nSides,wfstype=wfstype,modulation=modulation)
        
        # detector class
        self.detector = detector(pixel_scale,fov,binning=binning,spotFWHM=spotFWHM,\
                 nph=nph,bandwidth=bandwidth,transmittance=transmittance,dispersion=dispersion,\
                 ron=ron,gain=gain,dark=dark,sky=sky,excess=excess)
        
        # processing class
        self.processing = processing(algorithm=algorithm,settings=algo_param,noiseVar=noiseVar)