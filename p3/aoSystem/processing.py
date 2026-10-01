#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:19:06 2021

@author: omartin
"""
import numpy as np
from . import cpuArray

def is_auto_noise_var(noise_var) -> bool:
    """True if the noise variance is unset ([None]) and must be computed.

    Safe for lists and numpy/cupy arrays of any length, where a plain
    ``noise_var == [None]`` is element-wise and ambiguous for 2+ entries.
    """
    noise_arr = np.asarray(cpuArray(noise_var), dtype=object)
    return noise_arr.size == 1 and noise_arr.reshape(-1)[0] is None

class processing:
    """
        processing class to define the processing strategy to convert the detector image into an input for the RTC.
    """
    
    def __init__(self,algorithm='wcog',settings=[5,0.0,0.0],noiseVar=[None]):
        
        
        self.algorithm = algorithm
        self.settings = settings
        self.noiseVar = noiseVar
        
        if algorithm == 'wcog':
            if len(settings) !=3:
                print('Error : the number of parameters must be 3 : radius, threshold and new value')
                
    def __repr__(self):
        
        s = '__ PROCESSING __\n'
        s += '.Algorithm : %s\n'%(self.algorithm)
        if self.algorithm.upper() == 'WCOG':
            s += '.Window radius : %d pixels\n'%(self.settings[0])
            s += '.Threshold : %.2f e-\n'%(self.settings[1])
            s += '.New value : %.2f e-\n'%(self.settings[2])

        noise_var = cpuArray(self.noiseVar)
        if is_auto_noise_var(noise_var):
            s += '.Mean noise variance [rd^2] : %.f'%(0)
        else:
            s += '.Mean noise variance [rd^2] : %.f'%(float(np.mean(np.asarray(noise_var, dtype=float))))
        
        return s
        