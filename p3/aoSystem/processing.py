#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:19:06 2021

@author: omartin
"""
import numpy as np

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
            
        if np.any(self.noiseVar == [None]):
            s += '.Mean noise variance [rd^2] : %.f'%(0)
        else:
            s += '.Mean noise variance [rd^2] : %.f'%(np.mean(self.noiseVar))
        
        return s
        