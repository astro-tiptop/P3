#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:19:06 2021

@author: omartin
"""

class processing:
    """
        processing class to define the processing strategy to convert the detector image into an input for the RTC.
    """
    
    def __init__(self,algorithm='wcog',settings=[5,0.0,0.0],noiseVar=None):
        
        
        self.algorithm = algorithm
        self.settings = settings
        self.noiseVar = noiseVar
        
        if algorithm == 'wcog':
            if len(settings) !=3:
                print('Error : the number of parameters must be 3 : radius, threshold and new value')
                
