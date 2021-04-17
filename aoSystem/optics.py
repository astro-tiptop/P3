#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:05:23 2021

@author: omartin
"""

class optics:
    """
        optics class to define the optical element in front of the wavefront sensor detector
    """
    
    def __init__(self,nL=1,dsub=1,nSides=None,wfstype='Shack-Hartmann',modulation=None):
        
        
        self.nL = nL
        self.dsub = dsub
        self.nSides = nSides
        self.wfstype = wfstype
        self.modulation = modulation