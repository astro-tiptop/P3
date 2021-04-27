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
    
    def __init__(self,nL=1,dsub=1,nSides=4,wfstype='Shack-Hartmann',modulation=None):
        
        
        self.nL = nL
        self.dsub = dsub
        self.nSides = nSides
        self.wfstype = wfstype
        self.modulation = modulation
        
    def __repr__(self):
        
        s = '__ OPTICS __\n'
        if self.wfstype.upper() == 'SHACK-HARTMANN':
            s += 'Optics type : %dX%d %s\n'%(self.nL,self.nL,'lenslets array')
        else:
            s += 'Optics type : %d%s\n'%(self.nSides,'-faces Pyramid')
            s += 'Modulation : %d lambda/D\n'%(self.modulation)
            
        return s