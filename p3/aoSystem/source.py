#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 14:21:41 2018

@author: omartin
"""

import numpy as np
    
#class Attribute(object):
#    pass

class source:
    """
    """
    
    # DEPENDANT VARIABLES DEFINITION
    @property
    def direction(self):
        arcsec2rad = np.pi/180/3600
        deg2rad    = np.pi/180
        x          = np.tan(self.zenith*arcsec2rad)*np.cos(self.azimuth*deg2rad)
        y          = np.tan(self.zenith*arcsec2rad)*np.sin(self.azimuth*deg2rad)
        direction  = np.zeros((2,self.nSrc))
        direction[0,:] = x
        direction[1,:] = y
        return direction
            
    @property
    def waveNumber(self):
        return 2*np.pi/self.wvl
        
    # CONSTRUCTOR
    def __init__(self,wvl,zenith,azimuth,height=0,nSource=1,tag="SOURCE",verbose=False):
       
        # Vectorizing inputs is required  
        if np.isscalar(wvl):
            wvl= np.array([wvl])     
        if np.isscalar(zenith):
            zenith = np.array([zenith])
        if np.isscalar(azimuth):
            azimuth= np.array([azimuth])
        if np.isscalar(height):
            height= np.array([height])
        
        
        # PARSING INPUTS
        self.wvl       = wvl       # Wavelength value in meter             
        self.zenith    = zenith     # Zenith angle in arcsec
        self.azimuth   = azimuth    # Azimuth angle in degree
        self.height    = height     # Source height in meter
        self.nSource   = nSource
        self.tag       = tag
        self.verbose   = verbose        
        
        
        test= lambda x: len(x) == 1
        if (test(zenith)) and (test(azimuth)) and (test(wvl)):
            self.nSrc    = 1                        
        elif (test(zenith)) and (not test(azimuth)) and (test(wvl)):    
            self.nSrc    = len(azimuth)
            self.zenith  = zenith[0]*np.ones(self.nSrc)            
        elif (not test(zenith)) and (test(azimuth)) and (test(wvl)):   
            self.nSrc    = len(zenith)
            self.azimuth = azimuth[0]*np.ones(self.nSrc)           
        else:
            self.nSrc    = len(zenith)           
       
        # Vectorizes source properties
        if len(wvl) == 1 and self.nSrc>1:
            # print('Vectorize the wavelength value to cope with the number of ' + self.tag + ' sources')
            self.wvl = self.wvl[0]*np.ones(self.nSrc)
            self.nWvl=1
        else:
            self.nWvl = len(wvl)
            
        if self.height != 0:
            self.height = self.height*np.ones(self.nSrc)
        else:
            self.height = np.zeros(self.nSrc)
    
        # Put into array format
        self.wvl       = np.array(self.wvl)
        self.zenith    = np.array(self.zenith)
        self.azimuth   = np.array(self.azimuth)
        self.height    = np.array(self.height)
        
        if self.verbose:        
            self
        
    def __repr__(self):
        """Display object information: prints information about the source object
        """
       
        s = '___ ' + self.tag + '___\n'
        s += '--------------------------------------------------------------------------\n'
        s += ' Obj\t Zenith [arcsec]\t Azimuth [deg]\t height [m]\t wavelength [micron]\n'
        
        if self.nSrc > 0 and self.nWvl == 1: 
            for kObj in range(self.nSrc):
                s += ' %d\t\t\t %.2f\t\t\t %.2f\t\t\t %g\t\t\t %.3f\n'%(kObj,self.zenith[kObj],self.azimuth[kObj],
                                self.height[kObj],self.wvl[0]*1e6)
        elif self.nSrc == 1 and self.nWvl > 1:
            for kObj in range(self.nWvl):
                s += ' %d\t\t\t %.2f\t\t\t %.2f\t\t\t %g\t\t\t %.3f\n'%(kObj,self.zenith[0],self.azimuth[0],
                                self.height[0],self.wvl[kObj]*1e6)
        else:
            for kObj in range(self.nSrc):
                s += ' %d\t\t\t %.2f\t\t\t %.2f\t\t\t %g\t\t\t %s\n'%(kObj,self.zenith[kObj],self.azimuth[kObj],
                                self.height[kObj],str(self.wvl*1e6))
        s +='--------------------------------------------------------------------------\n'
        
        return s
    
    def polar(self):
        import matplotlib.pyplot as plt
        deg2rad = np.pi/180
        plt.figure()
        plt.polar(self.azimuth*deg2rad,self.zenith,'ro',markersize=7, label=self.tag + ' sources')
        plt.legend(bbox_to_anchor=(1.05, 1))
