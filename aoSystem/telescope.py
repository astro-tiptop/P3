#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:34:02 2018

@author: omartin
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import re
import fourier.FourierUtils as FourierUtils

    
class Attribute(object):
    pass

class telescope:
    """ Telescope class that defines the telescope characteristics.
    Inputs are:
        - D: telescope diameter
        - elevation: telescope elevation in degree
        - obsRatio: central obstruction ratio from 0 to 1
        - resolution: pupil pixels resolution
    """
    
    # DEPENDANT VARIABLES DEFINITION   
    def get_pupil(self):
        return self.p_pupil
                 
    def set_pupil(self,val):
        self.p_pupil = val
        self.resolution = val.shape[0]
        
    pupil = property(get_pupil,set_pupil)        
        
    @property
    def R(self):
        """Telescope radius in meter"""
        return self.D/2
    @property
    def area(self):
        """Telescope area in meter square"""
        return np.pi*self.R**2*(1-self.obsRatio**2)
    @property
    def pupilLogical(self):
        """Telescope area in meter square"""
        return self.pupil.astype(bool)
    
    @property
    def airmass(self):
        return 1/np.cos(self.zenith_angle*np.pi/180)
    
    # CONSTRUCTOR
    def __init__(self,D,zenith_angle,obsRatio,resolution,file = [],obj=[],verbose=True):
        
        # PARSING INPUTS
        self.D         = D          # in meter
        self.zenith_angle = zenith_angle  # in degree
        self.obsRatio  = obsRatio   # Ranges from 0 to 1
        self.resolution= resolution # In pixels
        self.verbose   = verbose
        self.file      = file
        # PUPIL DEFINITION        
        import os.path as ospath
        
        pupil = [] 
        if file!=[] and ospath.isfile(file) == True:
            self.verb = True
            if  re.search(".fits",file)!=None :
                pupil = fits.getdata(file)
                #pupil = pupil * (pupil>=1)
                if pupil.shape[0] != resolution:
                    pupil = FourierUtils.interpolateSupport(pupil,resolution,kind='nearest')
                self.pupil = pupil
        
        if len(pupil) ==0:
            x   = np.linspace(-D/2,D/2,resolution)
            X,Y = np.meshgrid(x,x)
            R   = np.hypot(X,Y)
            P   = (R <= self.R) * (R > self.R*self.obsRatio)
            self.pupil = P
            self.verb = False
    
        if self.verbose:
            self
                   
            
    def __repr__(self):
        s = "___TELESCOPE___\n ---------------------------------------- \n"
        s = s + ". Aperture diameter \t:{:.2f}m \n. Central obstruction \t:{:.2f}% \n. Collecting area \t\t:{:.2f}m^2\n. Pupil resolution \t\t:{:d}X{:d} pixels".format(self.D,self.obsRatio*1e2,self.area,self.resolution,self.resolution)
        s = s +"\n----------------------------------------\n"
        return s
        
    
    def displayPupil(self):
        """DISPLAYPUPIL Display the telescope pupil
        """
                    
        plt.figure()
        plt.imshow(self.pupil)
        plt.xlabel("Pixels")
        plt.ylabel("Pixels")
        plt.title("Telescope pupil")
            