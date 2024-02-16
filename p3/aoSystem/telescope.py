#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:34:02 2018

@author: omartin
"""
import numpy as nnp
from . import gpuEnabled

if not gpuEnabled:
    np = nnp
    from scipy.ndimage import rotate
else:
    import cupy as cp
    np = cp
    from cupyx.scipy.ndimage import rotate

from astropy.io import fits
import os.path as ospath
import re
import p3.aoSystem.FourierUtils as FourierUtils

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
        return 1/nnp.cos(self.zenith_angle*np.pi/180)
    
    # CONSTRUCTOR
    def __init__(self,D,resolution, zenith_angle=0.0,obsRatio=0.0,pupilAngle = 0.0,\
                 path_pupil='', path_static_on='', path_static_off='', path_static_pos='',\
                 path_apodizer='', path_statModes='', extraErrorNm=0.0, extraErrorExp=-2, \
                 extraErrorMin=0.0, extraErrorMax=0.0, extraErrorLoNm=0.0, extraErrorLoExp=-2, \
                 extraErrorLoMin=0.0, extraErrorLoMax=0.0, verbose=True):
        
        # PARSING INPUTS
        self.D                 = D            # primary mirror diameter in meters
        self.zenith_angle      = zenith_angle # telescope zenith angle in degrees
        self.obsRatio          = obsRatio     # secondary mirror diameter in D units
        self.resolution        = resolution   # pupil resolution in pixels
        self.verbose           = verbose
        self.pupilAngle        = pupilAngle
        self.path_pupil        = path_pupil
        self.path_static_on    = path_static_on
        self.path_static_pos   = path_static_off
        self.path_apodizer     = path_apodizer
        self.path_statModes    = path_statModes
        self.extraErrorNm      = extraErrorNm
        self.extraErrorExp     = extraErrorExp
        self.extraErrorMin     = extraErrorMin
        self.extraErrorMax     = extraErrorMax
        self.extraErrorLoNm    = extraErrorLoNm
        self.extraErrorLoExp   = extraErrorLoExp
        self.extraErrorLoMin   = extraErrorLoMin
        self.extraErrorLoMax   = extraErrorLoMax
        
        #----- PUPIL DEFINITION        
        
        pupil = [] 
        if path_pupil!= '' and ospath.isfile(path_pupil) == True and re.search(".fits",path_pupil)!=None:
            self.verb = True
            pupil = np.asarray(fits.getdata(path_pupil))
            pupil[pupil!=pupil] = 0
            if self.pupilAngle !=0.0:
                pupil = rotate(pupil,self.pupilAngle,reshape=False)
            if pupil.shape[0] != resolution:
                pupil = FourierUtils.interpolateSupport(pupil,resolution,kind='linear')
            self.pupil = pupil
        else:
            # build an annular pupil model
            th  = self.pupilAngle*np.pi/180
            x   = np.linspace(-D/2,D/2,resolution)
            X,Y = np.meshgrid(x,x)
            Xr  = X*np.cos(th) + Y*np.sin(th)
            Yr  = Y*np.cos(th) - X*np.sin(th)
            R   = np.hypot(Xr,Yr)
            P   = (R <= self.R) * (R > self.R*self.obsRatio)
            self.pupil = P
            self.verb = False
            self.path_pupil= ''
    
    
        #----- NCPA
        
        if path_static_on != None and ospath.isfile(path_static_on) == True and re.search(".fits",path_static_on)!=None:
            self.opdMap_on = np.asarray(fits.getdata(path_static_on))
            self.opdMap_on[self.opdMap_on!=self.opdMap_on] = 0
            if self.pupilAngle !=0.0:
                self.opdMap_on = rotate(self.opdMap_on,self.pupilAngle,reshape=False)
            self.opdMap_on = FourierUtils.interpolateSupport(self.opdMap_on,resolution,kind='linear')
        else:
            self.opdMap_on = None
            self.path_static_on= ''
            
        #----- FIELD-DEPENDANT STATIC ABERRATIONS
        if path_static_off != None and ospath.isfile(path_static_off) == True and re.search(".fits",path_static_off)!=None:   
            if path_static_pos != None and ospath.isfile(path_static_pos) == True and re.search(".fits",path_static_pos)!=None:   
                opdMap_off = np.asarray(fits.getdata(path_static_off))
                opdMap_pos = np.asarray(fits.getdata(path_static_pos))
                if opdMap_pos.shape[0] != opdMap_off[0]:
                    raise ValueError('You must provide as many positions values as maps')
                else:
                    self.opdMap_off = opdMap_off
                    self.opdMap_pos = opdMap_pos
            else:
                raise ValueError('Positions (zenith in arcsec, azimuth in radian) of the field-dependent aberrations must be provided as well as the maps')
        else:
            self.opdMap_off = None #center of the fov
            self.opdMap_pos = None #center of the fov
            self.path_static_off= ''
            self.path_static_pos= ''
            
        #----- APODIZER
        print(path_apodizer)
        if path_apodizer!= '' and ospath.isfile(path_apodizer) and re.search(".fits",path_apodizer)!=None:
            self.apodizer = np.asarray(fits.getdata(path_apodizer))
            self.apodizer[self.apodizer!=self.apodizer] = 0
            if self.pupilAngle !=0.0:
                self.apodizer = rotate(self.apodizer,self.pupilAngle,reshape=False)
            self.apodizer = FourierUtils.interpolateSupport(self.apodizer,resolution,kind='linear')
        else:
            self.apodizer = 1.0
            self.path_apodizer = ''
            
        #----- MODAL BASIS FOR TELESCOPE ABERRATIONS
        if path_statModes!='' and ospath.isfile(path_statModes) == True and re.search(".fits",path_statModes)!=None:                
            statModes = np.asarray(fits.getdata(path_statModes))
            s1,s2,s3 = statModes.shape
            if s1 != s2: # mode on first dimension
                tmp = np.transpose(statModes,(1,2,0))
            else:
                tmp = statModes
                    
            self.nModes = tmp.shape[-1]
            self.statModes = np.zeros((resolution,resolution,self.nModes))
                
            for k in range(self.nModes):
                mode = FourierUtils.interpolateSupport(tmp[:,:,k],resolution,kind='linear')
                if pupilAngle !=0:
                    mode = rotate(mode,pupilAngle,reshape=False)
                self.statModes[:,:,k] = mode
        else:
            self.statModes = None
            self.nModes = 0
            self.path_statModes = ''
            
        if self.verbose:
            self
                   
            
    def __repr__(self):
        s = "___TELESCOPE___\n -------------------------------------------------------------------------------- \n"
        s += '. Aperture diameter \t:%.2fm \n'%(self.D)
        s += '. Central obstruction \t:%.2f%s \n'%(self.obsRatio*1e2,'%')
        s += '. Collecting area \t:%.2fm^2\n'%(self.area)
        s += '. Pupil resolution \t:%dX%d pixels'%(self.resolution,self.resolution)
        if self.path_pupil != '':
            s+= '\n. User-defined pupil at :\n ' + self.path_pupil
        if self.path_apodizer != '':
            s+= '\n. User-defined amplitude apodizer at :\n ' + self.path_apodizer
        if self.path_static_on != '':
            s+= '\n. User-defined on-axis static aberrations map at :\n ' + self.path_static_on
        if self.path_static_off != '':
            s+= '\n. User-defined off-axis static aberrations maps at :\n ' + self.path_static_off
        if self.path_statModes != '':
            s+= '\n. User-defined modes of static aberrations at :\n ' + self.path_statModes
        s = s +"\n--------------------------------------------------------------------------------\n"
        return s
        
    
    def displayPupil(self):
        """DISPLAYPUPIL Display the telescope pupil
        """
               
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(self.pupil)
        plt.xlabel("Pixels")
        plt.ylabel("Pixels")
        plt.title("Telescope pupil")
            