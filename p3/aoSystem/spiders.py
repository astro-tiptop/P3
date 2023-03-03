#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:34:19 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from distutils.spawn import find_executable
from scipy.ndimage import label

import p3.aoSystem.FourierUtils as FourierUtils

#%% DISPLAY FEATURES
mpl.rcParams['font.size'] = 16

if find_executable('tex'): 
    usetex = True
else:
    usetex = False

plt.rcParams.update({
    "text.usetex": usetex,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
    
#%% EXAMPLES

def keckSpiderMask(display=False):
    D            = 10
    cobs         = 0.2375
    widthInMeter = 0.0254
    
    sp = spiders([0,60,120],widthInMeter,600,symetric=True,D=D,cobs=cobs,mskPup=True) 
    if display:
        plt.close('all')
        sp.displaySpiders()
        
    return sp    

def vltSpiderMask(display=False):
    D            = 8.1196
    cobs         = 0.6465 * 2
    widthInMeter = 0.04057
    w            = D*cobs/2 
    
    sp = spiders(np.array([52, 128, 231, 308]),widthInMeter,400,D=D,cobs=cobs,unit='m',mskPup=True,symetric=False, \
                 spidersInitCoor= [(-w,0), (w,0), (w,0), (-w,0)]) 
    if display:
        plt.close('all')
        sp.displaySpiders()
    return sp

#%% CLASS DEFINITION
class spiders:
    """ 
    Create a telescope spyders mask
    """
    
    @property
    def pixelRatio(self):
        """"""
        return self.D/self.nPixels
    
    def __init__(self,spidersAngle, spidersWidth,
                 spidersInitCoor=[], D=1, cobs=0,
                 mskPup=False, symetric=False, getPetal=False, nPixels=0):
        
        # PARSING INPUTS
        self.nSpiders = 0  
        self.spidersAngle = np.asarray(spidersAngle) * np.pi/180
        self.spidersWidth = spidersWidth
        self.spidersInitCoor = np.array(spidersInitCoor)
        self.D = D
        self.cobs = cobs
        self.mskPup = mskPup
        self.symetric = symetric
        self.getPetal = getPetal
        self.nPixels = nPixels
        
        if len(self.spidersInitCoor) == 0:
            self.nSpiders        = len(self.spidersAngle)
            self.spidersInitCoor = np.asarray(list(zip(np.zeros(self.nSpiders),np.zeros(self.nSpiders))))
            
        if len(self.spidersInitCoor) != len(self.spidersAngle):
            print('Error :  you must provide as many coordinates tuples as angle values')
        elif nPixels == 0:
            print('Caution :  spider class instantiated but no resolution given')   
        else:
            self.nSpiders = len(spidersAngle)
            self.makeSpiders(self.nPixels, getPetal=getPetal)
        
    def __repr__(self):
        s = ('___ SPIDERS ___\n' + 'nSpiders= %d\nWidth\t= %.1fm'%(self.nSpiders,self.spidersWidth)) 
        return s
    
    def makeSpiders(self, nPixels, getPetal=False):
        
        # DEFINING THE GEOMETRY
        self.nPixels = nPixels
        self.getPetal= getPetal
        if self.spidersWidth/self.pixelRatio < 1:
            print('Warning : the spiders width is smaller than 1 pixel, aborting')
            self.matrix = np.ones((self.nPixels,self.nPixels))
            return
        
        self.matrix  = np.ones((self.nPixels, self.nPixels))  
        x1D          = np.linspace(-1,1,self.nPixels)
        x2D,y2D      = np.meshgrid(x1D,x1D)
        
        # DEFINING THE SPIDERS MASK
        spCoor = self.spidersInitCoor * 2/self.nPixels/self.pixelRatio
        phi = np.arctan((self.spidersWidth/self.pixelRatio)/(2*self.nPixels))        
        theta  = [-phi, phi, -phi + np.pi, phi + np.pi]
        
        # EXTRUDE  
        xvAll = np.zeros((self.nSpiders,4))
        yvAll = np.zeros((self.nSpiders,4))
        for k in range(self.nSpiders):
            ck = spCoor[k]
            if self.symetric:
                xvAll[k] = ck[0] + 2 * np.cos(theta + self.spidersAngle[k])
                yvAll[k] = ck[1] + 2 * np.sin(theta + self.spidersAngle[k])
            else:
                xvAll[k] = ck[0] + 2 * (np.cos(theta + self.spidersAngle[k]) - np.cos(self.spidersAngle[k]))
                yvAll[k] = ck[1] + 2 * (np.sin(theta + self.spidersAngle[k]) - np.sin(self.spidersAngle[k]))
            # cumulative extrusion
            self.matrix         = self.matrix * (FourierUtils.inpolygon(x2D, y2D, xvAll[k], yvAll[k]) ==0)
            
        #INCLUDE THE PUPIL MASK
        if self.mskPup:
            R = np.hypot(x2D,y2D)*self.D/2
            self.matrix = self.matrix * (R <= self.D/2)*(R>= self.cobs*self.D/2)
        
        if self.matrix.shape[0] != self.nPixels:
            self.matrix = FourierUtils.interpolateSupport(self.matrix,self.nPixels,'linear')
        
        # GET PETAL AREAS
        if self.getPetal:
            a, self.nPetal = label(self.matrix)
            self.matrixPetal = np.zeros((self.nPetal,self.nPixels,self.nPixels))
            for k in range(self.nPetal):
                self.matrixPetal[k] = a * (a==k+1)/(k+1)
        
    def displaySpiders(self):
        plt.figure(figsize=(10,10), constrained_layout=True)
        plt.imshow(self.matrix)
        plt.title('Spiders')
        plt.xlabel('Position (m)')
        plt.ylabel('Position (m)')
