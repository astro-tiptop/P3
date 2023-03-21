#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:27:18 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from distutils.spawn import find_executable
from p3.aoSystem.FourierUtils import inpolygon
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
#%% CLASS DEFINITION

class segment:
    """ 
    Create a telescope segment giving number of sides, size in px, etc...
    """
    
    @property
    def pixelRatio(self):
        """"""
        return self.nPixels/(2*self.radius)
    
    def __init__(self,nSides,radius,nPixels,segAngle=0,posX=0,\
                 posY=0,angleError=0,posXError=0,posYError=0,phaseError=0,\
                 reflexivity=1,shrinkCoeff=1):
        
        # PARSING INPUTS
        self.nSides     = nSides
        self.radius     = radius        # in meters
        self.nPixels    = nPixels
        self.segAngle   = segAngle      # in degrees
        self.angleError = angleError    # in degrees
        self.phaseError = phaseError    # in rad
        self.reflexivity= reflexivity   # from 0 to 1
        self.shrinkCoeff= shrinkCoeff     # from 0 to 1
        
        # ADDITIONNAL INFORMATION
        self.posX       = posX          # in meters
        self.posY       = posY          # in meters
        self.posXError  = posXError     # in meters
        self.posYError  = posYError     # in meters
        
        if self.nSides == 0:
            self.nSides = np.inf
            
        # Create the segment
        self.matrix, self.segShape = self.makeSegment()

        
    def __repr__(self):
        if self.nSides == np.Inf:
            s = ('_________ SEGMENT _________\n' + 'nSides\t\t\t= %s\nradius\t\t\t= %.1f m\n#Pixels\t\t\t= %d\nPixel scale\t\t= %.1f mm/pix\nCoordinates(m)\t= (%.1f,%.1f)'%('Inf',self.radius,self.nPixels,1e3/self.pixelRatio,self.posX,self.posY)) 
        else:
            s = ('_________ SEGMENT __________\n' + 'nSides\t\t\t= %d\nradius\t\t\t= %.1f m\n#Pixels\t\t\t= %d\nPixel scale\t\t= %.1f mm/pix\nCoordinates (m)\t= (%.1f,%.1f)'%(self.nSides,self.radius,self.nPixels,1e3/self.pixelRatio,self.posX,self.posY)) 
        return s
   
    def makeSegment(self):
        
        nP = self.nPixels
        i  = complex(0,1)
        # Grid
        x1D      = np.linspace(-nP,nP,nP)
        x2D, y2D = np.meshgrid(x1D,x1D)
        
        # define segment borders
        if self.nSides == np.inf: #circle case
            segShape= np.hypot(x2D,y2D) <= self.sizeError*(nP + 0.1)
        else:
            theta   = np.linspace(0,2*np.pi,self.nSides+1)
            ang     = theta + (self.segAngle + self.angleError) * np.pi/180
            siz     = (nP+1)*self.shrinkCoeff
            xv      = siz*np.cos(ang)
            yv      = siz*np.sin(ang)
            segShape= inpolygon(x2D, y2D, xv, yv)
        # segment
        segShape = segShape*1
        matrix = segShape * self.reflexivity * np.exp(1*i*self.phaseError)
                
        return [matrix, segShape]
     
    def applyPhase(self, modesCube, modesCoeffs):
        
        self.modesCoeffs = np.asarray(modesCoeffs)
        nM = len(modesCoeffs)
        #if self.modesCoeffs.ndim ==1:
        #    self.modesCoeffs = np.asarray([self.modesCoeffs])
        #nCoeffs,nM       = self.modesCoeffs.shape
        
        if modesCube.ndim == 3:
            if modesCube.shape[0] >= nM:
                phase = 0
                for k in range(nM):
                    phase  += modesCube[k] * self.modesCoeffs[k]
            else:
                print('Number of coefficients doesn''t match the size of the cube of Modes')
                return [[],[]]   
        else:
            phase = modesCube * self.modesCoeffs
        matrix = self.segShape * self.reflexivity * np.exp(1*complex(0,1)*phase)
        return [matrix, phase]
              
        
    def applyReflexivity(self, R):
        '''
            Multiply the amplitude by the factor R
        '''
        matrix = self.segShape * R * np.exp(1*complex(0,1) * self.phaseError)
        return [matrix, R]     
    
    def displaySegment(self):
        
        fig,axs = plt.subplots(1,2,figsize=(10,10), constrained_layout=True)
        
        pcm = axs[0].imshow(np.abs(self.matrix),extent=[-self.radius,self.radius,-self.radius,self.radius])
        axs[0].set_title('Reflexivity')
        axs[0].set_xlabel('Position (m)')
        axs[0].set_ylabel('Position (m)')
        fig.colorbar(pcm,ax=axs[0],shrink=0.75)
        
        pcm = axs[1].imshow(np.angle(self.matrix),extent=[-self.radius,self.radius,-self.radius,self.radius])
        axs[1].set_title('Phase aberrations')
        axs[1].set_xlabel('Position (m)')
        axs[1].set_ylabel('Position (m)')
        fig.colorbar(pcm,ax=axs[1],shrink=0.75)
        plt.show()