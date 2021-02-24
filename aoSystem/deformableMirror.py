#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:59:35 2018

@author: omartin
"""

import numpy as np
import sys
import scipy.interpolate as interp
import scipy.sparse as sparse

def fprintf(stream, format_spec, *args):
    stream.write(format_spec % args)
    

    
class deformableMirror:
    """
    """
    # DEPENDENT PROPERTY
    @property
    def nValidActuator(self):
        return self.validActuator.sum()
    
    def __init__(self,nActu1D,pitch,mechCoupling,modes='gaussian',validActuator=0,offset = [0,0],resolution=None):
        # PARSING INPUTS
        self.nActu1D      = nActu1D
        self.pitch        = pitch
        self.mechCoupling = mechCoupling
        self.modes        = modes
        self.offset       = offset
        self.influenceCentre=0
        self.resolution   = resolution
        # DEFINE THE VALID ACTUATOR
        if np.isscalar(validActuator):
            self.validActuator = np.ones((nActu1D,nActu1D),dtype=bool)
        else:
            self.validActuator = validActuator
        # DEFINE THE MODES
        if np.isscalar(self.resolution):
            self.modes = self.setInfluenceFunction(self.resolution)
            
    
    def setInfluenceFunction(self,resolution):
        """SETINFLUENCEFUNCTION    
        """
        nIF = self.nActu1D 
        c  = 1/np.sqrt(np.log(1/self.mechCoupling))
        df = 1e-10
        mx = np.sqrt(-np.log(df)*c**2)
        x  = np.linspace(-mx,mx,1001)
        f  = np.exp(-x**2/c**2)
        spline = interp.BSpline(x*self.pitch, f,3)
                
        if self.influenceCentre==0:
            xIF = np.linspace(-1,1,nIF)*(nIF-1)/2*self.pitch - self.offset[0]
            yIF = np.linspace(-1,1,nIF)*(nIF-1)/2*self.pitch - self.offset[1]
            xIF2,yIF2 = np.meshgrid(xIF,yIF)
            self.actuatorCoord = yIF2 + complex(0,1)*np.flip(xIF2,axis=0)                                  
            u0 = np.linspace(-1,1,resolution)*(nIF-1)/2*self.pitch
        else:
            xIF = np.arange(0,nIF)
            yIF = np.arange(0,nIF)
            xIF2,yIF2 = np.meshgrid(xIF,yIF)
            self.actuatorCoord = xIF2 + complex(0,1)*yIF2
            u0 =  np.arange(0,nIF)          
        

        u           = np.transpose([u0])- [xIF]
        wu          = np.zeros((resolution,nIF))
        index_u     = (u >= -x[len(x)-1]*self.pitch) & (u <= x[len(x)-1]*self.pitch)
        wu[index_u] = spline(u[index_u])
                
        v           = np.transpose([u0])- [yIF]
        wv          = np.zeros((resolution,nIF))
        index_v     = (v >= -x[len(x)-1]*self.pitch) & (v <= x[len(x)-1]*self.pitch)
        wv[index_v] = spline(v[index_v])
             
        #m_modes = sparse.lil_matrix((resolution**2,self.nValidActuator))
        m_modes = np.zeros((resolution**2,self.nValidActuator))
        indIF = np.arange(0,nIF**2)
        idx = self.validActuator == False
        indIF[idx.ravel()] = []
        iIF,jIF = ind2sub((nIF,nIF),indIF)
        kIF = np.arange(0,self.nValidActuator)        
        wv = wv[:,iIF[kIF]]
        wu = wu[:,jIF[kIF]]
        
        for kIF in np.arange(0,self.nValidActuator):
            buffer = np.transpose([wv[:,kIF]])*wu[:,kIF]     
            m_modes[:,kIF] = buffer.ravel() #buffer.A.ravel()
                
            
        return m_modes
                
                
                    
    def display(self):
        """ DISPLAY prints information about the deformable mirror
        """
            
        print('___DEFORMABLE MIRROR ___')
        print('----------------------------------------------------------------')
        fprintf(sys.stdout,' %dX%d actuators deformable mirror with %d controlled actuators\n',\
                self.nActu1D,self.nActu1D,self.nValidActuator)
        print('----------------------------------------------------------------\n')
           
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    cols = ind % array_shape[1]
    rows = (ind - cols) / array_shape[0]
    
    return (rows.astype('int'), cols.astype('int'))