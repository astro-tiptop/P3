#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:59:35 2018

@author: omartin
"""

import numpy as np
import scipy.interpolate as interp

class deformableMirror:
    """
    """
    # DEPENDENT PROPERTY
    @property
    def nValidActuator(self):
        return self.validActuator.sum()
    
    def __init__(self,nActu1D,pitch,heights=[0.0],mechCoupling=[0.2],modes='gaussian',opt_dir=[[0.0],[0.0]], opt_weights=[1.0],\
                 opt_cond=1e2, n_rec=None, validActuator=None,offset = [[0,0]],AoArea='circle',resolution=None):
        # PARSING INPUTS
        if np.isscalar(nActu1D):
            nActu1D = [int(nActu1D)]
            
        self.nActu1D      = np.array(nActu1D)
        self.pitch        = np.array(pitch)
        self.heights      = np.array(heights)
        self.mechCoupling = np.array(mechCoupling)
        self.modes        = modes
        self.offset       = offset
        self.influenceCentre=0
        self.resolution   = resolution
        self.AoArea       = AoArea
        self.nRecLayers   = n_rec
        self.opt_dir      = np.array(opt_dir)
        self.opt_weights  = opt_weights
        self.nDMs         = len(pitch)
        
        # DEFINE THE VALID ACTUATOR
        if np.any(validActuator == None):
            self.validActuator = np.ones((nActu1D[0],nActu1D[0]),dtype=bool)
        else:
            self.validActuator = validActuator
        # DEFINE THE MODES
        if np.isscalar(self.resolution):
            self.modes = self.setInfluenceFunction(self.resolution)
            
    
    def setInfluenceFunction(self,resolution,ndm=0):
        """
            SETINFLUENCEFUNCTION    
        """
        nIF = self.nActu1D[ndm] 
        # defining the spline functions
        c  = 1/np.sqrt(np.log(1/self.mechCoupling[ndm]))
        df = 1e-10
        mx = np.sqrt(-np.log(df)*c**2)
        r  = np.linspace(-mx,mx,1001)
        
        if self.modes == 'gaussian':        
            f  = np.exp(-r**2/c**2)
        elif self.modes == 'xinetics':
            # defining main parameters
            m = 0.180267421;
            p = np.array([2.24506, 6.28464*m**2,-18.1956*m**4,31.2025*m**6,76.9336,-39.7956,m])
            tmp     = -150*p[6]**8 * r** 8
            w       = np.argwhere(r**8 < 1/(3*p[6]**8))
            mask    = 0*tmp
            mask[w] = np.exp(tmp[w])
            # Define sub function
            e  = (p[0] + p[1]*r**2 + p[2]*r**4 + p[3]*r**6)*mask
            re = (abs(r)**e) * p[6]**e
            # Get the influence function model
            f  = np.exp(-p[4]*re)*(1 + p[5]*re)*mask         
              
        spline = interp.BSpline(r*self.pitch, f,3)
        
        # managing the actuators positions
        if self.influenceCentre==0:
            xIF = np.linspace(-1,1,nIF)*(nIF-1)/2*self.pitch[ndm] - self.offset[ndm][0]
            yIF = np.linspace(-1,1,nIF)*(nIF-1)/2*self.pitch[ndm] - self.offset[ndm][1]
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
        index_u     = (u >= -r[len(r)-1]*self.pitch[ndm]) & (u <= r[len(r)-1]*self.pitch[ndm])
        wu[index_u] = spline(u[index_u])
                
        v           = np.transpose([u0])- [yIF]
        wv          = np.zeros((resolution,nIF))
        index_v     = (v >= -r[len(r)-1]*self.pitch[ndm]) & (v <= r[len(r)-1]*self.pitch[ndm])
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
                
                
    def __repr__(self):
        s = "___DEFORMABLE MIRROR___\n ---------------------------------------- \n"
        s+= ".%dX%d actuators deformable mirror with %d controlled actuators"%( self.nActu1D[0],self.nActu1D[0],self.nValidActuator)
        s = s +"\n----------------------------------------\n"
        return s
                            
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