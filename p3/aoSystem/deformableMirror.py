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
        return [int(self.validActuator[kObj].sum()) for kObj in range(self.nDMs)]    
    @property
    def nControlledRadialOrder(self):
        # nZernike = 0.5 * (nActu + 1) **2
        return np.sqrt(2*np.array(self.nValidActuator)).astype(int) - 1
    @property
    def nControlledRadialOrderTipTiltExcluded(self):
        return int(np.sqrt(2*np.array(self.nValidActuator))-3)
    
    def __init__(self,nActu1D,pitch,heights=[0.0],mechCoupling=[0.2],modes='gaussian',opt_dir=[[0.0],[0.0]], opt_weights=[1.0],\
                 opt_cond=1e2, n_rec=None, validActuator=None,offset = [[0,0]],AoArea='circle',resolution=None):
        # PARSING INPUTS
        if np.isscalar(nActu1D):
            nActu1D = [int(nActu1D)]
            
        # geometry
        self.nActu1D      = np.array(nActu1D)
        self.pitch        = np.array(pitch)
        self.heights      = np.array(heights)
        self.nDMs         = len(pitch)
        
        if (len(self.nActu1D) != self.nDMs) or (len(self.heights) != self.nDMs):
            raise ValueError("Please have the same size for nActu1D, pitch and heights input lists.")
            
        # influence function
        self.mechCoupling = np.array(mechCoupling)
        if len(self.mechCoupling) < self.nDMs:
            self.mechCoupling = np.full(shape=self.nDMs,fill_value=self.mechCoupling[0],dtype=type(self.mechCoupling))
            print("The first value of mechanical coupling ", self.mechCoupling[0] ," is set for all DMs")
        
        elif len(self.mechCoupling) > self.nDMs:
            self.mechCoupling = self.mechCoupling[0:self.nDMs]
            print("The list of mechanical coupling values is truncated up to the number od DMs.")
            
        self.modes        = modes
        self.offset       = offset
        self.influenceCentre=0
        self.resolution   = resolution
        
        # reconstruction
        self.nRecLayers   = n_rec
        self.opt_cond     = opt_cond
        self.AoArea       = AoArea

        # optimization
        self.opt_dir      = np.array(opt_dir)
        self.opt_weights  = opt_weights
        if (len(self.opt_dir[0]) != len(self.opt_weights)):
            raise ValueError("Please define as many weights as optimization directions.")
        
        # DEFINE THE VALID ACTUATOR
        if np.any(validActuator == None):
            self.validActuator = [np.ones((nActu1D[kObj],nActu1D[kObj]),dtype=bool) for kObj in range(self.nDMs)]
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
        m_modes = np.zeros((resolution**2,self.nValidActuator[ndm]))
        indIF = np.arange(0,nIF**2)
        idx = self.validActuator[ndm] == False
        indIF[idx.ravel()] = []
        iIF,jIF = ind2sub((nIF,nIF),indIF)
        kIF = np.arange(0,self.nValidActuator[ndm])        
        wv = wv[:,iIF[kIF]]
        wu = wu[:,jIF[kIF]]
        
        for kIF in np.arange(0,self.nValidActuator[ndm]):
            buffer = np.transpose([wv[:,kIF]])*wu[:,kIF]     
            m_modes[:,kIF] = buffer.ravel() #buffer.A.ravel()
                
            
        return m_modes
                
                
    def __repr__(self):
        """Display object information: prints information about the source object
        """
       
        s = '___ DEFORMABLE MIRROR___\n'
        s += '----------------------------------------------------------------------------------------------\n'
        s += ' Obj\t #Actuators\t Pitch [m]\t Heights [m]\t modes type\t Coupling\t #Controlled modes\n'
        for kObj in range(self.nDMs):
            if self.heights[kObj] == 0:
                s += ' %d\t %d\t\t %.4f\t\t %.1f\t\t %s\t %.2f\t\t %d\n'%(kObj,self.nActu1D[kObj],self.pitch[kObj],
                            float(self.heights[kObj]),self.modes,self.mechCoupling[kObj], self.nValidActuator[kObj])
            else:
                s += ' %d\t %d\t\t %.4f\t\t %.0f\t\t %s\t %.2f\t\t %d\n'%(kObj,self.nActu1D[kObj],self.pitch[kObj],
                            float(self.heights[kObj]),self.modes,self.mechCoupling[kObj], self.nValidActuator[kObj])
        s +='----------------------------------------------------------------------------------------------\n'
        
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