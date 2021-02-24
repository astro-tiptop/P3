#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:21:10 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
import math

#%% CLASS DEFINITION
class zernike:
    ''' Zernike polynomials
       
    obj = zernike(j,nPixel) creates a Zernike polynomials
    object from the modes vector and the the number of pixel across the
    polynomials diameter
        
    Example:
    zern = zernike([1,2,3,4,5,6],32);
    Computes Zernike polynomials from piston to astigmastism on an
    pupil sampled on a 32x32 grid 
    '''


    def __init__(self,jIndex,resolution,D=[],pupil=[], unitNorm=False,radius=[],angle=[],cobs=0):
        
        # PARSING INPUTS
        if type(jIndex) != list:
            print('Error : you must provide a list for the Noll indexes')
            return
        
        if not jIndex or not resolution:
            print('Error : you must provide valid inputs arguments')
            
        self.jIndex     = jIndex
        self.resolution = resolution
        self.D          = D
        self.cobs       = cobs
        self.pupil      = pupil
        self.radius     = radius
        self.angle      = angle
        
        # DEFINE GEOMETRY
        if not self.radius:
            u           = self.resolution
            x1D         = 2*np.linspace(-(u-1)/2,(u-1)/2,u)/u
            x2D, y2D    = np.meshgrid(x1D,x1D)
            self.radius = np.hypot(x2D,y2D)
            self.angle  = np.arctan2(y2D,x2D)
        else:
            self.resolution = len(self.radius)
    
        # PUPIL
        if len(self.pupil) == 0:
            self.pupil = np.logical_and(self.radius >= self.cobs,self.radius<=1)
           
        # GET RADIAL/AZIMUTHAL ORDERS
        self.n,self.m = self.findNM()
        self.nollNorm = np.sqrt((2-(self.m==0))*(self.n+1))
        self.coeffs   = np.zeros(self.nModes)
        
        # DEFINING POLYNOMIALS
        self.modes = self.polynomials(unitNorm=unitNorm)
        
        
    def findNM(self,jIndex=[]):
        '''Get the Zernike index from a Noll index.
        Parameters
        -------
        jIndex : int, the list of Noll's indexes.
        Returns
        -------
        n : int, The radial Zernike order.
        m : int. The azimuthal Zernike order.
        '''
        
        
        if not jIndex:
            jIndex = self.jIndex
        
        if type(jIndex) != list:
            print('Error : you must provide a list for the Noll indexes')
            return [[],[]]
        
        jIndex = np.asarray(jIndex).astype('int')
        self.nModes   = len(self.jIndex)             
        
        n = np.zeros(self.nModes)
        m = np.zeros(self.nModes)
        
        for k in range(self.nModes):
            n[k] = int(np.sqrt(2 * jIndex[k] - 1) + 0.5) - 1
            if n[k]%2:
                m[k] = 2 * int((2 * (jIndex[k] + 1) - n[k] * (n[k] + 1)) // 4) - 1
            else:
                m[k] = 2 * int((2 * jIndex[k] + 1 - n[k] * (n[k] + 1)) // 4)
            m[k] = m[k]
            
        return n, m
    
    def nModeFromRadialOrder(n):
        '''
        NMODEFROMRADIALORDER Number of Zernike polynomials
        out = zernike.nModeFromRadialOrder(n) returns the number of
        Zernike polynomials (n+1)(n+2)/2 for a given radial order n
        '''
        return (n+1)*(n+2)/2
            
#%% MODES DEFINITION    
    def polynomials(self,unitNorm=False):
        '''
        POLYNOMIALS Zernike polynomials
        fun = polynomes(obj) Computes the Zernike polynomials for the
        Zernike object  sampled on the polar coordinates arrays radius and
        angle. The radius must be normalized to 1.
        '''
        
        def R_fun(r,n,m):
            R = np.zeros(r.shape)
            s1 = int( (n + m)/2 )
            s2 = int( (n - m)/2 )

            for s in range(s2+1):
                ff = math.factorial(s) * math.factorial(max(0,s1-s)) * math.factorial(max(0,s2-s))
                ff = math.factorial(int(n-s))/ff
                R += (-1)**s * ff * r**(n-2*s)
                
            return R
        
        if len(self.radius)==0 or len(self.angle)==0 :
            return []
        else:
            nv      = self.n
            mv      = self.m
            nf      = self.nModes
            pupLog  = self.pupil
            modes   = np.zeros((nf,self.resolution,self.resolution))
            r       = self.radius[pupLog]
            o       = self.angle[pupLog]
            
            # Null azimuthal order
            ind_m = np.argwhere(mv==0)
            if len(ind_m) > 0:
                for cpt in ind_m:
                    n = nv[cpt]
                    m = mv[cpt]
                    modes[int(cpt),pupLog] = np.sqrt(n+1)*R_fun(r,n,m)
                
            mod_mode = np.asarray(self.jIndex).astype('int')%2
            
            # Even polynomes
            ind_m = np.argwhere(np.logical_and(np.logical_not(mod_mode),mv)) 
            if len(ind_m) > 0:
                for cpt in ind_m:
                    n = int(nv[cpt])
                    m = int(mv[cpt])
                    modes[int(cpt),pupLog] = np.sqrt(n+1)*R_fun(r,n,m)*np.sqrt(2)*np.cos(m*o)
                
           # Odd polynomes
            ind_m = np.argwhere(np.logical_and(mod_mode,mv))
            if len(ind_m) > 0:
                for cpt in ind_m:
                    n = int(nv[cpt])
                    m = int(mv[cpt])
                    modes[int(cpt),pupLog] = np.sqrt(n+1)*R_fun(r,n,m)*np.sqrt(2)*np.sin(m*o)     
                
        if unitNorm:
            modes = modes * np.diag(1/self.nollNorm)
        
        return modes    
            
        