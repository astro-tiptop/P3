#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:32:19 2021

@author: omartin
"""
import numpy as np
import scipy.interpolate as interp        
import scipy.ndimage as scnd

def interpolateSupport(image,nRes,kind='spline'):
    
    # Define angular frequencies vectors
    nx,ny = image.shape
        
    if np.isscalar(nRes):
        mx = my = nRes
    else:        
        mx = nRes[0]
        my = nRes[1]
                   
            
    if kind == 'nearest':
        tmpReal = scnd.zoom(np.real(image),min([mx/nx,my/ny]),order=0)
        if np.any(np.iscomplex(image)):
            tmpImag = scnd.zoom(np.imag(image),min([mx/nx,my/ny]),order=0)
            return tmpReal + complex(0,1)*tmpImag
        else:
            return tmpReal
    else:        
        
        
        # Initial frequencies grid    
        if nx%2 == 0:
            uinit = np.linspace(-nx/2,nx/2-1,nx)*2/nx
        else:
            uinit = np.linspace(-np.floor(nx/2),np.floor(nx/2),nx)*2/nx
        if ny%2 == 0:
            vinit = np.linspace(-ny/2,ny/2-1,ny)*2/ny
        else:
            vinit = np.linspace(-np.floor(ny/2),np.floor(ny/2),ny)*2/ny    
             
        # Interpolated frequencies grid                  
        if mx%2 == 0:
            unew = np.linspace(-mx/2,mx/2-1,mx)*2/mx
        else:
            unew = np.linspace(-np.floor(mx/2),np.floor(mx/2),mx)*2/mx
        if my%2 == 0:
            vnew = np.linspace(-my/2,my/2-1,my)*2/my
        else:
            vnew = np.linspace(-np.floor(my/2),np.floor(my/2),my)*2/my
                   
        # Interpolation
    
        if kind == 'spline':
            # Surprinsingly v and u vectors must be shifted when using
            # RectBivariateSpline. See:https://github.com/scipy/scipy/issues/3164
            tmpReal = interp.fitpack2.RectBivariateSpline(vinit, uinit, np.real(image))
            tmpImag = interp.fitpack2.RectBivariateSpline(vinit, uinit, np.imag(image))
        else:
            tmpReal = interp.interp2d(uinit, vinit, np.real(image),kind=kind)
            tmpImag = interp.interp2d(uinit, vinit, np.imag(image),kind=kind)
    
        if np.any(np.iscomplex(image)):
            return tmpReal(unew,vnew) + complex(0,1)*tmpImag(unew,vnew)
        else:
            return tmpReal(unew,vnew)