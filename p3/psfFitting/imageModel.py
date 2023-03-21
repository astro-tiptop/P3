#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:56:54 2021

@author: omartin
"""
import numpy as np

def imageModel(psf4D, spectralStacking = True, spatialStacking = True, saturation=np.inf):
    """
    Function to model an image of a stellar population fields from the psfModelInst class
    The model is: 
           î[x,y,l] = sum_i F_i x F[OTF_i[l]x FFTPHASOR[dx_i,dy_i]] + bkg_0
    where
        - l is wavelength
        - F_i, dx_i and dy_i are the photometry/astrometry for the ith source
        - OTF_i is the instrument Optical transfer function at the direction of the ith source
        - bkg_0 is a constant background
        - FFTPHASOR is the Fourier phasor to shift the PSF within the image
        - F[x] is the 2D Fourier transform of x.
        
    INPUTS:
        - psf4D: -  REQUIRED -  4D numpy array delivered by the P3 models: 
            - fourierModel
            - psfao21
            - psfR
        - Fluxes: - REQUIRED - list of flux values for each source
        - Positions: - REQUIRED - list of sources directions
        - Background: - OPTIONAL - background value
        - nPix: - OPTIONAL - size of the image if different from the OTF
        - antiAliasing: - OPTIONAL - if true, the OTF is zero-padded to mitigate FFT aliasing issues.
    """
    
    # MANAGE THE IMAGE SIZE
    #if nPix == None:
    #    nPix = psfModelInst.freq.nPix
    #    mode = 'psf-like'
    #else:
    #    if nPix > psfModelInst.freq.nPix:
    #        mode = 'wide-field'
    #    else:
    #        mode = 'cropping'
    
    # MANAGE THE DATA MODEL : 4D ARRAYS, CUBE OR IMAGE
    if (spectralStacking == False) and (spatialStacking == False):
        im = np.squeeze(psf4D)
    elif (spectralStacking == True) and (spatialStacking == False):
        im = np.squeeze(psf4D.sum(axis=3))
    elif (spectralStacking == False) and (spatialStacking == True):
        im = np.squeeze(psf4D.sum(axis=2))
    else:
        im = np.squeeze(psf4D.sum(axis=(2,3)))
        
    if saturation < np.inf:
        im[np.where(im>=saturation)] = saturation
                
    return im

                        
                        
        
        
        