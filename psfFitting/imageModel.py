#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:56:54 2021

@author: omartin
"""
import numpy as np
import numpy.fft as fft

def imageModel(psfModelInst, Fluxes, Positions, Background=0.0, spectralStacking = True, spatialStacking = True, nPix = None, antiAliasing=False):
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
        - psfModelInst: -  REQUIRED -  instance of one of the following PSF model: 
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
    if nPix is None:
        nPix = psfModelInst.freq.nPix
        mode = 'psf-like'
    else:
        if nPix > psfModelInst.freq.nPix:
            mode = 'wide-field'
        else:
            mode = 'cropping'
    
    # MANAGE THE DATA MODEL : 4D ARRAYS, CUBE OR IMAGE
    if (spectralStacking == False) and (spatialStacking == False):
        im = np.zeros((psfModelInst.ao.src.nSrc,psfModelInst.freq.nWvl,nPix,nPix))
    elif (spectralStacking == True) and (spatialStacking == False):
        im = np.zeros((psfModelInst.ao.src.nSrc,nPix,nPix))
    elif (spectralStacking == False) and (spatialStacking == True):
        im = np.zeros((psfModelInst.freq.nWvl,nPix,nPix))
    else:
        im = np.zeros((nPix,nPix))
            

    # POPULATE THE IMAGE        
    for iSrc in range(psfModelInst.ao.src.nSrc):
        # COMPUTE THE FFT PHASOR
        dx = Positions[0][iSrc]
        dy = Positions[1][iSrc]
        if dx !=0 or dy!=0:
            fftPhasor = np.exp(np.pi*complex(0,1)*(psfModelInst.freq.U_*dx + psfModelInst.freq.V_*dy))
        else:
            fftPhasor = 1
                        
                        
        # COMPUTE THE PSF
        psf_i = np.real(fft.fftshift(fft.ifft2(fft.fftshift(otf * fftPhasor))))
        
        
        