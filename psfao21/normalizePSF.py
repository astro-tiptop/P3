#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:57:19 2021

@author: omartin
"""
import aoSystem.fourier.FourierUtils as FourierUtils

def normalizePSF(psf_i,normType=1):
    ''' Returns the normalized PSF :
        normType =1 : flux normalization
        normtype=2 : min-max normalization
    '''
    if normType == 1:
        return psf_i/psf_i.sum()
    if normType == 2:
        return (psf_i-psf_i.min())/(psf_i.max() - psf_i.min())
    if normType == 3:
        Fest = abs(FourierUtils.getFlux(psf_i))
        return psf_i/Fest
    if normType == 4:
        Fest = psf_i[psf_i>0].sum()
        return psf_i/Fest
    else:
        return psf_i/normType
    