#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 13:57:21 2021

@author: omartin
"""

import numpy as np
from p3.aoSystem.optics import optics
from p3.aoSystem.detector import detector
from p3.aoSystem.processing import processing

class sensor:
    """
    Wavefront sensor class. This class is instantiated through three sub-classes:\
    optics, detector and processing.
    """
    
    def __init__(self,pixel_scale,fov,binning=1,spotFWHM=[0.0,0.0],\
                 nph=np.inf,bandwidth=0.0,transmittance=[1.0],dispersion=[[0.0],[0.0]],\
                 ron=0.0,gain=1.0,dark=0.0,sky=0.0,excess=1.0, \
                 wfstype='Shack-Hartmann',nL=[1],dsub=[1],nSides=4,modulation=None,\
                 algorithm='wcog', algo_param=[5,0,0], noiseVar=[None], tag='WAVEFRONT SENSOR'):
        
                
        self.wfstype = wfstype
        self.nWfs = len(nL)
        self.tag  = tag
        
        # optics class
        self.optics = [optics(nL=nL[k],dsub=dsub[k],nSides=nSides,wfstype=wfstype,modulation=modulation) for k in range(self.nWfs)]
        
        # detector class
        self.detector = detector(pixel_scale,fov,binning=binning,spotFWHM=spotFWHM,\
                 nph=nph,bandwidth=bandwidth,transmittance=transmittance,dispersion=dispersion,\
                 ron=ron,gain=gain,dark=dark,sky=sky,excess=excess)
        
        # processing class
        self.processing = processing(algorithm=algorithm,settings=algo_param,noiseVar=noiseVar)
    
    def __repr__(self):
        
        if self.nWfs == 1:
            s = '__ ' + self.tag + ' __\n' + '--------------------------------------------- \n'
        else:
            s = '__ '+str(self.nWfs) + ' WAVEFRONT SENSORS __\n' + '--------------------------------------------- \n'
        s += self.optics[0].__repr__() + '\n'
        s+= self.detector.__repr__() + '\n'
        s+= self.processing.__repr__() +'\n'
        s+= '---------------------------------------------\n'
        return s
        
    def NoiseVariance(self,r0,wvl):
        
        rad2arcsec = 3600 * 180 / np.pi

        # parsing inputs
        varNoise    = np.zeros(self.nWfs)
        nph         = np.array(self.detector.nph)
        pixelScale  = self.detector.psInMas/1e3 # in arcsec
        ron         = self.detector.ron
        for k in range(self.nWfs):
            # pixel per subaperture, N_s in Thomas et al. 2006
            # nPix**2 is the total number of pixels used in the CoG calculation
            nPix = self.detector.fovInPix
            dsub = self.optics[k].dsub
            
            if self.wfstype.upper() == 'SHACK-HARTMANN':
                # spot FWHM in pixels and without turbulence, N_samp in Thomas et al. 2006
                # The condition nD = 2 corresponds to the Nyquist sampling of the spots
                nD = max(1,rad2arcsec * wvl/dsub /pixelScale) 

                # The full width at half-maximum (FWHM) of the spot, N_T in Thomas et al. 2006
                # For diffraction-limited spots nT = nD = 2
                nT = max(1,np.hypot(max(self.detector.spotFWHM[0][0:2])/1e3,rad2arcsec*wvl/r0)/pixelScale)

                # for WCoG, Nw is the weighting function FWHM in pixel
                nW = self.processing.settings[0]
                if nW < nT:
                    print('WARNING: weighting function FWHM of the WCoG is smaller than the FWHM of the spot, forcing it to have the same size.')
                    nW = nT
              
                # tCoG parameters
                th = self.processing.settings[1]
                new_val_th = self.processing.settings[2]
                
                # read-out noise calculation & photo-noise calculation
                # from Thomas et al. 2006
                if self.processing.algorithm == 'cog':
                    varRON  = np.pi**2/3 * (ron**2 /nph[k]**2) * (nPix**2/nD)**2
                    varShot  = np.pi**2/(2*np.log(2)*nph[k]) * (nT/nD)**2
                if self.processing.algorithm == 'tcog':
                    # Here we consider that the pixel used in the computation
                    # are the ones where the PSF is above the 0.5 w.r.t. the maximum value,
                    # so, nPix**2 is subsituted by np.ceil(nT**2*np.pi/4)
                    varRON  = np.pi**2/3 * (ron**2 /nph[k]**2) * (np.ceil(nT**2*np.pi/4)/nD)**2
                    varShot  = np.pi**2/(2*np.log(2)*nph[k]) * (nT/nD)**2
                if self.processing.algorithm == 'wcog':
                    varRON  = np.pi**3/(32*np.log(2)**2) * (ron**2 /nph[k]**2) * (nT**2+nW**2)**4/(nD**2*nW**4)
                    varShot  = np.pi**2/(2*np.log(2)*nph[k]) * (nT/nD)**2 * (nT**2+nW**2)**4/((2*nT**2+nW**2)**2*nW**4)
                if self.processing.algorithm == 'qc':
                    if nT > nD:
                        k = np.sqrt(2*np.pi) * (nT/(2*np.sqrt(2*np.log(2))) / nD)
                    else:
                        k = 1
                    varRON  = k *  4*np.pi**2 * (ron/nph[k])**2
                    varShot  = k * np.pi**2/nph[k]
                    
                if varRON.any() > 3:
                    print('The read-out noise variance is very high (%.1f >3 rd^2), there is certainly smth wrong with your inputs, set to 0'%(varRON))
                    varRON = 0
                    
                if varShot.any() > 3:
                    print('The shot noise variance is very high (%.1f >3 rd^2), there is certainly smth wrong with your inputs, set to 0'%(varShot))
                    varShot = 0
                    
            if self.wfstype.upper() == 'PYRAMID':
                varRON  = 4*ron**2/np.mean(nph)**2
                varShot = nph[k]/np.mean(nph)**2

            varNoise[k] = varRON + self.detector.excess * varShot
        
        return varNoise
