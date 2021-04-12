#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:42:49 2021

@author: omartin
"""

# IMPORTING PYTHON LIBRAIRIES
import os.path as ospath
import time
from configparser import ConfigParser
import numpy as np
import re
from astropy.io import fits
from scipy.ndimage import rotate

# IMPORTING P3 MODULES
import fourier.FourierUtils as FourierUtils
from aoSystem.telescope import telescope
from aoSystem.atmosphere import atmosphere
from aoSystem.source import source

#INIT 
def defineAoSystem(psfModelInst,file,aoFilter='circle',nLayer=None):
                    
        tstart = time.time() 
    
        # verify if the file exists
        if ospath.isfile(file) == False:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The .ini file does not exist\n')
            return 0
        
        # open the .ini file
        config = ConfigParser()
        config.optionxform = str
        config.read(file)
        
        #%% Telescope
        psfModelInst.D = eval(config['telescope']['TelescopeDiameter'])
        if config.has_option('telescope','obscurationRatio'):
            obsRatio = eval(config['telescope']['obscurationRatio'])
        else:
            obsRatio = 0.0
        if config.has_option('telescope','zenithAngle'):
            zenithAngle = eval(config['telescope']['zenithAngle'])
        else:
            zenithAngle = 0.0
        if config.has_option('telescope','path_pupil'):
            path_pupil     = eval(config['telescope']['path_pupil'])
        else:
            path_pupil = []
        if config.has_option('telescope','path_static'):
            path_static = eval(config['telescope']['path_static'])
        else:
            path_static = []         
        if config.has_option('telescope','path_apodizer'):
            path_apodizer  = eval(config['telescope']['path_apodizer'])
        else:
            path_apodizer = []           
        #if config.has_option('telescope','CircleDiameter'):
        #    psfModelInst.Dcircle = eval(config['telescope']['CircleDiameter'])
        #else:
        #    psfModelInst.Dcircle = psfModelInst.D          
        if config.has_option('telescope','pupilAngle'):
            psfModelInst.pupilAngle= eval(config['telescope']['pupilAngle'])
        else:
            psfModelInst.pupilAngle= 0.0
    
        #%% Atmosphere
        wvlAtm         = eval(config['atmosphere']['atmosphereWavelength']) 
        r0             = 0.976*wvlAtm/eval(config['atmosphere']['seeing'])*3600*180/np.pi 
        L0             = eval(config['atmosphere']['L0']) 
        weights        = np.array(eval(config['atmosphere']['Cn2Weights']) )
        heights        = np.array(eval(config['atmosphere']['Cn2Heights']) )
        wSpeed         = np.array(eval(config['atmosphere']['wSpeed']) )
        wDir           = np.array(eval(config['atmosphere']['wDir']) )
        #-----  verification
        if not (len(weights) == len(heights) == len(wSpeed) == len(wDir)):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of atmospheric layers is not consistent in the parameters file\n')
            return 0
        #----- compressing
        if nLayer!=None: 
            _,wSpeed = FourierUtils.eqLayers(weights,wSpeed,nLayer)
            _,wDir   = FourierUtils.eqLayers(weights,wDir,nLayer,power=1)
            weights,heights = FourierUtils.eqLayers(weights,heights,nLayer)
            
            
        psfModelInst.atm = atmosphere(wvlAtm,r0,weights,heights/np.cos(zenithAngle*np.pi/180),wSpeed,wDir,L0)            
        
        #%% Sampling and field of view
        psfModelInst.psInMas = eval(config['PSF_DIRECTIONS']['psInMas'])
        psfModelInst.aoFilter= aoFilter
        psfModelInst.nPix    = eval(config['PSF_DIRECTIONS']['psf_FoV'])
        
        #%% PSF directions
        wvlSrc         = np.array(eval(config['PSF_DIRECTIONS']['ScienceWavelength']))
        zenithSrc      = np.array(np.array(eval(config['PSF_DIRECTIONS']['ScienceZenith'])))
        azimuthSrc     = np.array(np.array(eval(config['PSF_DIRECTIONS']['ScienceAzimuth'])))        
        # ----- verification
        psfModelInst.src = []
        if len(zenithSrc) != len(azimuthSrc):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of scientific sources is not consistent in the parameters file\n')
            return 0
        
        # MANAGING WAVELENGTHS
        if config.has_option('POLYCHROMATISM','spectralBandwidth'):
            psfModelInst.wvl_bw = eval(config['POLYCHROMATISM']['spectralBandwidth'])
        else:
            psfModelInst.wvl_bw = [0]
        
        if psfModelInst.wvl_bw != [0]:
            # CASE 1: SINGLE POLYCHROMATIC PSF
            psfModelInst.wvl_dpx = np.array(eval(config['POLYCHROMATISM']['dispersionX']))
            psfModelInst.wvl_dpy = np.array(eval(config['POLYCHROMATISM']['dispersionY']))
            psfModelInst.wvl_tr  = np.array(eval(config['POLYCHROMATISM']['transmittance']))
            if (len(psfModelInst.wvl_dpx) == len(psfModelInst.wvl_dpx)) and (len(psfModelInst.wvl_dpx) == len(psfModelInst.wvl_tr)):
                psfModelInst.nWvl    = len(psfModelInst.wvl_tr)
                psfModelInst.wvlRef  = wvlSrc - psfModelInst.wvl_bw/2
                psfModelInst.wvlMax  = wvlSrc + psfModelInst.wvl_bw/2
                psfModelInst.wvl     = np.linspace(psfModelInst.wvlRef,psfModelInst.wvlMax,psfModelInst.nWvl)
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('The number of spectral bins are not consistent in the parameters file\n')
                return 0
        else:
            # CASE 2 : DATA CUBE OF MONOCHROMATIC PSF
            psfModelInst.wvl     = np.unique(wvlSrc)
            psfModelInst.nWvl    = len(psfModelInst.wvl)
            psfModelInst.wvl_tr  = np.ones(psfModelInst.nWvl)
            psfModelInst.wvl_dpx = np.zeros(psfModelInst.nWvl)
            psfModelInst.wvl_dpy = np.zeros(psfModelInst.nWvl)
            psfModelInst.wvlRef  = psfModelInst.wvl.min()
            
            
        psfModelInst.wvlCen = np.mean(psfModelInst.wvl)
        #PUPIL RESOLUTION
        if config.has_option('telescope','resolution'):
            psfModelInst.nPup = eval(config['telescope']['resolution'])
        else:
            psfModelInst.nPup = 2*int(np.ceil(psfModelInst.nPix*psfModelInst.kRef_/psfModelInst.samp.min()/2))
                
        psfModelInst.nAct    = np.floor(psfModelInst.D/np.array(eval(config['DM']['DmPitchs']))+1)
        psfModelInst.src     = source(wvlSrc,zenithSrc,azimuthSrc,types="SCIENTIFIC STAR",verbose=True)   
        
        #%% UPDATING PUPIL RESOLUTION
        psfModelInst.tel     = telescope(psfModelInst.D,zenithAngle,obsRatio,psfModelInst.nPup,pupilAngle = psfModelInst.pupilAngle,file=path_pupil)                     
    
        # APODIZER
        psfModelInst.apodizer = 1
        if path_apodizer != [] and ospath.isfile(path_apodizer) == True:
             if  re.search(".fits",path_apodizer)!=None :
                psfModelInst.apodizer = fits.getdata(path_apodizer)
                psfModelInst.apodizer = FourierUtils.interpolateSupport(psfModelInst.apodizer,psfModelInst.nPup,kind='linear')
                
        # EXTERNAL STATIC MAP
        psfModelInst.opdMap_ext = 0
        if path_static != [] and ospath.isfile(path_static) == True:
             if  re.search(".fits",path_static)!=None :
                psfModelInst.opdMap_ext = fits.getdata(path_static)
                psfModelInst.opdMap_ext = FourierUtils.interpolateSupport(psfModelInst.opdMap_ext,psfModelInst.nPup,kind='linear')
        
        # STATIC ABERRATIONS
        psfModelInst.statModes = None
        psfModelInst.nModes = 1
        psfModelInst.isStatic = False
        if config.has_option('telescope', 'path_statModes'):
            psfModelInst.path_statModes = eval(config['telescope']['path_statModes'])
        else:
            psfModelInst.path_statModes = []
            
        if psfModelInst.path_statModes:
            if ospath.isfile(psfModelInst.path_statModes) == True and re.search(".fits",psfModelInst.path_statModes)!=None:                
                psfModelInst.statModes = fits.getdata(psfModelInst.path_statModes)
                s1,s2,s3 = psfModelInst.statModes.shape
                if s1 != s2: # mode on first dimension
                    tmp = np.transpose(psfModelInst.statModes,(1,2,0))  
                else:
                    tmp = psfModelInst.statModes
                    
                psfModelInst.nModes = tmp.shape[-1]
                psfModelInst.statModes = np.zeros((psfModelInst.nPup,psfModelInst.nPup,psfModelInst.nModes))
                
                for k in range(psfModelInst.nModes):
                    mode = FourierUtils.interpolateSupport(tmp[:,:,k],psfModelInst.nPup,kind='linear')
                    if psfModelInst.pupilAngle !=0:
                        mode = rotate(mode,psfModelInst.pupilAngle,reshape=False)
                    psfModelInst.statModes[:,:,k] = psfModelInst.tel.pupil * mode
                psfModelInst.isStatic = True
        
        #%% GUIDE STARS
        
        wvlGs      = np.unique(np.array(eval(config['SENSOR_HO']['SensingWavelength_HO'])))
        zenithGs   = np.array(eval(config['GUIDESTARS_HO']['GuideStarZenith_HO']))
        azimuthGs  = np.array(eval(config['GUIDESTARS_HO']['GuideStarAzimuth_HO']))
        heightGs   = eval(config['GUIDESTARS_HO']['GuideStarHeight_HO'])
        # ----- verification
        if len(zenithGs) != len(azimuthGs):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of guide stars for high-order sensing is not consistent in the parameters file\n')
            return 0
        # ----- creating the source class
        if heightGs == 0:
            psfModelInst.ngs = source(wvlGs,zenithGs,azimuthGs,height=heightGs,types="NGS",verbose=True)   
            psfModelInst.lgs = None
        else:
            psfModelInst.lgs = source(wvlGs,zenithGs,azimuthGs,height=heightGs,types="LGS",verbose=True)   
            if (not config.has_section('GUIDESTARS_LO')) | (not config.has_section('SENSOR_LO')):
                print('%%%%%%%% WARNING %%%%%%%%')
                print('No information about the tip-tilt star can be retrieved\n')
                psfModelInst.ngs = None
            else:
                wvlGs      = np.unique(np.array(eval(config['SENSOR_LO']['SensingWavelength_LO'])))
                zenithGs   = np.array(eval(config['GUIDESTARS_LO']['GuideStarZenith_LO']))
                azimuthGs  = np.array(eval(config['GUIDESTARS_LO']['GuideStarAzimuth_LO']))
                # ----- verification
                if len(zenithGs) != len(azimuthGs):
                    print('%%%%%%%% ERROR %%%%%%%%')
                    print('The number of guide stars for high-order sensing is not consistent in the parameters file\n')
                    return 0
                psfModelInst.ngs = source(wvlGs,zenithGs,azimuthGs,types="NGS",verbose=True)   
        
        
        psfModelInst.t_getParam = 1000*(time.time() - tstart)
        return 1