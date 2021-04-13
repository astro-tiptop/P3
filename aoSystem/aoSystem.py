#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:42:49 2021

@author: omartin
"""

# IMPORTING PYTHON LIBRAIRIES
import os.path as ospath
from configparser import ConfigParser
import numpy as np

# IMPORTING P3 MODULES
import fourier.FourierUtils as FourierUtils
from aoSystem.telescope import telescope
from aoSystem.atmosphere import atmosphere
from aoSystem.source import source
from aoSystem.deformableMirror import deformableMirror
from aoSystem.detector import detector

#INIT 

class aoSystem():
    
    

    def __init__(self,path_ini,nLayer=None):
                            
        self.error = False
        # verify if the file exists
        if ospath.isfile(path_ini) == False:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The .ini file does not exist\n')
            self.error = True
            return
        
        # open the .ini file
        config = ConfigParser()
        config.optionxform = str
        config.read(path_ini)
            
        #%% Telescope
        #----- grabbing main parameters
        if config.has_option('telescope','TelescopeDiameter'):
            D = eval(config['telescope']['TelescopeDiameter'])
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the telescope diameter\n')
            self.error = True
            return
        
        if config.has_option('telescope','ZenithAngle'):
            zenithAngle = eval(config['telescope']['ZenithAngle'])
        else:
            zenithAngle = 0.0
            
        if config.has_option('telescope','ObscurationRatio'):
            obsRatio = eval(config['telescope']['ObscurationRatio'])
        else:
            obsRatio = 0.0
        
        if config.has_option('telescope','Resolution'):
            nPup = eval(config['telescope']['Resolution'])
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the pupil resolution\n')
            self.error = True
            return
            
        #----- PUPIL
        if config.has_option('telescope','PathPupil'):
            path_pupil = eval(config['telescope']['PathPupil'])
        else:
            path_pupil = ''
                  
        if config.has_option('telescope','PupilAngle'):
            pupilAngle = eval(config['telescope']['PupilAngle'])
        else:
            pupilAngle = 0.0
        
        if config.has_option('telescope','PathStatic'):
            path_static = eval(config['telescope']['PathStatic'])
        else:
            path_static = None       
                         
        #----- APODIZER
        if config.has_option('telescope','PathApodizer'):
            path_apodizer = eval(config['telescope']['PathApodizer'])
        else:
            path_apodizer = ''
                
        #----- TELESCOPE ABERRATIONS
        if config.has_option('telescope', 'PathStatModes'):
            path_statModes = eval(config['telescope']['PathStatModes'])
        else:
            path_statModes = ''
            
        #----- class definition     
        self.tel = telescope(D,nPup,zenith_angle=zenithAngle,obsRatio=obsRatio,\
                        pupilAngle=pupilAngle,path_pupil=path_pupil,path_static=path_static,\
                        path_apodizer=path_apodizer,path_statModes=path_statModes)                     

        #%% Atmosphere
        
        if config.has_option('atmosphere','AtmosphereWavelength'):
            wvlAtm = eval(config['atmosphere']['AtmosphereWavelength']) 
        else:
            wvlAtm = 500e-9
        
        if config.has_option('atmosphere','Seeing'):
            r0 = 0.976*wvlAtm/eval(config['atmosphere']['Seeing'])*3600*180/np.pi 
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the seeing\n')
            self.error = True
            return
        
        if config.has_option('atmosphere','L0'):
            L0 = eval(config['atmosphere']['L0']) 
        else:
            L0 = 25
            
        if config.has_option('atmosphere','Cn2Weights'):
            weights = eval(config['atmosphere']['Cn2Weights']) 
        else:
            weights = [1.0]
        
        if config.has_option('atmosphere','Cn2Heights'):
            heights = eval(config['atmosphere']['Cn2Heights']) 
        else:
            heights = [0.0]
            
        if config.has_option('atmosphere','WindSpeed'):
            wSpeed = eval(config['atmosphere']['WindSpeed']) 
        else:
            wSpeed = [10.0]
            
        if config.has_option('atmosphere','WindDirection'):
            wDir = eval(config['atmosphere']['WindDirection']) 
        else:
            wDir = [0.0]

        #-----  verification
        if not (len(weights) == len(heights) == len(wSpeed) == len(wDir)):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of atmospheric layers is not consistent in the parameters file\n')
            self.error = True
            return
        #----- class definition
        self.atm = atmosphere(wvlAtm,r0,weights,heights/np.cos(zenithAngle*np.pi/180),wSpeed,wDir,L0)            
        
        #%%  SCIENCE SOURCES
        
        if config.has_option('sources_science','Wavelength'):
            wvlSrc     = np.array(eval(config['sources_science']['Wavelength']))
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the wavelength of the science source\n')
            self.error = True
            return
        
        if config.has_option('sources_science','Zenith'):
            zenithSrc = eval(config['sources_science']['Zenith']) 
        else:
            zenithSrc = [0.0]
            
        if config.has_option('sources_science','Azimuth'):
            azimuthSrc = eval(config['sources_science']['Azimuth']) 
        else:
            azimuthSrc = [0.0]
            
        #----- verification
        if len(zenithSrc) != len(azimuthSrc):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of scientific sources is not consistent in the parameters file\n')
            self.error = True
            return
        #----- class definition
        self.src  = source(wvlSrc,zenithSrc,azimuthSrc,tag="SCIENCE SOURCE",verbose=True)   
 
        #%%  GUIDE STARS 
        if config.has_option('sources_HO','Wavelength'):
            wvlGs     = np.unique(np.array(eval(config['sources_HO']['Wavelength'])))
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the wavelength of the science source\n')
            return 0
        
        if config.has_option('sources_HO','Zenith'):
            zenithGs = eval(config['sources_HO']['Zenith']) 
        else:
            zenithGs = [0.0]
            
        if config.has_option('sources_HO','Azimuth'):
            azimuthGs = eval(config['sources_HO']['Azimuth']) 
        else:
            azimuthGs = [0.0]
            
        if config.has_option('sources_HO','Height'):
            heightGs  = eval(config['sources_HO']['Height']) 
        else:
            heightGs = 0.0
                         
        # ----- verification
        if len(zenithGs) != len(azimuthGs):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of guide stars for high-order sensing is not consistent in the parameters file\n')
            self.error = True
            return
        # ----- creating the source class
        if heightGs == 0:
            self.ngs = source(wvlGs,zenithGs,azimuthGs,height=heightGs,tag="NGS",verbose=True)   
            self.lgs = None
        else:
            self.lgs = source(wvlGs,zenithGs,azimuthGs,height=heightGs,tag="LGS",verbose=True)   
            if (not config.has_section('sources_LO')) | (not config.has_section('sources_LO')):
                print('%%%%%%%% WARNING %%%%%%%%')
                print('No information about the tip-tilt star can be retrieved\n')
                self.ngs = None
            else:
                if config.has_option('sources_LO','Wavelength'):
                    wvlGs = np.unique(np.array(eval(config['sources_LO']['Wavelength'])))
                else:
                    print('%%%%%%%% ERROR %%%%%%%%')
                    print('You must provide a value for the wavelength of the science source\n')
                    self.error = True
                    return
        
                zenithGs   = np.array(eval(config['sources_LO']['Zenith']))
                azimuthGs  = np.array(eval(config['sources_LO']['Azimuth']))
                # ----- verification
                if len(zenithGs) != len(azimuthGs):
                    print('%%%%%%%% ERROR %%%%%%%%')
                    print('The number of guide stars for high-order sensing is not consistent in the parameters file\n')
                    self.error = True
                    return
                self.ngs = source(wvlGs,zenithGs,azimuthGs,tag="NGS",verbose=True)   
                
        #%% SCIENCE INSTRUMENT
        
        if config.has_option('sensor_science','PixelScale'):
            psInMas = eval(config['sensor_science']['PixelScale'])
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the science detector pixel scale\n')
            self.error = True
            return
        
        if config.has_option('sensor_science','FiedOfView'):
            fov = eval(config['sensor_science']['FiedOfView'])
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the science detector field of view\n')
            self.error = True
            return
        
        if config.has_option('sensor_science','spotFWHM'):
            spotFWHM = eval(config['sensor_science']['spotFWHM'])
        else:
            spotFWHM = [[0.0, 0.0]]
            
        if config.has_option('sensor_science','FiedOfView'):
            bw = eval(config['sensor_science']['FiedOfView'])
        else:
            bw = 0.0
        if config.has_option('sensor_science','Transmittance'):
            tr = eval(config['sensor_science']['Transmittance'])
        else:
            tr = [1.0]
        if config.has_option('sensor_science','Dispersion'):
            disp = eval(config['sensor_science']['Dispersion'])
        else:
            disp = [[0.0],[0.0]]
        
        if config.has_option('sensor_science','NumberPhotons'):
            nph = eval(config['sensor_science']['NumberPhotons'])
        else:
            nph = np.inf
        
        if config.has_option('sensor_science','SigmaRON'):
            ron = eval(config['sensor_science']['SigmaRON'])
        else:
            ron = 0.0
        
        if config.has_option('sensor_science','SkyBackground'):
            sky = eval(config['sensor_science']['SkyBackground'])
        else:
            sky = 0.0
        
        if config.has_option('sensor_science','Dark'):
            dark = eval(config['sensor_science']['Dark'])
        else:
            dark = 0.0
        
        self.cam = detector(psInMas,fov,bandwidth=bw,transmittance=tr,dispersion=disp,\
                       spotFWHM=spotFWHM,nph=nph,ron=ron,sky=sky,dark=dark,tag="SCIENCE DETECTOR")
              
        #%% HIGH-ORDER WAVEFRONT SENSOR
        
        if config.has_option('sensor_HO','PixelScale'):
            psInMas = eval(config['sensor_HO']['PixelScale'])
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the HO detector pixel scale\n')
            self.error = True
            return
        
        if config.has_option('sensor_HO','FiedOfView'):
            fov = eval(config['sensor_HO']['FiedOfView'])
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the science detector field of view\n')
            self.error = True
            return
        
        if config.has_option('sensor_HO','spotFWHM'):
            spotFWHM = eval(config['sensor_HO']['spotFWHM'])
        else:
            spotFWHM = [[0.0, 0.0]]
            
        if config.has_option('sensor_HO','NumberPhotons'):
            nph = eval(config['sensor_HO']['NumberPhotons'])
        else:
            nph = np.inf
        
        if config.has_option('sensor_HO','SigmaRON'):
            ron = eval(config['sensor_HO']['SigmaRON'])
        else:
            ron = 0.0
        
        if config.has_option('sensor_HO','SkyBackground'):
            sky = eval(config['sensor_HO']['SkyBackground'])
        else:
            sky = 0.0
        
        if config.has_option('sensor_HO','Dark'):
            dark = eval(config['sensor_HO']['Dark'])
        else:
            dark = 0.0
            
        if config.has_option('sensor_HO','ExcessNoiseFactor'):
            excess = eval(config['sensor_HO']['ExcessNoiseFactor'])
        else:
            excess = 1.0
            
        if config.has_option('sensor_HO','NumberLenslets'):
            NumberLenslets = eval(config['sensor_HO']['NumberLenslets'])
        else:
            NumberLenslets = 20
            
        if config.has_option('sensor_HO','LoopGain'):
            LoopGain = eval(config['sensor_HO']['LoopGain'])
        else:
            LoopGain = 0.5
            
        if config.has_option('sensor_HO','SensorFrameRate'):
            SensorFrameRate = eval(config['sensor_HO']['SensorFrameRate'])
        else:
            SensorFrameRate = 500.0
            
        if config.has_option('sensor_HO','LoopDelaySteps'):
            LoopDelaySteps = eval(config['sensor_HO']['LoopDelaySteps'])
        else:
            LoopDelaySteps = 2
            
        if config.has_option('sensor_HO','NoiseVariance'):
            NoiseVariance = eval(config['sensor_HO']['NoiseVariance'])
        else:
            NoiseVariance = None
            
        if config.has_option('sensor_HO','SlopeAlgorithm'):
            SlopeAlgorithm = eval(config['sensor_HO']['SlopeAlgorithm'])
        else:
            SlopeAlgorithm = 'window'
            
        if config.has_option('sensor_HO','WindowRadiusWCoG'):
            wr = eval(config['sensor_HO']['WindowRadiusWCoG'])
        else:
            wr = 5.0
            
        if config.has_option('sensor_HO','ThresholdWCoG = 0.0'):
            thr = eval(config['sensor_HO']['ThresholdWCoG = 0.0'])
        else:
            thr = 0.0
            
        if config.has_option('sensor_HO','NewValueThrPix'):
            nv = eval(config['sensor_HO']['NewValueThrPix'])
        else:
            nv = 0.0
            
        self.wfs = detector(psInMas,fov,dispersion=disp,nph=nph,spotFWHM=spotFWHM,ron=ron,sky=sky,dark=dark,excess=excess,\
                       nL=NumberLenslets, loopGain=LoopGain, SensorFrameRate=SensorFrameRate,delay=LoopDelaySteps,\
                       noise=NoiseVariance,SlopeAlgorithm=SlopeAlgorithm,wcog=[wr,thr,nv],tag="HO WFS")
        
        #%% TIP-TILT SENSORS
        if config.has_section('sensor_LO'):
        
        
            if config.has_option('sensor_LO','PixelScale'):
                psInMas = eval(config['sensor_LO']['PixelScale'])
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the HO detector pixel scale\n')
                self.error = True
                return
            
            if config.has_option('sensor_LO','FiedOfView'):
                fov = eval(config['sensor_LO']['FiedOfView'])
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the science detector field of view\n')
                self.error = True
                return
            
            if config.has_option('sensor_LO','spotFWHM'):
                spotFWHM = eval(config['sensor_LO']['spotFWHM'])
            else:
                spotFWHM = [[0.0, 0.0]]
                
            if config.has_option('sensor_LO','NumberPhotons'):
                nph = eval(config['sensor_LO']['NumberPhotons'])
            else:
                nph = np.inf
            
            if config.has_option('sensor_LO','SigmaRON'):
                ron = eval(config['sensor_LO']['SigmaRON'])
            else:
                ron = 0.0
            
            if config.has_option('sensor_LO','SkyBackground'):
                sky = eval(config['sensor_LO']['SkyBackground'])
            else:
                sky = 0.0
            
            if config.has_option('sensor_LO','Dark'):
                dark = eval(config['sensor_LO']['Dark'])
            else:
                dark = 0.0
                
            if config.has_option('sensor_LO','ExcessNoiseFactor'):
                excess = eval(config['sensor_LO']['ExcessNoiseFactor'])
            else:
                excess = 1.0
                
            if config.has_option('sensor_LO','NumberLenslets'):
                NumberLenslets = eval(config['sensor_LO']['NumberLenslets'])
            else:
                NumberLenslets = 20
                
            if config.has_option('sensor_LO','LoopGain'):
                LoopGain = eval(config['sensor_LO']['LoopGain'])
            else:
                LoopGain = 0.5
                
            if config.has_option('sensor_LO','SensorFrameRate'):
                SensorFrameRate = eval(config['sensor_LO']['SensorFrameRate'])
            else:
                SensorFrameRate = 500.0
                
            if config.has_option('sensor_LO','LoopDelaySteps'):
                LoopDelaySteps = eval(config['sensor_LO']['LoopDelaySteps'])
            else:
                LoopDelaySteps = 2
                
            if config.has_option('sensor_LO','NoiseVariance'):
                NoiseVariance = eval(config['sensor_LO']['NoiseVariance'])
            else:
                NoiseVariance = None
                
            if config.has_option('sensor_LO','SlopeAlgorithm'):
                SlopeAlgorithm = eval(config['sensor_LO']['SlopeAlgorithm'])
            else:
                SlopeAlgorithm = 'window'
                
            if config.has_option('sensor_LO','WindowRadiusWCoG'):
                wr = eval(config['sensor_LO']['WindowRadiusWCoG'])
            else:
                wr = 5.0
                
            if config.has_option('sensor_LO','ThresholdWCoG = 0.0'):
                thr = eval(config['sensor_LO']['ThresholdWCoG = 0.0'])
            else:
                thr = 0.0
                
            if config.has_option('sensor_LO','NewValueThrPix'):
                nv = eval(config['sensor_LO']['NewValueThrPix'])
            else:
                nv = 0.0
                
            if config.has_option('sensor_LO','Technical_FoV'):
                tech_fov = eval(config['sensor_LO']['Technical_FoV']) 
            else:
                tech_fov = 120
                
            self.tts = detector(psInMas,fov,dispersion=disp,nph=nph,spotFWHM=spotFWHM,ron=ron,sky=sky,dark=dark,excess=excess,\
                       nL=NumberLenslets, loopGain=LoopGain, SensorFrameRate=SensorFrameRate,delay=LoopDelaySteps,\
                       noise=NoiseVariance,SlopeAlgorithm=SlopeAlgorithm,wcog=[wr,thr,nv],tech_fov=tech_fov,tag="TT WFS") 
        else:
            self.tts = None
               
        #%% DEFORMABLE MIRRORS
        if config.has_option('DM','NumberActuators'):
            NumberActuators = eval(config['DM']['NumberActuators'])
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the Dm actuators pitch\n')
            self.error = True
            return
        
        if config.has_option('DM','DmPitchs'):
            DmPitchs = np.array(eval(config['DM']['DmPitchs']))
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('You must provide a value for the Dm actuators pitch\n')
            self.error = True
            return
        
        if config.has_option('DM','InfModel'):
            InfModel = eval(config['DM']['InfModel']) 
        else:
            InfModel = 'gaussian'
            
        if config.has_option('DM','InfCoupling'):
            InfCoupling = eval(config['DM']['InfCoupling']) 
        else:
            InfCoupling = [0.2]
            
        if config.has_option('DM','DmHeights'):
            DmHeights = eval(config['DM']['DmHeights']) 
        else:
            DmHeights = [0.0]
            
        if config.has_option('DM','OptimizationWeight'):
            OptimizationWeight = eval(config['DM']['OptimizationWeight']) 
        else:
            OptimizationWeight = [0.0]
            
        if config.has_option('DM','OptimizationAzimuth'):
            OptimizationAzimuth = eval(config['DM']['OptimizationAzimuth']) 
        else:
            OptimizationAzimuth = [0.0]
            
        if config.has_option('DM','OptimizationZenith'):
            OptimizationZenith = eval(config['DM']['OptimizationZenith']) 
        else:
            OptimizationZenith = [0.0]

         # ----- verification
        if (len(OptimizationZenith) != len(OptimizationZenith)) or (len(OptimizationZenith) != len(OptimizationWeight)):
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of optimization directions is not consistent in the parameters file\n')
            self.error = True
            return
              
        if config.has_option('DM','OptimizationConditioning'):
            OptimizationConditioning = eval(config['DM']['OptimizationConditioning']) 
        else:
            OptimizationConditioning = 100.0
            
        if config.has_option('DM','NumberReconstructedLayers'):
            NumberReconstructedLayers = eval(config['DM']['NumberReconstructedLayers']) 
        else:
            NumberReconstructedLayers = 10
            
        if config.has_option('DM','AoArea'):
            AoArea = eval(config['DM']['AoArea']) 
        else:
            AoArea = 'circle'
            
        # ----- creating the dm class
        self.dms = deformableMirror(NumberActuators,DmPitchs,heights=DmHeights,mechCoupling=InfCoupling,modes=InfModel,\
                              opt_dir=[OptimizationZenith,OptimizationAzimuth],opt_weights=OptimizationWeight,\
                              opt_cond=OptimizationConditioning,n_rec = NumberReconstructedLayers,AoArea=AoArea)
      