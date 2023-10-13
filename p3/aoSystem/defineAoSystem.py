#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:42:49 2021

@author: omartin

# CHECK - Repetition between defineAOSystem and aoSystem
# CHECK - Do the defineAoSystem and aoSystem serve the same purpose? 
"""

# IMPORTING PYTHON LIBRAIRIES
import os.path as ospath
from configparser import ConfigParser
import numpy as np

import yaml

# IMPORTING P3 MODULES
import p3.aoSystem.FourierUtils as FourierUtils
from p3.aoSystem.telescope import telescope
from p3.aoSystem.atmosphere import atmosphere
from p3.aoSystem.source import source
from p3.aoSystem.deformableMirror import deformableMirror
from p3.aoSystem.detector import detector

#INIT 


class aoSystem():
    
    def check_section_key(self, primary):        
        return primary in self.my_data_map.keys()
    
    def check_config_key(self, primary, secondary):
        if primary in self.my_data_map.keys():
            return secondary in self.my_data_map[primary].keys()
        else:
            return False

    def get_config_value(self, primary, secondary):
        return self.my_data_map[primary][secondary]

    def __init__(path_config, nLayer=None):
                            
            # verify if the file exists
            if ospath.isfile(path_config) == False:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('The .ini or .yml file does not exist\n')
                return 0
            
            
            if path_config[-4::]=='.ini':
                # open the .ini file
                config = ConfigParser()
                config.optionxform = str
                config.read(path_config)
                self.my_data_map = {} 
                for section in config.sections():
                    self.my_data_map[section] = {}
                    for name,value in config.items(section):
                        self.my_data_map[section].update({name:eval(value)})            
                
            elif path_config[-4::]=='.yml':
                with open(path_config) as f:
                    my_yaml_dict = yaml.safe_load(f)        
                self.my_data_map = my_yaml_dict

                
            #%% Telescope
            #----- grabbing main parameters
            if self.check_config_key('telescope','TelescopeDiameter'):
                D = self.get_config_value('telescope','TelescopeDiameter')
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the telescope diameter\n')
                return 0
            
            if self.check_config_key('telescope','ZenithAngle'):
                zenithAngle = self.get_config_value('telescope','ZenithAngle')
            else:
                zenithAngle = 0.0
                
            if self.check_config_key('telescope','ObscurationRatio'):
                obsRatio = self.get_config_value('telescope','ObscurationRatio')
            else:
                obsRatio = 0.0
            
            if self.check_config_key('telescope','Resolution'):
                nPup = self.get_config_value('telescope','Resolution')
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the pupil resolution\n')
                return 0
                
            #----- PUPIL
            if self.check_config_key('telescope','PathPupil'):
                path_pupil = self.get_config_value('telescope','PathPupil')
            else:
                path_pupil = ''
                      
            if self.check_config_key('telescope','PupilAngle'):
                pupilAngle = self.get_config_value('telescope','PupilAngle')
            else:
                pupilAngle = 0.0
            
            if self.check_config_key('telescope','PathStatic'):
                path_static = self.get_config_value('telescope','PathStatic')
            else:
                path_static = None       
                             
            #----- APODIZER
            if self.check_config_key('telescope','PathApodizer'):
                path_apodizer = self.get_config_value('telescope','PathApodizer')
            else:
                path_apodizer = ''
                    
            #----- TELESCOPE ABERRATIONS
            if self.check_config_key('telescope', 'PathStatModes'):
                path_statModes = self.get_config_value('telescope','PathStatModes')
            else:
                path_statModes = ''
                
            #----- class definition     
            self.tel = telescope(D,nPup,zenith_angle=zenithAngle,obsRatio=obsRatio,\
                            pupilAngle=pupilAngle,path_pupil=path_pupil,path_static=path_static,\
                            path_apodizer=path_apodizer,path_statModes=path_statModes)                     
    
            #%% Atmosphere
            
            if self.check_config_key('atmosphere','AtmosphereWavelength'):
                wvlAtm = self.get_config_value('atmosphere','AtmosphereWavelength') 
            else:
                wvlAtm = 500e-9
            
            if self.check_config_key('atmosphere','Seeing'):
                r0 = 0.976*wvlAtm/self.get_config_value('atmosphere','Seeing')*3600*180/np.pi 
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the seeing\n')
                return 0
            
            if self.check_config_key('atmosphere','L0'):
                L0 = self.get_config_value('atmosphere','L0') 
            else:
                L0 = 25
                
            if self.check_config_key('atmosphere','Cn2Weights'):
                weights = self.get_config_value('atmosphere','Cn2Weights') 
            else:
                weights = [1.0]
            
            if self.check_config_key('atmosphere','Cn2Heights'):
                heights = self.get_config_value('atmosphere','Cn2Heights') 
            else:
                heights = [0.0]
                
            if self.check_config_key('atmosphere','WindSpeed'):
                wSpeed = self.get_config_value('atmosphere','WindSpeed') 
            else:
                wSpeed = [10.0]
                
            if self.check_config_key('atmosphere','WindDirection'):
                wDir = self.get_config_value('atmosphere','WindDirection') 
            else:
                wDir = [0.0]
    
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
            #----- class definition
            self.atm = atmosphere(wvlAtm,r0,weights,heights/np.cos(zenithAngle*np.pi/180),wSpeed,wDir,L0)            
            
            #%%  SCIENCE SOURCES
            
            if self.check_config_key('sources_science','Wavelength'):
                wvlSrc     = np.array(self.get_config_value('sources_science','Wavelength'))
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the wavelength of the science source\n')
                return 0
            
            if self.check_config_key('sources_science','Zenith'):
                zenithSrc = self.get_config_value('sources_science','Zenith') 
            else:
                zenithSrc = [0.0]
                
            if self.check_config_key('sources_science','Azimuth'):
                azimuthSrc = self.get_config_value('sources_science','Azimuth') 
            else:
                azimuthSrc = [0.0]
                
            #----- verification
            src = []
            if len(zenithSrc) != len(azimuthSrc):
                print('%%%%%%%% ERROR %%%%%%%%')
                print('The number of scientific sources is not consistent in the parameters file\n')
                return 0
            #----- class definition
            self.src  = source(wvlSrc,zenithSrc,azimuthSrc,tag="SCIENCE SOURCE",verbose=True)   
     
            #%%  GUIDE STARS 
            if self.check_config_key('sources_HO','Wavelength'):
                wvlGs     = np.unique(np.array(self.get_config_value('sources_HO','Wavelength')))
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the wavelength of the science source\n')
                return 0
            
            if self.check_config_key('sources_HO','Zenith'):
                zenithGs = self.get_config_value('sources_HO','Zenith') 
            else:
                zenithGs = [0.0]
                
            if self.check_config_key('sources_HO','Azimuth'):
                azimuthGs = self.get_config_value('sources_HO','Azimuth') 
            else:
                azimuthGs = [0.0]
                
            if self.check_config_key('sources_HO','Height'):
                heightGs  = self.get_config_value('sources_HO','Height') 
            else:
                heightGs = 0.0
                             
            # ----- verification
            if len(zenithGs) != len(azimuthGs):
                print('%%%%%%%% ERROR %%%%%%%%')
                print('The number of guide stars for high-order sensing is not consistent in the parameters file\n')
                return 0
            # ----- creating the source class
            if heightGs == 0:
                self.ngs = source(wvlGs,zenithGs,azimuthGs,height=heightGs,tag="NGS",verbose=True)   
                self.lgs = None
            else:
                self.lgs = source(wvlGs,zenithGs,azimuthGs,height=heightGs,tag="LGS",verbose=True)   
                if not check_section_key('sources_LO'):
                    print('%%%%%%%% WARNING %%%%%%%%')
                    print('No information about the tip-tilt star can be retrieved\n')
                    self.ngs = None
                else:
                    if self.check_config_key('sources_LO','Wavelength'):
                        wvlGs = np.unique(np.array(self.get_config_value('sources_LO','Wavelength')))
                    else:
                        print('%%%%%%%% ERROR %%%%%%%%')
                        print('You must provide a value for the wavelength of the science source\n')
                        return 0
            
                    zenithGs   = np.array(self.get_config_value('sources_LO','Zenith'))
                    azimuthGs  = np.array(self.get_config_value('sources_LO','Azimuth'))
                    # ----- verification
                    if len(zenithGs) != len(azimuthGs):
                        print('%%%%%%%% ERROR %%%%%%%%')
                        print('The number of guide stars for high-order sensing is not consistent in the parameters file\n')
                        return 0
                    self.ngs = source(wvlGs,zenithGs,azimuthGs,tag="NGS",verbose=True)   
                    
            #%% SCIENCE INSTRUMENT
            
            if self.check_config_key('sensor_science','PixelScale'):
                psInMas = self.get_config_value('sensor_science','PixelScale')
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the science detector pixel scale\n')
                return 0
            
            if self.check_config_key('sensor_science','FiedOfView'):
                fov = self.get_config_value('sensor_science','FiedOfView')
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the science detector field of view\n')
                return 0
            
            if self.check_config_key('sensor_science','spotFWHM'):
                spotFWHM = self.get_config_value('sensor_science','spotFWHM')
            else:
                spotFWHM = [[0.0, 0.0]]
                
            if self.check_config_key('sensor_science','FiedOfView'):
                bw = self.get_config_value('sensor_science','FiedOfView')
            else:
                bw = 0.0
            if self.check_config_key('sensor_science','Transmittance'):
                tr = self.get_config_value('sensor_science','Transmittance')
            else:
                tr = [1.0]
            if self.check_config_key('sensor_science','Dispersion'):
                disp = self.get_config_value('sensor_science','Dispersion')
            else:
                disp = [[0.0],[0.0]]
            
            if self.check_config_key('sensor_science','NumberPhotons'):
                nph = self.get_config_value('sensor_science','NumberPhotons')
            else:
                nph = np.inf
            
            if self.check_config_key('sensor_science','SigmaRON'):
                ron = self.get_config_value('sensor_science','SigmaRON')
            else:
                ron = 0.0
            
            if self.check_config_key('sensor_science','SkyBackground'):
                sky = self.get_config_value('sensor_science','SkyBackground')
            else:
                sky = 0.0
            
            if self.check_config_key('sensor_science','Dark'):
                dark = self.get_config_value('sensor_science','Dark')
            else:
                dark = 0.0
            
            self.cam = detector(psInMas,fov,bandwidth=bw,transmittance=tr,dispersion=disp,\
                           spotFWHM=spotFWHM,nph=nph,ron=ron,sky=sky,dark=dark,tag="SCIENCE DETECTOR")
                  
            #%% HIGH-ORDER WAVEFRONT SENSOR
            
            if self.check_config_key('sensor_HO','PixelScale'):
                psInMas = self.get_config_value('sensor_HO','PixelScale')
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the HO detector pixel scale\n')
                return 0
            
            if self.check_config_key('sensor_HO','FiedOfView'):
                fov = self.get_config_value('sensor_HO','FiedOfView')
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the science detector field of view\n')
                return 0
            
            if self.check_config_key('sensor_HO','spotFWHM'):
                spotFWHM = self.get_config_value('sensor_HO','spotFWHM')
            else:
                spotFWHM = [[0.0, 0.0]]
                
            if self.check_config_key('sensor_HO','NumberPhotons'):
                nph = self.get_config_value('sensor_HO','NumberPhotons')
            else:
                nph = np.inf
            
            if self.check_config_key('sensor_HO','SigmaRON'):
                ron = self.get_config_value('sensor_HO','SigmaRON')
            else:
                ron = 0.0
            
            if self.check_config_key('sensor_HO','SkyBackground'):
                sky = self.get_config_value('sensor_HO','SkyBackground')
            else:
                sky = 0.0
            
            if self.check_config_key('sensor_HO','Dark'):
                dark = self.get_config_value('sensor_HO','Dark')
            else:
                dark = 0.0
                
            if self.check_config_key('sensor_HO','ExcessNoiseFactor'):
                excess = self.get_config_value('sensor_HO','ExcessNoiseFactor')
            else:
                excess = 1.0
                
            if self.check_config_key('sensor_HO','NumberLenslets'):
                NumberLenslets = self.get_config_value('sensor_HO','NumberLenslets')
            else:
                NumberLenslets = 20
                
            if self.check_config_key('sensor_HO','LoopGain'):
                LoopGain = self.get_config_value('sensor_HO','LoopGain')
            else:
                LoopGain = 0.5
                
            if self.check_config_key('sensor_HO','SensorFrameRate'):
                SensorFrameRate = self.get_config_value('sensor_HO','SensorFrameRate')
            else:
                SensorFrameRate = 500.0
                
            if self.check_config_key('sensor_HO','LoopDelaySteps'):
                LoopDelaySteps = self.get_config_value('sensor_HO','LoopDelaySteps')
            else:
                LoopDelaySteps = 2
                
            if self.check_config_key('sensor_HO','NoiseVariance'):
                NoiseVariance = self.get_config_value('sensor_HO','NoiseVariance')
            else:
                NoiseVariance = None
                
            if self.check_config_key('sensor_HO','SlopeAlgorithm'):
                SlopeAlgorithm = self.get_config_value('sensor_HO','SlopeAlgorithm')
            else:
                SlopeAlgorithm = 'window'
                
            if self.check_config_key('sensor_HO','WindowRadiusWCoG'):
                wr = self.get_config_value('sensor_HO','WindowRadiusWCoG')
            else:
                wr = 5.0
                
            if self.check_config_key('sensor_HO','ThresholdWCoG = 0.0'):
                thr = self.get_config_value('sensor_HO','ThresholdWCoG = 0.0')
            else:
                thr = 0.0
                
            if self.check_config_key('sensor_HO','NewValueThrPix'):
                nv = self.get_config_value('sensor_HO','NewValueThrPix')
            else:
                nv = 0.0
                
            self.wfs = detector(psInMas,fov,dispersion=disp,nph=nph,spotFWHM=spotFWHM,ron=ron,sky=sky,dark=dark,excess=excess,\
                           nL=NumberLenslets, loopGain=LoopGain, SensorFrameRate=SensorFrameRate,delay=LoopDelaySteps,\
                           noise=NoiseVariance,SlopeAlgorithm=SlopeAlgorithm,wcog=[wr,thr,nv],tag="HO WFS")
            
            #%% TIP-TILT SENSORS
            if check_section_key('sensor_LO'):            
                if self.check_config_key('sensor_LO','PixelScale'):
                    psInMas = self.get_config_value('sensor_LO','PixelScale')
                else:
                    print('%%%%%%%% ERROR %%%%%%%%')
                    print('You must provide a value for the HO detector pixel scale\n')
                    return 0
                
                if self.check_config_key('sensor_LO','FiedOfView'):
                    fov = self.get_config_value('sensor_LO','FiedOfView')
                else:
                    print('%%%%%%%% ERROR %%%%%%%%')
                    print('You must provide a value for the science detector field of view\n')
                    return 0
                
                if self.check_config_key('sensor_LO','spotFWHM'):
                    spotFWHM = self.get_config_value('sensor_LO','spotFWHM')
                else:
                    spotFWHM = [[0.0, 0.0]]
                    
                if self.check_config_key('sensor_LO','NumberPhotons'):
                    nph = self.get_config_value('sensor_LO','NumberPhotons')
                else:
                    nph = np.inf
                
                if self.check_config_key('sensor_LO','SigmaRON'):
                    ron = self.get_config_value('sensor_LO','SigmaRON')
                else:
                    ron = 0.0
                
                if self.check_config_key('sensor_LO','SkyBackground'):
                    sky = self.get_config_value('sensor_LO','SkyBackground')
                else:
                    sky = 0.0
                
                if self.check_config_key('sensor_LO','Dark'):
                    dark = self.get_config_value('sensor_LO','Dark')
                else:
                    dark = 0.0
                    
                if self.check_config_key('sensor_LO','ExcessNoiseFactor'):
                    excess = self.get_config_value('sensor_LO','ExcessNoiseFactor')
                else:
                    excess = 1.0
                    
                if self.check_config_key('sensor_LO','NumberLenslets'):
                    NumberLenslets = self.get_config_value('sensor_LO','NumberLenslets')
                else:
                    NumberLenslets = 20
                    
                if self.check_config_key('sensor_LO','LoopGain'):
                    LoopGain = self.get_config_value('sensor_LO','LoopGain')
                else:
                    LoopGain = 0.5
                    
                if self.check_config_key('sensor_LO','SensorFrameRate'):
                    SensorFrameRate = self.get_config_value('sensor_LO','SensorFrameRate')
                else:
                    SensorFrameRate = 500.0
                    
                if self.check_config_key('sensor_LO','LoopDelaySteps'):
                    LoopDelaySteps = self.get_config_value('sensor_LO','LoopDelaySteps')
                else:
                    LoopDelaySteps = 2
                    
                if self.check_config_key('sensor_LO','NoiseVariance'):
                    NoiseVariance = self.get_config_value('sensor_LO','NoiseVariance')
                else:
                    NoiseVariance = None
                    
                if self.check_config_key('sensor_LO','SlopeAlgorithm'):
                    SlopeAlgorithm = self.get_config_value('sensor_LO','SlopeAlgorithm')
                else:
                    SlopeAlgorithm = 'window'
                    
                if self.check_config_key('sensor_LO','WindowRadiusWCoG'):
                    wr = self.get_config_value('sensor_LO','WindowRadiusWCoG')
                else:
                    wr = 5.0
                    
                if self.check_config_key('sensor_LO','ThresholdWCoG = 0.0'):
                    thr = self.get_config_value('sensor_LO','ThresholdWCoG = 0.0')
                else:
                    thr = 0.0
                    
                if self.check_config_key('sensor_LO','NewValueThrPix'):
                    nv = self.get_config_value('sensor_LO','NewValueThrPix')
                else:
                    nv = 0.0
                    
                if self.check_config_key('sensor_LO','Technical_FoV'):
                    tech_fov = self.get_config_value('sensor_LO','Technical_FoV') 
                else:
                    tech_fov = 120
                    
                self.tts = detector(psInMas,fov,dispersion=disp,nph=nph,spotFWHM=spotFWHM,ron=ron,sky=sky,dark=dark,excess=excess,\
                           nL=NumberLenslets, loopGain=LoopGain, SensorFrameRate=SensorFrameRate,delay=LoopDelaySteps,\
                           noise=NoiseVariance,SlopeAlgorithm=SlopeAlgorithm,wcog=[wr,thr,nv],tech_fov=tech_fov,tag="TT WFS") 
            else:
                self.tts = None
                   
            #%% DEFORMABLE MIRRORS
            if self.check_config_key('DM','NumberActuators'):
                NumberActuators = np.array(self.get_config_value('DM','NumberActuators'))
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the Dm actuators pitch\n')
                return 0
            
            if self.check_config_key('DM','DmPitchs'):
                DmPitchs = np.array(self.get_config_value('DM','DmPitchs'))
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('You must provide a value for the Dm actuators pitch\n')
                return 0
            
            if self.check_config_key('DM','InfModel'):
                InfModel = self.get_config_value('DM','InfModel') 
            else:
                InfModel = 'gaussian'
                
            if self.check_config_key('DM','InfCoupling'):
                InfCoupling = self.get_config_value('DM','InfCoupling') 
            else:
                InfCoupling = [0.2]
                
            if self.check_config_key('DM','DmHeights'):
                DmHeights = self.get_config_value('DM','DmHeights') 
            else:
                DmHeights = [0.0]
                
            if self.check_config_key('DM','OptimizationWeight'):
                OptimizationWeight = self.get_config_value('DM','OptimizationWeight') 
            else:
                OptimizationWeight = [0.0]
                
            if self.check_config_key('DM','OptimizationAzimuth'):
                OptimizationAzimuth = self.get_config_value('DM','OptimizationAzimuth') 
            else:
                OptimizationAzimuth = [0.0]
                
            if self.check_config_key('DM','OptimizationZenith'):
                OptimizationZenith = self.get_config_value('DM','OptimizationZenith') 
            else:
                OptimizationZenith = [0.0]
    
             # ----- verification
            if (len(OptimizationZenith) != len(OptimizationZenith)) or (len(OptimizationZenith) != len(OptimizationWeight)):
                print('%%%%%%%% ERROR %%%%%%%%')
                print('The number of optimization directions is not consistent in the parameters file\n')
                return 0
                  
            if self.check_config_key('DM','OptimizationConditioning'):
                OptimizationConditioning = self.get_config_value('DM','OptimizationConditioning') 
            else:
                OptimizationConditioning = 100.0
                
            if self.check_config_key('DM','NumberReconstructedLayers'):
                NumberReconstructedLayers = self.get_config_value('DM','NumberReconstructedLayers') 
            else:
                NumberReconstructedLayers = 10
                
            # ----- creating the dm class
            self.dms = deformableMirror(NumberActuators,DmPitchs,heights=DmHeights,mechCoupling=InfCoupling,modes=InfModel,\
                                  opt_dir=[OptimizationZenith,OptimizationAzimuth],opt_weights=OptimizationWeight,\
                                  opt_cond=OptimizationConditioning,n_rec = NumberReconstructedLayers)
          