#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:42:49 2021

@author: omartin
"""

# IMPORTING PYTHON LIBRAIRIES
import os.path as ospath
import pathlib
from configparser import ConfigParser
import numpy as np

import yaml

# IMPORTING P3 MODULES
from p3.aoSystem.telescope import telescope
from p3.aoSystem.atmosphere import atmosphere
from p3.aoSystem.source import source
from p3.aoSystem.deformableMirror import deformableMirror
from p3.aoSystem.detector import detector
from p3.aoSystem.sensor import sensor
from p3.aoSystem.rtc import rtc
import p3.aoSystem.anisoplanatismModel as anisoplanatismModel

#%%
class aoSystem():

    def raiseMissingRequiredOpt(self,sec,opt):
        raise ValueError("'{}' is missing from section '{}'"
                         .format(opt,sec))
        
    def raiseMissingRequiredSec(self,sec):
        raise ValueError("The section '{}' is missing from the parameter file"
                         .format(sec))
        
    def raiseNotSameLength(self,sec,opt):
        raise ValueError("'{}' in section '{}' must have the same length"
                         .format(*opt,sec))

    def check_section_key(self, primary):        
        return primary in self.my_data_map.keys()
    
    def check_config_key(self, primary, secondary):
        if primary in self.my_data_map.keys():
            return secondary in self.my_data_map[primary].keys()
        else:
            return False

    def get_config_value(self, primary, secondary):
        return self.my_data_map[primary][secondary]
    
    def __init__(self,path_config,path_root='',nLayer=None,getPSDatNGSpositions=False,coo_stars=None):

        self.coo_stars = coo_stars
        self.error = False
        # verify if the file exists
        if ospath.isfile(path_config) == False:
            raise ValueError('The parameter file (.ini or .yml) could not be found.')
                
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

        self.getPSDatNGSpositions = getPSDatNGSpositions
        
        #%% TELESCOPE
        #----- grabbing main parameters
        if not(self.check_section_key('telescope')):
            self.raiseMissingRequiredSec('telescope')
        
        if self.check_config_key('telescope','TelescopeDiameter'):
            self.D = self.get_config_value('telescope','TelescopeDiameter')
        else:
            self.raiseMissingRequiredOpt('telescope','TelescopeDiameter')
        
        if self.check_config_key('telescope','ZenithAngle'):
            zenithAngle = self.get_config_value('telescope','ZenithAngle')
            if zenithAngle >= 90:
                raise ValueError("'zenithAngle' [deg] cannot be greater than or equal to 90.")
        else:
            zenithAngle = 0.0
        
        airmass = 1/np.cos(zenithAngle*np.pi/180)
        
        if self.check_config_key('telescope','ObscurationRatio'):
            obsRatio = self.get_config_value('telescope','ObscurationRatio')
        else:
            obsRatio = 0.0
        
        if self.check_config_key('telescope','Resolution'):
            nPup = self.get_config_value('telescope','Resolution')
        else:
            self.raiseMissingRequiredOpt('telescope','Resolution')
            
        path_p3 = str(pathlib.Path(__file__).parent.parent.absolute())
            
        #----- PUPIL
        if self.check_config_key('telescope','PathPupil'):
            PathPupil = self.get_config_value('telescope','PathPupil')
            if path_root == '' and PathPupil[0:9]=='/aoSystem' :
                path_pupil = path_p3 + PathPupil
            elif path_root == '' and PathPupil[0:8]=='aoSystem' :
                path_pupil = path_p3 +'/'+ PathPupil
            else:
                path_pupil = path_root + PathPupil
        else:
            path_pupil = ''
                  
        if self.check_config_key('telescope','PupilAngle'):
            pupilAngle = self.get_config_value('telescope','PupilAngle')
        else:
            pupilAngle = 0.0
        
        if self.check_config_key('telescope','PathStaticOn'):
            PathStaticOn = self.get_config_value('telescope','PathStaticOn')
            if path_root == '' and PathStaticOn[0:9]=='/aoSystem' :
                path_static_on = path_p3 + PathStaticOn
            elif path_root == '' and PathStaticOn[0:8]=='aoSystem' :
                path_static_on = path_p3 +'/'+ PathStaticOn
            else:
                path_static_on = path_root + PathStaticOn
        else:
            path_static_on = None       
        
        if self.check_config_key('telescope','PathStaticOff'):
            PathStaticOff = self.get_config_value('telescope','PathStaticOff')
            if path_root == '' and PathStaticOff[0:9]=='/aoSystem' :
                path_static_off = path_p3 + PathStaticOff
            elif path_root == '' and PathStaticOff[0:8]=='aoSystem' :
                path_static_off = path_p3 +'/'+ PathStaticOff
            else:
                path_static_off = path_root + PathStaticOff
        else:
            path_static_off = None
        
        if self.check_config_key('telescope','PathStaticPos'):
            PathStaticPos = self.get_config_value('telescope','PathStaticPos')
            if path_root == '' and PathStaticPos[0:9]=='/aoSystem' :
                path_static_pos = path_p3 + PathStaticPos
            elif path_root == '' and PathStaticPos[0:8]=='aoSystem' :
                path_static_pos = path_p3 +'/'+ PathStaticPos
            else:
                path_static_pos = path_root + PathStaticPos
        else:
            path_static_pos = None
            
        #----- APODIZER
        if self.check_config_key('telescope','PathApodizer'):
            PathApodizer = self.get_config_value('telescope','PathApodizer')
            if path_root == '' and PathApodizer[0:9]=='/aoSystem' :
                path_apodizer = path_p3 + PathApodizer
            elif path_root == '' and PathApodizer[0:8]=='aoSystem' :
                path_apodizer = path_p3 +'/'+ PathApodizer
            else:
                path_apodizer = path_root + PathApodizer
        else:
            path_apodizer = ''
                
        #----- TELESCOPE ABERRATIONS
        if self.check_config_key('telescope', 'PathStatModes'):
            PathStatModes = self.get_config_value('telescope','PathStatModes')
            if path_root == '' and PathStatModes[0:9]=='/aoSystem' :
                path_statModes = path_p3 + PathStatModes
            elif path_root == '' and PathStatModes[0:8]=='aoSystem' :
                path_statModes = path_p3 +'/'+ PathStatModes
            else:
                path_statModes = path_root + PathStatModes
        else:
            path_statModes = ''
            
        #----- EXTRA ERROR
        if self.check_config_key('telescope', 'extraErrorNm'):       
            extraErrorNm = self.get_config_value('telescope','extraErrorNm')
        else:
            extraErrorNm = 0
            
        if self.check_config_key('telescope', 'extraErrorExp'):
            extraErrorExp = self.get_config_value('telescope','extraErrorExp')
        else:
            extraErrorExp = -2
            
        if self.check_config_key('telescope', 'extraErrorMin'):
            extraErrorMin = self.get_config_value('telescope','extraErrorMin')
        else:
            extraErrorMin = 0
            
        if self.check_config_key('telescope', 'extraErrorMax'):
            extraErrorMax = self.get_config_value('telescope','extraErrorMax')
        else:
            extraErrorMax = 0
            
        # ----- class definition     
        self.tel = telescope(self.D, nPup,
                             zenith_angle=zenithAngle,
                             obsRatio=obsRatio,
                             pupilAngle=pupilAngle,
                             path_pupil=path_pupil,
                             path_static_on=path_static_on,
                             path_static_off=path_static_off,
                             path_static_pos=path_static_pos,
                             path_apodizer=path_apodizer,
                             path_statModes=path_statModes,
                             extraErrorNm=extraErrorNm,
                             extraErrorExp=extraErrorExp,
                             extraErrorMin=extraErrorMin,
                             extraErrorMax=extraErrorMax)                     

        #%% ATMOSPHERE
        if not(self.check_section_key('atmosphere')):
            self.raiseMissingRequiredSec('atmosphere')
            
        if self.check_config_key('atmosphere','Wavelength'):
            wvlAtm = self.get_config_value('atmosphere','Wavelength') 
        else:
            wvlAtm = 500e-9
        
        if self.check_config_key('atmosphere','Seeing'):
            r0 = 0.976*wvlAtm/self.get_config_value('atmosphere','Seeing')*3600*180/np.pi 
        else:
            if self.check_config_key('atmosphere','r0_value'):
                r0 = self.get_config_value('atmosphere','r0_value') 
            else:
                self.raiseMissingRequiredOpt('atmosphere','Seeing')
        
        if self.check_config_key('atmosphere','L0'):
            L0 = self.get_config_value('atmosphere','L0') 
        else:
            L0 = 25
            
        if self.check_config_key('atmosphere','Cn2Weights'):
            weights = self.get_config_value('atmosphere','Cn2Weights') 
            if np.abs(np.sum(weights) - 1) > 1e-3:
                raise ValueError("Sum of 'Cn2Weights' in section 'atmosphere' is not equal to 1")
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
            self.raiseNotSameLength('atmosphere',['Cn2Weights','Cn2Heights','WindSpeed','WindDirection'])

        #----- class definition
        self.atm = atmosphere(wvlAtm, r0*airmass**(-3.0/5.0),
                              weights,
                              np.array(heights)*airmass,
                              wSpeed,
                              wDir,
                              L0)            
        
        #%%  GUIDE STARS 
        if not(self.check_section_key('sources_HO')):
            self.raiseMissingRequiredSec('sources_HO')
            
        if self.check_config_key('sources_HO','Wavelength'):
            self.wvlGs     = np.unique(np.array(self.get_config_value('sources_HO','Wavelength')))
        else:
            self.raiseMissingRequiredOpt('sources_HO', 'Wavelength')
            return 0
        
        if self.check_config_key('sources_HO','Zenith'):
            self.zenithGs = self.get_config_value('sources_HO','Zenith') 
        else:
            self.zenithGs = [0.0]
            
        if self.check_config_key('sources_HO','Azimuth'):
            self.azimuthGs = self.get_config_value('sources_HO','Azimuth') 
        else:
            self.azimuthGs = [0.0]
            
        if self.check_config_key('sources_HO','Height'):
            heightGs  = self.get_config_value('sources_HO','Height') 
        else:
            heightGs = 0.0
                         
        # ----- verification
        if len(self.zenithGs) != len(self.azimuthGs):
            self.raiseNotSameLength('sources_HO', ['Azimuth','Zenith'])

        self.ngs = None
        self.lgs = None
        # ----- creating the source class
        if heightGs == 0 and not self.check_section_key('sources_LO'):
            self.ngs = source(self.wvlGs,
                              self.zenithGs,self.azimuthGs,
                              tag="NGS",verbose=True)
        else:
            self.lgs = source(self.wvlGs,
                              self.zenithGs,self.azimuthGs,
                              height=heightGs*airmass,
                              tag="LGS",verbose=True)

        self.configLO()
        self.configLO_SC()

    def configLO_SC(self):
        #%%  SCIENCE SOURCES
        if not(self.check_section_key('sources_science')):
            self.raiseMissingRequiredSec('sources_science') 
            
        if self.check_config_key('sources_science','Wavelength'):
            wvlSrc = np.array(self.get_config_value('sources_science','Wavelength'))
        else:
            self.raiseMissingRequiredOpt('sources_science', 'Wavelength')
        
        if self.check_config_key('sources_science','Zenith'):
            zenithSrc = self.get_config_value('sources_science','Zenith') 
        else:
            zenithSrc = [0.0]
            
        if self.check_config_key('sources_science','Azimuth'):
            azimuthSrc = self.get_config_value('sources_science','Azimuth') 
        else:
            azimuthSrc = [0.0]
        
        if np.any(self.coo_stars):
            zenithSrc = np.hypot(self.coo_stars[0],self.coo_stars[1])
            azimuthSrc = np.arctan2(self.coo_stars[0],self.coo_stars[1])
        
        #----- verification
        if len(zenithSrc) != len(azimuthSrc):
            self.raiseNotSameLength('sources_science', ['Zenith','Azimuth'])
            return

        if self.getPSDatNGSpositions and self.check_config_key('sources_LO','Wavelength'):
            zenithSrc = zenithSrc +  (self.get_config_value('sources_LO','Zenith'))
            azimuthSrc = azimuthSrc + (self.get_config_value('sources_LO','Azimuth'))
            
        #----- class definition
        self.src = source(wvlSrc,
                          zenithSrc, azimuthSrc,
                          tag="SCIENCE",verbose=True)   

        #%% HIGH-ORDER WAVEFRONT SENSOR
        if not(self.check_section_key('sensor_HO')):
            self.raiseMissingRequiredSec('sensor_HO')
            
        if self.check_config_key('sensor_HO','PixelScale'):
            psInMas = self.get_config_value('sensor_HO','PixelScale')
        else:
            self.raiseMissingRequiredOpt('sensor_HO', 'PixelScale')
        
        if self.check_config_key('sensor_HO','FieldOfView'):
            fov = self.get_config_value('sensor_HO','FieldOfView')
        else:
            self.raiseMissingRequiredOpt('sensor_HO', 'FieldOfView') 
        
        if self.check_config_key('sensor_HO','Binning'):
            Binning = self.get_config_value('sensor_HO','Binning')
        else:
            Binning = 1
            
        if self.check_config_key('sensor_HO','SpotFWHM'):
            spotFWHM = self.get_config_value('sensor_HO','SpotFWHM')
        else:
            spotFWHM = [[0.0, 0.0]]
            
        if self.check_config_key('sensor_HO','NumberPhotons'):
            nphHO = self.get_config_value('sensor_HO','NumberPhotons')
        else:
            nphHO = [np.inf]
        
        if self.check_config_key('sensor_HO','SigmaRON'):
            ron = self.get_config_value('sensor_HO','SigmaRON')
        else:
            ron = 0.0
        
        if self.check_config_key('sensor_HO','Gain'):
            self.detectorGainHO = self.get_config_value('sensor_HO','Gain')
        else:
            self.detectorGainHO = 1
            
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
         
        if self.check_config_key('sensor_HO','SpectralBandwidth'):
            bw = self.get_config_value('sensor_HO','SpectralBandwidth')
        else:
            bw = 0.0
        if self.check_config_key('sensor_HO','Transmittance'):
            tr = self.get_config_value('sensor_HO','Transmittance')
        else:
            tr = [1.0]
        if self.check_config_key('sensor_HO','Dispersion'):
            disp = self.get_config_value('sensor_HO','Dispersion')
        else:
            disp = [[0.0], [0.0]]
                
        if self.check_config_key('sensor_HO','WfsType'):
            wfstype = self.get_config_value('sensor_HO','WfsType')
        else:
            wfstype = 'Shack-Hartmann'
            
        if self.check_config_key('sensor_HO','NumberLenslets'):
            nL = self.get_config_value('sensor_HO','NumberLenslets')
        else:
            nL = [20]
            
        if self.check_config_key('sensor_HO','SizeLenslets'):
            dsub = self.get_config_value('sensor_HO','SizeLenslets')
        else:
            dsub = list(self.D/np.array(nL))
            
        if self.check_config_key('sensor_HO','Modulation'):
            modu = self.get_config_value('sensor_HO','Modulation')
        else:
            modu = None
            
        if self.check_config_key('sensor_HO','NoiseVariance'):
            NoiseVar = np.array(self.get_config_value('sensor_HO','NoiseVariance'))
        else:
            NoiseVar = [None]
            
        if self.check_config_key('sensor_HO','Algorithm'):
            algorithm = self.get_config_value('sensor_HO','Algorithm')
        else:
            algorithm = 'wcog'
            
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
         
        if self.check_config_key('sensor_HO','ExcessNoiseFactor'):
            excess = self.get_config_value('sensor_HO','ExcessNoiseFactor')
        else:
            excess = 1.0
            
        self.wfs = sensor(psInMas, fov,
                          binning=Binning, spotFWHM=spotFWHM,
                          nph=nphHO, bandwidth=bw, transmittance=tr, dispersion=disp,
                          gain=self.detectorGainHO, ron=ron, sky=sky, dark=dark, excess=excess,
                          nL=nL, dsub=dsub, wfstype=wfstype, modulation=modu,
                          noiseVar=NoiseVar, algorithm=algorithm,
                          algo_param=[wr,thr,nv], tag="HO WFS")
#%% TIP-TILT SENSORS
        if self.check_section_key('sensor_LO'):
            self.configLOsensor()
        else:
            self.tts = None

 #%% REAL-TIME-COMPUTER
     
        if self.check_config_key('RTC','LoopGain_HO'):
            self.LoopGain_HO = self.get_config_value('RTC','LoopGain_HO')
        else:
            self.LoopGain_HO = 0.5
            
        if self.check_config_key('RTC','SensorFrameRate_HO'):
            frameRate_HO = self.get_config_value('RTC','SensorFrameRate_HO')
        else:
            frameRate_HO = 500.0
            
        if self.check_config_key('RTC','LoopDelaySteps_HO'):
            delay_HO = self.get_config_value('RTC','LoopDelaySteps_HO')
        else:
            delay_HO = 2
                     
        if self.check_config_key('RTC','LoopGain_LO'):
            temp = self.get_config_value('RTC','LoopGain_LO')
            if temp != 'optimize':
                self.LoopGain_LO = temp
            else:
                self.LoopGain_LO = None
        else:
            self.LoopGain_LO = None
            
        if self.check_config_key('RTC','SensorFrameRate_LO'):
            frameRate_LO = self.get_config_value('RTC','SensorFrameRate_LO')
        else:
            frameRate_LO = None
            
        if self.check_config_key('RTC','LoopDelaySteps_LO'):
            delay_LO = self.get_config_value('RTC','LoopDelaySteps_LO')
        else:
            delay_LO = None
            
        if self.check_config_key('RTC','ResidualError'):
            wfe = self.get_config_value('RTC','ResidualError')
        else:
            wfe = None
            
        self.rtc = rtc(self.LoopGain_HO, frameRate_HO, delay_HO, wfe=wfe,
                       loopGainLO=self.LoopGain_LO, frameRateLO=frameRate_LO, delayLO=delay_LO)
               
#%% DEFORMABLE MIRRORS
        if not(self.check_section_key('DM')):
            self.raiseMissingRequiredSec('DM')

        if self.check_config_key('DM','NumberActuators'):
            nActu = self.get_config_value('DM','NumberActuators')
        else:
            self.raiseMissingRequiredOpt('DM', 'NumberActuators')
        
        if self.check_config_key('DM','DmPitchs'):
            DmPitchs = np.array(self.get_config_value('DM','DmPitchs'))
        else:
            self.raiseMissingRequiredOpt('DM','DmPitchs')
            self.error = True
            return
        
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
            opt_w = self.get_config_value('DM','OptimizationWeight') 
        else:
            opt_w = [0.0]
            
        if self.check_config_key('DM','OptimizationAzimuth'):
            opt_az = self.get_config_value('DM','OptimizationAzimuth') 
        else:
            opt_az = [0.0]
            
        if self.check_config_key('DM','OptimizationZenith'):
            opt_zen = self.get_config_value('DM','OptimizationZenith') 
        else:
            opt_zen = [0.0]

         # ----- verification
        if (len(opt_zen) != len(opt_az)) or (len(opt_zen) != len(opt_w)):
            self.raiseNotSameLength('DM', ['OptimizationZenith','OptimizationAzimuth','OptimizationWeight'])
              
        if self.check_config_key('DM','OptimizationConditioning'):
            cond = self.get_config_value('DM','OptimizationConditioning') 
        else:
            cond = 100.0
            
        if self.check_config_key('DM','NumberReconstructedLayers'):
            nrec = self.get_config_value('DM','NumberReconstructedLayers') 
        else:
            nrec = 10
            
        if self.check_config_key('DM','AoArea'):
            AoArea = self.get_config_value('DM','AoArea') 
        else:
            AoArea = 'circle'
        
        # ----- creating the dm class
        self.dms = deformableMirror(nActu, DmPitchs,
                                    heights=DmHeights, mechCoupling=InfCoupling,
                                    modes=InfModel,
                                    opt_dir=[opt_zen,opt_az],
                                    opt_weights=opt_w,
                                    opt_cond=cond,n_rec = nrec,
                                    AoArea=AoArea)
      
        #%% SCIENCE DETECTOR
        if not(self.check_section_key('sensor_science')):
            self.raiseMissingRequiredSec('sensor_science')
        
        if self.check_config_key('sensor_science','Name'):
            camName = self.get_config_value('sensor_science','Name')
        else:
            camName = 'SCIENCE CAM'
            
        if self.check_config_key('sensor_science','PixelScale'):
            psInMas = self.get_config_value('sensor_science','PixelScale')
        else:
            self.raiseMissingRequiredOpt('sensor_science', 'PixelScale')
        
        if self.check_config_key('sensor_science','FieldOfView'):
            fov = self.get_config_value('sensor_science','FieldOfView')
        else:
            self.raiseMissingRequiredOpt('sensor_science', 'FieldOfView')
        
        if self.check_config_key('sensor_science','Binning'):
            Binning = self.get_config_value('sensor_science','Binning')
        else:
            Binning = 1
            
        if self.check_config_key('sensor_science','SpotFWHM'):
            spotFWHM = self.get_config_value('sensor_science','SpotFWHM')
        else:
            spotFWHM = [[0.0,0.0,0.0]]
            
        if self.check_config_key('sensor_science','SpectralBandwidth'):
            bw = self.get_config_value('sensor_science','SpectralBandwidth')
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
            nphSC = self.get_config_value('sensor_science','NumberPhotons')
        else:
            nphSC = np.inf
        
        if self.check_config_key('sensor_science','Saturation'):
            saturation = self.get_config_value('sensor_science','Saturation')
        else:
            saturation = np.inf
            
        if self.check_config_key('sensor_science','SigmaRON'):
            ron = self.get_config_value('sensor_science','SigmaRON')
        else:
            ron = 0.0
        
        if self.check_config_key('sensor_science','Gain'):
            self.detectorGainScience = self.get_config_value('sensor_science','Gain')
        else:
            self.detectorGainScience = 1
            
        if self.check_config_key('sensor_science','SkyBackground'):
            sky = self.get_config_value('sensor_science','SkyBackground')
        else:
            sky = 0.0
        
        if self.check_config_key('sensor_science','Dark'):
            dark = self.get_config_value('sensor_science','Dark')
        else:
            dark = 0.0
        
        if self.check_config_key('sensor_science','ExcessNoiseFactor'):
            excess = self.get_config_value('sensor_science','ExcessNoiseFactor')
        else:
            excess = 1.0
        
        self.cam = detector(psInMas, fov,
                            binning=Binning, spotFWHM=spotFWHM, saturation=saturation,
                            nph=nphSC, bandwidth=bw, transmittance=tr, dispersion=disp,
                            gain=self.detectorGainScience, ron=ron, sky=sky, dark=dark, excess=excess,
                            tag=camName)
        
        # %% AO mode
        self.aoMode = 'SCAO'
        
        if self.lgs:
            if self.lgs.nSrc > 1:
                if self.dms.nDMs > 1:
                    self.aoMode = 'MCAO'
                else:
                    if self.dms.nRecLayers >1:
                        self.aoMode = 'LTAO' 
                    else:
                        self.aoMode = 'GLAO'                  
            else:
                self.aoMode = 'SLAO'
    
        # %% ERROR BREAKDOWN
        if self.rtc.holoop['gain'] > 0:
            self.errorBreakdown()


    def configLO(self):
        
        if not self.check_section_key('sources_LO'):
        
            print('Warning: No information about the tip-tilt star can be retrieved')            
        else:
            if self.check_config_key('sources_LO','Wavelength'):
                self.wvlGsLO = np.unique(np.array(self.get_config_value('sources_LO','Wavelength')))
            else:
                self.raiseMissingRequiredOpt('sources_LO','Wavelength')

            self.zenithGsLO   = self.get_config_value('sources_LO','Zenith')
            self.azimuthGsLO  = self.get_config_value('sources_LO','Azimuth')
            # ----- verification
            if len(self.zenithGsLO) != len(self.azimuthGsLO):
                self.raiseNotSameLength('sources_LO', ['Zenith','Azimuth'])

            self.ngs = source(self.wvlGsLO,self.zenithGsLO,self.azimuthGsLO,tag="NGS",verbose=True)

    
    def configLOsensor(self):
            
        if self.check_config_key('sensor_LO','PixelScale'):
            psInMas = self.get_config_value('sensor_LO','PixelScale')
        else:
            self.raiseMissingRequiredOpt('sensor_LO', 'PixelScale')
            self.error = True
            return

        if self.check_config_key('sensor_LO','FieldOfView'):
            fov = self.get_config_value('sensor_LO','FieldOfView')
        else:
            self.raiseMissingRequiredOpt('sensor_LO', 'FieldOfView')
            self.error = True
            return

        if self.check_config_key('sensor_LO','Binning'):
            Binning = self.get_config_value('sensor_LO','Binning')
        else:
            Binning = 1

        if self.check_config_key('sensor_LO','SpotFWHM'):
            spotFWHM = self.get_config_value('sensor_LO','SpotFWHM')
        else:
            spotFWHM = [[0.0, 0.0]]

        if self.check_config_key('sensor_LO','NumberPhotons'):
            nphLO = self.get_config_value('sensor_LO','NumberPhotons')
        else:
            nphLO = np.inf

        if self.check_config_key('sensor_LO','Gain'):
            self.detectorGainLO = self.get_config_value('sensor_LO','Gain')
        else:
            self.detectorGainLO = 1
            
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

        if self.check_config_key('sensor_LO','SpectralBandwidth'):
            bw = self.get_config_value('sensor_LO','SpectralBandwidth')
        else:
            bw = 0.0
        if self.check_config_key('sensor_LO','Transmittance'):
            tr = self.get_config_value('sensor_LO','Transmittance')
        else:
            tr = [1.0]
        if self.check_config_key('sensor_LO','Dispersion'):
            disp = self.get_config_value('sensor_LO','Dispersion')
        else:
            disp = [[0.0], [0.0]]

        if self.check_config_key('sensor_LO','NumberLenslets'):
            nL = self.get_config_value('sensor_LO','NumberLenslets')
        else:
            nL = [1]

        if self.check_config_key('sensor_LO','NoiseVariance'):
            NoiseVar = self.get_config_value('sensor_LO','NoiseVariance')
        else:
            NoiseVar = [None]

        if self.check_config_key('sensor_LO','Algorithm'):
            algorithm = self.get_config_value('sensor_LO','Algorithm')
        else:
            algorithm = 'wcog'

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
            
        self.tts = sensor(psInMas, fov,
                          binning=Binning, spotFWHM=spotFWHM,
                          nph=nphLO, bandwidth=bw, transmittance=tr, dispersion=disp,
                          gain=self.detectorGainLO, ron=ron, sky=sky, dark=dark, excess=excess,
                          nL=nL, dsub=list(self.D/np.array(nL)),
                          wfstype='Shack-Hartmann', noiseVar=NoiseVar,
                          algorithm=algorithm, algo_param=[wr,thr,nv], tag="TT WFS")

    
    def __repr__(self):
        
        s = '\t\t\t\t________________________'+ self.aoMode + ' SYSTEM ________________________\n\n'
        s += self.src.__repr__() + '\n'
        if self.lgs:
            s += self.lgs.__repr__() + '\n'
        if self.ngs:
            s += self.ngs.__repr__() + '\n'
        s += self.atm.__repr__() + '\n'
        s+= self.tel.__repr__() + '\n'
        s+= self.wfs.__repr__() +'\n'
        if self.tts:
            s+= self.tts.__repr__() +'\n'
        
        s+= self.rtc.__repr__() +'\n'
        s+= self.dms.__repr__() +'\n'
        s+= self.cam.__repr__()

        return s
    
    def errorBreakdown(self):
        """
            Computing the AO wavefront error breakdown based on theoretical formula
        """
        
        rad2nm = lambda x:  np.sqrt(x) * self.atm.wvl*1e9/2/np.pi
        self.wfe = dict()
        
        Dr053 = (self.tel.D/self.atm.r0)**(5/3)
        dactur053 = (self.dms.pitch[0]/self.atm.r0)**(5/3)
        dsubr053 = (self.wfs.optics[0].dsub/self.atm.r0)**(5/3)
        
        # DM fitting error
        self.wfe['DM fitting'] = rad2nm(0.23*dactur053)
        
        # Aliasing error
        self.wfe['WFS aliasing'] = rad2nm(0.07*dsubr053)      
        
        # Servo-lag errors
        ff = np.pi*0.5**2 # to account for the loss of valid actuator outside the pupil
        nMax = int(np.sqrt(ff)*(self.dms.nControlledRadialOrder[0]+1))
        if hasattr(self.rtc,'ttloop') and self.tts !=None :
            nMin = 3
        else:
            nMin = 1
            
        nrad  = np.array(range(nMin,nMax))
        self.wfe['HO Servo-lag'] = rad2nm(0.04 * (self.atm.meanWind/self.tel.D/self.rtc.holoop['bandwidth'])\
                                    * Dr053 * np.sum((nrad+1)**(-2/3)))
        
        # Noise errors        
        if self.wfs.processing.noiseVar == [None]:
            varNoise = self.wfs.NoiseVariance(self.atm.r0 ,self.atm.wvl)
        else:
            varNoise = self.wfs.processing.noiseVar
            
        self.wfe['HO Noise'] = rad2nm(np.mean(varNoise))
        
        if hasattr(self.rtc,'ttloop') and self.tts !=None:
            self.wfe['TT Servo-lag'] = rad2nm(0.04 * (self.atm.meanWind/self.tel.D/self.rtc.ttloop['bandwidth'])* Dr053 * 2**(-2/3))
            
            if self.tts.processing.noiseVar == [None]:
                varNoise = self.tts.NoiseVariance(self.atm.r0 ,self.atm.wvl)
            else:
                varNoise = self.tts.processing.noiseVar
            
            self.wfe['TT Noise']     = rad2nm(np.mean(varNoise))
        else:
            self.wfe['TT Servo-lag'] = 0
            self.wfe['TT Noise'] = 0
        
        # Focal anisoplanatism
        if self.lgs and self.lgs.height[0] > 0:
            self.wfe['Focal anisoplanatism'] = anisoplanatismModel.focal_anisoplanatism_variance(self.tel,self.atm,self.lgs)
        else:
            self.wfe['Focal anisoplanatism'] = 0
            
        # TO be added : angular anisoplanatisms
           
        self.wfe['Total'] = np.sqrt(self.wfe['DM fitting']**2 + self.wfe['WFS aliasing']**2\
                                    + self.wfe['HO Servo-lag']**2 + self.wfe['HO Noise']**2\
                                    + self.wfe['TT Servo-lag']**2 + self.wfe['TT Noise']**2\
                                    + self.wfe['Focal anisoplanatism']**2)
        self.wfe['Strehl'] = np.exp(-self.wfe['Total']**2 * (2*np.pi*1e-9/self.src.wvl[0])**2)
        
