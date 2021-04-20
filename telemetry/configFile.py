#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:04:35 2021

@author: omartin
"""
import numpy as np
import os
from configparser import ConfigParser

class configFile():
    

    def __init__(self,trs):
        
        # create the .ini file
        file = trs.tel.name + '_' + trs.cam.name + '_' + trs.obsdate + '_' + trs.acqtime + '.ini'
        self.path_ini = trs.path_save + '/'+ file
        if not os.path.exists(self.path_ini):
            with open(trs.path_ini, 'w'): pass
        else:
            print('WARNING: the .ini file already exists')

        # open the .ini file
        parser = ConfigParser()
        parser.optionxform = str
        parser.read(self.path_ini)
        
        # updating the telescope parameters
        if not parser.has_section('telescope'):
            parser.add_section('telescope')
        parser.set('telescope','TelescopeDiameter',str(trs.tel.D))
        parser.set('telescope','ObscurationRatio', str(trs.tel.cobs))
        parser.set('telescope','ZenithAngle', str(trs.tel.zenith_angle))
        parser.set('telescope','Resolution', str(trs.tel.resolution))
        parser.set('telescope','PupilAngle', str(trs.tel.pupilAngle))
        parser.set('telescope','PathPupil','\'' + trs.tel.path_pupil + '\'')
        parser.set('telescope','PathStatic','\'' + trs.cam.path_ncpa + '\'')
        parser.set('telescope','PathStatModes','\'' + trs.tel.path_telstat + '\'')
        
        # updating the atmosphere config
        if not parser.has_section('atmosphere'):
            parser.add_section('atmosphere')
        parser.set('atmosphere','AtmosphereWavelength', str(trs.atm.wvl))
        parser.set('atmosphere','Seeing', str(trs.atm.seeing))
        parser.set('atmosphere','L0', str(trs.atm.L0))
        parser.set('atmosphere','Cn2Weights', str(trs.atm.Cn2Weights))
        parser.set('atmosphere','Cn2Heights', str(trs.atm.Cn2Heights))
        parser.set('atmosphere','WindSpeed', str(trs.atm.wSpeed))
        parser.set('atmosphere','WindDirection', str(trs.atm.wDir))

        # updating the HO WFS config
        if not parser.has_section('sensor_HO'):
            parser.add_section('sensor_HO')
        parser.set('sensor_HO','PixelScale', str(trs.wfs.pixel_scale))
        parser.set('sensor_HO','FiedOfView', str(trs.wfs.fov))
        parser.set('sensor_HO','NumberLenslets', str([trs.wfs.nSubap]))
        parser.set('sensor_HO','NoiseVariance', str([0.0]))
        
        # GUIDE STARS
        if not parser.has_section('sources_HO'):
            parser.add_section('sources_HO')
            parser.set('sources_HO','Wavelength', str(trs.wfs.wvl))

        if self.aoMode == 'LGS':
            parser.set('sources_HO','Height', str(trs.lgs.height))
            parser.set('sources_HO','Zenith',str(trs.lgs.zenith))
            parser.set('sources_HO','Azimuth',str(trs.lgs.azimuth))
            if not parser.has_section('sources_LO'):
                parser.add_section('sources_LO')
            parser.set('sources_LO','Zenith',str(trs.ngs.zenith))
            parser.set('sources_LO','Azimuth',str(trs.ngs.azimuth))
        else:
            parser.set('sources_HO','Height',str(0))
            parser.set('sources_HO','Zenith',str(trs.ngs.zenith))
            parser.set('sources_HO','Azimuth',str(trs.ngs.azimuth))
            
        # updating the DM config
        if not parser.has_section('DM'):
            parser.add_section('DM')
        parser.set('DM','NumberActuators', str(trs.dm.nActuators))
        parser.set('DM','DmPitchs', str(trs.dm.pitch))
        parser.set('DM','InfModel', '\'xinetics\'')
        parser.set('DM','InfCoupling', str([0.11]))
        
        # updating the imager config
        if not parser.has_section('sources_science'):
            parser.add_section('sources_science')
        parser.set('sources_science','Zenith', str(trs.cam.zenith))
        parser.set('sources_science','Azimuth', str(trs.cam.azimuth))
        parser.set('sources_science','Wavelength', str([np.mean(trs.cam.wvl)]))
        
        if not parser.has_section('sensor_science'):
            parser.add_section('sensor_science')
        parser.set('sensor_science','FiedOfView', str(trs.cam.fov))
        parser.set('sensor_science','PixelScale', str(trs.cam.psInMas))

        if len(self.cam.wvl) > 1:
            parser.set('sensor_science','Transmittance', str(trs.cam.transmission))
            parser.set('sensor_science','SpectralBandwidth', str(trs.cam.bw))
            parser.set('sensor_science','Dispersion', str(trs.cam.dispersion))
            
            
        if not parser.has_section('RTC'):
            parser.add_section('RTC')
        parser.set('RTC','LoopGain_HO', str(trs.holoop.gain))
        parser.set('RTC','SensorFrameRate_HO', str(trs.holoop.freq))
        parser.set('RTC','LoopDelaySteps_HO', str(trs.holoop.lat * trs.holoop.freq))
        if hasattr(self,'ttloop'):
            parser.set('RTC','LoopGain_LO', str(trs.ttloop.gain))
            parser.set('RTC','SensorFrameRate_LO', str(trs.ttloop.freq))
            parser.set('RTC','LoopDelaySteps_LO', str(trs.ttloop.lat * trs.ttloop.freq))
        
        with open(self.path_ini, 'w') as configfile:
            parser.write(configfile)