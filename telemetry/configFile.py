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
    

    def __init__(self,sysdiag):
        
        # create the .ini file
        file = sysdiag.trs.tel.name + '_' + sysdiag.trs.cam.name + '_' + sysdiag.trs.obsdate + '_' + sysdiag.trs.acqtime + '.ini'
        self.path_ini = sysdiag.trs.path_save + '/'+ file
        if not os.path.exists(self.path_ini):
            with open(sysdiag.trs.path_ini, 'w'): pass
        else:
            print('WARNING: the .ini file already exists')

        # open the .ini file
        parser = ConfigParser()
        parser.optionxform = str
        parser.read(self.path_ini)
        
        #%% TELESCOPE 
        if not parser.has_section('telescope'):
            parser.add_section('telescope')
        parser.set('telescope','TelescopeDiameter',str(sysdiag.trs.tel.D))
        parser.set('telescope','ObscurationRatio', str(sysdiag.trs.tel.cobs))
        parser.set('telescope','ZenithAngle', str(sysdiag.trs.tel.zenith_angle))
        parser.set('telescope','Resolution', str(sysdiag.trs.tel.resolution))
        parser.set('telescope','PupilAngle', str(sysdiag.trs.tel.pupilAngle))
        parser.set('telescope','PathPupil','\'' + sysdiag.trs.tel.path_pupil + '\'')
        parser.set('telescope','PathStatic','\'' + sysdiag.trs.cam.path_ncpa + '\'')
        parser.set('telescope','PathStatModes','\'' + sysdiag.trs.tel.path_telstat + '\'')
        
        #%% ATMOSPHERE 
        
        # updating the atmosphere config
        if not parser.has_section('atmosphere'):
            parser.add_section('atmosphere')
        parser.set('atmosphere','Wavelength', str(sysdiag.trs.atm.wvl))
        parser.set('atmosphere','Seeing', str(sysdiag.trs.atm.seeing))
        parser.set('atmosphere','L0', str(sysdiag.trs.atm.L0))
        parser.set('atmosphere','Cn2Weights', str(sysdiag.trs.atm.Cn2Weights))
        parser.set('atmosphere','Cn2Heights', str(sysdiag.trs.atm.Cn2Heights))
        parser.set('atmosphere','WindSpeed', str(sysdiag.trs.atm.wSpeed))
        parser.set('atmosphere','WindDirection', str(sysdiag.trs.atm.wDir))

        #%% WFS 
        # updating the HO WFS config
        if not parser.has_section('sensor_HO'):
            parser.add_section('sensor_HO')
        parser.set('sensor_HO','PixelScale', str(sysdiag.trs.wfs.pixel_scale))
        parser.set('sensor_HO','FiedOfView', str(sysdiag.trs.wfs.fov))
        parser.set('sensor_HO','NumberLenslets', str(sysdiag.trs.wfs.nSubap))
        parser.set('sensor_HO','NoiseVariance', str(sysdiag.trs.wfs.noiseVar))
        
        #%% GUIDE STARS
        if not parser.has_section('sources_HO'):
            parser.add_section('sources_HO')
            parser.set('sources_HO','Wavelength', str(sysdiag.trs.wfs.wvl))

        if sysdiag.trs.aoMode == 'LGS':
            parser.set('sources_HO','Height', str(sysdiag.trs.lgs.height))
            parser.set('sources_HO','Zenith',str(sysdiag.trs.lgs.zenith))
            parser.set('sources_HO','Azimuth',str(sysdiag.trs.lgs.azimuth))
            if not parser.has_section('sources_LO'):
                parser.add_section('sources_LO')
            parser.set('sources_LO','Zenith',str(sysdiag.trs.ngs.zenith))
            parser.set('sources_LO','Azimuth',str(sysdiag.trs.ngs.azimuth))
        else:
            parser.set('sources_HO','Height',str(0))
            parser.set('sources_HO','Zenith',str(sysdiag.trs.ngs.zenith))
            parser.set('sources_HO','Azimuth',str(sysdiag.trs.ngs.azimuth))
            
         #%% DMS
         
        # updating the DM config
        if not parser.has_section('DM'):
            parser.add_section('DM')
        parser.set('DM','NumberActuators', str(sysdiag.trs.dm.nActuators))
        parser.set('DM','DmPitchs', str(sysdiag.trs.dm.pitch))
        parser.set('DM','InfModel', '\'xinetics\'')
        parser.set('DM','InfCoupling', str([0.11]))
        
        #%% SCIENCE SOURCES AND DETECTOR    
        
        # updating the imager config
        if not parser.has_section('sources_science'):
            parser.add_section('sources_science')
        parser.set('sources_science','Zenith', str(sysdiag.trs.cam.zenith))
        parser.set('sources_science','Azimuth', str(sysdiag.trs.cam.azimuth))
        parser.set('sources_science','Wavelength', str([np.mean(sysdiag.trs.cam.wvl)]))
        
        if not parser.has_section('sensor_science'):
            parser.add_section('sensor_science')
        parser.set('sensor_science','FiedOfView', str(sysdiag.trs.cam.fov))
        parser.set('sensor_science','PixelScale', str(sysdiag.trs.cam.psInMas))

        if len(sysdiag.trs.cam.wvl) > 1:
            parser.set('sensor_science','Transmittance', str(sysdiag.trs.cam.transmission))
            parser.set('sensor_science','SpectralBandwidth', str(sysdiag.trs.cam.bw))
            parser.set('sensor_science','Dispersion', str(sysdiag.trs.cam.dispersion))
       
        # jitter
        Cj = np.sqrt(np.dot(sysdiag.trs.tipTilt.slopes.T,sysdiag.trs.tipTilt.slopes)/sysdiag.trs.tipTilt.slopes.shape[0])
        Cj*= 1000/sysdiag.trs.tipTilt.tilt2meter
        psInMas = sysdiag.trs.cam.psInMas 
        #1 mas = rad2mas * 4/D * 1e-9 * 1nm
        parser.set('sensor_science','spotFWHM', str([[np.hypot(Cj[1,1],psInMas/2), np.hypot(Cj[0,0],psInMas/2), np.hypot(Cj[0,1],psInMas/2)]]))
        #%% RTC    
        if not parser.has_section('RTC'):
            parser.add_section('RTC')
        parser.set('RTC','LoopGain_HO', str(sysdiag.trs.holoop.gain))
        parser.set('RTC','SensorFrameRate_HO', str(sysdiag.trs.holoop.freq))
        parser.set('RTC','LoopDelaySteps_HO', str(sysdiag.trs.holoop.lat * sysdiag.trs.holoop.freq))
        if hasattr(self,'ttloop'):
            parser.set('RTC','LoopGain_LO', str(sysdiag.trs.ttloop.gain))
            parser.set('RTC','SensorFrameRate_LO', str(sysdiag.trs.ttloop.freq))
            parser.set('RTC','LoopDelaySteps_LO', str(sysdiag.trs.ttloop.lat * sysdiag.trs.ttloop.freq))
        
        with open(self.path_ini, 'w') as configfile:
            parser.write(configfile)
            
        sysdiag.trs.path_ini = self.path_ini