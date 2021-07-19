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
        if os.path.exists(self.path_ini):
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
        parser.set('telescope','PathStaticOn','\'' + sysdiag.trs.cam.path_ncpa + '\'')
        parser.set('telescope','PathStatModes','\'' + sysdiag.trs.tel.path_telstat + '\'')
        
        #%% ATMOSPHERE         
        if not parser.has_section('atmosphere'):
            parser.add_section('atmosphere')
            
        parser.set('atmosphere','Wavelength', str(sysdiag.trs.atm.wvl))
        parser.set('atmosphere','Seeing', str(sysdiag.trs.atm.seeing))
        parser.set('atmosphere','L0', str(sysdiag.trs.atm.L0))
        parser.set('atmosphere','Cn2Weights', str(sysdiag.trs.atm.Cn2Weights))
        parser.set('atmosphere','Cn2Heights', str(sysdiag.trs.atm.Cn2Heights))
        parser.set('atmosphere','WindSpeed', str(sysdiag.trs.atm.wSpeed))
        parser.set('atmosphere','WindDirection', str(sysdiag.trs.atm.wDir))

        #%% SCIENCE SOURCES
        # updating the imager config
        if not parser.has_section('sources_science'):
            parser.add_section('sources_science')
        parser.set('sources_science','Wavelength', str([np.mean(sysdiag.trs.cam.wvl)]))
        parser.set('sources_science','Zenith', str(sysdiag.trs.cam.zenith))
        parser.set('sources_science','Azimuth', str(sysdiag.trs.cam.azimuth))
        
        
        #%% GUIDE STARS
        if not parser.has_section('sources_HO'):
            parser.add_section('sources_HO')
            parser.set('sources_HO','Wavelength', str(sysdiag.trs.wfs.wvl))

        if sysdiag.trs.aoMode == 'LGS':
            parser.set('sources_HO','Zenith',str(sysdiag.trs.lgs.zenith))
            parser.set('sources_HO','Azimuth',str(sysdiag.trs.lgs.azimuth))
            parser.set('sources_HO','Height', str(sysdiag.trs.lgs.height))
            if not parser.has_section('sources_LO'):
                parser.add_section('sources_LO')
            parser.set('sources_LO','Wavelength', str(sysdiag.trs.tipTilt.wvl))
            parser.set('sources_LO','Zenith',str(sysdiag.trs.ngs.zenith))
            parser.set('sources_LO','Azimuth',str(sysdiag.trs.ngs.azimuth))
        else:
            parser.set('sources_HO','Zenith',str(sysdiag.trs.ngs.zenith))
            parser.set('sources_HO','Azimuth',str(sysdiag.trs.ngs.azimuth))
            parser.set('sources_HO','Height',str(0))
            
         #%% WFS 
        # updating the HO WFS config
        if not parser.has_section('sensor_HO'):
            parser.add_section('sensor_HO')
        parser.set('sensor_HO','WfsType', '\''+sysdiag.trs.wfs.type+'\'')
        parser.set('sensor_HO','Modulation', str(sysdiag.trs.wfs.modulation))
        parser.set('sensor_HO','PixelScale', str(sysdiag.trs.wfs.pixel_scale))
        parser.set('sensor_HO','FieldOfView', str(sysdiag.trs.wfs.fov))
        parser.set('sensor_HO','Binning', str(sysdiag.trs.wfs.binning))
        parser.set('sensor_HO','NumberPhotons', str(sysdiag.trs.wfs.nph))
        parser.set('sensor_HO','SpotFWHM', str(sysdiag.trs.wfs.spot_fwhm))
        parser.set('sensor_HO','SpectralBandwidth', str(sysdiag.trs.wfs.bw))
        parser.set('sensor_HO','Transmittance', str(sysdiag.trs.wfs.tr))
        parser.set('sensor_HO','Dispersion', str(sysdiag.trs.wfs.disp))
        parser.set('sensor_HO','SigmaRON', str(sysdiag.trs.wfs.ron))
        parser.set('sensor_HO','Dark', str(sysdiag.trs.wfs.dark))
        parser.set('sensor_HO','SkyBackground', str(sysdiag.trs.wfs.sky))
        parser.set('sensor_HO','Gain', str(sysdiag.trs.wfs.gain))
        parser.set('sensor_HO','ExcessNoiseFactor', str(sysdiag.trs.wfs.excess))
        parser.set('sensor_HO','NumberLenslets', str(sysdiag.trs.wfs.nSubap))
        parser.set('sensor_HO','SizeLenslets', str(sysdiag.trs.wfs.dsub))
        parser.set('sensor_HO','NoiseVariance', str(sysdiag.trs.wfs.noiseVar))
        parser.set('sensor_HO','Algorithm', '\''+sysdiag.trs.wfs.algo+'\'')
        parser.set('sensor_HO','WindowRadiusWCoG', str(sysdiag.trs.wfs.win))
        parser.set('sensor_HO','ThresholdWCoG', str(sysdiag.trs.wfs.thres))
        parser.set('sensor_HO','NewValueThrPix', str(sysdiag.trs.wfs.new))
        
        if sysdiag.trs.aoMode == 'LGS':
            if not parser.has_section('sensor_LO'):
                parser.add_section('sensor_LO')
            parser.set('sensor_LO','PixelScale', str(sysdiag.trs.tipTilt.pixel_scale))
            parser.set('sensor_LO','FieldOfView', str(sysdiag.trs.tipTilt.fov))
            parser.set('sensor_LO','Binning', str(sysdiag.trs.tipTilt.binning))
            parser.set('sensor_LO','NumberPhotons', str(sysdiag.trs.tipTilt.nph))
            parser.set('sensor_LO','SpotFWHM', str(sysdiag.trs.tipTilt.spot_fwhm))
            parser.set('sensor_LO','SpectralBandwidth', str(sysdiag.trs.tipTilt.bw))
            parser.set('sensor_LO','Transmittance', str(sysdiag.trs.tipTilt.tr))
            parser.set('sensor_LO','Dispersion', str(sysdiag.trs.tipTilt.disp))
            parser.set('sensor_LO','SigmaRON', str(sysdiag.trs.tipTilt.ron))
            parser.set('sensor_LO','Dark', str(sysdiag.trs.tipTilt.dark))
            parser.set('sensor_LO','SkyBackground', str(sysdiag.trs.tipTilt.sky))
            parser.set('sensor_LO','Gain', str(sysdiag.trs.tipTilt.gain))
            parser.set('sensor_LO','ExcessNoiseFactor', str(sysdiag.trs.tipTilt.excess))
            parser.set('sensor_LO','NumberLenslets', str(sysdiag.trs.tipTilt.nSubap))
            parser.set('sensor_LO','SizeLenslets', str(sysdiag.trs.tipTilt.dsub))
            parser.set('sensor_LO','NoiseVariance', str(sysdiag.trs.tipTilt.noiseVar))
            parser.set('sensor_LO','Algorithm', '\''+sysdiag.trs.tipTilt.algo+'\'')
            parser.set('sensor_LO','WindowRadiusWCoG', str(sysdiag.trs.tipTilt.win))
            parser.set('sensor_LO','ThresholdWCoG', str(sysdiag.trs.tipTilt.thres))
            parser.set('sensor_LO','NewValueThrPix', str(sysdiag.trs.tipTilt.new))
        
        
         #%% DMS
        if not parser.has_section('DM'):
            parser.add_section('DM')
        parser.set('DM','NumberActuators', str(sysdiag.trs.dm.nActuators))
        parser.set('DM','DmPitchs', str(sysdiag.trs.dm.pitch))
        parser.set('DM','InfModel', '\''+sysdiag.trs.dm.modes+'\'')
        parser.set('DM','InfCoupling', str(sysdiag.trs.dm.mechCoupling))
        
        parser.set('DM','DmHeights', str(sysdiag.trs.dm.heights))
        parser.set('DM','OptimizationZenith', str(sysdiag.trs.dm.opt_zen))
        parser.set('DM','OptimizationAzimuth', str(sysdiag.trs.dm.opt_azi))
        parser.set('DM','OptimizationWeight', str(sysdiag.trs.dm.opt_weight))
        parser.set('DM','OptimizationConditioning', str(sysdiag.trs.dm.opt_cond))
        parser.set('DM','NumberReconstructedLayers', str(sysdiag.trs.dm.nrec))
        parser.set('DM','AoArea', '\''+sysdiag.trs.dm.area+'\'')
        
        #%% RTC    
        if not parser.has_section('RTC'):
            parser.add_section('RTC')
        parser.set('RTC','LoopGain_HO', str(sysdiag.trs.holoop.gain))
        parser.set('RTC','SensorFrameRate_HO', str(sysdiag.trs.holoop.freq))
        parser.set('RTC','LoopDelaySteps_HO', str(sysdiag.trs.holoop.lat * sysdiag.trs.holoop.freq))
        parser.set('RTC','ResidualError', str(sysdiag.trs.rec.wfe))
        if hasattr(sysdiag.trs,'ttloop'):
            parser.set('RTC','LoopGain_LO', str(sysdiag.trs.ttloop.gain))
            parser.set('RTC','SensorFrameRate_LO', str(sysdiag.trs.ttloop.freq))
            parser.set('RTC','LoopDelaySteps_LO', str(sysdiag.trs.ttloop.lat * sysdiag.trs.ttloop.freq))
            
        #%% SCIENCE SOURCES AND DETECTOR    
        if not parser.has_section('sensor_science'):
            parser.add_section('sensor_science')
        parser.set('sensor_science','Name', str('\''+sysdiag.trs.cam.name+'\''))
        parser.set('sensor_science','FieldOfView', str(sysdiag.trs.cam.fov))
        parser.set('sensor_science','PixelScale', str(sysdiag.trs.cam.psInMas))
        parser.set('sensor_science','SigmaRON', str(sysdiag.trs.cam.ron))
        parser.set('sensor_science','Gain', str(sysdiag.trs.cam.gain))
        parser.set('sensor_science','Transmittance', str(list(sysdiag.trs.cam.transmission)))
        parser.set('sensor_science','SpectralBandwidth', str(sysdiag.trs.cam.bw))
        parser.set('sensor_science','Dispersion', str(sysdiag.trs.cam.dispersion))
       
        # jitter
        Cj = np.dot(sysdiag.trs.tipTilt.slopes.T,sysdiag.trs.tipTilt.slopes)/sysdiag.trs.tipTilt.slopes.shape[0]
        Cj*= (1000/sysdiag.trs.tipTilt.tilt2meter)**2
        #psInMas = sysdiag.trs.cam.psInMas 
        #1 mas = rad2mas * 4/D * 1e-9 * 1nm
        parser.set('sensor_science','SpotFWHM', \
        str([[np.sqrt(Cj[0,0]), np.sqrt(Cj[1,1]), 
              np.sign(Cj[0,1]) * np.sqrt(abs(Cj[0,1])) ]]))
    
        #%% EXTERNAL PROFILER
        if not parser.has_section('external'):
            parser.add_section('external')
        parser.set('external','Seeing', str(sysdiag.trs.atm.seeing_dimm))
        parser.set('external','SeeingAlt', str(sysdiag.trs.atm.seeing_mass))
        parser.set('external','Cn2Weights', str(sysdiag.trs.atm.Cn2_mass))
        
        #%% WRITING
        with open(self.path_ini, 'w') as configfile:
            parser.write(configfile)
            
        sysdiag.trs.path_ini = self.path_ini