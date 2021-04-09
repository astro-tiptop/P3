#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:24:36 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
from scipy.io import readsav
import os
from astropy.io import fits
from configparser import ConfigParser

from aoSystem.deformableMirror import deformableMirror
import psfr.keckUtils as keckUtils
from psfr.massdimm import fetch_data
from psfr.massdimm import DIMM
from psfr.massdimm import MASS
from psfr.massdimm import MASSPROF
from psfr.massdimm import CombineMASSandDIMM

#%%
class structtype():
    pass

class telemetry:
    
    
    def __init__(self,path_trs,path_img,path_calib,path_save='./',verbose=False):
        
        # Check the telemetry path
        self.path_trs = path_trs
        if path_trs == None or path_trs == [] or path_trs == '':
            print('No telemetry file')
            self.path_trs = None
            self.path_calib = None
        else:
            # Check the telemetry file
            if os.path.isfile(path_trs) == False:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('The self file does not exist\n')
                return
            
            
        # Check the presence of the calibration folder
        self.path_calib = path_calib
        if os.path.isdir(path_calib) == False:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The calibration folder does not exist\n')
            return
            
        # Check the image file
        self.path_img = path_img
        if os.path.isfile(path_img) == False:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The image file does not exist\n')
            return
        
        self.path_save = path_save
        
        #1\ instantiating the field
        self.instantiatingFields()
        
        #2\ restoring instrument data
        self.restoringInstrumentData()
    
        if self.path_trs != None:
            #3\ restoring calibration data
            self.restoringCalibrationData()
            
            #4\ CREATE TELEMETRY LEVEL 0
            self.restoringTelemetry(verbose=verbose)
        
            #5\ CREATE TELEMETRY LEVEL 1
        
            #6\ CREATE TELEMETRY LEVEL 2
            if path_save!=None:
                self.CreateTelemetryLevel2()
        
        
    def instantiatingFields(self):
        """
            Instantiating the data structures for the Keck AO system"
        """
        #NOTE : SO FAR I"M USING STRUCTURES, BUT WE COULD RELY ON CLASSES IN THE aoSystem FOLDER
        # telescope
        self.tel = structtype()
        self.tel.D           = 10.5  # DM actuators pitch multiplied by number of actuators
        self.tel.cobs        = 0.2311 # central obstruction ratio
        self.tel.resolution  = 100    
        # atmosphere
        self.atm = structtype()
        # wfs
        self.src = structtype()
        # NGS
        self.ngs = structtype()
        self.ngs.zenith = [0.0]
        self.ngs.azimuth = [0.0]
        # LGS
        self.lgs = structtype()
        self.lgs.height = 90e3
        self.lgs.zenith = [0.0]
        self.lgs.azimuth = [0.0]
        # wfs
        self.wfs = structtype()  
        self.wfs.nSubap = 20
        self.wfs.nSl    = 608 # Number of slopes measurements within the pupil (x and y)
        self.wfs.theta  = 90
        # tipTilt
        self.tipTilt = structtype()
        self.tipTilt.tilt2meter = 12.68e-6 # arcsec of tilt to OPD over the Keckpupil 
        # dm
        self.dm = structtype()
        self.dm.volt2meter   = 0.6e-6 # conversion factor from volts to meter OPD
        self.dm.nActuators   = 21     # 1D Number of actuators                                            
        self.dm.nCom         = 349   # Number of total actuators within the pupil
        self.dm.pitch        = 0.5625
        self.dm.mechCoupling = 0.11
        self.dm.modes        = 'xinetics'
        #self.dm.resolution   = 2*self.dm.nActuators-1
        self.dm.condmax      = 1e2
        # cam
        self.cam = structtype()
        # rec
        self.rec = structtype()
        # mat
        self.mat = structtype()
        # holoop
        self.holoop = structtype()
        self.holoop.tf = structtype()
        # ttloop
        self.ttloop = structtype()
        self.ttloop.tf = structtype()
        #results
        self.seeing = structtype()
        self.noise  = structtype()
        self.noise.Cn_ao = 0
        self.noise.Cn_tt = 0
       
    def restoringInstrumentData(self):
        """
        """
        hdr = fits.getheader(self.path_img)
        
        #1\ Dates
        self.obsdate = hdr['DATE-OBS'].replace('-','')
        self.acqtime = hdr['EXPSTOP'].replace(':','_')
        
        # 2\ Restore the telescope configuration
        self.tel.name         = hdr['TELESCOP'].replace(' ','_')
        self.tel.zenith_angle, self.tel.airmass = keckUtils.getTelescopeAirmass(hdr)
        self.tel.pupilAngle   = keckUtils.getPA(hdr)
        self.tel.pupilMaskName= keckUtils.getPupilMask(hdr)
        _,self.aoMode         = keckUtils.getStagePositionWFS(hdr)        
        self.tel.path_pupil   = self.path_calib + '/NIRC2_MASK/keck_pupil_largeHex_272px.fits'
        
        # 3\ Restore the instrument configuration
        self.cam.name = keckUtils.getInstName(hdr)
        self.cam.psInMas = keckUtils.getScale(hdr)
        
        if self.cam.name == 'NIRC2':
            # wavelengths and transmission
            self.path_filter = self.path_calib + '/NIRC2_FILTERS/'
            self.cam.wvl, self.cam.bw, self.cam.transmission, self.cam.dispersion = keckUtils.samplingFilterProfile(self.path_filter,hdr)
            # exposure configuration
            self.cam.ittime,self.cam.coadds,self.cam.sampmode,self.cam.nreads,self.cam.ron,self.cam.gain = keckUtils.getExposureConfig(hdr)
            self.cam.saturation = keckUtils.getSaturationLevel(self.cam.name)
            # NCPA
            self.cam.path_ncpa = self.path_calib + '/NIRC2_STAT/' +  keckUtils.getNCPAstr(hdr)
            # image
            self.cam.image = fits.getdata(self.path_img)
            self.cam.fov   = self.cam.image.shape[0] #square image
            # positions
            self.cam.zenith = [0.0]
            self.cam.azimuth = [0.0]
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('THE OSIRIS CASE IS NOT YET IMPLEMENTED\n')
            return 0
        
    def restoringCalibrationData(self):
        """
            Collecting calibration data for the Keck AO system
        """
        #1\ Valid apertures/actuators
        file = open(self.path_calib + '/KECKAO/keckValidActuatorMap.txt','r')
        self.dm.validActuators = np.array(np.genfromtxt(file)).astype('bool')
        file.close()
        
        file = open(self.path_calib + '/KECKAO/keckValidSubapMap.txt','r')
        self.wfs.validSubaperture = np.array(np.genfromtxt(file)).astype('bool')
        file.close()
        
        #2\ DM influence functions and filters
        if self.path_trs != None:
            dm = deformableMirror(self.dm.nActuators,self.dm.pitch,self.dm.mechCoupling,modes=self.dm.modes)
            self.mat.dmIF = dm.setInfluenceFunction(self.tel.resolution)
            self.mat.dmIF_inv = np.linalg.pinv(self.mat.dmIF,rcond=1/self.dm.condmax)
            self.mat.Hdm  = np.matmul(self.mat.dmIF,self.mat.dmIF_inv)
        
        #3\ MASS/DIMM
        # managing the saving folder
        if not os.path.isdir(self.path_calib+'/MASSDIMM/'):
            os.mkdir(self.path_calib+'/MASSDIMM/')
        self.path_massdimm = self.path_calib+'/MASSDIMM/'+self.obsdate+'/'
        if not os.path.isfile(self.path_massdimm):
            status = fetch_data(self.obsdate,self.path_massdimm)
        
        # median conditions at MaunaKea
        self.atm.wvl        = 500e-9
        self.atm.wSpeed     = [6.8, 6.9, 7.1, 7.5, 10.0, 26.9, 18.5]
        self.atm.wDir       = list(np.zeros(len(self.atm.wSpeed)))
        self.atm.L0         = 25
        self.atm.Cn2Heights = [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]

        if status == 0:
            # default atmosphere : median conditions
            self.atm.seeing     = 3600*180/np.pi * 0.98*500e-9/0.16
            self.atm.Cn2Weights = [0.517, 0.119, 0.063, 0.061, 0.105, 0.081, 0.054]
        else:
            def AcqTime2hhmmss(acqtime):
                StrTime = (acqtime.split('_')) 
                return float(StrTime[0]) + float(StrTime[1])/60 + float(StrTime[2])/3600
               
            hhmmss = AcqTime2hhmmss(self.acqtime)
            # read the DIMM data to get the seeing
            dimm = DIMM(self.path_massdimm + '/' + self.obsdate + '.dimm.dat')
            self.atm.seeing,_   = dimm.indexTime([hhmmss])
            
            # read the MASS data to get the Cn2 profile
            mass        = MASS(self.path_massdimm + '/' + self.obsdate + '.mass.dat')
            SeeingAlt,_ = mass.indexTime([hhmmss])
            massprof    = MASSPROF(self.path_massdimm + '/' + self.obsdate + '.masspro.dat')
            Cn2Alt,_    = massprof.indexTime([hhmmss])
            
            # combine MASS and DIMM data
            self.atm.Cn2 = CombineMASSandDIMM(self.atm.seeing,SeeingAlt,Cn2Alt,wvl=self.atm.wvl)
            self.atm.Cn2Weights = self.atm.Cn2/self.atm.Cn2.sum()
            
    def restoringTelemetry(self,verbose=False):
              
        
        # 1\ Restore telemetry data and header
        trsData = readsav(self.path_trs,verbose=verbose)
        hdr     = fits.getheader(self.path_img)
        
        # 2\ Get AO control loop data                 
    
        #2.1. Get slopes in pixels unit
        self.wfs.slopes = trsData.A['OFFSETCENTROID'][0]
        self.wfs.nSl    = self.wfs.slopes.shape[1]
        self.wfs.nExp   = self.wfs.slopes.shape[0]
    
        #2.2 z-position of the WFS stage and AO mode
        self.wfs.zstage_defocus, self.aoMode = keckUtils.getStagePositionWFS(hdr)
        
        if np.any(self.wfs.validSubaperture):   
            self.wfs.intensity = np.zeros((self.wfs.nExp,400))
            self.wfs.intensity[:,self.wfs.validSubaperture.reshape(-1)] = trsData.A['SUBAPINTENSITY'][0]
            self.wfs.intensity = self.wfs.intensity.reshape((self.wfs.nExp,self.wfs.nSubap,self.wfs.nSubap))
        else:
            self.wfs.intensity = trsData.A['SUBAPINTENSITY'][0]
        self.wfs.wvl = keckUtils.getWFSwavelength(hdr)
        
        #2.3. Get DMs commands in OPD units
        self.dm.com = trsData.A['DMCOMMAND'][0]*self.dm.volt2meter
        self.dm.nCom= self.dm.com.shape[1]
        
        #2.4. Get tip-tilt measurements and conversion into OPD
        if 'b' in trsData:
            self.tipTilt.slopes    = trsData.B['DTTCENTROIDS'][0]
            self.tipTilt.com       = trsData.B['DTTCOMMANDS'][0]
            self.tipTilt.intensity = trsData.B['APDCOUNTS'][0]
            self.tipTilt.tilt2meter= 3.2*1.268e-05
        else:
            self.tipTilt.tilt2meter = 12.68e-6
            self.tipTilt.slopes  = trsData.A['RESIDUALWAVEFRONT'][0][:,self.dm.nCom:self.dm.nCom+1]# %angle in arcsec
            self.tipTilt.com     = trsData.A['TTCOMMANDS'][0]
       
        self.tipTilt.slopes  = self.tipTilt.tilt2meter*self.tipTilt.slopes
        self.tipTilt.slopes -= np.mean(self.tipTilt.slopes,axis=0) 
        self.tipTilt.com     = self.tipTilt.tilt2meter*self.tipTilt.com
        self.tipTilt.com    -= np.mean(self.tipTilt.com,axis=0) 
        self.tipTilt.nExp    = self.tipTilt.slopes.shape[0]
        
        # 3\ Get system matrices and reconstructed wavefront
        
        #3.1\ Get DM commands reconstructors from slopes
        MC            = np.reshape(trsData['rx'],(self.wfs.nSl,self.dm.nCom+3,trsData['NREC'])) #command matrix
        self.mat.R    = self.dm.volt2meter*np.transpose(MC[:,:self.dm.nCom,:],[1,0,2])
        self.mat.Rtt  = self.tipTilt.tilt2meter*np.transpose(MC[:,self.dm.nCom:self.dm.nCom+1,:],[1,0,2])
        
        #3.2\ Get the reconstructed wavefront in OPD and in the actuators space
        self.rec.res   = self.dm.volt2meter*trsData.A['RESIDUALWAVEFRONT'][0][:,0:self.dm.nCom]
        self.rec.res  -= np.mean(self.rec.res,axis=0) 
        self.rec.focus = trsData.A['RESIDUALWAVEFRONT'][0][-1,:]
        self.rec.res  -= np.mean(self.rec.focus,axis=0) 
        
        # fill vector to get 21x21 actuators
        if np.any(self.dm.validActuators):   
            idG          = self.dm.validActuators.reshape(-1)
            u            = np.zeros((self.wfs.nExp,self.dm.nActuators**2))
            u[:,idG]     = self.rec.res
            self.rec.res = u
            u            = np.zeros((self.wfs.nExp,self.dm.nActuators**2))
            u[:,idG]     = self.dm.com
            self.dm.com  = u
        else:
            self.rec.res = self.rec.res
            self.dm.com  = self.dm.com
        

        # 4\ Get the loop status and model transfer function
        #4.1. Delays
        self.holoop.lat,self.ttloop.lat = keckUtils.estimateLoopDelay(hdr)
        
        #4.2. Frequency
        self.holoop.freq = 1/(100e-9*np.mean(np.diff(trsData.A['TIMESTAMP'][0])))
        self.ttloop.freq =self.holoop.freq
        if 'b' in trsData:
            self.ttloop.freq =  1/(100e-9*np.mean(np.diff(trsData.B['TIMESTAMP'][0])))
        
        #4.3. RTC controller HO loop
        self.holoop.gain   = trsData['DM_SERVO'][0]
        self.holoop.tf.num = trsData['DM_SERVO'][0:3]
        self.holoop.tf.den = trsData['DM_SERVO'][4:]
        self.holoop.noise  = 0.0
        self.ttloop.gain   = trsData['DT_SERVO'][0]
        self.ttloop.tf.num = trsData['DT_SERVO'][0:3]
        self.ttloop.tf.den = trsData['DT_SERVO'][4:]
        
        
    def CreateTelemetryLevel1(self):
        """
        """
        pass
    
    def CreateTelemetryLevel2(self):
        
        # create the .ini file
        file = self.tel.name + '_' + self.cam.name + '_' + self.obsdate + '_' + self.acqtime + '.ini'
        self.path_ini = self.path_save + '/'+ file
        if not os.path.exists(self.path_ini):
            with open(self.path_ini, 'w'): pass
        else:
            print('WARNING: the .ini file already exists')

        # open the .ini file
        parser = ConfigParser()
        parser.optionxform = str
        parser.read(self.path_ini)
        
        # updating the telescope parameters
        if not parser.has_section('telescope'):
            parser.add_section('telescope')
        parser.set('telescope','TelescopeDiameter',str(self.tel.D))
        parser.set('telescope','obscurationRatio', str(self.tel.cobs))
        parser.set('telescope','zenithAngle', str(self.tel.zenith_angle))
        parser.set('telescope','resolution', str(self.tel.resolution))
        parser.set('telescope','pupilAngle', str(self.tel.pupilAngle))
        parser.set('telescope','path_pupil','\'' + self.tel.path_pupil + '\'')
        parser.set('telescope','path_static','\'' + self.cam.path_ncpa + '\'')
        
        # updating the atmosphere config
        if not parser.has_section('atmosphere'):
            parser.add_section('atmosphere')
        parser.set('atmosphere','atmosphereWavelength', str(self.atm.wvl))
        parser.set('atmosphere','seeing', str(self.atm.seeing))
        parser.set('atmosphere','L0', str(self.atm.L0))
        parser.set('atmosphere','Cn2Weights', str(self.atm.Cn2Weights))
        parser.set('atmosphere','Cn2Heights', str(self.atm.Cn2Heights))
        parser.set('atmosphere','wSpeed', str(self.atm.wSpeed))
        parser.set('atmosphere','wDir', str(self.atm.wDir))

        # updating the HO WFS config
        if not parser.has_section('SENSOR_HO'):
            parser.add_section('SENSOR_HO')
        parser.set('SENSOR_HO','nLenslet_HO', str(self.wfs.nSubap))
        parser.set('SENSOR_HO','SensingWavelength_HO', str(self.wfs.wvl))
        parser.set('SENSOR_HO','loopGain_HO', str(self.holoop.gain))
        parser.set('SENSOR_HO','SensorFrameRate_HO', str(self.holoop.freq))
        parser.set('SENSOR_HO','loopDelaySteps_HO', str(self.holoop.lat * self.holoop.freq))
        parser.set('SENSOR_HO','noiseVariance_HO', str(self.holoop.noise))
        # GUIDE STARS
        if not parser.has_section('GUIDESTARS_HO'):
            parser.add_section('GUIDESTARS_HO')
        if self.aoMode == 'LGS':
            parser.set('GUIDESTARS_HO','GuideStarHeight_HO', str(self.lgs.height))
            parser.set('GUIDESTARS_HO','GuideStarZenith_HO',str(self.lgs.zenith))
            parser.set('GUIDESTARS_HO','GuideStarAzimuth_HO',str(self.lgs.azimuth))
            if not parser.has_section('GUIDESTARS_LO'):
                parser.add_section('GUIDESTARS_LO')
            parser.set('GUIDESTARS_LO','GuideStarZenith_LO',str(self.ngs.zenith))
            parser.set('GUIDESTARS_LO','GuideStarAzimuth_LO',str(self.ngs.azimuth))
        else:
            parser.set('GUIDESTARS_HO','GuideStarHeight_HO',str(0))
            parser.set('GUIDESTARS_HO','GuideStarZenith_HO',str(self.ngs.zenith))
            parser.set('GUIDESTARS_HO','GuideStarAzimuth_HO',str(self.ngs.azimuth))
            
        # updating the DM config
        if not parser.has_section('DM'):
            parser.add_section('DM')
        parser.set('DM','DmPitchs', str(self.dm.pitch))
        
        # updating the imager config
        if not parser.has_section('PSF_DIRECTIONS'):
            parser.add_section('PSF_DIRECTIONS')
        parser.set('PSF_DIRECTIONS','ScienceZenith', str(self.cam.zenith))
        parser.set('PSF_DIRECTIONS','ScienceAzimuth', str(self.cam.azimuth))
        parser.set('PSF_DIRECTIONS','psf_FoV', str(self.cam.fov))
        parser.set('PSF_DIRECTIONS','psInMas', str(self.cam.psInMas))
        parser.set('PSF_DIRECTIONS','ScienceWavelength', str(np.mean(self.cam.wvl)))
        
        if len(self.cam.wvl) > 1:
            if not parser.has_section('POLYCHROMATISM'):
                parser.add_section('POLYCHROMATISM')
            parser.set('POLYCHROMATISM','transmittance', str(self.cam.transmission))
            parser.set('POLYCHROMATISM','spectralBandwidth', str(self.cam.bw))
            parser.set('POLYCHROMATISM','dispersionX', str(self.cam.dispersion[0]))
            parser.set('POLYCHROMATISM','dispersionY', str(self.cam.dispersion[1]))
            
        with open(self.path_ini, 'w') as configfile:
            parser.write(configfile)
            