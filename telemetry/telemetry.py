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
import fourier.FourierUtils as FourierUtils
import telemetry.keckUtils as keckUtils
from telemetry.massdimm import fetch_data
from telemetry.massdimm import DIMM
from telemetry.massdimm import MASS
from telemetry.massdimm import MASSPROF
from telemetry.massdimm import CombineMASSandDIMM

#%%
class structtype():
    pass

class telemetry:
    
    
    def __init__(self,path_trs,path_img,path_calib,path_save='./',nLayer=1,verbose=False):
        
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
            self.restoringCalibrationData(nLayer=nLayer)
            
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
        self.wfs.ron    = 3.0
        self.wfs.pixel_scale = 800
        self.wfs.fov = 3200
        # tipTilt
        self.tipTilt = structtype()
        self.tipTilt.tilt2meter = 12.68e-6 # arcsec of tilt to OPD over the Keckpupil 
        self.tipTilt.ron    = 3.0
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
            # image
            self.cam.image = fits.getdata(self.path_img) #in DN
            self.cam.fov   = self.cam.image.shape[0] #square image
            # positions
            self.cam.zenith = [0.0]
            self.cam.azimuth = [0.0]
            
            # wavelengths and transmission
            self.path_filter = self.path_calib + '/NIRC2_FILTERS/'
            self.cam.wvl, self.cam.bw, self.cam.transmission, self.cam.dispersion = keckUtils.samplingFilterProfile(self.path_filter,hdr)
            
            # exposure configuration
            self.cam.ittime,self.cam.coadds,self.cam.sampmode,self.cam.nreads,self.cam.ron,self.cam.gain = keckUtils.getExposureConfig(hdr)
            self.cam.saturation = keckUtils.getSaturationLevel(self.cam.name)
            if self.cam.image.max() > self.cam.saturation*self.cam.coadds:
                print('WARNING : the image is likely to be saturated')
            # ron in e- and gain in e-/DN
            self.cam.ronDN = np.sqrt(self.cam.coadds)*self.cam.ron/self.cam.gain
            # NCPA
            self.cam.path_ncpa = self.path_calib + '/NIRC2_STAT/' +  keckUtils.getNCPAstr(hdr)
            
            
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('THE OSIRIS CASE IS NOT YET IMPLEMENTED\n')
            return 0
        
    def restoringCalibrationData(self,nLayer=1):
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
            
        # compressing
        if nLayer < len(self.atm.Cn2Heights):
            _,self.atm.wSpeed = FourierUtils.eqLayers(self.atm.Cn2Weights,np.array(self.atm.wSpeed),nLayer)
            _,self.atm.wDir   = FourierUtils.eqLayers(self.atm.Cn2Weights,np.array(self.atm.wDir),nLayer,power=1)
            self.atm.Cn2Weights,self.atm.Cn2Heights = FourierUtils.eqLayers(self.atm.Cn2Weights,np.array(self.atm.Cn2Heights),nLayer)
            
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
            self.tipTilt.slopes  = trsData.A['RESIDUALWAVEFRONT'][0][:,self.dm.nCom:self.dm.nCom+2]# %angle in arcsec
            self.tipTilt.com     = trsData.A['TTCOMMANDS'][0]
       
        self.tipTilt.slopes  = self.tipTilt.tilt2meter*self.tipTilt.slopes
        self.tipTilt.slopes -= np.mean(self.tipTilt.slopes,axis=0) 
        self.tipTilt.com     = self.tipTilt.tilt2meter*self.tipTilt.com
        self.tipTilt.com    -= np.mean(self.tipTilt.com,axis=0) 
        self.tipTilt.nExp    = self.tipTilt.slopes.shape[0]
        
        # 3\ Get system matrices and reconstructed wavefront
        
        #3.1\ Get DM commands reconstructors from slopes
        MC            = np.reshape(trsData['rx'],(self.dm.nCom+3,self.wfs.nSl,trsData['NREC'])) #command matrix
        self.mat.R    = self.dm.volt2meter*MC[:self.dm.nCom,:,:]
        self.mat.Rtt  = self.tipTilt.tilt2meter*MC[self.dm.nCom:self.dm.nCom+2,:,:]
        
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
        self.holoop.tf.num = trsData['DM_SERVO'][0:4]
        self.holoop.tf.den = trsData['DM_SERVO'][4:]
        self.holoop.noise  = 0.0
        self.ttloop.gain   = trsData['DT_SERVO'][0]
        self.ttloop.tf.num = trsData['DT_SERVO'][0:4]
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
        parser.set('telescope','ObscurationRatio', str(self.tel.cobs))
        parser.set('telescope','ZenithAngle', str(self.tel.zenith_angle))
        parser.set('telescope','Resolution', str(self.tel.resolution))
        parser.set('telescope','PupilAngle', str(self.tel.pupilAngle))
        parser.set('telescope','PathPupil','\'' + self.tel.path_pupil + '\'')
        parser.set('telescope','PathStatic','\'' + self.cam.path_ncpa + '\'')
        
        # updating the atmosphere config
        if not parser.has_section('atmosphere'):
            parser.add_section('atmosphere')
        parser.set('atmosphere','AtmosphereWavelength', str(self.atm.wvl))
        parser.set('atmosphere','Seeing', str(self.atm.seeing))
        parser.set('atmosphere','L0', str(self.atm.L0))
        parser.set('atmosphere','Cn2Weights', str(self.atm.Cn2Weights))
        parser.set('atmosphere','Cn2Heights', str(self.atm.Cn2Heights))
        parser.set('atmosphere','WindSpeed', str(self.atm.wSpeed))
        parser.set('atmosphere','WindDirection', str(self.atm.wDir))

        # updating the HO WFS config
        if not parser.has_section('sensor_HO'):
            parser.add_section('sensor_HO')
        parser.set('sensor_HO','PixelScale', str(self.wfs.pixel_scale))
        parser.set('sensor_HO','FiedOfView', str(self.wfs.fov))
        parser.set('sensor_HO','NumberLenslets', str(self.wfs.nSubap))
        parser.set('sensor_HO','LoopGain', str(self.holoop.gain))
        parser.set('sensor_HO','SensorFrameRate', str(self.holoop.freq))
        parser.set('sensor_HO','LoopDelaySteps', str(self.holoop.lat * self.holoop.freq))
        parser.set('sensor_HO','NoiseVariance', str(0.0))
        
        # GUIDE STARS
        if not parser.has_section('sources_HO'):
            parser.add_section('sources_HO')
            parser.set('sources_HO','Wavelength', str(self.wfs.wvl))

        if self.aoMode == 'LGS':
            parser.set('sources_HO','Height', str(self.lgs.height))
            parser.set('sources_HO','Zenith',str(self.lgs.zenith))
            parser.set('sources_HO','Azimuth',str(self.lgs.azimuth))
            if not parser.has_section('sources_LO'):
                parser.add_section('sources_LO')
            parser.set('sources_LO','Zenith',str(self.ngs.zenith))
            parser.set('sources_LO','Azimuth',str(self.ngs.azimuth))
        else:
            parser.set('sources_HO','Height',str(0))
            parser.set('sources_HO','Zenith',str(self.ngs.zenith))
            parser.set('sources_HO','Azimuth',str(self.ngs.azimuth))
            
        # updating the DM config
        if not parser.has_section('DM'):
            parser.add_section('DM')
        parser.set('DM','NumberActuators', str(self.dm.nActuators))
        parser.set('DM','DmPitchs', str(self.dm.pitch))
        parser.set('DM','InfModel', '\'xinetics\'')
        parser.set('DM','InfCoupling', str(0.11))
        
        # updating the imager config
        if not parser.has_section('sources_science'):
            parser.add_section('sources_science')
        parser.set('sources_science','Zenith', str(self.cam.zenith))
        parser.set('sources_science','Azimuth', str(self.cam.azimuth))
        parser.set('sources_science','Wavelength', str([np.mean(self.cam.wvl)]))
        
        if not parser.has_section('sensor_science'):
            parser.add_section('sensor_science')
        parser.set('sensor_science','FiedOfView', str(self.cam.fov))
        parser.set('sensor_science','PixelScale', str(self.cam.psInMas))

        if len(self.cam.wvl) > 1:
            parser.set('sensor_science','Transmittance', str(self.cam.transmission))
            parser.set('sensor_science','SpectralBandwidth', str(self.cam.bw))
            parser.set('sensor_science','Dispersion', str(self.cam.dispersion))
            
        with open(self.path_ini, 'w') as configfile:
            parser.write(configfile)
            