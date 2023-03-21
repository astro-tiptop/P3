#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:24:36 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
import os
from scipy.io import readsav
from astropy.io import fits
from p3.aoSystem.deformableMirror import deformableMirror
import p3.aoSystem.FourierUtils as FourierUtils
import p3.telemetry.keckUtils as keckUtils
from p3.telemetry.massdimm import fetch_data, DIMM, MASS, MASSPROF, CombineMASSandDIMM
from scipy.signal import medfilt2d

#%%
class structtype():
    pass

class telemetryKeck:
    
    
    def __init__(self,path_trs,path_img,path_calib,path_save='./',
                 nLayer=1,verbose=False,addNCPA=True,cam_fov = 150):
        
        # Check the telemetry path
        self.path_trs = path_trs
        if path_trs == None or path_trs == [] or path_trs == '':
            print('No telemetry file')
            self.path_trs = None
            self.path_calib = None
        else:
            # Check the telemetry file
            if os.path.isfile(path_trs) == False:
                raise ValueError('Error : the sav file does not exist\n')
            
        # Check the presence of the calibration folder
        self.path_calib = path_calib
        if os.path.isdir(path_calib) == False:
            raise ValueError('Error : the calibration folder does not exist\n')
            
        # Check the image file
        self.path_img = path_img
        if os.path.isfile(path_img) == False:
            raise ValueError('Error : the image file does not exist\n')
        
        self.path_save = path_save
        self.addNCPA   = addNCPA
        #1\ instantiating the field
        self.instantiatingFields()
        
        #2\ restoring instrument data
        self.restoringInstrumentData(cam_fov=cam_fov)
    
        if self.path_trs != None:
            #3\ restoring calibration data
            self.restoringCalibrationData(nLayer=nLayer)
            
            #4\ CREATE TELEMETRY LEVEL 1 KECK-SPECIFIC
            self.restoringTelemetry(verbose=verbose)
            
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
        self.lgs.zenith = [0.0]
        self.lgs.azimuth = [0.0]
        self.lgs.height = 90e3

        # wfs
        self.wfs             = structtype()  
        self.wfs.type        = 'Shack-Hartmann'
        self.wfs.modulation  = None
        self.wfs.pixel_scale = 800
        self.wfs.fov         = 3200
        self.wfs.binning     = 1
        self.wfs.nph         = None
        self.wfs.spot_fwhm   = [[0.0,0.0,0.0]]
        self.wfs.bw          = 0.0
        self.wfs.tr          = [1.0]
        self.wfs.disp        = [[0.0],[0.0]]
        self.wfs.ron         = 3.0
        self.wfs.dark        = 0.0
        self.wfs.sky         = 0.0
        self.wfs.gain        = 1.0
        self.wfs.excess      = 1.0
        self.wfs.nSubap      = [20]
        self.wfs.dsub        = [0.5625]
        self.wfs.noiseVar    = [None]
        self.wfs.algo        = 'cog'
        self.wfs.win         = 2.0
        self.wfs.thres       = 0.0
        self.wfs.new         = 0.0
        self.wfs.nSl         = [608] # Number of slopes measurements within the pupil (x and y)
        self.wfs.theta       = [90]
        
        # tipTilt
        self.tipTilt             = structtype()
        self.tipTilt.tilt2meter  = [] # arcsec of tilt to OPD over the Keckpupil 
        self.tipTilt.pixel_scale = 800
        self.tipTilt.fov         = 3200
        self.tipTilt.nph         = None
        self.tipTilt.binning     = 1
        self.tipTilt.bw          = 0.0
        self.tipTilt.tr          = [1.0]
        self.tipTilt.disp        = [[0.0],[0.0]]
        self.tipTilt.spot_fwhm   = [[0.0,0.0,0.0]]
        self.tipTilt.ron         = 3.0
        self.tipTilt.dark        = 0.0
        self.tipTilt.sky         = 0.0
        self.tipTilt.gain        = 1.0
        self.tipTilt.excess      = 1.0
        self.tipTilt.nSubap      = [1]
        self.tipTilt.dsub        = [10.5]
        self.tipTilt.noiseVar    = [None]
        self.tipTilt.algo        = 'cog'
        self.tipTilt.win         = 2.0
        self.tipTilt.thres       = 0.0
        self.tipTilt.new         = 0.0
        
        # dm
        self.dm              = structtype()
        self.dm.volt2meter   = 0.4095e-6#0.6e-6 # conversion factor from volts to meter OPD
        self.dm.nActuators   = [21]   # 1D Number of actuators                                            
        self.dm.nCom         = []     # Number of total actuators within the pupil
        self.dm.pitch        = [0.5625]
        self.dm.mechCoupling = [0.11]
        self.dm.heights      = [0.0]
        self.dm.modes        = 'xinetics'        
        self.dm.opt_cond     = 1e2
        self.dm.opt_zen      = [0.0]
        self.dm.opt_azi      = [0.0]
        self.dm.opt_weight   = [1.0]
        self.dm.nrec         = 7
        self.dm.area         = 'circle'
        
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
       
    def restoringInstrumentData(self,cam_fov=None):
        """
        """

        hdr = fits.getheader(self.path_img,
                             ignore_missing_end=True,
                             ignore_missing_simple=True)
        
        #1\ Dates
        self.obsdate = hdr['DATE-OBS'].replace('-','')
        self.acqtime = hdr['EXPSTOP'].replace(':','_')
        
        # 2\ Restore the telescope configuration
        self.tel.name         = hdr['TELESCOP'].replace(' ','')
        self.tel.zenith_angle, self.tel.airmass = keckUtils.getTelescopeAirmass(hdr)
        self.tel.pupilAngle   = keckUtils.getPA(hdr)
        self.tel.pupilMaskName= keckUtils.getPupilMask(hdr)
        _,self.aoMode         = keckUtils.getStagePositionWFS(hdr)   
        if self.tel.pupilMaskName.upper() == 'LARGEHEX':
            self.tel.path_pupil   = self.path_calib + 'NIRC2_MASK/keck_pupil_largeHex_272px.fits'
        else:
            self.tel.path_pupil   = self.path_calib + 'NIRC2_MASK/keck_pupil_open2_240px.fits'
        self.tel.path_telstat = self.path_calib + 'KECK_STAT/keck_piston_modes_200px.fits'
        
        # 3\ Restore the instrument configuration
        self.cam.name = keckUtils.getInstName(hdr)
        self.cam.psInMas = keckUtils.getScale(hdr)
        
        if self.cam.name == 'OSIRIS':
            raise ValueError('THE OSIRIS CASE IS NOT YET IMPLEMENTED\n')
        elif self.cam.name == 'NIRSPEC':
            raise ValueError('THE NIRSPEC CASE IS NOT YET IMPLEMENTED\n')
        else:
            # image
            self.cam.image = fits.getdata(self.path_img,
                                          ignore_missing_end=True,
                                          ignore_missing_simple=True) #in DN
            
            if cam_fov and (cam_fov < min(self.cam.image.shape)) :
                self.cam.fov   = cam_fov
                # retrieving the cog and cropping the image
                im_tmp         = medfilt2d(self.cam.image.astype(float),kernel_size=5)
                y0 , x0        = np.unravel_index(im_tmp.argmax(),im_tmp.shape)
                self.cam.image = self.cam.image[int(y0-cam_fov/2):int(y0+cam_fov/2),
                                                int(x0-cam_fov/2):int(x0+cam_fov/2)]
            else:
                self.cam.fov = self.cam.image.shape[0] #square image
            # positions
            self.cam.zenith  = [0.0]
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
            if self.addNCPA:
                self.cam.path_ncpa = self.path_calib + '/NIRC2_STAT/' +  keckUtils.getNCPAstr(hdr)
            else:
                self.cam.path_ncpa = ''

        
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
            dm = deformableMirror(self.dm.nActuators,self.dm.pitch,heights=self.dm.heights\
                                  ,mechCoupling=self.dm.mechCoupling,modes=self.dm.modes)
            self.mat.dmIF     = dm.setInfluenceFunction(self.tel.resolution)
            self.mat.dmIF_inv = np.linalg.pinv(self.mat.dmIF,rcond=1/self.dm.opt_cond)
            self.mat.Hdm      = np.matmul(self.mat.dmIF,self.mat.dmIF_inv)
        
        #3\ MASS/DIMM
        # managing the saving folder
        if not os.path.isdir(self.path_calib+'/MASSDIMM/'):
            os.mkdir(self.path_calib+'/MASSDIMM/')
        self.path_massdimm = self.path_calib+'/MASSDIMM/'+self.obsdate +'/'
        if not os.path.isdir(self.path_massdimm):
            status = fetch_data(self.obsdate,self.path_massdimm)
        else:
            if not os.path.isfile(self.path_massdimm + '/' + self.obsdate+'.dimm.dat'):
                status = 0
            else:
                status = 1
        
        # median conditions at MaunaKea
        self.atm.wvl        = 500e-9
        self.atm.wSpeed     = [6.8, 6.9, 7.1, 7.5, 10.0, 26.9, 18.5]
        self.atm.wDir       = list(np.zeros(len(self.atm.wSpeed)))
        self.atm.L0         = 25
        self.atm.Cn2Heights = [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
        self.atm.Cn2_mass    = [-1]
        self.atm.seeing_dimm = -1
        self.atm.seeing_mass = -1
        
       
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
            self.atm.seeing  = dimm.indexTime([hhmmss])[0][0]
            
            # read the MASS data to get the Cn2 profile
            mass        = MASS(self.path_massdimm + '/' + self.obsdate + '.mass.dat')
            SeeingAlt   = mass.indexTime([hhmmss])[0][0]
            massprof    = MASSPROF(self.path_massdimm + '/' + self.obsdate + '.masspro.dat')
            Cn2Alt      = massprof.indexTime([hhmmss])[0].reshape(-1)
            if np.all(Cn2Alt == -1) or len(np.nonzero(Cn2Alt != Cn2Alt)[0]):
                self.atm.seeing     = 3600*180/np.pi * 0.98*500e-9/0.16
                self.atm.Cn2Weights = [0.517, 0.119, 0.063, 0.061, 0.105, 0.081, 0.054]
            else:
                if SeeingAlt == -1:
                    rad2arc = 3600*180/np.pi
                    mu = (0.976*rad2arc)**(5/3)*0.423*4*np.pi*np.pi/self.atm.wvl**(1/3)
                    SeeingAlt = (mu * np.sum(Cn2Alt)) ** (3/5)
                # combine MASS and DIMM data
                self.atm.Cn2 = CombineMASSandDIMM(self.atm.seeing,SeeingAlt,Cn2Alt,wvl=self.atm.wvl)
                self.atm.Cn2Weights = list(self.atm.Cn2/self.atm.Cn2.sum())
                
                # copying MASS/DIM data
                self.atm.Cn2_mass    = np.copy(self.atm.Cn2Weights)
                self.atm.seeing_dimm = np.copy(self.atm.seeing)
                self.atm.seeing_mass = np.copy(SeeingAlt)
                
        # compressing
        if nLayer < len(self.atm.Cn2Heights):
            self.atm.wSpeed = list(FourierUtils.eqLayers(np.array(self.atm.Cn2Weights),np.array(self.atm.wSpeed),nLayer)[1])
            self.atm.wDir   = list(FourierUtils.eqLayers(np.array(self.atm.Cn2Weights),np.array(self.atm.wDir),nLayer,power=1)[1])
            self.atm.Cn2Weights,self.atm.Cn2Heights = FourierUtils.eqLayers(self.atm.Cn2Weights,np.array(self.atm.Cn2Heights),nLayer)
            
    def restoringTelemetry(self,verbose=False):
              
        
        # 1\ Restore telemetry data and header
        trsData = readsav(self.path_trs,verbose=verbose)
        hdr = fits.getheader(self.path_img,
                             ignore_missing_end=True,
                             ignore_missing_simple=True)
        
        # 2\ Get AO control loop data                 
    
        #2.1. Get slopes in pixels unit
        self.wfs.slopes = np.copy(trsData.A['OFFSETCENTROID'][0])
        self.wfs.nSl    = self.wfs.slopes.shape[1]
        self.wfs.nExp   = self.wfs.slopes.shape[0]
    
        #2.2 z-position of the WFS stage and AO mode
        self.wfs.zstage_defocus, self.aoMode = keckUtils.getStagePositionWFS(hdr)
        
        if np.any(self.wfs.validSubaperture):   
            self.wfs.intensity = np.zeros((self.wfs.nExp,400))
            self.wfs.intensity[:,self.wfs.validSubaperture.reshape(-1)] = trsData.A['SUBAPINTENSITY'][0]
            self.wfs.intensity = self.wfs.intensity.reshape((self.wfs.nExp,self.wfs.nSubap[0],self.wfs.nSubap[0]))
        else:
            self.wfs.intensity = np.copy(trsData.A['SUBAPINTENSITY'][0])
        if self.aoMode == 'NGS':
            self.wfs.wvl = keckUtils.getWFSwavelength(hdr)
        else:
            self.wfs.wvl = 589e-9
            self.tipTilt.wvl = keckUtils.getWFSwavelength(hdr)
            self.tipTilt.intensity = trsData.B['APDCOUNTS'][0]
            
        #2.3. Get DMs commands in OPD units
        self.dm.com = trsData.A['DMCOMMAND'][0]*self.dm.volt2meter
        self.dm.nCom= self.dm.com.shape[1]
        
        #2.4. Get tip-tilt measurements and conversion into OPD
        if 'b' in trsData:
            self.tipTilt.tilt2meter= 3.2*1.268e-05 # to be verified
            #self.tipTilt.tilt2meter= 1.268e-05 # to be verified
            self.tipTilt.slopes    = np.copy(trsData.B['DTTCENTROIDS'][0])
            self.tipTilt.com       = np.copy(trsData.B['DTTCOMMANDS'][0])
            self.tipTilt.intensity = np.copy(trsData.B['APDCOUNTS'][0])
            unit_tt                = 0.18
        else:
            unit_tt                 = 1
            self.tipTilt.tilt2meter = 12.68e-6 # should be np.pi*tel.D/4/3600/180
            self.tipTilt.slopes     = np.copy(trsData.A['RESIDUALWAVEFRONT'][0][:,self.dm.nCom:self.dm.nCom+2])# %angle in arcsec
            self.tipTilt.com        = np.copy(trsData.A['TTCOMMANDS'][0])
            self.tipTilt.intensity  = None 
            
        self.tipTilt.slopes  = unit_tt * self.tipTilt.tilt2meter*self.tipTilt.slopes
        self.tipTilt.slopes -= np.mean(self.tipTilt.slopes,axis=0) 
        self.tipTilt.com     = self.tipTilt.tilt2meter*self.tipTilt.com
        self.tipTilt.com    -= np.mean(self.tipTilt.com,axis=0) 
        self.tipTilt.nExp    = self.tipTilt.slopes.shape[0]
        
        # 3\ Get system matrices and reconstructed wavefront
        
        #3.1\ Get DM commands reconstructors from slopes
        MC            = np.copy(trsData['rx'])
        MC            = np.reshape(MC,(self.dm.nCom+3,self.wfs.nSl,trsData['NREC'])) #command matrix
        self.mat.R    = self.dm.volt2meter*MC[:self.dm.nCom,:,:]
        self.mat.Rtt  = self.tipTilt.tilt2meter*MC[self.dm.nCom:self.dm.nCom+2,:,:]
        
        #3.2\ Get the reconstructed wavefront in OPD and in the actuators space
        self.rec.res   = self.dm.volt2meter*trsData.A['RESIDUALWAVEFRONT'][0][:,0:self.dm.nCom]
        self.rec.res  -= np.mean(self.rec.res,axis=0) 
        self.rec.focus = np.copy(trsData.A['RESIDUALWAVEFRONT'][0][:,-1]) #in nm, but TBC 
        self.rec.focus-= np.mean(self.rec.focus,axis=0) 
        
        # compute the residual error
        u = np.std(self.rec.res,axis=0)
        u = u[u.nonzero()]
        self.rec.wfe = np.sqrt(np.sum(u**2/len(u)))*1e9
        
        # fill vector to get 21x21 actuators
        if np.any(self.dm.validActuators):   
            idG          = self.dm.validActuators.reshape(-1)
            u            = np.zeros((self.wfs.nExp,self.dm.nActuators[0]**2))
            u[:,idG]     = self.rec.res
            self.rec.res = u
            u            = np.zeros((self.wfs.nExp,self.dm.nActuators[0]**2))
            u[:,idG]     = self.dm.com
            self.dm.com  = u

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