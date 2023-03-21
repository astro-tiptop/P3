#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 18:48:47 2021

@author: omartin
"""

#%%
import numpy as np
import os
from scipy.io import loadmat
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.sparse import csc_matrix

#%%
class structtype():
    pass

class telemetryKASP:
    
    def __init__(self,path_mat,path_save='./'):
        
        # Check the telemetry path
        self.path_mat = path_mat
        if path_mat == None or path_mat == [] or path_mat == '':
            print('No telemetry file')
            self.path_mat = None
        else:
            # Check the telemetry file
            if os.path.isfile(path_mat) == False:
                raise ValueError('Error : the .mat file does not exist\n')
                return
        self.path_save = path_save
        
        # instantiating the fields
        self.instantiating_fields()
        
        # reading the mat file
        self.data_struct = loadmat(path_mat,struct_as_record=True)['data_struct']
        
        # populating fields
        self.populating_fields()
        
        
    def instantiating_fields(self):
        """
            Instantiating the data structures for the Keck AO system"
        """
        #NOTE : SO FAR I"M USING STRUCTURES, BUT WE COULD RELY ON CLASSES IN THE aoSystem FOLDER
        # telescope
        self.tel = structtype()
        self.tel.D           = None  # DM actuators pitch multiplied by number of actuators
        self.tel.cobs        = None # central obstruction ratio
        self.tel.resolution  = None   
        self.tel.pupilAngle  = 0
        self.tel.path_pupil  = ''
        self.tel.path_telstat= ''
        
        # atmosphere
        self.atm = structtype()
        self.atm.seeing_dimm = -1
        self.atm.seeing_mass = -1
        self.atm.Cn2_mass    = -1
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
        self.wfs.pixel_scale = None
        self.wfs.fov         = None
        self.wfs.ron         = 0.0
        self.wfs.nSl         = [None] # Number of slopes measurements within the pupil (x and y)
        self.wfs.nSubap      = [None]
        self.wfs.dsub        = [None]
        self.wfs.algo        = 'cog'
        
        self.wfs.type        = 'Shack-Hartmann'
        self.wfs.modulation  = None
        self.wfs.binning     = 1
        self.wfs.nph         = None
        self.wfs.spot_fwhm   = [[0.0,0.0,0.0]]
        self.wfs.bw          = 0.0
        self.wfs.tr          = [1.0]
        self.wfs.disp        = [[0.0],[0.0]]
        self.wfs.dark        = 0.0
        self.wfs.sky         = 0.0
        self.wfs.gain        = 1.0
        self.wfs.excess      = 1.0
        self.wfs.noiseVar    = [None]
        self.wfs.win         = 2.0
        self.wfs.thres       = 0.0
        self.wfs.new         = 0.0
        self.wfs.theta       = [0]
        
        # tipTilt
        self.tipTilt             = structtype()
        self.tipTilt.pixel_scale = None
        self.tipTilt.fov         = None
        self.tipTilt.ron         = 0.0
        self.tipTilt.nSubap      = [1]
        self.tipTilt.dsub        = [None]
        self.tipTilt.algo        = 'cog'
        self.tipTilt.tilt2meter  = 1
        self.tipTilt.binning     = 1
        self.tipTilt.nph         = None
        self.tipTilt.spot_fwhm   = [[0.0,0.0,0.0]]
        self.tipTilt.bw          = 0.0
        self.tipTilt.tr          = [1.0]
        self.tipTilt.disp        = [[0.0],[0.0]]
        self.tipTilt.dark        = 0.0
        self.tipTilt.sky         = 0.0
        self.tipTilt.gain        = 1.0
        self.tipTilt.excess      = 1.0
        self.tipTilt.nSubap      = [1]
        self.tipTilt.noiseVar    = [None]
        self.tipTilt.win         = 2.0
        self.tipTilt.thres       = 0.0
        self.tipTilt.new         = 0.0
        
        # dm
        self.dm              = structtype()
        self.dm.nActuators   = [None]   # 1D Number of actuators                                            
        self.dm.nCom         = [None]     # Number of total actuators within the pupil
        self.dm.pitch        = [None]
        self.dm.mechCoupling = [None]
        self.dm.heights      = [0.0]
        self.dm.modes        = ''        
        self.dm.opt_cond     = 1e2
        self.dm.opt_zen      = [0.0]
        self.dm.opt_azi      = [0.0]
        self.dm.opt_weight   = [1.0]
        self.dm.nrec         = 7
        self.dm.area         = 'square'
        
        # cam
        self.cam = structtype()
        self.cam.spot_fwhm   = [[0.0,0.0,0.0]]
        self.cam.bw          = 0.0
        self.cam.transmission= [1.0]
        self.cam.dispersion  = [[0.0],[0.0]]
        self.cam.ron         = 0.0
        self.cam.gain        = 1.0
        
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
        
    def populating_fields(self):
        '''
            Populating fields of the telemetryKASP object from the .mat file
        '''
        rad2arc = 3600 * 180/np.pi
        
        # TELESCOPE
        self.obsdate          = self.data_struct['simu_date'][0,0][0]
        self.acqtime          = self.data_struct['simu_time'][0,0][0]
        self.tel.name         = 'OOMAO'
        self.tel.D            = float(self.data_struct['D'])
        self.tel.cobs         = float(self.data_struct['cobs'])
        self.tel.resolution   = int(self.data_struct['resolution'])
        self.tel.zenith_angle = float(self.data_struct['zenith_angle'])
        self.tel.airmass      = float(self.data_struct['airmass'])
        self.tel.pupil        = np.array(self.data_struct['pupil'][0,0])
        self.tel.map_ncpa     = np.array(self.data_struct['map_ncpa'][0,0])
        
        # ATMOSPHERE
        self.atm.wvl        = float(self.data_struct['wvl_atm'])
        self.atm.r0         = float(self.data_struct['r0'])
        self.atm.seeing     = 3600 * 180/np.pi * self.atm.wvl/self.atm.r0
        self.atm.L0         = float(self.data_struct['L0'])
        self.atm.Cn2Heights = list(self.data_struct['cn2_alt'][0,0][0])
        self.atm.Cn2Weights = np.array(self.data_struct['cn2_weight'][0,0][0])
        self.atm.Cn2        = list(self.atm.Cn2Weights * self.atm.r0**(-5/3))
        self.atm.Cn2Weights = list(self.atm.Cn2Weights)
        self.atm.wSpeed     = list(self.data_struct['wind_speed'][0,0][0])
        self.atm.wDir       = list(self.data_struct['wind_dir'][0,0][0])
        
        # CAMERA
        self.cam.name       = 'OOMAO'
        self.cam.wvl        = [float(self.data_struct['src_wvl'])]
        self.cam.zenith     = [float(self.data_struct['src_zenith'])*rad2arc]
        self.cam.azimuth    = [float(self.data_struct['src_azimuth'])]
        self.cam.image      = np.array(self.data_struct['psf'][0,0])
        self.cam.strehl     = float(self.data_struct['strehl'][0,0])
        self.cam.psInMas    = float(self.data_struct['cam_psInMas'])
        self.cam.fov        = int(self.data_struct['cam_fov'])*2
        #note : the factor 2 allows to avoid aliasing effect
        self.cam.ittime     = float(self.data_struct['cam_ittime'])
        self.cam.ron        = float(self.data_struct['cam_ron'])
        self.cam.path_ncpa  = ''
        
        # SOURCES
        self.ngs.wvl        = float(self.data_struct['ngs_wvl'])
        self.ngs.zenith     = [float(self.data_struct['ngs_zenith'])*rad2arc]
        self.ngs.azimuth    = [float(self.data_struct['ngs_azimuth'])]
        
        if len(self.data_struct['lgs_azimuth'][0,0]) > 0:
            self.lgs.wvl        = float(self.data_struct['lgs_wvl'])
            self.lgs.zenith     = [float(self.data_struct['lgs_zenith'])*rad2arc]
            self.lgs.azimuth    = [float(self.data_struct['lgs_azimuth'])]
            self.lgs.height     = float(self.data_struct['lgs_height'])
            self.wfs.wvl        = self.lgs.wvl
            self.aoMode         = 'LGS'
        else:
            self.lgs.wvl        = None
            self.lgs.zenith     = None
            self.lgs.azimuth    = None
            self.wfs.wvl        = self.ngs.wvl
            self.aoMode         = 'NGS'
            
        # SYSTEM MATRICES
        mech2opt = 2
        self.mat.M          = mech2opt*np.array(self.data_struct['mat_rec'][0,0]) 
        self.mat.DMTTRem    = np.array(self.data_struct['mat_dmttrem'][0,0]) 
        self.mat.SlopeTTRem = np.array(self.data_struct['mat_slttrem'][0,0]) 
        self.mat.TT2COM     = np.array(self.data_struct['mat_tt2com'][0,0]) 
        self.mat.SL2TT      = np.array(self.data_struct['mat_sl2tt'][0,0]) 
        self.mat.COM2TT     = np.array(self.data_struct['mat_com2tt'][0,0]) 
        self.mat.dmIF       = csc_matrix.toarray(self.data_struct['mat_dmif'][0,0])
        self.mat.Hdm        = np.array(self.data_struct['mat_dmfilter'][0,0])
        
        # HO WFS
        self.wfs.pixel_scale = float(self.data_struct['wfs_psInMas'])
        self.wfs.fov         = int(self.data_struct['wfs_fov'])
        self.wfs.ron         = float(self.data_struct['wfs_ron'])
        self.wfs.nSubap      = [int(self.data_struct['wfs_nsubap'])]
        self.wfs.dsub        = [float(self.data_struct['wfs_dsub'])]
        self.wfs.nSlopes     = int(self.data_struct['wfs_nslopes'])
        self.wfs.nExp        = int(self.data_struct['wfs_nexp'][0])
        self.wfs.slopes      = np.squeeze(self.data_struct['wfs_slopes'][0,0]).T
        self.wfs.slopes     -= np.mean(self.wfs.slopes,axis=0)
        
        # DM
        # note : the factor 2 allows to convert the mechanical OPD to the optical OPD
        self.dm.pitch           = [float(self.data_struct['dm_pitch'])]
        self.dm.com             = mech2opt*np.array(np.dot(self.mat.DMTTRem,np.squeeze(self.data_struct['dm_com'][0,0] ))).T
        self.dm.com            -= np.mean(self.dm.com,axis=0)
        self.dm.validActuators  = self.data_struct['dm_validactu'][0,0]
        self.dm.nActuators      = [int(self.data_struct['dm_nactu'])]
        self.dm.nCom            = int(self.data_struct['dm_ncom'])
        self.dm.mechCoupling    = [float(self.data_struct['dm_coupling'])]
        self.dm.heights         = [float(self.data_struct['dm_height'])]
        
        # RESIDUAL WAVEFRONT
        self.mat.R              = np.array(np.dot(self.mat.DMTTRem,np.dot(self.mat.M,self.mat.SlopeTTRem))).T
        self.rec.res            = np.dot(self.wfs.slopes,self.mat.R)
        self.rec.res           -= np.mean(self.rec.res,axis=0)
        self.rec.wfe            = 1e9*np.sqrt(np.sum(np.std(self.rec.res,axis=0)**2)/self.dm.nCom)
        
        # TIP-TILT
        if len(self.data_struct['lgs_azimuth'][0,0]) > 0:
            self.tipTilt.wvl         = float(self.data_struct['ngs_wvl'])
            self.tipTilt.pixel_scale = float(self.data_struct['tt_psInMas'])
            self.tipTilt.fov         = int(self.data_struct['tt_fov'])
            self.tipTilt.ron         = float(self.data_struct['tt_ron'])
            self.tipTilt.nExp        = int(self.data_struct['tt_nexp'][0])
            self.wfs.slopes          = np.dot(self.mat.SlopeTTRem,self.wfs.slopes.T).T
            self.tipTilt.tilt2meter  = self.tipTilt.pixel_scale * self.tel.D/1e3/206264.8
            self.tipTilt.slopes      = self.tipTilt.tilt2meter*np.array(np.squeeze(self.data_struct['tt_slopes'][0,0] )).T
        else:
            self.tipTilt.nExp        = self.wfs.nExp
            self.tipTilt.tilt2meter  = self.wfs.pixel_scale * self.tel.D/1e3/206264.8
            self.tipTilt.slopes      = np.dot(self.mat.COM2TT,np.dot(self.mat.M,self.wfs.slopes.T)).T
            self.wfs.slopes          = np.dot(self.mat.SlopeTTRem,self.wfs.slopes.T).T
        
        self.tipTilt.com = np.array(np.squeeze(self.data_struct['tt_com'][0,0] )).T
        self.tipTilt.com-= np.mean(self.tipTilt.com,axis=0)
        
        # TRANSFER FUNCTION
        self.holoop.lat      = float(self.data_struct['lat_ho'])
        self.holoop.freq     = float(self.data_struct['rate_ho'])
        self.holoop.gain     = float(self.data_struct['gain_ho'])
        self.holoop.tf.num   = [self.holoop.gain,0,0,0]
        self.holoop.tf.den   = [-1,0,0]     
        self.ttloop.lat      = float(self.data_struct['lat_tt'])
        self.ttloop.freq     = float(self.data_struct['rate_tt'])
        self.ttloop.gain     = float(self.data_struct['gain_tt'])
        self.ttloop.tf.num   = [self.ttloop.gain,0,0,0]
        self.ttloop.tf.den   = [-1,0,0]
        
        
        

        