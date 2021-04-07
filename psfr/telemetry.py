#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:24:36 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
import numpy.fft as fft
import fourier.FourierUtils as FourierUtils
from scipy.io import readsav
import os.path as ospath
from astropy.io import fits

#%%
class structtype():
    pass

class telemetry:
    
    
    def __init__(self,path_trs,path_img,verbose=False):
        
        # Check the telemetry file
        self.path_trs = path_trs
        if ospath.isfile(path_trs) == False:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The self file does not exist\n')
            return 0
        
        # Check the image file
        self.path_img = path_img
        if ospath.isfile(path_img) == False:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The image file does not exist\n')
            return 0
        
        # instantiating the field
        self.instantiatingFields()
        
        # restore telemetry
        self.restoreKeckTelemetry(verbose=verbose)
        
    def instantiatingFields(self):
        
        # telescope
        self.tel = structtype()
        self.tel.zenith_angle = 0
        self.tel.airmass      = 1
        
        # wfs
        self.src = structtype()
        
        # wfs
        self.ngs = structtype()
        
        # wfs
        self.lgs = structtype()
        
        # wfs
        self.wfs = structtype()
        
        # tipTilt
        self.tipTilt = structtype()
        
        # dm
        self.dm = structtype()
        
        # cam
        self.cam = structtype()
        
        # rec
        self.rec = structtype()
        
        # mat
        self.mat = structtype()
        
        # holoop
        self.holoop = structtype()
        
        # ttloop
        self.ttloop = structtype()
        
    def restoreKeckTelemetry(self,verbose=False):
              
        def estimateLoopDelay(wssmprg,aoMode):
            
            if wssmprg == 0:
                t0 = 8000e-6
            if wssmprg == 1:
                t0 = 2600e-6
            if wssmprg == 2:
                t0 = 850e-6
            if wssmprg == 3:
                t0 = 700e-6
            if wssmprg == 4:
                t0 = 28000e-6
            if wssmprg == 5:
                t0 = 9500e-6
            if wssmprg == 6:
                t0 = 2000e-6
            if wssmprg == 7:
                t0 = 1350e-6
        
            # Total delays
            t8    = 40e-6
            delay = t0+t8

            if aoMode == 'NGS':
                t9        = 1e-6
                t10       = 2e-6
                textra_tt = 200e-6 # retrieved from rejection transfer function fit
                delay_tt  = t0 + t9 + t10 + textra_tt
            else:
                delay_tt  = 1e-3 #STRAP is much faster

            return delay, delay_tt
            
            
        # 1\ Restore telemetry data and header
        trsData = readsav(self.path_trs,verbose=verbose)
        hdr     = fits.getheader(self.path_img)
        
        # 2\ Get fits header and restore observing conditions
        self.aoMode = 'NGS'
        if 'b' in trsData:
            self.aoMode = 'LGS'
            self.tipTilt = []
            self.tipTilt.tilt2meter = 3.2*1.268e-05
        
        self.tel.zenith_angle = 90 - float(hdr['EL'])
        self.tel.airmass = 1/np.cos(self.tel.zenith_angle*np.pi/180)

        # 3\ Get AO control loop data                 
        
        #3.1. Get slopes in pixels unit
        self.wfs.slopes = trsData.A['OFFSETCENTROID'][0]
        self.wfs.nSl    = self.wfs.slopes.shape[0]
        self.wfs.nExp   = self.wfs.slopes.shape[1]
        
        OBWF   = float(hdr['OBWF'])
        LSPROP = float(hdr['LSPROP']) 
        if LSPROP == 'yes':
            AOFCLGFO = hdr['AOFCLGFO']*1e3 #mm
            self.wfs.zstage_defocus = abs(OBWF - AOFCLGFO)
        else:
            AOFCNGFO = hdr['AOFCNGFO']*1e3
            self.wfs.zstage_defocus = abs(OBWF - AOFCNGFO)
        
        self.wfs.intensity = np.zeros((400,self.wfs.nExp))
        self.wfs.intensity[self.wfs.validSubaperture,:] = trsData.A['SUBAPINTENSITY'][0]
        self.wfs.wavelength = hdr['GUIDWAVE']*1e-6
        
        #3.2. Get DMs commands in OPD units
        self.dm.volt2meter = 0.5e-6
        self.dm.com = trsData.A['DMCOMMAND'][0]*self.dm.volt2meter
        self.dm.nCom= self.dm.com.shape[0]
        
        #3.3. Get tip-tilt measurements and conversion into OPD
        if 'b' in trsData:
            self.tipTilt.slopes    = trsData.B['DTTCENTROIDS'][0]
            self.tipTilt.com       = trsData.B['DTTCOMMANDS'][0]
            self.tipTilt.intensity = trsData.B['APDCOUNTS'][0]
        else:
            self.tipTilt.slopes  = trsData.A['RESIDUALWAVEFRONT'][0][self.dm.nCom:self.dm.nCom+1,:]# %angle in arcsec
            self.tipTilt.com     = trsData.A['TTCOMMANDS']
       
        self.tipTilt.slopes  = self.tipTilt.tilt2meter*self.tipTilt.slopes;
        self.tipTilt.slopes -= np.mean(self.tipTilt.slopes,axis=1) 
        self.tipTilt.com     = self.tipTilt.tilt2meter*self.tipTilt.com;
        self.tipTilt.com    -= np.mean(self.tipTilt.com,axis=1) 
        self.tipTilt.nExp    = self.tipTilt.slopes.shape[1]
        
        # 4\ Get system matrices and reconstructed wavefront
        
        #4.1\ Get DM commands reconstructors from slopes
        MC            = np.reshape(trsData['RX'][0],self.wfs.nSl ,self.dm.nCom+3,trsData['NREC']) #command matrix
        self.mat.R    = self.dm.volt2meter*np.permute(MC[:,:self.dm.nCom,:],[1,0,2])
        self.mat.Rtt  = self.tipTilt.tilt2meter*np.permute(MC[:,self.dm.nCom:self.dm.nCom+1,:],[1,0,2])
        
        #4.2\ Get the reconstructed wavefront in OPD and in the actuators space
        self.rec.res   = self.dm.volt2meter*trsData.A['RESIDUALWAVEFRONT'][0][:self.dm.nCom,:]
        self.rec.res  -= np.mean(self.rec.res,axis=1) 
        self.rec.focus = trsData.A['RESIDUALWAVEFRONT'][0][-1,:]
        self.rec.res  -= np.mean(self.rec.focus,axis=1) 
        
        # fill vector to get 21x21 actuators
        idG          = self.dm.validActuators
        u            = np.zeros((self.dm.nActuators**2,self.wfs.nExp))
        u[idG,:]     = self.rec.res
        self.rec.res = u
        u            = np.zeros((self.dm.nActuators**2,self.wfs.nExp))
        u[idG,:]     = self.dm.com
        self.dm.com  = u
        

        # 5\ Get the loop status and model transfer function
        #5.1. Delays
        wssmprg = hdr['WSSMPRG']
        self.holoop.lat,self.ttloop.lat = estimateLoopDelay(wssmprg,self.aoMode)
        
        #5.2. Frequency
        self.holoop.freq = 1/(100e-9*mean(np.diff(trsData.A['TIMESTAMP'])))
        self.ttloop.freq =self.holoop.freq;
        if b in trsData:
            self.ttloop.freq =  1/(100e-9*np.mean(np.diff(trsData.B['TIMESTAMP'])))
        
        #5.3. RTC controller HO loop
        self.holoop.gain   = trsData['DM_SERVO'][0]
        self.holoop.tf.num = trsData['DM_SERVO'][0:3]
        self.holoop.tf.den = trsData['DM_SERVO'][4:]
        self.ttloop.gain   = trsData['DT_SERVO'][0]
        self.ttloop.tf.num = trsData['DT_SERVO'][0:3]
        self.ttloop.tf.den = trsData['DT_SERVO'][4:]