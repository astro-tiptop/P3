#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:17:49 2021

@author: omartin
"""

#%% IMPORTING LIBRARIES
import sys
import time

from aoSystem.aoSystem import aoSystem
from aoSystem.pupil import pupil
from aoSystem.segment import segment
from aoSystem.spiders import spiders
from aoSystem.frequencyDomain import frequencyDomain

import psfao21 as psfao21Main
from telemetry.telemetryKeck import telemetryKeck
from telemetry.systemDiagnosis import systemDiagnosis
from telemetry.configFile import configFile
from psfr.psfR import psfR
from psfao21.psfao21 import psfao21
from psfFitting.psfFitting import psfFitting
from psfFitting.psfFitting import displayResults

#%% TEST THE TELEMETRY READING
def TestReadTelemetry():
    
    path_mod = '/'.join(psfao21Main.__file__.split('/')[0:-2])
    path_trs   = path_mod + '/data/n0004_fullNGS_trs.sav'
    path_img   = path_mod + '/data/20130801_n0004.fits'
    path_calib = path_mod + '/aoSystem/data/KECK_CALIBRATION/'
    trs = telemetryKeck(path_trs,path_img,path_calib,nLayer=1)
    return trs
    

#%% TEST THE PSF-FITTING

def TestPsfFitting():
    
    return 0