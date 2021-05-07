#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 17:37:49 2021

@author: omartin
"""

#%% IMPORTING LIBRARIES
import sys
import time

import aoSystem as aoSystemMain

from aoSystem.aoSystem import aoSystem
from aoSystem.pupil import pupil
from aoSystem.segment import segment
from aoSystem.spiders import spiders
from aoSystem.frequencyDomain import frequencyDomain
from aoSystem.fourierModel import fourierModel



#%% TEST THE PUPIL MAKER

def MakeKeckPupil(nargout=0):
    path_txt = 'aoSystem/_txtFile/Keck_segmentVertices.txt'
    
    t0 = time.time()
    spiRef   = spiders([0,60,120],0.0254,symetric=True,D=10.5) 
    keckPup  = pupil(segClass=segment(6,0.9,200),segCoord=path_txt,D=10.5,cobs=0.2311,spiderClass=spiRef)
    print("Pupil creation in %.2f s  "%(time.time() - t0))
    keckPup.displayPupil()
    if nargout ==1:
        return keckPup
    
def MakeELTPupil(nargout=0):
    path_txt = 'aoSystem/_txtFile/ELT_segmentVertices.txt'
    
    t0 = time.time()
    spiRef   = spiders([0,60,120],0.5,symetric=True,D=39) 
    eltPup   = pupil(segClass=segment(6,1.3/2,25),segCoord=path_txt,D=39,cobs=0.2375,spiderClass=spiRef)
    print("Pupil creation in %.2f s  "%(time.time() - t0))
    eltPup.displayPupil()
    if nargout ==1:
        return eltPup
    
    
#%% TEST THE AOSYSTEM CALL
def InitSys(sysName,nargout=0):
    
    # RETIEVING THE .INI FILE
    path_ini = 'aoSystem/parFiles/' + sysName + '.ini'
        
    # INIT THE AO SYSTEM
    t0 = time.time()
    ao = aoSystem(path_ini)
    print(sysName + " system instantiation in %.2f ms  "%(1000*(time.time() - t0)))
    print(ao.__repr__())
    
    # INIT THE FREQUENCY DOMAIN
    t0 = time.time()
    freq = frequencyDomain(ao)
    print("Frequency domain instantiation in %.2f ms  "%(1000*(time.time() - t0)))
    print(freq.__repr__())
    
    if nargout==1:
        return ao
    elif nargout == 2:
        return ao, freq
    
def TestInitSys():
    
    # NIRC2
    InitSys('nirc2')
    # IRDIS
    InitSys('irdis')
    # ERIS
    InitSys('eris')
    # MAVIS
    InitSys('mavis')
    # HARMONI SCAO
    InitSys('HarmoniSCAO')
    # HARMONI LTAO
    InitSys('HarmoniLTAO')
    
#%% TEST THE FOURIER MODELING
    
def TestFourierModel(sysName,calcPSF=False,getMetrics=False,nargout=0):

    # RETIEVING THE .INI FILE
    path_ini = 'aoSystem/parFiles/' + sysName + '.ini'
        
    # INIT THE fourierModel object
    typeData = 'PSD'
    if calcPSF == True:
        typeData = 'PSF'
        
    t0 = time.time()
    fao = fourierModel(path_ini,calcPSF=calcPSF,verbose=True,display=False,getErrorBreakDown=True,\
        getFWHM=getMetrics,getEncircledEnergy=getMetrics,getEnsquaredEnergy=getMetrics,displayContour=getMetrics)
    print(sysName +  ' ' + typeData +  ' model computation %.2f ms\n '%(1000*(time.time() - t0)))
    
    if nargout == 1:
        return fao
    
    
def TestPSD():

    # NIRC2
    TestFourierModel('nirc2')
    # IRDIS
    TestFourierModel('irdis')
    # ERIS
    TestFourierModel('eris')
    # MAVIS
    TestFourierModel('mavis')
    # HARMONI SCAO
    TestFourierModel('HarmoniSCAO')
    # HARMONI LTAO
    TestFourierModel('HarmoniLTAO')

def TestPSF():

    # NIRC2
    TestFourierModel('nirc2',calcPSF=True, getMetrics=True)
    # IRDIS
    TestFourierModel('irdis',calcPSF=True, getMetrics=True)
    # ERIS
    TestFourierModel('eris',calcPSF=True, getMetrics=True)
    # MAVIS
    TestFourierModel('mavis',calcPSF=True, getMetrics=True)
    # HARMONI SCAO
    TestFourierModel('HarmoniSCAO',calcPSF=True, getMetrics=True)
    # HARMONI LTAO
    TestFourierModel('HarmoniLTAO',calcPSF=True, getMetrics=True)