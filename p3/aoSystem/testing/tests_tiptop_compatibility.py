#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:00:00 2022

@author: gagapito
"""

#%% IMPORTING LIBRARIES
import sys
import time

import p3.aoSystem as aoSystemMain

from p3.aoSystem.aoSystem import aoSystem
from p3.aoSystem.pupil import pupil
from p3.aoSystem.segment import segment
from p3.aoSystem.spiders import spiders
from p3.aoSystem.frequencyDomain import *
from p3.aoSystem.fourierModel import *

import numpy as np

import unittest

#%% TEST COMPATIBILITY WITH TIPTOP
class TestFourierModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path_p3 = '/'.join(aoSystemMain.__file__.split('/')[0:-2])
        inputDir = path_p3+'/aoSystem/parFiles/'
        parametersFile = 'MavisMCAO'
        fullPathFilename = inputDir + parametersFile + '.ini'
        # High-order PSD caculations at the science directions and NGSs directions
        TestFourierModel.fao = fourierModel(fullPathFilename, calcPSF=False, verbose=False, display=False, getPSDatNGSpositions=True)      

class TestTipTopInterface(TestFourierModel):
    def test_interface(self):
        PSD           = TestFourierModel.fao.powerSpectrumDensity() # in nm^2
        PSD           = PSD.transpose()
        N             = PSD[0].shape[0]
        freq_range    = TestFourierModel.fao.ao.cam.fovInPix*TestFourierModel.fao.freq.PSDstep
        pitch         = 1/freq_range
        grid_diameter = pitch*N
        dk            = 1e9*TestFourierModel.fao.freq.kcMax_/TestFourierModel.fao.freq.resAO
        psInMas       = TestFourierModel.fao.freq.psInMas[0]
        
        print('np.sum(PSD)',np.sum(PSD))
        print('size of PSD',PSD.shape)
        print('freq_range',freq_range)
        print('dk',dk)
        print('psInMas',psInMas)
        
        self.assertTrue(np.sum(PSD)>0)
        self.assertTrue(N>0)
        self.assertTrue(freq_range>0)
        self.assertTrue(pitch>0)
        self.assertTrue(grid_diameter>0)
        self.assertTrue(dk>0)
        self.assertTrue(psInMas>0)
    
def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestTipTopInterface('test_interface'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

