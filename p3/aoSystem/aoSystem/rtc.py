#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:34:11 2021

@author: omartin
"""
import numpy as np

class rtc:
    """
        processing class to define the processing strategy to convert the detector image into an input for the RTC.
    """
    
    def __init__(self,loopGainHO, frameRateHO, delayHO,wfe=None,\
                 loopGainLO=None, frameRateLO=None, delayLO=None):
        
        
        # HIGH-ORDER LOOP
        self.holoop = dict()
        self.holoop['gain']  = loopGainHO
        self.holoop['rate']  = frameRateHO
        self.holoop['delay'] = delayHO       
        self.holoop['bandwidth'] = frameRateHO/2/np.pi * np.sqrt(loopGainHO/(1 + 2*delayHO))
        self.holoop['wfe']   = wfe
        
        # LOW-ORDER LOOP
        if loopGainLO!=None and frameRateLO!=None and delayLO!=None:
            self.ttloop = dict()
            self.ttloop['gain']  = loopGainLO
            self.ttloop['rate']  = frameRateLO
            self.ttloop['delay'] = delayLO
            self.ttloop['bandwidth'] = frameRateLO/2/np.pi * np.sqrt(loopGainLO/(1 + 2*delayLO))

    def __repr__(self):
        
        s = ('___ RTC ___\n') + '-------------------------------------------------------------------------------------- \n'
        
        if hasattr(self,'ttloop'):
            s+= '. HIGH-ORDER LOOP:\n'
            s+= '\t Gain : %.2f'%(self.holoop['gain'])
            s+= '\t Rate : %.2f Hz'%(self.holoop['rate'])
            s+= '\t Delay : %.2f frames'%(self.holoop['delay'])
            s+= '\t Delay : %.2f ms'%(1e3*self.holoop['delay']/self.holoop['rate'])
            s+= '\t Bandwidth : %.2f Hz\n'%(self.holoop['bandwidth'])
            s+= '. LOW-ORDER LOOP:\n'
            s+= '\t Gain : %.2f'%(self.ttloop['gain'])
            s+= '\t Rate : %.2f Hz'%(self.ttloop['rate'])
            s+= '\t Delay : %.2f frames'%(self.ttloop['delay'])
            s+= '\t Delay : %.2f ms'%(1e3*self.ttloop['delay']/self.holoop['rate'])
            s+= '\t Bandwidth : %.2f Hz\n'%(self.ttloop['bandwidth'])
        else:
            s+= '. AO LOOP:\n'
            s+= '\t Gain : %.2f'%(self.holoop['gain'])
            s+= '\t Rate : %.2f Hz'%(self.holoop['rate'])
            s+= '\t Delay : %.2f frames'%(self.holoop['delay'])
            s+= '\t Delay : %.2f ms'%(1e3*self.holoop['delay']/self.holoop['rate'])
            s+= '\t Bandwidth : %.2f Hz\n'%(self.holoop['bandwidth'])
        s+= '--------------------------------------------------------------------------------------\n' 
        return s+'\n'