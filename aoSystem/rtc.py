#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:34:11 2021

@author: omartin
"""


class rtc:
    """
        processing class to define the processing strategy to convert the detector image into an input for the RTC.
    """
    
    def __init__(self,loopGainHO, frameRateHO, delayHO,\
                 loopGainLO=None, frameRateLO=None, delayLO=None):
        
        
        # HIGH-ORDER LOOP
        self.holoop = dict()
        self.holoop['gain']  = loopGainHO
        self.holoop['rate']  = frameRateHO
        self.holoop['delay'] = delayHO
        
        # LOW-ORDER LOOP
        
        if loopGainLO!=None and frameRateLO!=None and delayLO!=None:
            self.ttloop = dict()
            self.ttloop['gain']  = loopGainLO
            self.ttloop['rate']  = frameRateLO
            self.ttloop['delay'] = delayLO
