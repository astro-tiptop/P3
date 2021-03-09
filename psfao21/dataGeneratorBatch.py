#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:43:20 2021

@author: omartin
"""
import argparse
from dataGenerator import generatePSF

def commandLine():
     # CREATE ArgumentParser OBJECT :
    parser = argparse.ArgumentParser(description='PSFAO 21 DATA GENERATOR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # INI FILE
    parser.add_argument('--ini', help='Path to the .ini file for instantiating the model',
                        default='/home/omartin/Projects/APPLY/CODES/_ANN/dataGen/nirc2.ini', type=str)     
    # FOLDERS ARCHITECTURE
    parser.add_argument('--nIntervals', help='Number of intervals per primary parameters',
                        default=10, type=int)     
    parser.add_argument('--nPSFperFolder', help='Number of PSF per sub-folders',
                        default=3500, type=int) 
    parser.add_argument('--rootPath', help='Saving main folder',
                        default='', type=str) 
    # MODEL
    parser.add_argument('--addStatic', help='If true, add static aberrations to the model',
                        default=0, type=int) 
    parser.add_argument('--mag', help='If not none, add Shot noise from sky/star and read-out noise',
                        default=0, type=int) 
    parser.add_argument('--zP', help='Zero point in mag/s',
                        default=25.44, type=float) 
    parser.add_argument('--DIT', help='Exposure time per frame in seconds',
                        default=0.5, type=float) 
    parser.add_argument('--nDIT', help='Number of frames',
                        default=50, type=int) 
    parser.add_argument('--skyMag', help='Sky magnitude in mag/arcsec^2/s',
                        default=13.6, type=int) 
    parser.add_argument('--ron', help='read-out noise in e-',
                        default=40, type=int)  
    parser.add_argument('--normType', help='Normalization strategy, 1:sum, 2:min-max, 3:sum inner circle, 4: sum positive, otherwise:normType',
                        default=1, type=int)  
    
    args = parser.parse_args()
    return (args)
    

#%% MAIN PROGRAM
if __name__=='__main__':
    
    args = commandLine()
    generatePSF(args.ini,nIntervals=args.nIntervals,nPSFperFolder=args.nPSFperFolder,addStatic=args.addStatic,\
                mag=args.mag,zP=args.zP,DIT=args.DIT,nDIT=args.nDIT,skyMag=args.skyMag,ron=args.ron,\
                normType=args.normType,rootPath=args.rootPath)