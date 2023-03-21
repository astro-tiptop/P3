#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:43:20 2021

@author: omartin
"""
import argparse
from dataGenerator import generate_psf
import sys

# grab the path
import p3.aoSystem as aoSystemMain

if sys.platform[0:3] == 'win':
    path_mod = '\\'.join(aoSystemMain.__file__.split('\\')[0:-1])
    path_file = path_mod + '\parFiles\\'
else:
    path_mod = '/'.join(aoSystemMain.__file__.split('/')[0:-1])
    path_file = path_mod + '/parFiles/'
        
        
def commandLine():
     # CREATE ArgumentParser OBJECT :
    parser = argparse.ArgumentParser(description='PSFAO 21 DATA GENERATOR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # INI FILE
    parser.add_argument('--ini', help='Path to the .ini file for instantiating the model',
                        default=path_file+'nirc2_monochromatic.ini', type=str)     
    # FOLDERS ARCHITECTURE
    parser.add_argument('--nIntervals', help='Number of intervals per primary parameters',
                        default=10, type=int)     
    parser.add_argument('--nPSFperFolder', help='Number of PSF per sub-folders',
                        default=3500, type=int) 
    parser.add_argument('--savePath', help='Path to the saving folder',
                        default='', type=str) 
    parser.add_argument('--ntest', help='Number of test data',
                        default=0, type=float) 
    # STATIC
    parser.add_argument('--addStatic', help='If true, add static aberrations to the model',
                        default=0, type=int) 
    parser.add_argument('--nmodes', help='Number of modes to describe the static aberrations',
                        default=9, type=int) 
    # DETECTOR
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
                        default=60, type=int)  
    parser.add_argument('--normType', help='Normalization strategy, 1:sum, 2:min-max, 3:sum inner circle, 4: sum positive, otherwise:normType',
                        default=1, type=int)  
    parser.add_argument('--bounds', help='Lower and upper bounds for r0 (500mn), C, A (nm), ax, p, beta and static aberrations (in wave ratio)',
                        default=[[0.05,1e-3,80,1e-3,0.5,1.1,-0.1],
                                 [0.3,5e-2,390,5e-2,2,3.0,0.1]], type=list) 
    args = parser.parse_args()
    return (args)
    

#%% MAIN PROGRAM
if __name__=='__main__':
    
    args = commandLine()
    generate_psf(args.ini, n_inter=args.nIntervals, n_psf_folder=args.nPSFperFolder,
                 add_static=args.addStatic, nmodes=args.nmodes, mag=args.mag, 
                 zp=args.zP, dit=args.DIT, ndit=args.nDIT, sky_mag=args.skyMag, 
                 ron=args.ron, norm=args.normType, save_path=args.savePath,
                 bounds=args.bounds, n_test=args.ntest)
