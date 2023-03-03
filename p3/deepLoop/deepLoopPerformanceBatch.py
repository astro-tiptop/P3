#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 07:46:14 2021

@author: omartin
"""

import argparse
from .deepLoopPerformance import deepLoopPerformance

def commandLine():
     # CREATE ArgumentParser OBJECT :
    parser = argparse.ArgumentParser(description='DEEP LOOP PERFORMANCE',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # TXT FILE
    parser.add_argument('--txt', help='Path to the .txt file containing deeploop results',
                        default='',type=str)

    # INI FILE
    parser.add_argument('--ini', help='Path to the .ini file for instantiating the model',
                        default='', type=str)

    # FIT
    parser.add_argument('--nPSF', help='Number of fitted PSFs',
                        default=100, type=int)
    parser.add_argument('--fit', help='To fit the PSF',
                        default=0, type=int)
    parser.add_argument('--init', help='option to define the initial guess for the PSF-fitting; -1=median, 0=ground truth, float>0=precision on the estimates',
                        default=0, type=float)

    # PATHS
    parser.add_argument('--savePath', help='Path to the saving folder',
                        default='', type=str)
    parser.add_argument('--rootPath', help='Root path to extend relative paths in the .ini file',
                        default='', type=str)

    # MODEL
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

    # FORMAT
    parser.add_argument('--fontsize', help='Font size on figure',
                        default=22, type=int)
    parser.add_argument('--nstd', help='number of times the std of distribution to be plotted',
                        default=5, type=float)
    parser.add_argument('--nBins', help='number of bins in the histograms',
                        default=100, type=int)
    args = parser.parse_args()

    return (args)


#%% MAIN PROGRAM
if __name__=='__main__':

    args = commandLine()
    print(args.txt)
    print(args)

    # manage the init field
    if args.init < 0:
        init = 'median'
    elif args.init == 0:
        init = 'truth'
    else:
        init = args.init

    # instantiate deepLoopPerformance
    dlp = deepLoopPerformance(args.txt,path_ini=args.ini,path_save=args.savePath,path_root=args.rootPath,\
                nPSF=args.nPSF,fit=args.fit,init=init,
                mag=args.mag,zP=args.zP,DIT=args.DIT,nDIT=args.nDIT,skyMag=args.skyMag,ron=args.ron)

    # run the display
    dlp(fontsize=args.fontsize,nstd=args.nstd)
