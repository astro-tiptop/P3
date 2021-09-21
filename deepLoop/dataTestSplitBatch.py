#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:31:05 2021

@author: omartin
"""

import argparse
from dataGenerator import split_test_data

def commandLine():
     # CREATE ArgumentParser OBJECT :
    parser = argparse.ArgumentParser(description='DATA SPLITTER',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # INI FILE
    parser.add_argument('--folder', help='Path to the main data folder',
                        default='.', type=str)
    parser.add_argument('--nfolder', help='Number of sub-folders',
                        default=10, type=int)
    parser.add_argument('--mode', help='permissions mode',
                        default=511, type=int)
    args = parser.parse_args()
    return (args)


#%% MAIN PROGRAM
if __name__=='__main__':

    args = commandLine()
    split_test_data(args.folder, n_sub_folder=args.nfolder, mode=args.mode)
