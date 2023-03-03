#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:44:51 2021

@author: omartin
"""

#%% IMPORTAING LIBRARIES
import os
import numpy as np
import argparse

#%%
def commandLine():
    parser = argparse.ArgumentParser(description='RECOVER LIST',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # INI FILE
    parser.add_argument('--path_data', help='Path to the data folder',
                        default='./PSFAO21_NONOISE_NOSTATIC', type=str)  
    args = parser.parse_args()
    return (args)

def recover_main_parameters_values_from_subfolders(path_data):
    '''
        Recover the list of parameters from the names of subfolders 
        that contains simulated data for training DEEPLOOP
    '''

    # verify that the folder exists
    if not os.path.isdir(path_data):
        raise ValueError('The path does not exist')
        
    # list the folders
    list_folders = [ name for name in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, name)) ]
    nFolders     = len(list_folders)
    # instantiate the lists of values
    r0           = np.zeros(nFolders)
    sig2         = np.zeros(nFolders)
    # loopmon folders
    for k in range(nFolders):
        tmp   = list_folders[k].split('_')
        idR0  = 1
        # grab parameters
        r0[k]   = float(tmp[idR0])
        sig2[k] = float(tmp[idR0 + 2])
        
    # takes the unique values
    return np.unique(r0) , np.unique(sig2)

#%% MAIN PROGRAM
if __name__=='__main__':
    args = commandLine()
    r0, sig2 = recover_main_parameters_values_from_subfolders(args.path_data)
    np.set_printoptions(precision=20)
    print(r0)
    print(sig2)