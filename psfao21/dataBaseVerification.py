#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 21:06:21 2021

@author: omartin
"""

#%% IMPORTING LIBRAIRIES
import numpy as np
import os
from astropy.io import fits
from psfao21.psfao21 import psfao21
from psfFitting.psfFitting import psfFitting

#%%
def verifyDataBase(savePath,path_ini=None,nTest=10):
    
    # verify the presence of the saving folder
    if not os.path.isdir(savePath):
        print('ERROR : The data folder does not exist')
        return -1
    
    # counting the number of subfolders
    listSub = os.listdir(savePath)
    nSubFold= len(listSub)
    print('There are %d sub-folders in this directory'%nSubFold)
    
    # grabbing intervals values
    r0_int = np.zeros(nSubFold)
    A_int  = np.zeros(nSubFold)
    for t in range(nSubFold):
        s = listSub[t].split('_')
        r0_int[t] = s[1]
        A_int[t]  = s[3]    
    r0_int = np.unique(r0_int)
    nInter = len(r0_int)
    dr0    = abs(r0_int[0] - r0_int[1])
    r0_int = np.linspace(r0_int.min(),r0_int.max()+dr0,nInter+1) 
    A_int  = np.unique(A_int)
    dA     = abs(A_int[0] - A_int[1])
    A_int  = np.linspace(A_int.min(),A_int.max()+dA,nInter+1) 

    # Verification #1 : number of PSF per folder is constant
    nSubPSF = np.zeros(nSubFold)
    for t in range(nSubFold):
        path_data   = savePath + '/' + listSub[t] + '/'
        listPSF     = os.listdir(path_data)
        # counting number of PSF
        nSubPSF[t]  = len(listPSF)
        
    nSubPSF = np.unique(nSubPSF)
    if len(nSubPSF) != 1:
        print('ERROR : There are %d sub-folders in this directory'%nSubFold)
        return -1
    else:
        print('There are %d PSFs per sub-folders'%nSubPSF)
     
    # Verification #2 : parameters values are consistent over all sub-folders
    
    # counting the number of parameters
    s = listPSF[0].split('_')
    if 'mode1' in s:
        k=1
        while 'mode'+str(k) in s:
           k+=1 
        nStatic = k-1
        print('The data include %d modes of static aberrations'%nStatic)
    else:
        nStatic = 0
        print('The data do not include static aberrations')

    if 'Mag' in s or 'mag' in s:
        print('The data include noise')
        nonoise = False
    else:
        print('The data do not include noise')
        nonoise = True

    # grabbing parameters values
    nParam   = int(6 + nStatic)
    nSubPSF  = int(nSubPSF)
    nSubFold = int(nSubFold)
    param  = np.zeros((nParam,nSubPSF,nSubFold))
    for t in range(nSubFold):
        path_data = savePath + '/' + listSub[t] + '/'
        listPSF   = os.listdir(path_data)
        for j in range(nSubPSF):
            s         = listPSF[j].split('_')
            param[:,j,t] = np.array(s[5:5+2*nParam:2]).astype(float)
    
    # verifying r0/A varies within predefined intervals
    r0 = np.sort(param[0],axis=1)
    A  = np.sort(param[2],axis=1)

    for k in range(nInter):    
        idk_m = 0+k*nInter
        idk_M = (k+1)*nInter-1
        if r0[:,idk_m:idk_M].max() > r0_int[k+1] or r0[:,idk_m:idk_M].min() < r0_int[k]:
           print('ERROR : the r0 value in label exceeds the intervals defined from the sub-folder name')
           return -1 
        if A[:,idk_m:idk_M].max() > A_int[k+1] or A[:,idk_m:idk_M].min() < A_int[k]:
           print('ERROR : the r0 value in label exceeds the intervals defined from the sub-folder name')
           return -1 
    #verifying that other parameters are identically distributed from one subfolder to another
    secondaryList = [1,3,4,5] + list(np.linspace(6,nParam-1,nStatic,dtype='int'))
    for p in secondaryList:
        pp = np.sort(param[p],axis=0)
        if len(np.unique(pp)) != nSubPSF:
            print('ERROR : The secondary parameters are not identically distributed over the sub-folders')
            return -1
    
    # Verification #3 : perform PSF-fitting i no noise
    if nonoise == True:
        fixed = (False,)*5 + (True,) + (False,) + (True,)*7 + (False,)*nStatic
        if path_ini !=None:
            psfao = psfao21(path_ini)
            print('Run PSF-fitting on %d psfs'%nTest)
            for n in range(nTest):
                path_data = savePath + listSub[n] + '/'
                listPSF   = os.listdir(path_data)[0]
                s         = listPSF.split('_')
                x0 = np.array([float(s[5])**(-5/3)] + list(np.array(s[7:15:2]).astype(float)) + [0] + [float(s[15])]+ [0,0,0,1,0,0,0] + list(np.array(s[17:5+2*nParam:2]).astype(float)))
                psf_i = fits.getdata(path_data+listPSF)
                res   = psfFitting(psf_i,psfao,x0,fixed=fixed,verbose=-1)
                if any(1e2*(res.x[res.x!=0] - x0[res.x!=0])/x0[res.x!=0] > 1):
                    print('ERROR : The labels does not correspond to the PSF or the fit did not sucessfully converge')
                    return -1
        else:
            print('WARNING : PSF-fitting can not be performed : the field path_ini is empty ')
    else:
        print('Do not perform PSF-fitting on noisy images')

    print('All verifications are sucessfully passed')
