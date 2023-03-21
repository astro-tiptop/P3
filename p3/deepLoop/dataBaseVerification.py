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
from p3.psfao21.psfao21 import psfao21
from p3.psfFitting.psfFitting import psfFitting

#%%
def verifyDataBase(savePath,path_ini=None,nTest=10,nPSD=6):
    
    # verify the presence of the saving folder
    if not os.path.isdir(savePath):
        raise ValueError('ERROR : The data folder does not exist')
    
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
        raise ValueError('ERROR : There are %d sub-folders in this directory'%nSubFold)
    else:
        print('There are %d PSFs per sub-folders'%nSubPSF)
     
    # Verification #2 : parameters values are consistent over all sub-folders
    
    # counting the number of parameters
    s = listPSF[0].split('_')
    if 'm1' in s:
        k=1
        while 'm'+str(k) in s:
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
    nParam   = int(nPSD + nStatic)
    nSubPSF  = int(nSubPSF)
    nSubFold = int(nSubFold)
    param  = np.zeros((nParam,nSubPSF,nSubFold))
    for t in range(nSubFold):
        path_data = savePath + '/' + listSub[t] + '/'
        listPSF   = os.listdir(path_data)
        for j in range(nSubPSF):
            s         = listPSF[j].split('_')
            nstart    = int(np.where([s[jj] == 'r0' for jj in range(len(s)) ])[0][0] + 1)
            param[:,j,t] = np.array(s[nstart:nstart+2*nParam:2]).astype(float)
          
    # verifying r0/A varies within predefined intervals
    r0 = np.sort(param[0],axis=1)
    A  = np.sort(param[2],axis=1)

    # find the number of decimal
    nR = len(str(r0.min()).split('.')[1])
    nA = len(str(A.min()).split('.')[1])
    
    for k in range(nInter):    
        idk_m = 0+k*nInter
        idk_M = (k+1)*nInter-1
        if r0[:,idk_m:idk_M].max() > r0_int[k+1] or r0[:,idk_m:idk_M].min() < np.round(r0_int[k],nR):
           raise ValueError('ERROR : the r0 value in label exceeds the intervals defined from the sub-folder name')

        if A[:,idk_m:idk_M].max() > A_int[k+1] or A[:,idk_m:idk_M].min() < np.round(A_int[k],nA):
           raise ValueError('ERROR : the A value in label exceeds the intervals defined from the sub-folder name')

    #verifying that other parameters are identically distributed from one subfolder to another
    secondaryList = [1,3,4,5] + list(np.linspace(nPSD,nParam-1,nStatic,dtype='int'))
    for p in secondaryList:
        pp = np.sort(param[p],axis=0)
        if len(np.unique(pp)) != nSubPSF:
            raise ValueError('ERROR : The secondary parameters are not identically distributed over the sub-folders')
    
    # Verification #3 : perform PSF-fitting i no noise
    if nonoise == True:
        fixed = (False,)*5 + (True,) + (False,) + (True,)*3 + (True,)*4 + (False,)*nStatic
        if path_ini !=None:
            psfao = psfao21(path_ini)
            print('Run PSF-fitting on %d psfs'%nTest)
            for nn in range(nTest):
                # get the PSF
                path_data = savePath + listSub[nn] + '/'
                listPSF   = os.listdir(path_data)[0]
                psf_i     = fits.getdata(path_data+listPSF)
                
                # get the corresponding parameters from the label
                s         = listPSF.split('_')
                nstart    = int(np.where([s[jj] == 'r0' for jj in range(len(s)) ])[0][0] + 1)
                r0        = float(s[nstart])
                x0_psd    = list(np.array(s[nstart+2:nstart+2+2*(nPSD-2):2]).astype(float)) + [0] + [float(s[nstart+2+2*(nPSD-2)])]
                xstat     = list(np.array(s[nstart+2+2*(nPSD-1):nPSD+2*(nParam-2):2]).astype(float))
                x0        = np.array([r0] + x0_psd + [0,0,0,1,0,0,0] + xstat)
                
                # comparing psf
                psf_ref   = np.squeeze(psfao(x0))
                if 1e2*np.sqrt(np.sum(abs(psf_ref - psf_i)**2))/psf_ref.sum() > 0.2:
                    raise ValueError('The generated PSF foes not match the PSFAO21 model with the same parameters')
                    
                # fitting
                res       = psfFitting(psf_i,psfao,x0,fixed=fixed,verbose=-1)
                if any(1e2*(res.x[res.x!=0] - x0[res.x!=0])/x0[res.x!=0] > 1):
                    raise ValueError('The labels does not correspond to the PSF or the fit did not sucessfully converge')
                    
        else:
            print('WARNING : PSF-fitting can not be performed : the field path_ini is empty ')
    else:
        print('Do not perform PSF-fitting on noisy images')

    print('All verifications are sucessfully passed')
