#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 14:57:57 2021

@author: omartin
"""

#%% IMPORTING LIBRAIRIES
import numpy as np
from numpy.random import uniform
import numpy.fft as fft
import os
import time

from astropy.io import fits
from psfao21.psfao21 import psfao21
import aoSystem.FourierUtils as FourierUtils
np.random.seed(1)


def generatePSF(path_ini,nIntervals=10,nPSFperFolder=3500,addStatic=0,mag=0,zP=25.44,DIT=0.5,nDIT=50,skyMag=13.6,ron=60,\
                normType=1,savePath='',nModes=9,nRound=7,fullName=False,\
                bounds=[[0.05,1e-3,100,1e-3,0.5,1.1,-0.1],[0.4,5e-2,390,1e-1,2,3.0,0.1]]):
    
    tstart = time.time()
    #%% MODEL INSTANTIATION
    psfao    = psfao21(path_ini) 
    wvl      = psfao.freq.wvlRef
    wvlFact  = 2*np.pi*1e-9/wvl

    #%% CHECKING THE NUMBER OF MODES
    if addStatic == True  and nModes > psfao.ao.tel.nModes:
        print('WARNING : the inputs number of modes is too high and then cropped')
        nModes = psfao.ao.tel.nModes
        
    #%% DEFINITION OF PARAMETERS DOMAIN

    # ------------------ DEFINING FOLDERS
    nMain         = 2 # r0 and A are the primary parameters
    nSubFolder    = nIntervals ** nMain
    nPSFperFolder = nPSFperFolder
    
    # DEFINING THE DATA FOLDER
    if savePath != '':
        idNOISE = 'NO'
        idSTATIC= 'NO'
        if addStatic:
            idSTATIC = ''
        if mag != 0:
            Flux    = 10 ** (-0.4*(mag - zP))*DIT*nDIT
            skyFlux = 10 ** (-0.4*(skyMag - zP)) * nDIT * DIT * (psfao.ao.cam.psInMas/1e3) ** 2
            ronStack= ron * np.sqrt(nDIT)
            idNOISE = ''      
        # defining prefix   
        instru    = path_ini.split('/')[-1].split('.')[0]
        if fullName == True:
            psfPrefix = 'psfao21_' + instru + '_wvl_' + str(round(wvl*1e6,3)) + 'µm'
        else:
            psfPrefix = 'psf'
            
        # creating tha main data folder
        savePath = savePath + '/PSFAO21_' + idNOISE + 'NOISE_' + idSTATIC + 'STATIC/'
        if not os.path.isdir(savePath):
            os.mkdir(savePath)
        
        
    # ------------------- DEFINING BOUNDS
    if (len(bounds[0]) == 0) or (len(bounds[1]) == 0):
        raise ValueError('Bounds can not be empty')
    if len(bounds[0]) != len(bounds[1]):
        raise ValueError('Bounds must be of same size')
    if (not addStatic) and ((len(bounds[0]) < 6) or (len(bounds[1]) < 6)):
        raise ValueError('You must provide bounds values of the 6 parameters')
    if addStatic and ((len(bounds[0]) < 7) or (len(bounds[1]) < 7)):
        raise ValueError('Bounds on static aberrations must be provided as well')
     
    r0_lb_500nm = bounds[0][0];r0_ub_500nm = bounds[1][0]
    A_lb_nm    = bounds[0][2];A_ub_nm    = bounds[1][2]
    C_lb    = bounds[0][1]; C_ub    = bounds[1][1]
    ax_lb   = bounds[0][3]; ax_ub   = bounds[1][3]
    p_lb    = bounds[0][4]; p_ub    = bounds[1][4]
    beta_lb = bounds[0][5]; beta_ub = bounds[1][5]
    if addStatic:
        stat_lb = bounds[0][6] * wvl*1e9
        stat_ub = bounds[1][6] * wvl*1e9
    # ------------------- DEFINING R0 AND A DISTRIBUTIONS OVER THE WHOLE DATA SETS
    # bounds
    r0_lb = r0_lb_500nm*(wvl/500e-9)**1.2
    r0_ub = r0_ub_500nm*(wvl/500e-9)**1.2
    A_lb  = (wvlFact*A_lb_nm)**2
    A_ub  = (wvlFact*A_ub_nm)**2
    # intervals
    r0_int = np.round(np.linspace(r0_lb,r0_ub,nIntervals+1),nRound)
    A_int  = np.round(np.linspace(A_lb,A_ub,nIntervals+1),nRound)
    # uniform distributions
    t    =-1
    r0   = np.zeros((nPSFperFolder,nSubFolder))
    A    = np.zeros((nPSFperFolder,nSubFolder))
    idr0 = np.zeros((nPSFperFolder,nSubFolder),dtype='int')
    idA  = np.zeros((nPSFperFolder,nSubFolder),dtype='int')

    for k in range(nIntervals):
        for j in range(nIntervals):
            t = t+1
            # uniform distribution for each interval
            r0[:,t]   =  r0_int[k] + (r0_int[k+1] - r0_int[k])*uniform(size=nPSFperFolder)
            A[:,t]    = A_int[j] + (A_int[j+1] - A_int[j])*uniform(size=nPSFperFolder)
            idr0[:,t] = k*np.ones(nPSFperFolder)
            idA[:,t]  = j*np.ones(nPSFperFolder)
            # CREATING SUBFOLDERS
            if savePath != '':
                # define subfolders name
                idsub = 'r0_' + str(r0_int[k]) + '_sig2_' + str(A_int[j])
                # create subdirectories
                path_folder = savePath + idsub
                if not os.path.isdir(path_folder):
                    os.mkdir(path_folder)
                    
    r053 = r0 ** (-5.0/3.0)

    # -------------------------- DEFINING SECONDARY PARAMETERS OVER A SINGLE SUBFOLDER AND REPEAT    
    C    = uniform(low=C_lb,high=C_ub,size=nPSFperFolder)
    ax   = uniform(low=ax_lb,high=ax_ub,size=nPSFperFolder)
    pe   = np.linspace(p_lb,p_ub,nPSFperFolder)
    be   = uniform(low=beta_lb,high=beta_ub,size=nPSFperFolder)
    if addStatic:
        stat = uniform(low=stat_lb,high=stat_ub,size=(nModes,nPSFperFolder))

    #%% LOOPS
    pix2freq = 1/(psfao.ao.tel.D * psfao.freq.sampRef)**2
    otfDL    = fft.fftshift(psfao.freq.otfDL)
    idStat   = ''
    idNoise  = ''
    # PRE-CALCULATE PSD
    for j in range(nPSFperFolder):
        # DERIVING THE AO-CORRECTED PSD
        psdIn = psfao.freq.pistonFilter_ * psfao.moffat(psfao.freq.kx_,psfao.freq.ky_,[ax[j],pe[j],0,be[j],0,0])
        Aemp  = np.trapz(np.trapz(psdIn,psfao.freq.ky_[0]),psfao.freq.ky_[0])
        psdIn = psdIn/Aemp
        
        # DERIVING THE TELESCOPE OTF
        if addStatic:
            otfStat,_,_ = FourierUtils.getStaticOTF(psfao.ao.tel,psfao.freq.nOtf,psfao.freq.sampRef,psfao.freq.wvlRef,xStat=stat[:,j])
            otfStat = fft.fftshift(otfStat)
        else:
            otfStat = otfDL
                
        for k in range(nSubFolder):                        
            # Total PSD
            psd = r053[j,k] * (psfao.freq.psdKolmo_) + psfao.freq.mskIn_ * (C[j] + psdIn * A[j,k] )        
            # COMPUTING THE PSF        
            Bphi  = fft.fft2(fft.fftshift(psd)) * pix2freq
            Dphi  = np.real(2*(Bphi.max() - Bphi))
            psf_i = np.real(fft.fftshift(fft.ifft2(otfStat * np.exp(-0.5*Dphi))))
                        
            # ADDING THE NOISE
            if mag != 0:
                #noise_sky = np.random.poisson(skyFlux*np.ones_like(psf_i)) - skyFlux
                noise_sky = np.random.poisson(skyFlux*np.ones_like(psf_i))
                noise_dec = ronStack*np.random.randn(psf_i.shape[0],psf_i.shape[1])
                noise_dec-= noise_dec.min()
                psf_i = np.random.poisson(Flux*psf_i) + noise_sky  + noise_dec
                
            # NORMALIZING PSF
            psf_i,_ = FourierUtils.normalizeImage(psf_i,normType=normType)    
            
            # SAVING
            if savePath != '':
                # psf name
                if addStatic == True:
                    s = ['_m'+str(nn+1)+'_' + str(round(stat[nn,j],nRound)) for nn in range(nModes)]
                    idStat = ''.join(s)
                if mag != 0:
                    idNoise = '_mag_' + str(mag)# + '_ron_' + str(ron) + '_skyMag_' + str(skyMag) + '_DIT_' + str(DIT) + '_NDIT_' + str(nDIT) 
                    
                idpsf = psfPrefix + '_r0_' \
                            + str(round(r0[j,k],nRound))\
                            + '_bg_' + str(round(C[j],nRound))\
                            + '_s2_' + str(round(A[j,k],nRound))\
                            + '_ax_' + str(round(ax[j],nRound))\
                            + '_pe_' + str(round(pe[j],nRound))\
                            + '_be_' + str(round(be[j],nRound))\
                            + idStat + idNoise  +  '_norm_'  + str(normType)
                
                # corresponding subfolders
                idsub = 'r0_' + str(r0_int[idr0[j,k]]) + '_sig2_' + str(A_int[idA[j,k]])
                # save
                hdu = fits.PrimaryHDU(psf_i)
                hdu.writeto(savePath + idsub + '/' + idpsf + '.fits',overwrite=True)
                
    print('Simulation of %d PSF done in %.2fs'%(nPSFperFolder*nSubFolder,time.time()-tstart))
