#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:02:21 2021

@author: omartin
"""

#%% IMPORTING LIBRAIRIES
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from distutils.spawn import find_executable
import matplotlib.ticker as mtick
import time
from astropy.table import QTable
from scipy.stats import linregress

import aoSystem.FourierUtils as FourierUtils
from psfao21.psfao21 import psfao21
from psfFitting.psfFitting import psfFitting
np.random.seed(1)

#%% CLASS
class deepLoopPerformance:
    ''' neuralNetworkPerformance class that allows to assess the 
    reconstruction performance of DEEPLOOP from the output .txt files
    '''
    
    def __init__(self,path_txt,path_ini=None,path_save=None,path_root='',nPSF=1000,
                 fit=False,mag=0,zP=25.44,DIT=0.5,nDIT=50,skyMag=13.6,ron=60):
        '''
        path_txt can be either:
            - a string of characters that gives the path to the txt file
            - a tuple of string for multi-configurations analyses
        '''
        # PARSING INPUTS
        # txt files created by DEEPLOOP
        if type(path_txt) == str:
            path_txt = [path_txt]
        self.path_txt = np.unique(path_txt)
        self.nCases   = len(self.path_txt)
        # verifying the non-emptiness of the path_txt field
        if self.nCases == 0:
            print('WARNING : the field path_txt is empty; no analyses possible.')
            return
        # verifying the existence of the txt files
        for k in range(self.nCases):
            if not os.path.isfile(self.path_txt[k]):
                print('ERROR : the path #%d does not exist'%(k+1))
                return
            
        # .ini file : needed for PSF computation
        self.path_ini = path_ini
        if self.path_ini == None:
            print('WARNING : the field path_ini is None : no analyses on PSFs possible')
        
        # IDENTIFYING DATA
        self.idNN     = np.empty(self.nCases,dtype=list)
        self.idData   = np.empty(self.nCases,dtype=list)
        self.dataType = np.empty(self.nCases,dtype=list)
        self.nParam   = np.empty(self.nCases,dtype=list)
            
        for n in range(self.nCases):
            # data identification from the file name
            s = self.path_txt[n].split('/')[-1]
            s = s.split('_')
            self.idNN[n]     = s[1]
            self.idData[n]   = s[2]
            self.dataType[n] = s[0][3:]
            # reading the first line to count the number of parameters
            self.nParam[n]   = self.read_txt_files(self.path_txt[n],getParamNumberOnly=True)
            
        self.nNetworks = len(self.idNN)
        self.nDataSets = len(self.idData)
        print('Test of %d network architectures performance on %d data sets'%(self.nNetworks,self.nDataSets))
        
        # INSTANTIATING DATA STRUCTURES
        self.gtruth = np.empty(self.nCases,dtype=tuple)
        self.nnest  = np.empty(self.nCases,dtype=tuple)
        self.labels = np.empty(self.nCases,dtype=tuple)
        self.metrics_param = np.empty(self.nCases,dtype=tuple)
        for n in range(self.nCases):
            # get parameters
            self.gtruth[n],self.nnest[n],self.labels[n] = self.read_txt_files(self.path_txt[n])
            # get metrics
            self.metrics_param[n] = np.zeros((self.nParam[n],7))
            for k in range(self.nParam[n]):
                self.metrics_param[n][k] = self.get_parameters_metrics(self.gtruth[n][k],self.nnest[n][k])
                
            if path_save:
                Tab = QTable(names=self.labels[n])   
                for k in range(7):
                    Tab.add_row(self.metrics_param[n][:,k])
                Tab.write(path_save + self.idNN[n] + '_' + self.idData[n] + '_' + self.dataType[n] + '.csv',overwrite=True)
            
        
        # COMPUTING PSF METRICS
        if self.path_ini:
            self.psfao = psfao21(path_ini,path_root=path_root)
            self.get_psf_metrics(nPSF=nPSF,fit=fit,mag=mag,zP=zP,DIT=DIT,nDIT=nDIT,skyMag=skyMag,ron=ron)
        
    def __call__(self,fontsize=16,fontfamily='serif',fontserif='Palatino',
                 figsize=(20,20),getPSF=False,constrained_layout=True,nBins=100,nstd=5):
        '''
        Display DEEPLOOP performance
        '''
        # managing display configuration
        mpl.rcParams['font.size'] = fontsize
        if find_executable('tex'): 
            usetex = True
        else:
            usetex = False
        plt.rcParams.update({
                "text.usetex": usetex,
                "font.family": fontfamily,
                "font.serif": fontserif,
                })
        plt.close('all')
        
        #number of axes in scientific notation
        formatter = mtick.ScalarFormatter(useMathText=False)
        formatter.set_scientific(True) 

        # ------ VERSUS PLOTS
        for n in range(self.nCases):
            # creating the figure
            nP = self.nParam[n]
            k1 = int(np.sqrt(nP))
            k2 = int(nP/k1)
            fig , axs = plt.subplots(k1,k2,figsize=figsize,constrained_layout=constrained_layout)
            a=-1
            for m in range(nP):
                b = m%k2
                if b==0:
                    a+=1                    
                
                mn = min(self.gtruth[n][m].min(),self.nnest[n][m].min())
                mx = max(self.gtruth[n][m].max(),self.nnest[n][m].max())
                axs[a,b].plot(self.gtruth[n][m],self.nnest[n][m],'bo')
                axs[a,b].plot([mn,mx],[mn,mx],'k--')
                axs[a,b].set_xlabel(self.labels[n][m] + ' simulated')
                axs[a,b].set_ylabel(self.labels[n][m] + ' reconstructed')
                if mx > 10 or mn < 0.1:
                    formatter.set_powerlimits((-1,1)) 
                    axs[a,b].yaxis.set_major_formatter(formatter) 
                    axs[a,b].xaxis.set_major_formatter(formatter) 
            
            
        # ------ HISTOGRAMS
        for n in range(self.nCases):
            # creating the figure
            nP = self.nParam[n]
            k1 = int(np.sqrt(nP))
            k2 = int(nP/k1)
            fig , axs = plt.subplots(k1,k2,figsize=figsize,constrained_layout=constrained_layout)
            a=-1
            for m in range(nP):
                b = m%k2
                if b==0:
                    a+=1 
                b = m%k2
                
                if m < 6:
                    err = 1e2*(self.nnest[n][m] - self.gtruth[n][m])/self.gtruth[n][m]
                    lab = '\%'
                else:
                    err = self.nnest[n][m] - self.gtruth[n][m]
                    lab = 'nm'
                    
                err = err[abs(err)<5*err.std()]
                axs[a,b].hist(err, weights=np.ones_like(err) / len(err), bins=nBins)
                axs[a,b].set_xlabel(self.labels[n][m] + ' error ('+lab+')')
                axs[a,b].set_ylabel('Probability')
                axs[a,b].set_xlim([-nstd*err.std(),nstd*err.std()])

        # ------ PSFs
        if self.path_ini:
            for n in range(self.nCases):
                # creating the figure
                fig , axs = plt.subplots(2,2,figsize=figsize,constrained_layout=constrained_layout)
                # MSE
                err = self.mse[n]
                nPSF = len(err)
                axs[0,0].hist(err, weights=np.ones_like(err) / len(err), bins=int(min(100,nBins)))
                axs[0,0].set_xlabel('Mean square error (\%)')
                axs[0,0].set_ylabel('Probability')
                axs[0,0].set_xlim([0,nstd*err.std()])
                # SR
                err = 1e2*(self.SR[n][1] - self.SR[n][0])/self.SR[n][0]
                axs[0,1].hist(err, weights=np.ones_like(err) / len(err), bins=int(min(100,nBins)))
                axs[0,1].set_xlabel('Strehl-ratio error (\%)')
                axs[0,1].set_ylabel('Probability')
                axs[0,1].set_xlim([-nstd*err.std(),nstd*err.std()])
                # FWHM
                err = 1e2*(self.FWHM[n][1] - self.FWHM[n][0])/self.FWHM[n][0]
                axs[1,0].hist(err, weights=np.ones_like(err) / len(err), bins=int(min(100,nBins)))
                axs[1,0].set_xlabel('FWHM error (\%)')
                axs[1,0].set_ylabel('Probability')
                axs[1,0].set_xlim([-nstd*err.std(),nstd*err.std()])
                # Photometry
                err = self.mag_err[n]
                axs[1,1].hist(err, weights=np.ones_like(err) / len(err), bins=int(min(100,nBins)))
                axs[1,1].set_xlabel('Photometric error (mag)')
                axs[1,1].set_ylabel('Probability')
                axs[1,1].set_xlim([-nstd*err.std(),nstd*err.std()])
                
                plt.figure()
                nPx = self.psf_mean[n].shape[0]
                fov = nPx * self.psfao.ao.cam.psInMas
                x = np.linspace(-fov/2,fov/2,num=nPx)
                plt.semilogy(x,self.psf_mean[n][nPx//2,:],'k',label='Mean PSF')
                plt.semilogy(x,self.psf_diff_mean[n][nPx//2,:],'b',label='Mean differential PSF')
                plt.semilogy(x,self.psf_diff_std[n][nPx//2,:],'r',label='Std differential PSF')
                plt.legend()
            
    def read_txt_files(self,path_txt,getParamNumberOnly=False):
        '''
        Reading the .txt input file and populating the gtruth and nnest arrays
        '''
        def is_float(string):
            try:
                float(string)
                return True
            except ValueError:
                return
            
        # GETTING THE NUMBER OF PARAMETERS
        tmp = open(path_txt)
        firstLine = tmp.readline().split()
        lab       = tmp.readline().split()
        labels    = lab[1:int((len(lab)-1)/2+1)]
        nParam = 0
        for n in range(len(firstLine)):
            if is_float(firstLine[n]):
                nParam+=1
        if getParamNumberOnly:
            tmp.close()
            return nParam
        
        # GETTING THE NUMBER OF LINES
        nData = sum(1 for line in tmp) -1
        tmp.close()
        
        # READING THE ENTIRE FILE
        groundTruth = np.zeros((nParam,nData))
        nnEstimates = np.zeros((nParam,nData))
        tmp = open(path_txt)
        lines = tmp.readlines()[2:]
        for n in range(nData):
            tmpP = np.array(lines[n][0:-1].split('\t')).astype(float)
            groundTruth[:,n] = tmpP[0:nParam]
            nnEstimates[:,n] = tmpP[nParam:]
            
        tmp.close()
        return groundTruth,nnEstimates,labels
            
    def get_parameters_metrics(self,gtruth,nnEst):
        '''
        getting median, mean, std and Peason coefficients
        '''
        
        def get_sub_metrics(vec):
            return np.mean(vec), np.median(vec), np.std(vec)
        
        # absolute difference
        dabs = nnEst - gtruth
        dabs_mean, dabs_med, dabs_std = get_sub_metrics(dabs)
        # relative difference
        drel = 1e2 * dabs/gtruth
        drel_mean, drel_med, drel_std = get_sub_metrics(drel)
        # Pearson
        coeff_pearson = np.corrcoef(gtruth,nnEst)

        return dabs_mean, dabs_med, dabs_std,drel_mean, drel_med, drel_std,coeff_pearson[0,1]
        
    def get_psf_metrics(self,nPSF=1000,fit=False,mag=0,zP=25.44,DIT=0.5,nDIT=50,skyMag=13.6,ron=60):
        
        t0 = time.time()
        print('Regenerating the PSFs ... ')
        p2x = lambda x: list(x[0:5]) + [0] + [x[5]] + [0,0,0,1,0,0,0] + list(x[6:-1])
        
        self.mag_err = np.empty(self.nCases,dtype=list)
        self.mse   = np.empty(self.nCases,dtype=list)
        self.SR    = np.empty(self.nCases,dtype=list)
        self.FWHM  = np.empty(self.nCases,dtype=list)
        
        self.psf_mean = np.empty(self.nCases,dtype=list)
        self.psf_diff_mean = np.empty(self.nCases,dtype=list)
        self.psf_diff_std  = np.empty(self.nCases,dtype=list)
        if fit:
            self.psf_diff_mean_fit = np.empty(self.nCases,dtype=list)
            self.psf_diff_std_fit  = np.empty(self.nCases,dtype=list)
        nPx = self.psfao.ao.cam.fovInPix
        nC  = 2 + fit*1
        
        for n in range(self.nCases):
            # instantiating outputs
            nPSFtot          = self.gtruth[n].shape[1]
            self.mag_err[n]  = np.zeros((nC-1,nPSF))
            self.mse[n]      = np.zeros((nC-1,nPSF))
            self.SR[n]       = np.zeros((nC,nPSF))
            self.FWHM[n]     = np.zeros((nC,nPSF))
            
            self.psf_mean[n] = np.zeros((nPx,nPx))
            self.psf_diff_mean[n] = np.zeros((nPx,nPx))
            self.psf_diff_std[n]  = np.zeros((nPx,nPx))
            if fit:
                self.psf_diff_mean_fit[n] = np.zeros((nPx,nPx))
                self.psf_diff_std_fit[n]  = np.zeros((nPx,nPx))  
                
            # down-selection 
            idx = np.random.randint(0,high=nPSFtot,size=nPSF)
            
            for k in range(nPSF):
                # GETTING THE GROUND TRUTH AND THE ESTIMATED PSF
                psf_true = np.squeeze(self.psfao( p2x(self.gtruth[n][:,idx[k]]) ))
                psf_ml   = np.squeeze(self.psfao( p2x(self.nnest[n][:,idx[k]]) )) 
                
                # PHOTOMETRY
                tmp,_,_,_,_     = linregress(psf_true.reshape(-1),psf_ml.reshape(-1))
                self.mag_err[n][0,k] = -2.5*np.log10(tmp)
                
                # AVERAGE PSF
                self.psf_mean[n]      += psf_true/nPSF
                self.psf_diff_mean[n] += abs(psf_ml - psf_true)/nPSF
                self.psf_diff_std[n]  += (psf_ml - psf_true)**2 /nPSF
                
                # GETTING METRICS
                self.mse[n][0,k]    = 1e2 * np.sqrt(np.sum((psf_ml-psf_true)**2))/psf_true.sum()
                self.SR[n][0,k]   = FourierUtils.getStrehl(psf_true,self.psfao.ao.tel.pupil,self.psfao.freq.sampRef) 
                self.SR[n][1,k]   = FourierUtils.getStrehl(psf_ml,self.psfao.ao.tel.pupil,self.psfao.freq.sampRef) 
                self.FWHM[n][0,k] = FourierUtils.getFWHM(psf_true,self.psfao.ao.cam.psInMas,nargout=1) 
                self.FWHM[n][1,k] = FourierUtils.getFWHM(psf_ml,self.psfao.ao.cam.psInMas,nargout=1) 
                
                # PSF-FITTING
                if fit:
                    weights = 1
                    if mag != 0:
                        # noising the image
                        Flux      = 10 ** (-0.4*(mag - zP))*DIT*nDIT
                        skyFlux   = 10 ** (-0.4*(skyMag - zP)) * nDIT * DIT * (self.psfao.ao.cam.psInMas/1e3) ** 2
                        ronStack  = ron * np.sqrt(nDIT)
                        noise_sky = np.random.poisson(skyFlux*np.ones_like(psf_true))
                        noise_dec = ronStack*np.random.randn(psf_true.shape[0],psf_true.shape[1])
                        noise_dec-= noise_dec.min()
                        im_noise  = np.random.poisson(Flux*psf_true) + noise_sky  + noise_dec
                        # computing the weights
                        weights   = 1.0/np.sqrt(ronStack**2 + psf_true)   
                    else:
                        im_noise = psf_true
                        
                    # defining the initial guess
                    x0   = p2x(self.gtruth[n][:,idx[k]])
                    fixed= (False,)*5 + (True,False,) + (True,)*3 + (False,)*4 + (False,)*self.psfao.ao.tel.nModes
                    # fit the image
                    res  = psfFitting(im_noise,self.psfao,x0,fixed=fixed,verbose=-1,weights=weights)
                    
                    # averaged fitted-PSF
                    self.psf_diff_mean_fit[n] += abs(res.psf - psf_true)/nPSF
                    self.psf_diff_std_fit[n]  += (res.psf - psf_true)**2 /nPSF
                
                    # compare PSFs
                    tmp,_,_,_,_     = linregress(psf_true.reshape(-1),res.psf.reshape(-1))
                    self.mag_err[n][1,k] = -2.5*np.log10(tmp)
                    self.mse[n][1,k]    = res.mse
                    self.SR[n][2,k]     = FourierUtils.getStrehl(res.psf,self.psfao.ao.tel.pupil,self.psfao.freq.sampRef) 
                    self.FWHM[n][2,k]   = FourierUtils.getFWHM(res.psf,self.psfao.ao.cam.psInMas,nargout=1)
                
                
            self.psf_diff_std[n] = np.sqrt(self.psf_diff_std[n])
            if fit:
                self.psf_diff_std_fit[n] = np.sqrt(self.psf_diff_std_fit[n])
                
            print('... Done in %.2f s ! '%(time.time() - t0))