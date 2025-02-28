#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:02:21 2021

@author: omartin
"""

#%% IMPORTING LIBRAIRIES
import time
import os
from distutils.spawn import find_executable
import matplotlib.ticker as mtick
from tqdm import tqdm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import QTable
from scipy.stats import linregress

import p3.aoSystem.FourierUtils as FourierUtils
from p3.psfao21.psfao21 import psfao21
from p3.psfFitting.psfFitting import psfFitting
np.random.seed(1)

#%% CLASS
class deepLoopPerformance:
    ''' neuralNetworkPerformance class that allows to assess the
    reconstruction performance of DEEPLOOP from the output .txt files
    '''

    def __init__(self, path_txt, path_ini=None, path_save=None, path_root='',
                 nPSF=100, fit=False, mag=0, zP=25.44, DIT=0.5, nDIT=50,
                 skyMag=13.6, ron=60, init='truth'):
        '''
        path_txt can be either:
            - a string of characters that gives the path to the txt file
            - a tuple of string for multi-configurations analyses
        '''
        # PARSING INPUTS
        # txt files created by DEEPLOOP
        if type(path_txt)==str:
            path_txt = [path_txt]
        self.path_txt = np.unique(path_txt)
        self.nCases = len(self.path_txt)
        # verifying the non-emptiness of the path_txt field
        if not self.nCases:
            raise ValueError('The field path_txt is empty; no analyses possible.')

        # verifying the existence of the txt files
        for k in range(self.nCases):
            if not os.path.isfile(self.path_txt[k]):
                raise ValueError('ERROR : the path #%d does not exist'%(k+1))

        # .ini file : needed for PSF computation
        self.path_ini = path_ini
        if self.path_ini == None:
            print('WARNING : the field path_ini is None : no analyses on PSFs possible')

        # IDENTIFYING DATA
        self.idNN = np.empty(self.nCases,dtype=list)
        self.idData = np.empty(self.nCases,dtype=list)
        self.dataType = np.empty(self.nCases,dtype=list)
        self.nParam = np.empty(self.nCases,dtype=list)

        for n in range(self.nCases):
            # data identification from the file name
            s = self.path_txt[n].split('/')[-1]
            s = s.split('_')
            self.idNN[n] = s[1]
            self.idData[n] = s[2]
            self.dataType[n] = s[0][3:]
            # reading the first line to count the number of parameters
            self.nParam[n] = self.read_txt_files(self.path_txt[n],getParamNumberOnly=True)

        self.nNetworks = len(self.idNN)
        self.nDataSets = len(self.idData)
        print('Test of %d network architectures performance on %d data sets'%(self.nNetworks,self.nDataSets))

        # INSTANTIATING DATA STRUCTURES
        self.gtruth = np.empty(self.nCases,dtype=tuple)
        self.nnest = np.empty(self.nCases,dtype=tuple)
        self.labels = np.empty(self.nCases,dtype=tuple)
        self.metrics_param = np.empty(self.nCases,dtype=tuple)

        for n in range(self.nCases):
            # get parameters
            self.gtruth[n],self.nnest[n],self.labels[n] = self.read_txt_files(self.path_txt[n])
            # get metrics
            self.metrics_param[n] = np.zeros((self.nParam[n],7))
            for k in range(self.nParam[n]):
                self.metrics_param[n][k] = self.get_parameters_metrics(self.gtruth[n][k],self.nnest[n][k])

            self.path_save = path_save
            if self.path_save:
                Tab = QTable(names=self.labels[n])
                for k in range(7):
                    Tab.add_row(self.metrics_param[n][:,k])
                Tab.write(path_save + self.idNN[n] + '_' + self.idData[n] + '_' + self.dataType[n] + '.csv',overwrite=True)


        # COMPUTING PSF METRICS
        if self.path_ini:
            self.fit = fit
            self.psfao = psfao21(path_ini, path_root=path_root)
                #managing bounds
            b_low = [-np.inf,]*(self.nParam[0]+1)
            b_up = [np.inf,]*(self.nParam[0]+1)
            self.psfao.bounds = (b_low, b_up)
            self.psfao.psfao_19.bounds = (b_low, b_up)

            self.init_fit = init
            self.get_psf_metrics(nPSF=nPSF, fit=self.fit, init=self.init_fit,
                                 mag=mag, zP=zP, DIT=DIT, nDIT=nDIT,
                                 skyMag=skyMag, ron=ron)

    def __call__(self, fontsize=22, fontfamily='normal', fontserif='Palatino',
                 figsize=20, constrained_layout=True, nBins=100, nstd=np.inf):
        '''
        Display DEEPLOOP performance
        '''
        # managing display configuration

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
            fig , axs = plt.subplots(k1,k2,figsize=(int(figsize*k2),int(figsize*k1)),constrained_layout=constrained_layout)
            a=-1
            for m in range(nP):
                b = m%k2
                if b==0:
                    a+=1

                mn = min(self.gtruth[n][m].min(),self.nnest[n][m].min())
                mx = max(self.gtruth[n][m].max(),self.nnest[n][m].max())
                axs[a,b].plot(self.gtruth[n][m],self.nnest[n][m],'bo')
                axs[a,b].plot([mn,mx],[mn,mx],'k--')
                axs[a,b].set_xlabel(self.labels[n][m] + ' simulated',fontsize=fontsize)
                axs[a,b].set_ylabel(self.labels[n][m] + ' reconstructed',fontsize=fontsize)
                #axs[a,b].set_aspect('equal')
                if mx > 10 or mn < 0.1:
                    formatter.set_powerlimits((-1,1))
                    axs[a,b].yaxis.set_major_formatter(formatter)
                    axs[a,b].xaxis.set_major_formatter(formatter)

            # saving
            if self.path_save:
                path_fig = self.path_save + 'versusplots_parameters_' + self.dataType[n]\
                            + 'data_' + self.idNN[n] + '_' + str(self.idData[n])\
                            + '_' + str(len(self.nnest[n][0]))
                plt.savefig(path_fig)

        # ------ HISTOGRAMS
        for n in range(self.nCases):
            # creating the figure
            nP = self.nParam[n]
            k1 = int(np.sqrt(nP))
            k2 = int(nP/k1)
            fig , axs = plt.subplots(k1, k2, figsize=(int(figsize*k2), int(figsize*k1)),
                                     constrained_layout=constrained_layout)
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

                err = err[abs(err)<nstd*err.std()]
                axs[a,b].hist(err, weights=np.ones_like(err) / len(err), bins=nBins)
                axs[a,b].set_xlabel(self.labels[n][m] + ' error ('+lab+')')
                axs[a,b].set_ylabel('Probability')
                axs[a,b].plot([0,0],axs[a,b].get_ylim(),'k--')
                if nstd != np.inf:
                    axs[a,b].set_xlim([-nstd*err.std(),nstd*err.std()])

            # saving
            if self.path_save:
                path_fig = self.path_save + 'histograms_parameters_' + self.dataType[n]\
                            + 'data_' + self.idNN[n] + '_' + str(self.idData[n])\
                            + '_' + str(len(self.nnest[n][0]))
                plt.savefig(path_fig)

        # ------ PSFs METRICS
        if self.path_ini:
            for n in range(self.nCases):
                # creating the figure
                fig, axs = plt.subplots(2, 2, figsize=(figsize, figsize), constrained_layout=constrained_layout)
                # MSE
                err = self.mse[n]
                nPSF = err.shape[1]
                if self.fit:
                    err = self.mse[n][0]
                    errfit = self.mse[n][1]
                    err = err[err==err]
                    errfit = errfit[errfit==errfit]
                    err = err[err<=nstd*err.std()]
                    errfit = errfit[errfit<=nstd*errfit.std()]
                    mn = 0
                    mx = max([max(err),max(errfit)])
                    bins = np.linspace(mn, mx, num=int(min(nPSF/10, nBins)))
                    axs[0,0].hist(err, weights=np.ones_like(err)/len(err),
                                  bins=bins, label='DEEPLOOP', alpha=0.5)
                    axs[0,0].hist(errfit, weights=np.ones_like(errfit)/len(errfit),
                                  bins=bins, label='PSF-FITTING', alpha=0.5)
                    axs[0,0].legend()
                else:
                    err = err[0][err[0]==err[0]]
                    axs[0,0].hist(err, weights=np.ones_like(err)/len(err), bins=int(min(nPSF/10,nBins)))

                axs[0,0].set_xlabel('Mean square error (\%)')
                axs[0,0].set_ylabel('Probability')

                # SR
                if self.fit:
                    err = 1e2*(self.SR[n][1] - self.SR[n][0])/self.SR[n][0]
                    errfit = 1e2*(self.SR[n][2] - self.SR[n][0])/self.SR[n][0]
                    err = err[err==err]
                    errfit = errfit[errfit==errfit]
                    err = err[abs(err) <= np.median(err) + nstd*err.std()]
                    errfit = errfit[abs(errfit) <= np.median(errfit) + nstd*errfit.std()]
                    mn = min([min(err),min(errfit)])
                    mx = max([max(err),max(errfit)])
                    bins = np.linspace(mn,mx,num=int(min(nPSF/10,nBins)))
                    axs[0,1].hist(err, weights=np.ones_like(err)/len(err),
                                  bins=bins,label='DEEPLOOP',alpha=0.5)
                    axs[0,1].hist(errfit, weights=np.ones_like(errfit)/len(errfit),
                                  bins=bins, label='PSF-FITTING', alpha=0.5)
                    axs[0,1].legend()
                else:
                    err = 1e2*(self.SR[n][1] - self.SR[n][0])/self.SR[n][0]
                    err = err[err==err]
                    axs[0,1].hist(err, weights=np.ones_like(err)/len(err),
                                  bins=int(min(nPSF/10, nBins)))

                axs[0,1].set_xlabel('Strehl-ratio error (\%)')
                axs[0,1].set_ylabel('Probability')
                axs[0,1].plot([0,0],axs[0,1].get_ylim(),'k--')

                # FWHM
                if self.fit:
                    err = 1e2*(self.FWHM[n][1] - self.FWHM[n][0])/self.FWHM[n][0]
                    errfit = 1e2*(self.FWHM[n][2] - self.FWHM[n][0])/self.FWHM[n][0]
                    err = err[err==err]
                    errfit = errfit[errfit==errfit]
                    err = err[abs(err) <= np.median(err) + nstd*err.std()]
                    errfit = errfit[abs(errfit) <= np.median(errfit) + nstd*errfit.std()]
                    mn = min([min(err),min(errfit)])
                    mx = max([max(err),max(errfit)])
                    bins = np.linspace(mn,mx,num=int(min(nPSF/10, nBins)))
                    axs[1,0].hist(err, weights=np.ones_like(err)/len(err),
                                  bins=bins, label='DEEPLOOP', alpha=0.5)
                    axs[1,0].hist(errfit, weights=np.ones_like(errfit)/len(errfit),
                                  bins=bins, label='PSF-FITTING', alpha=0.5)
                    axs[1,0].legend()
                else:
                    err = 1e2*(self.FWHM[n][1] - self.FWHM[n][0])/self.FWHM[n][0]
                    err = err[err==err]
                    axs[1,0].hist(err, weights=np.ones_like(err)/len(err),
                                  bins=int(min(nPSF/10, nBins)))

                axs[1,0].set_xlabel('FWHM error (\%)')
                axs[1,0].set_ylabel('Probability')
                axs[1,0].plot([0,0],axs[1,0].get_ylim(),'k--')

                # Photometry
                if self.fit:
                    err = self.mag_err[n][0]
                    errfit = self.mag_err[n][1]
                    err = err[err==err]
                    errfit = errfit[errfit==errfit]
                    err = err[abs(err) <= np.median(err) + nstd*err.std()]
                    errfit = errfit[abs(errfit) <= np.median(errfit) + nstd*errfit.std()]
                    mn = min([min(err),min(errfit)])
                    mx = max([max(err),max(errfit)])
                    bins = np.linspace(mn,mx,num=int(min(nPSF/10,nBins)))
                    axs[1,1].hist(err, weights=np.ones_like(err)/len(err),
                                  bins=bins, label='DEEPLOOP', alpha=0.5)
                    axs[1,1].hist(errfit, weights=np.ones_like(errfit)/len(errfit),
                                  bins=bins, label='PSF-FITTING', alpha=0.5)
                    axs[1,1].legend()
                else:
                    err = self.mag_err[n][0]
                    err = err[err==err]
                    axs[1,1].hist(err, weights=np.ones_like(err)/len(err),
                                  bins=int(min(nPSF/10, nBins)))

                axs[1,1].set_xlabel('Photometric error (mag)')
                axs[1,1].set_ylabel('Probability')
                axs[1,1].plot([0,0],axs[1,1].get_ylim(),'k--')

                # saving
                if self.path_save:
                    path_fig = self.path_save + 'histograms_psf_' + self.dataType[n]\
                                + 'data_' + self.idNN[n] + '_' + str(self.idData[n])\
                                + '_' + str(len(self.nnest[n][0])) + '_init_' + str(self.init_fit) +'.png'
                    plt.savefig(path_fig)


                # ------ PSFs PLOTS
                plt.figure(figsize=(figsize,figsize), constrained_layout=constrained_layout)
                nPx = self.psf_mean[n].shape[0]
                fov = nPx * self.psfao.ao.cam.psInMas
                x = np.linspace(-fov/2,fov/2,num=nPx)
                plt.semilogy(x, self.psf_mean[n][nPx//2,:],'k',label='Mean PSF')
                plt.semilogy(x, self.psf_diff_mean[n][nPx//2,:],'b',label='Mean differential PSF')
                plt.semilogy(x, self.psf_diff_std[n][nPx//2,:],'r',label='Std differential PSF')
                if self.fit:
                    plt.semilogy(x, self.psf_diff_mean_fit[n][nPx//2,:],'g--',label='Mean differential PSF - fitting')
                    plt.semilogy(x, self.psf_diff_std_fit[n][nPx//2,:],'m--',label='Std differential PSF - fitting')
                plt.legend()
                plt.xlabel('Separation from the optical axis [mas]')
                # saving
                if self.path_save:
                    path_fig = self.path_save + 'psf_plots' + self.dataType[n]\
                                + 'data_' + self.idNN[n] + '_' + str(self.idData[n])\
                                + '_' + str(len(self.nnest[n][0])) + '_init_' + str(self.init_fit) +'.png'
                    plt.savefig(path_fig)

                # ---- histograms of errors deeploop vs psf-fitting
                if self.fit:
                    # creating the figure
                    nP = self.nParam[n]
                    k1 = int(np.sqrt(nP))
                    k2 = int(nP/k1)
                    fig , axs = plt.subplots(k1, k2, figsize=(int(figsize*k2), int(figsize*k1)),
                                             constrained_layout=constrained_layout)
                    a=-1
                    for m in range(nP):
                        b = m%k2
                        if b==0:
                            a+=1
                        b = m%k2

                        if m < 6:
                            err_dl = 1e2*(self.nnest_subfit[n][m] - self.gtruth_subfit[n][m])/self.gtruth_subfit[n][m]
                            err_fit = 1e2*(self.fitEstimates[n][m] - self.gtruth_subfit[n][m])/self.gtruth_subfit[n][m]
                            lab = '\%'
                        else:
                            err_dl = self.nnest_subfit[n][m] - self.gtruth_subfit[n][m]
                            err_fit = self.fitEstimates[n][m] - self.gtruth_subfit[n][m]
                            lab = 'nm'

                        err_dl = err_dl[abs(err_dl)<nstd*err_dl.std()]
                        err_fit = err_fit[abs(err_fit)<nstd*err_fit.std()]
                        axs[a,b].hist(err_dl, weights=np.ones_like(err_dl)/len(err_dl),
                                      bins=int(min(nPSF/10, nBins)), label="DEEPLOOP", alpha=0.5)
                        axs[a,b].hist(err_fit, weights=np.ones_like(err_fit)/len(err_fit),
                                      bins=int(min(nPSF/10,nBins)), label="PSF-FITTING", alpha=0.5)
                        axs[a,b].set_xlabel(self.labels[n][m] + ' error ('+lab+')')
                        axs[a,b].set_ylabel('Probability')
                        axs[a,b].plot([0,0],axs[a,b].get_ylim(),'k--')
                        if nstd!=np.inf:
                            axs[a,b].set_xlim([-nstd*max(err_dl.std(), err_fit.std()),
                                                nstd*max(err_dl.std(), err_fit.std())])
                    plt.legend()
                    # saving
                    if self.path_save:
                        path_fig = self.path_save + 'histograms_parameters_DLvFit_' + self.dataType[n]\
                                    + 'data_' + self.idNN[n] + '_' + str(self.idData[n])\
                                    + '_' + str(len(self.nnest[n][0]))
                        plt.savefig(path_fig)


    def read_txt_files(self, path_txt, getParamNumberOnly=False):
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
        lab = tmp.readline().split()
        labels = lab[1:int((len(lab)-1)/2+1)]
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
        groundTruth = np.zeros((nParam, nData))
        nnEstimates = np.zeros((nParam, nData))
        tmp = open(path_txt)
        lines = tmp.readlines()[2:]
        for n in range(nData):
            tmpP = np.array(lines[n][0:-1].split('\t')).astype(float)
            groundTruth[:,n] = tmpP[0:nParam]
            nnEstimates[:,n] = tmpP[nParam:]

        tmp.close()
        return groundTruth, nnEstimates, labels

    def get_parameters_metrics(self, gtruth, nnEst):
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

    def get_psf_metrics(self, nPSF=1000, fit=False, mag=0, init=0.1, zP=25.44,
                        DIT=0.5, nDIT=50, skyMag=13.6, ron=60):

        t0 = time.time()
        print('Regenerating the PSFs ... ')
        p2x = lambda x: list(x[0:5]) + [0] + [x[5]] + [0, 0, 0, 1, 0, 0, 0] + list(x[6:-1])

        self.mag_err = np.empty(self.nCases, dtype=list)
        self.mse = np.empty(self.nCases, dtype=list)
        self.SR = np.empty(self.nCases, dtype=list)
        self.FWHM = np.empty(self.nCases, dtype=list)

        self.psf_mean = np.empty(self.nCases, dtype=list)
        self.psf_diff_mean = np.empty(self.nCases, dtype=list)
        self.psf_diff_std = np.empty(self.nCases, dtype=list)
        if fit:
            self.psf_diff_mean_fit = np.empty(self.nCases, dtype=list)
            self.psf_diff_std_fit = np.empty(self.nCases, dtype=list)
            self.nnest_subfit = np.empty(self.nCases, dtype=list)
            self.gtruth_subfit = np.empty(self.nCases, dtype=list)
            self.fitEstimates= np.empty(self.nCases, dtype=list)
            p_median = np.median(self.gtruth[0], axis=1)
        nPx = self.psfao.ao.cam.fovInPix
        nC = 2 + fit*1

        for n in range(self.nCases):
            # instantiating outputs
            nPSFtot = self.gtruth[n].shape[1]
            self.mag_err[n] = np.zeros((nC-1, nPSF))
            self.mse[n] = np.zeros((nC-1, nPSF))
            self.SR[n] = np.zeros((nC, nPSF))
            self.FWHM[n] = np.zeros((nC, nPSF))

            self.psf_mean[n] = np.zeros((nPx, nPx))
            self.psf_diff_mean[n] = np.zeros((nPx, nPx))
            self.psf_diff_std[n] = np.zeros((nPx, nPx))
            if fit:
                self.nnest_subfit[n] = np.zeros((self.nParam[n], nPSF))
                self.gtruth_subfit[n] = np.zeros((self.nParam[n], nPSF))
                self.fitEstimates[n] = np.zeros((self.nParam[n], nPSF))
                self.psf_diff_mean_fit[n] = np.zeros((nPx, nPx))
                self.psf_diff_std_fit[n] = np.zeros((nPx, nPx))
                if type(init)==float:
                    frandn = np.random.normal(1, 0.2, (nPSF, self.nParam[n]))

            # down-selection
            idx = np.random.randint(0,high=nPSFtot,size=nPSF)
            if self.nParam[n]==6:
                nModes = 0
            else:
                nModes = self.psfao.ao.tel.nModes

            for k in tqdm(range(nPSF)):
                # GETTING THE GROUND TRUTH AND THE ESTIMATED PSF
                psf_true = np.squeeze(self.psfao(p2x(self.gtruth[n][:,idx[k]])))
                psf_ml = np.squeeze(self.psfao(p2x(self.nnest[n][:,idx[k]])))

                # PHOTOMETRY
                tmp,_,_,_,_  = linregress(psf_true.reshape(-1),psf_ml.reshape(-1))
                self.mag_err[n][0,k] = -2.5*np.log10(tmp)

                # AVERAGE PSF
                self.psf_mean[n] += psf_true/nPSF
                self.psf_diff_mean[n] += abs(psf_ml - psf_true)/nPSF
                self.psf_diff_std[n] += (psf_ml - psf_true)**2 /nPSF

                # GETTING METRICS
                self.mse[n][0,k] = 1e2 * np.sqrt(np.sum((psf_ml-psf_true)**2))/psf_true.sum()
                self.SR[n][0,k] = FourierUtils.getStrehl(psf_true,self.psfao.ao.tel.pupil,self.psfao.freq.sampRef)
                self.SR[n][1,k] = FourierUtils.getStrehl(psf_ml,self.psfao.ao.tel.pupil,self.psfao.freq.sampRef)
                self.FWHM[n][0,k] = FourierUtils.getFWHM(psf_true,self.psfao.ao.cam.psInMas,nargout=1)
                self.FWHM[n][1,k] = FourierUtils.getFWHM(psf_ml,self.psfao.ao.cam.psInMas,nargout=1)

                # PSF-FITTING
                if fit:
                    self.gtruth_subfit[n][:,k] = self.gtruth[n][:, idx[k]]
                    self.nnest_subfit[n][:,k] = self.nnest[n][:, idx[k]]
                    weights = np.ones_like(psf_true)
                    if mag:
                        # noising the image
                        Flux = 10 ** (-0.4*(mag - zP))*DIT*nDIT
                        skyFlux = 10 ** (-0.4*(skyMag - zP)) * nDIT * DIT * (self.psfao.ao.cam.psInMas/1e3) ** 2
                        ronStack = ron * np.sqrt(nDIT)
                        noise_sky = np.random.poisson(skyFlux*np.ones_like(psf_true))
                        noise_dec = ronStack*np.random.randn(psf_true.shape[0],psf_true.shape[1])
                        noise_dec-= noise_dec.min()
                        im_noise = np.random.poisson(Flux*psf_true) + noise_sky  + noise_dec
                        # computing the weights
                        weights = 1.0/np.sqrt(ronStack**2 + psf_true)
                    else:
                        im_noise = psf_true

                    # defining the initial guess
                    if init == 'truth':
                        x0 = p2x(self.gtruth[n][:,idx[k]])
                    elif init == 'median':
                        x0 = p2x(p_median)
                    elif type(init) == float:
                        if init > 1:
                            init/=100
                        p_r= self.gtruth[n][:,idx[k]] * frandn[k]
                        p_r= np.maximum(p_r,self.gtruth[n].min(axis=1))
                        p_r= np.minimum(p_r,self.gtruth[n].max(axis=1))
                        x0 = p2x(p_r)
                    else:
                        raise ValueError("The init field must be 'truth', 'median' or a float")
                    # fit the image; the image musn't be normalized to properly account for the Poisson noise

                    if mag:
                        x0[10] = Flux
                        normType = 0
                    else:
                        normType = 1
                    fixed= (False,)*5 + (True,False,) + (True,)*3 + (False,)*4 + (False,)*nModes

                    res = psfFitting(im_noise, self.psfao, x0, fixed=fixed,
                                      verbose=-1, weights=weights, normType=normType)

                    # Get parameters
                    self.fitEstimates[n][:,k] = list(res.x[0:5]) + [res.x[6]] + list(res.x[14:])

                    # averaged fitted-PSF
                    self.psf_fit_last = res.psf
                    self.psf_true_last = psf_true
                    self.psf_diff_mean_fit[n] += abs(res.psf - psf_true)/nPSF
                    self.psf_diff_std_fit[n] += (res.psf - psf_true)**2 /nPSF

                    # compare PSFs
                    tmp,_,_,_,_  = linregress(psf_true.reshape(-1),res.psf.reshape(-1))
                    self.mag_err[n][1,k]= -2.5*np.log10(tmp)
                    self.mse[n][1,k] = 1e2 * np.sqrt(np.sum((res.psf-psf_true)**2))/psf_true.sum()
                    self.SR[n][2,k] = FourierUtils.getStrehl(res.psf,self.psfao.ao.tel.pupil,self.psfao.freq.sampRef)
                    self.FWHM[n][2,k] = FourierUtils.getFWHM(res.psf,self.psfao.ao.cam.psInMas,nargout=1)

            self.psf_diff_std[n] = np.sqrt(self.psf_diff_std[n])
            if fit:
                self.psf_diff_std_fit[n] = np.sqrt(self.psf_diff_std_fit[n])

            print('... Done in %.2f s ! '%(time.time() - t0))

# 35421.20 for 10k cases, init=0.1
def measure_accuracy_metrics(path_test_results, path_ini=None, nPSF=100):
    """
    Read the multiple .txt files from the TEST folder of a given simulation to measure
    the accuracy on predicted parameters.
    """
    # grabbing the paths to the files
    if not os.path.isdir(path_test_results):
        raise ValueError("The folder does not exist")

    list_txt = os.listdir(path_test_results)
    list_txt_all = [path_test_results + file for file in list_txt if '_all' in file]
    list_txt_all.sort()
    n_tests = len(list_txt_all)

    # running deepLoopPerformance to get metrics in percents
    dlp = deepLoopPerformance(list_txt_all, path_ini=path_ini, nPSF=nPSF)
    metrics_bias = np.array([dlp.metrics_param[n][:,4] for n in range(n_tests)])
    metrics_std = np.array([dlp.metrics_param[n][:,5] for n in range(n_tests)])

    # save results
    bias_mean = np.mean(metrics_bias, axis=0)
    bias_std = np.std(metrics_bias, axis=0)
    std_mean = np.mean(metrics_std, axis=0)
    std_std = np.std(metrics_std, axis=0)
    d = {'label':dlp.labels[0], 'mean bias [%]':bias_mean, 'bias precision [%]':bias_std,
         'mean std [%]':std_mean, 'std precision [%]':std_std}
    df_param = pd.DataFrame(data=d)

    df_psf = None
    if path_ini is not None:
        def bias_and_std(err):
            bias = np.mean(err, axis=0)
            std = np.std(err, axis=0)
            return bias.mean(), bias.std(), std.mean(), std.std()
        SR_err = np.array([(dlp.SR[n][0] - dlp.SR[n][1])/dlp.SR[n][0] for n in range(n_tests)])
        FW_err = np.array([(dlp.FWHM[n][0] - dlp.FWHM[n][1])/dlp.FWHM[n][0] for n in range(n_tests)])
        mg_err = np.array([dlp.mag_err[n][0] for n in range(n_tests)])

        SR_bias_mean, SR_bias_std, SR_std_mean, SR_std_std = bias_and_std(SR_err)
        FW_bias_mean, FW_bias_std, FW_std_mean, FW_std_std = bias_and_std(FW_err)
        mg_bias_mean, mg_bias_std, mg_std_mean, mg_std_std = bias_and_std(mg_err)


        d = {'SR bias mean[%]':[SR_bias_mean], 'SR bias std[%]':[SR_bias_std],
             'SR std mean[%]':[SR_std_mean], 'SR std std[%]':[SR_std_std],
             'FWHM bias mean[%]':[FW_bias_mean], 'FWHM bias std[%]':[FW_bias_std],
             'FWHM std mean[%]':[FW_std_mean], 'FWHM std std[%]':[FW_std_std],
             'mg bias mean[mag]':[mg_bias_mean], 'mg bias std[mag]':[mg_bias_std],
             'mg std mean[mag]':[mg_std_mean], 'mg std std[mag]':[mg_std_std]}
        df_psf = pd.DataFrame(data=d)

    return df_param, df_psf

def measure_accuracy_metrics_multiple_folders(path_folder, path_ini=None, nPSF=100):
    """
    Grab the accuracy on metrics for multiple tests case and concatenate results
    """
    if not os.path.isdir(path_folder):
        raise ValueError("The folder does not exist")

    tmp = os.listdir(path_folder)
    list_fold = [path_folder+file for file in tmp if os.path.isdir(path_folder+file)]
    n_psf = [int(fold.split('_')[-1]) for fold in list_fold]
    idx = np.argsort(n_psf)
    list_fold = np.take(list_fold, idx)
    n_psf = np.take(n_psf, idx)
    # loop
    df_param = pd.DataFrame()
    df_psf = pd.DataFrame()
    for fold in list_fold:
        df_pa, df_ps = measure_accuracy_metrics(fold+"/TEST/", path_ini=path_ini, nPSF=nPSF)
        df_param = df_param.append(df_pa)
        df_psf = df_psf.append(df_ps)

    return df_param, df_psf, n_psf

def display_metrics_accuracy(df_param, n_psf, df_psf=None, fontsize=22,
                             fontfamily='normal', fontserif='Palatino',
                             figsize=20, constrained_layout=True):
    """
    Display metrics accuracy versus number of training data.
    """
    # format
    font = {'weight' : 'normal',
    'size'   : fontsize,
    'family': fontfamily,
    'serif': fontserif}
    mpl.rc('font', **font)
    # latex
    usetex = False
    if find_executable('tex'):
        usetex = True
    text = {'usetex': usetex}
    mpl.rc('text',**text)


    # ----------- DISPLAYING BIAS AND PRECISION ON PARAMEERS ERRORS
    # extracting and transforming data
    r0 = df_param.loc[0];r0["n_psf"] = n_psf
    bg = df_param.loc[1];bg["n_psf"] = n_psf
    s2 = df_param.loc[2];s2["n_psf"] = n_psf
    ax = df_param.loc[3];ax["n_psf"] = n_psf
    pe = df_param.loc[4];pe["n_psf"] = n_psf
    be = df_param.loc[5];be["n_psf"] = n_psf

    # plot
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(figsize,figsize),
                            constrained_layout=constrained_layout)
    # bias
    axs[0].errorbar(n_psf, (r0["mean bias [%]"]), yerr=r0["bias precision [%]"], fmt='bo-', label='r0')
    axs[0].errorbar(n_psf, (bg["mean bias [%]"]), yerr=bg["bias precision [%]"], fmt='gs-', label='bg')
    axs[0].errorbar(n_psf, (s2["mean bias [%]"]), yerr=s2["bias precision [%]"], fmt='r8-', label='s2')
    axs[0].errorbar(n_psf, (ax["mean bias [%]"]), yerr=ax["bias precision [%]"], fmt='cp-', label='ax')
    axs[0].errorbar(n_psf, (pe["mean bias [%]"]), yerr=pe["bias precision [%]"], fmt='mP-', label='pe')
    axs[0].errorbar(n_psf, (be["mean bias [%]"]), yerr=be["bias precision [%]"], fmt='yD-', label='be')
    axs[0].legend()
    axs[0].set_xlabel("\# training PSF")
    axs[0].set_ylabel("Relative bias [\%]")
    axs[0].grid('on')

    axs[1].errorbar(n_psf, (r0["mean std [%]"]), yerr=r0["std precision [%]"], fmt='bo-', label='r0')
    axs[1].errorbar(n_psf, (bg["mean std [%]"]), yerr=bg["std precision [%]"], fmt='gs-', label='bg')
    axs[1].errorbar(n_psf, (s2["mean std [%]"]), yerr=s2["std precision [%]"], fmt='r8-', label='s2')
    axs[1].errorbar(n_psf, (ax["mean std [%]"]), yerr=ax["std precision [%]"], fmt='cp-', label='ax')
    axs[1].errorbar(n_psf, (pe["mean std [%]"]), yerr=pe["std precision [%]"], fmt='mP-', label='pe')
    axs[1].errorbar(n_psf, (be["mean std [%]"]), yerr=be["std precision [%]"], fmt='yD-', label='be')
    axs[1].legend()
    axs[1].set_xlabel("\# training PSF")
    axs[1].set_ylabel("Relative precision [\%]")
    axs[1].set_yscale('log')
    axs[1].grid('on')

    # ----------- DISPLAYING BIAS AND PRECISION ON PSF METRICS
    if df_psf is not None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(figsize,figsize),
                                constrained_layout=constrained_layout)
        axs[0].errorbar(n_psf, df_psf["SR bias mean[%]"], yerr=df_psf["SR bias std[%]"], fmt='bo-', label='Strehl-ratio')
        axs[0].errorbar(n_psf, df_psf["FWHM bias mean[%]"], yerr=df_psf["FWHM bias std[%]"], fmt='r8-', label='FWHM')
        axs[0].legend()
        axs[0].set_xlabel("\# training PSF")
        axs[0].set_ylabel("Relative bias [\%]")
        axs[0].grid('on')

        axs[1].errorbar(n_psf, df_psf["SR std mean[%]"], yerr=df_psf["SR std std[%]"], fmt='bo-', label='Strehl-ratio')
        axs[1].errorbar(n_psf, df_psf["FWHM std mean[%]"], yerr=df_psf["FWHM std std[%]"], fmt='r8-', label='FWHM')
        axs[1].legend()
        axs[1].set_xlabel("\# training PSF")
        axs[1].set_ylabel("Relative precision [\%]")
        axs[1].set_yscale('log')
        axs[1].grid('on')