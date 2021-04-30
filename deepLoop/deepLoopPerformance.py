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
from astropy.table import QTable

from psfao21.psfao21 import psfao21

#%% CLASS
class deepLoopPerformance:
    ''' neuralNetworkPerformance class that allows to assess the 
    reconstruction performance of DEEPLOOP from the output .txt files
    '''
    
    def __init__(self,path_txt,path_ini=None,path_save=None):
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
            self.nParam[n]   = self.readTxtFiles(self.path_txt[n],getParamNumberOnly=True)
            
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
            self.gtruth[n],self.nnest[n],self.labels[n] = self.readTxtFiles(self.path_txt[n])
            # get metrics
            self.metrics_param[n] = np.zeros((self.nParam[n],7))
            for k in range(self.nParam[n]):
                self.metrics_param[n][k] = self.getParametersMetrics(self.gtruth[n][k],self.nnest[n][k])
                
            if path_save:
                Tab = QTable(names=self.labels[n])   
                for k in range(7):
                    Tab.add_row(self.metrics_param[n][:,k])
                Tab.write(path_save + self.idNN[n] + '_' + self.idData[n] + '_' + self.dataType[n] + '.csv',overwrite=True)
            
            
    def __call__(self,fontsize=16,fontfamily='serif',fontserif='Palatino',figsize=(20,20),getPSF=False,constrained_layout=True):
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

        for n in range(self.nCases):
            # creating the figure
            nP = self.nParam[n]
            k1 = int(np.sqrt(nP))
            k2 = int(nP/k1)
            fig , axs = plt.subplots(k1,k2,figsize=figsize,constrained_layout=constrained_layout)
            for m in range(nP):
                if m >= k2:
                    a=1
                else:
                    a=0
                b = m%k2
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
            
            # Display metrics on parameters
            
            
    def readTxtFiles(self,path_txt,getParamNumberOnly=False):
        '''
        Reading the .txt input file and populating the gtruth and nnest arrays
        '''
        def isfloat(string):
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
            if isfloat(firstLine[n]):
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
            
    def getParametersMetrics(self,gtruth,nnEst):
        '''
        getting median, mean, std and Peason coefficients
        '''
        
        def getSubMetrics(vec):
            return np.mean(vec), np.median(vec), np.std(vec)
        
        # absolute difference
        dabs = nnEst - gtruth
        dabs_mean, dabs_med, dabs_std = getSubMetrics(dabs)
        # relative difference
        drel = 1e2 * dabs/gtruth
        drel_mean, drel_med, drel_std = getSubMetrics(drel)
        # Pearson
        coeff_pearson = np.corrcoef(gtruth,nnEst)

        return dabs_mean, dabs_med, dabs_std,drel_mean, drel_med, drel_std,coeff_pearson[0,1]
        
        