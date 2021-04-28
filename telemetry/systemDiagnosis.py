#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:04:21 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import numpy as np
import numpy.fft as fft
import aoSystem.FourierUtils as FourierUtils
from aoSystem.zernike import zernike
from astropy.io import fits
from scipy.ndimage import rotate
from scipy.optimize import least_squares
from psfFitting.confidenceInterval import confidence_interval

# 
# TO DO : 
# - verifying the Zernike reconstruction
# - WFS pupil mask
# - noise estimation, test the rtf approach
# - implement estimation of the wind speed

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000
arc2rad = 1/rad2arc

class systemDiagnosis:

    def __init__(self,trs,noiseMethod='autocorrelation',nshift=1,nfit=2):
        
        self.trs = trs
        
        # Transfer function
        self.TemporalTransferFunction()
        
        # noise
        self.noiseMethod = noiseMethod
        if hasattr(self.trs.wfs,'intensity'):
            self.trs.wfs.nph,self.trs.wfs.ron,self.trs.tipTilt.nph,self.trs.tipTilt.ron = self.GetNumberPhotons()
        self.trs.wfs.Cn_ao,self.trs.tipTilt.Cn_tt = self.GetNoiseVariance(noiseMethod=self.noiseMethod,nshift=nshift,nfit=nfit)
        
        # GET THE VARIANCE
        Cnn = np.diag(self.trs.wfs.Cn_ao)
        Cnn = Cnn[Cnn!=0]
        self.trs.wfs.noiseVar = [(2*np.pi/self.trs.wfs.wvl)**2 *np.mean(Cnn),]*len(self.trs.wfs.nSubap)
        
        # Zernike
        self.trs.wfs.Cz_ao, self.trs.tipTilt.Cz_tt = self.ReconstructZernike()
        
        # Atmosphere statistics
        r0, L0, tau0, dr0, dL0, dtau0 = self.GetAtmosphereStatistics()
        self.trs.atm.r0_tel   = r0
        self.trs.atm.L0_tel   = L0
        self.trs.atm.tau0_tel = tau0
        self.trs.atm.dr0_tel  = dr0
        self.trs.atm.dL0_tel  = dL0
        self.trs.atm.dtau0_tel= dtau0
#%%    
    def TemporalTransferFunction(self):
        def delayTransferFunction(freq,fSamp,delay):
            return np.exp(-2*complex(0,1)*freq*delay/fSamp)
        def wfsTransferFunction(freq,fSamp):
            return np.sinc(freq/fSamp/np.pi)*np.exp(-1*complex(0,1)*freq/fSamp)
        def servoTransferFunction(freq,fSamp,num,den):
            z = np.exp(-2*complex(0,1)*np.pi*freq/fSamp)
            ho_num = num[0] + num[1]*z + num[2]*z**2 + num[3]*z**3
            ho_den = 1 + den[0]*z + den[1]*z**2 + den[2]*z**3
            return ho_num/ho_den
        
        #1\ HIGH-ORDER LOOP
        self.trs.holoop.tf.freq = np.linspace(self.trs.holoop.freq/self.trs.wfs.nExp,self.trs.wfs.nExp/2,num=self.trs.wfs.nExp//2)*self.trs.holoop.freq/self.trs.wfs.nExp
        #1.2 wfs TF
        self.trs.holoop.tf.wfs  = wfsTransferFunction(self.trs.holoop.tf.freq,self.trs.holoop.freq)
        #1.3. Lag tf
        self.trs.holoop.tf.lag = delayTransferFunction(self.trs.holoop.tf.freq,self.trs.holoop.freq,self.trs.holoop.lat*self.trs.holoop.freq)
        #1.4 servo tf
        self.trs.holoop.tf.servo = servoTransferFunction(self.trs.holoop.tf.freq,self.trs.holoop.freq,self.trs.holoop.tf.num,self.trs.holoop.tf.den)
        #1.5. open loop tf
        self.trs.holoop.tf.ol = self.trs.holoop.tf.wfs*self.trs.holoop.tf.servo*self.trs.holoop.tf.lag
        #1.6. closed loop tf
        self.trs.holoop.tf.ctf = self.trs.holoop.tf.ol/(1+ self.trs.holoop.tf.ol)
        #1.7. rejection tf
        self.trs.holoop.tf.rtf = 1 - self.trs.holoop.tf.ctf
        #1.8. noise transfer function
        self.trs.holoop.tf.ntf = np.squeeze(self.trs.holoop.tf.servo/(1+self.trs.holoop.tf.ol))
        self.trs.holoop.tf.pn  = (np.trapz(self.trs.holoop.tf.freq,abs(self.trs.holoop.tf.ntf)**2)*2/self.trs.holoop.freq)
        
        # 2\ TT LOOP
        #2.1. Define the frequency vectors
        self.trs.ttloop.tf.freq = np.linspace(self.trs.ttloop.freq/self.trs.tipTilt.nExp,self.trs.tipTilt.nExp/2,self.trs.tipTilt.nExp//2)*self.trs.ttloop.freq/self.trs.tipTilt.nExp
        #2.2 wfs TF
        self.trs.ttloop.tf.wfs  = wfsTransferFunction(self.trs.ttloop.tf.freq,self.trs.ttloop.freq)
        #2.3 TT Lag tf
        self.trs.ttloop.tf.lag = delayTransferFunction(self.trs.ttloop.tf.freq,self.trs.ttloop.freq,self.trs.ttloop.lat*self.trs.ttloop.freq)
        #2.4 TT servo tf
        self.trs.ttloop.tf.servo = servoTransferFunction(self.trs.ttloop.tf.freq,self.trs.ttloop.freq,self.trs.ttloop.tf.num,self.trs.ttloop.tf.den)
        #2.5 open loop tf
        self.trs.ttloop.tf.ol  = self.trs.ttloop.tf.wfs*self.trs.ttloop.tf.servo*self.trs.ttloop.tf.lag
        #2.6 closed loop tf
        self.trs.ttloop.tf.ctf = self.trs.ttloop.tf.ol/(1+self.trs.ttloop.tf.ol)
        #2.7 rejection tf
        self.trs.ttloop.tf.rtf = 1 - self.trs.ttloop.tf.ctf
        #2.8 noise transfer function
        self.trs.ttloop.tf.ntf = np.squeeze(self.trs.ttloop.tf.servo/(1+self.trs.ttloop.tf.ol))
        self.trs.ttloop.tf.pn  = (np.trapz(self.trs.ttloop.tf.freq,abs(self.trs.ttloop.tf.ntf)**2)*2/self.trs.ttloop.freq)

    def GetNumberPhotons(self):
        """
        ESTIMATE THE NUMBER OF PHOTONS FROM WFS MAPS
        """
        nph = ron = nph_tt = ron_tt = 0
        
        nExp, nS, _ = self.trs.wfs.intensity.shape
        maps = self.trs.wfs.intensity.reshape((nExp,nS**2))
        if np.any(maps):
            pixInt = np.sort(maps/self.trs.wfs.gain ,axis=1)
            nSl_c  = np.count_nonzero(self.trs.mat.R[100,:,0])
            pixCog = pixInt[:,-nSl_c//2:]
            nph = np.mean(pixCog) # in photo-events/frame/subap            
            # Read-out noise estimation
            ron = np.median(np.std(pixCog,axis=0))
 
        
        if self.trs.aoMode == 'LGS':
              nph = np.mean(self.trs.tipTilt.intensity/self.trs.tipTilt.gain)
              ron = np.std(self.trs.tipTilt.intensity/self.trs.tipTilt.gain)
        else:
            nph_tt = nph
            ron_tt =ron
        
        return nph, ron, nph_tt, ron_tt
          
    def GetNoiseVariance(self,noiseMethod='autocorrelation',nshift=1,nfit=2):
        """
            ESTIMATE THE WFS NOISE VARIANCE
        """                    
        
        
        def SlopesToNoise(u,noiseMethod='autocorrelation',nshift=1,nfit=2,rtf=None):
        
            nF,nU      = u.shape
            u         -= np.mean(u,axis=0)
            Cnn        = np.zeros((nU,nU))
            validInput = np.argwhere(u.std(axis=0)!=0)
            
            if noiseMethod == 'rtf':
                
                fftS = fft.fft(u,axis=0)
                fftS = fftS[0:nF//2,:]/rtf
                # truncate to keep high-spatial frequencies only
                sub_dim = np.floor(9*nF/2/10)
                fftS = fftS[sub_dim:,:]
                cfftS= np.conj(fftS)
       
                for i in range(nU): 
                    if validInput[i]:
                        # Get the cross PSD
                        crossPSD  = fftS[i,:] * cfftS[i:,:]
                        # Estimate the noise plateau
                        Cnn[i,i:]  = np.median(np.real(crossPSD),axis=0)
                    
                Cnn = (np.transpose(Cnn) + Cnn - np.diag(np.diag(Cnn)))//nF
                
            if noiseMethod == 'interpolation':
                # Polynomial fitting procedure
                delay   = np.linspace(0,1,nfit+1)
                for i in np.arange(0,nU,1):
                    g      = fft.ifft(fft.fft(u[i,:])*np.conjugate(fft.fft(u[i,:])))/nF
                    mx     = g.max()
                    fit    = np.polyfit(delay[1:nfit+1],g[1:nfit+1],nfit)
                    yfit   = 0*delay
                        
                    for k in np.arange(0,nfit+1,1):
                        yfit = yfit + fit[k]*delay**(nfit-k)
                        
                    Cnn[i,i] = mx - yfit[0]              
        
            if noiseMethod == 'autocorrelation':
                du_n  = u - np.roll(u,-nshift,axis=0)
                du_p  = u - np.roll(u,nshift,axis=0)
                Cnn   = 0.5*(np.matmul(u.T,du_n + du_p))/nF
            
            return Cnn
            
        if noiseMethod == 'nonoise':
            Cn_ao = 0
            Cn_tt = 0
            
        else:
            rtf  = self.trs.holoop.tf.ctf/self.trs.holoop.tf.wfs
            Cn_ao = SlopesToNoise(self.trs.dm.com,noiseMethod=noiseMethod,nshift=nshift,nfit=nfit,rtf=rtf)
            if self.trs.aoMode == 'LGS':
                rtf  = self.trs.ttloop.tf.ctf/self.trs.ttloop.tf.wfs
                Cn_tt = SlopesToNoise(self.trs.tipTilt.com,noiseMethod=noiseMethod,nshift=nshift,nfit=nfit)
            else:
                Cn_tt = 0
                
        return Cn_ao, Cn_tt
     
#%%    
    def ReconstructZernike(self,nZer=30,wfsMask=None,j0=4):
        """
        Reconstrut the covariance matrix of the Zernike coefficients from the reconstructed open-loop wavefront
        """
        # defining Zernike modes
        self.trs.wfs.jIndex = list(range(j0,nZer+j0))
            
        # defining the pupil mask and the Zernike modes
        if wfsMask==None:
            wfsMask = fits.getdata(self.trs.tel.path_pupil)
            ang     = self.trs.wfs.theta[0] + self.trs.tel.pupilAngle
            wfsMask = FourierUtils.interpolateSupport(rotate(wfsMask,ang,reshape=False),self.trs.tel.resolution).astype(bool)
        
        self.z = zernike(self.trs.wfs.jIndex,self.trs.tel.resolution,pupil=wfsMask)
        
        # computing the Zernike reconstructor
        u2ph = self.trs.mat.dmIF
        zM   = self.z.modes.T.reshape((self.trs.tel.resolution**2,nZer))
        ph2z = np.dot(np.linalg.pinv(np.dot(zM.T,zM)),zM.T)
        self.trs.mat.u2z = np.dot(ph2z,u2ph)
        
        # open-loop reconstruction
        dt    = self.trs.holoop.lat * self.trs.holoop.freq
        dt_up = int(np.ceil(dt))
        dt_lo = int(np.floor(dt))
        self.trs.dm.com_delta = (1-dt_up+dt)*np.roll(self.trs.rec.res,(-dt_up,0)) + (1-dt+dt_lo)*np.roll(self.trs.rec.res,(-dt_lo,0))
        u_ol  = self.trs.dm.com + self.trs.dm.com_delta
        u_ol  -= np.mean(u_ol,axis=1)[:,np.newaxis]
        
        # reconstructing the amplitude of Zernike coefficients
        self.trs.wfs.coeffs = np.dot(self.trs.mat.u2z,u_ol.T)
        self.trs.wfs.coeffs -= np.mean(self.trs.wfs.coeffs,axis=1)[:,np.newaxis]
        Cz_ho  = np.dot(self.trs.wfs.coeffs,self.trs.wfs.coeffs.T)/self.trs.wfs.nExp
        
        #tip-tilt case
        if self.trs.aoMode == 'LGS':
            Cz_tt = np.matmul(self.trs.tipTilt.com.T,self.trs.tipTilt.com)/self.trs.tipTilt.nExp
        else:
            Cz_tt = Cz_ho[0:1,0:1]
            
        return Cz_ho, Cz_tt
    
    def GetAtmosphereStatistics(self,ftol=1e-8,xtol=1e-8,gtol=1e-8,max_nfev=100,verbose=2):
        
        # DEFINING THE COST FUNCTIONS
        z = self.z
        D = self.trs.tel.D
        class CostClass(object):
            def __init__(self):
                self.iter = 0
            def __call__(self,y):
                if (self.iter%3)==0 and (verbose == 0 or verbose == 1): print("-",end="")
                self.iter += 1
                var_mod = z.CoefficientsVariance(D/y)
                return (var_mod - var_emp)
        cost = CostClass()
    
        # GETTING THE INPUT : VARIANCE OF ZERNIKE MODE
        self.trs.wfs.var_meas      = np.diag(self.trs.wfs.Cz_ao)
        self.trs.wfs.var_noise_zer = 0*np.diag(np.dot(np.dot(self.trs.mat.u2z,self.trs.wfs.Cn_ao),self.trs.mat.u2z.T))
        var_emp                    = (2*np.pi/self.trs.atm.wvl)**2 *(self.trs.wfs.var_meas - self.trs.wfs.var_noise_zer)
        
        # DATA-FITTING WITH A LEVENBERG-MARQUARDT ALGORITHM
        # minimization
        x0  = np.array([0.2,self.trs.atm.L0])
        res = least_squares(cost,x0,method='lm',ftol=ftol, xtol=xtol, gtol=gtol,max_nfev=max_nfev,verbose=max(verbose,0))
        
        # FIT UNCERTAINTIES
        res.xerr   = confidence_interval(res.fun,res.jac)
        
        # UNPACKING RESULTS
        r0 = res.x[0] # r0 los at the atm wavelength
        L0 = res.x[1]
        
        # coherence time
        _,v0   = FourierUtils.eqLayers(np.array(self.trs.atm.Cn2Weights),np.array(self.trs.atm.wSpeed),1)
        tau0 = 0.314 * r0/v0[0]
        
        # uncertainties
        dr0 = res.xerr[0]
        dL0 = res.xerr[1]
        dtau0 = 0.314*dr0/v0[0]
        
        return r0, L0, tau0, dr0, dL0, dtau0