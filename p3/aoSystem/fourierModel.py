#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:31:39 2020

@author: omartin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:00:44 2018

@author: omartin
"""

import numpy as nnp
from . import gpuEnabled, np, nnp, fft, spc, cpuArray, trapz
import matplotlib as mpl
import matplotlib.pyplot as plt

import time
from distutils.spawn import find_executable

import p3.aoSystem.FourierUtils as FourierUtils
from p3.aoSystem.aoSystem import aoSystem
from p3.aoSystem.atmosphere import atmosphere
from p3.aoSystem.frequencyDomain import frequencyDomain

#%% DISPLAY FEATURES
mpl.rcParams['font.size'] = 16

if find_executable('tex'):
    usetex = True
else:
    usetex = False

plt.rcParams.update({
    "text.usetex": usetex,
    "font.family": "serif",
    "font.serif": ["Palatino","DejaVu Sans"],
})

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000
deg2rad = np.pi/180


def besselj__n(n, z):
    if n<0:
        return -1**(-n) * besselj__n(-n, z)
    if n==0:
        return spc.j0(z)
    elif n==1:
        return spc.j1(z)
    elif n>=2:
        return 2*(n-1)*besselj__n(int(n)-1, z)/z - besselj__n(int(n)-2, z)

class fourierModel:
    """
    Fourier class gathering the PSD calculation for PSF reconstruction and
    fast analytic simulations.
    """

    # CONTRUCTOR
    def __init__(self, path_ini, calcPSF=True, verbose=False, display=True,
                 path_root=None, normalizePSD=False, displayContour=False,
                 getPSDatNGSpositions=False, getErrorBreakDown=False, getFWHM=False,
                 getEnsquaredEnergy=False, getEncircledEnergy=False, fftphasor=False,
                 MV=0, nyquistSampling=False, addOtfPixel=False, freq=None, ao=None,
                 computeFocalAnisoCov=True, TiltFilter=False, doComputations=True,
                 psdExpansion=False, reduce_memory=False):

        tstart = time.time()

        # COLLECTING INPUTS
        self.verbose = verbose
        self.path_ini = path_ini
        self.display = display
        self.displayContour = displayContour
        self.getErrorBreakDown = getErrorBreakDown
        self.get_metrics = getFWHM or getEnsquaredEnergy or getEncircledEnergy
        self.calcPSF = calcPSF
        self.tag = 'TIPTOP'
        self.addOtfPixel = addOtfPixel
        self.nyquistSampling = nyquistSampling
        self.computeFocalAnisoCov = computeFocalAnisoCov
        self.MV = MV
        self.TiltFilterP = TiltFilter
        self.normalizePSD = normalizePSD
        self.fftphasor = fftphasor
        self.getFWHM = getFWHM
        self.getEnsquaredEnergy = getEnsquaredEnergy
        self.getEncircledEnergy = getEncircledEnergy
        self.reduce_memory = reduce_memory

        if freq is not None:
            self.freq = freq

        # DEFINING THE NUMBER OF PSF PARAMETERS
        self.tag = "TIPTOP"
        self.param_labels = ['jitterX', 'jitterY', 'jitterXY',
                             'F', 'dx', 'dy', 'bkg', 'stat']
        self.n_param_atm = 0
        self.n_param_dphi = 0

        # GRAB PARAMETERS
        if ao is None:
            self.ao = aoSystem(path_ini, path_root=path_root,
                               getPSDatNGSpositions=getPSDatNGSpositions,
                               psdExpansion=psdExpansion)
        else:
            self.ao = ao
        self.my_data_map = self.ao.my_data_map

        self.t_initAO = 1000*(time.time() - tstart)

        self.t_init = 0
        self.t_initFreq = 0
        self.t_atmo = 0
        self.t_powerSpectrumDensity = 0
        self.t_reconstructor = 0
        self.t_finalReconstructor = 0
        self.t_tomo = 0
        self.t_opt = 0
        self.t_controller = 0
        self.t_fittingPSD = 0
        self.t_aliasingPSD = 0
        self.t_noisePSD = 0
        self.t_spatioTemporalPSD = 0
        self.t_windShakePSD = 0
        self.t_focalAnisoplanatism = 0
        self.t_errorBreakDown = 0
        self.t_getPsfMetrics = 0
        self.t_displayResults = 0
        self.t_getPSF = 0
        self.t_focalAnisoplanatism = 0
        self.t_mcaoWFsensCone = 0
        self.t_extra = 0
        self.t_extraLo = 0
        self.t_tiltFilter = 0
        self.t_focusFilter = 0
        self.t_errorBreakDown = 0
        self.t_getPsfMetrics = 0
        self.nGs = 0

        if doComputations:
            self.initComputations()


    def initComputations(self):

        tstart = time.time()

        if self.ao.error is False:

            # DEFINING THE FREQUENCY DOMAIN
            self.wvl, self.nwvl = FourierUtils.create_wavelength_vector(self.ao)
            if not hasattr(self, 'freq'):
                self.freq = frequencyDomain(
                    self.ao,
                    nyquistSampling=self.nyquistSampling,
                    computeFocalAnisoCov=self.computeFocalAnisoCov
                )
            self.t_initFreq = 1000*(time.time() - tstart)

            # DEFINING THE GUIDE STAR AND THE STRECHING FACTOR
            if self.ao.lgs:
                self.gs = self.ao.lgs
                self.nGs = self.ao.lgs.nSrc
                if self.gs.height[0]!=0:
                    self.strechFactor = 1.0/(1.0 - self.ao.atm.heights/self.gs.height[0])
                else:
                    self.strechFactor = 1.0
            else:
                self.gs = self.ao.ngs
                self.nGs = self.ao.ngs.nSrc
                self.strechFactor = 1.0

            # DEFINING THE MODELED ATMOSPHERE
            if (self.ao.dms.nRecLayers!=None) and (self.ao.dms.nRecLayers < len(self.ao.atm.weights)):
                weights_mod,heights_mod = FourierUtils.eqLayers(
                    self.ao.atm.weights,
                    self.ao.atm.heights,
                    self.ao.dms.nRecLayers
                )
                if self.ao.dms.nRecLayers == 1:
                    heights_mod = [0.0]
                wSpeed_mod = cpuArray(
                    np.linspace(min(self.ao.atm.wSpeed),
                                max(self.ao.atm.wSpeed),
                                num=self.ao.dms.nRecLayers)
                )
                wDir_mod   = cpuArray(
                    np.linspace(min(self.ao.atm.wDir),
                                max(self.ao.atm.wDir),
                                num=self.ao.dms.nRecLayers)
                )
                if self.ao.lgs:
                    # Recalculate stretch factor for modeled atmosphere
                    self.strechFactor_mod = 1.0/(1.0 - heights_mod/self.gs.height[0])
                else:
                    self.strechFactor_mod = 1.0
            else:
                weights_mod    = self.ao.atm.weights
                heights_mod    = self.ao.atm.heights
                wSpeed_mod     = self.ao.atm.wSpeed
                wDir_mod       = self.ao.atm.wDir
                self.strechFactor_mod = self.strechFactor

            self.atm_mod = atmosphere(
                self.ao.atm.wvl,
                self.ao.atm.r0,
                cpuArray(weights_mod),
                cpuArray(heights_mod),
                cpuArray(wSpeed_mod),
                cpuArray(wDir_mod),
                self.ao.atm.L0
            )

            #updating the atmosphere wavelength !
            self.ao.atm.wvl  = self.freq.wvlRef
            self.atm_mod.wvl = self.freq.wvlRef

            self.t_atmo = 1000*(time.time() - self.t_initFreq/1000 - tstart)

            vv = np.asarray(self.freq.psInMas)
            kc = np.asarray(self.freq.kcInMas)
            if vv.size == 1:
                # Single-wavelength path: vv may be 0-D or length-1 array
                rr = 2.0 * kc / vv
            else:
                # Multi-wavelength case: take the worst requirement
                # for each DM across all wavelengths
                rr = np.max(2.0 * kc[:, None] / vv[None, :], axis=1)
            # FoV check: ensure the worst case across all DMs
            if np.max(rr) > self.freq.nOtf:
                raise ValueError('Error : the PSF field of view is too small to simulate the AO correction area\n')

            # DEFINING THE NOISE AND ATMOSPHERE PSD
            if self.ao.wfs.processing.noiseVar == [None]:
                self.ao.wfs.processing.noiseVar = self.ao.wfs.NoiseVariance(self.ao.atm.r0 ,self.ao.atm.wvl)

            self.Wn   = np.mean(self.ao.wfs.processing.noiseVar)/(2*self.freq.kcMax_)**2
            self.Wphi = self.ao.atm.spectrum(np.sqrt(self.freq.k2AO_))

            # DEFINE THE RECONSTRUCTOR
            self.spatialReconstructor(MV=self.MV)

            # DEFINE THE CONTROLLER
            self.controller(display=self.display)

            #set tilt filter key before computing the PSD
            self.applyTiltFilter = self.TiltFilterP

            # COMPUTE THE PSD
            if self.normalizePSD == True:
                wfe = self.ao.rtc.holoop['wfe']
            else:
                wfe = None
            self.PSD = self.powerSpectrumDensity(wfe=wfe)

            # COMPUTE THE PHASE STRUCTURE FUNCTION
            self.SF  = self.phaseStructureFunction()

            # COMPUTE THE PSF
            if self.calcPSF:
                self.PSF, self.SR = self.point_spread_function(
                    verbose=self.verbose,fftphasor=self.fftphasor,addOtfPixel=self.addOtfPixel
                )

                # GETTING METRICS
                if self.getFWHM == True or self.getEnsquaredEnergy==True or self.getEncircledEnergy==True:
                    self.getPsfMetrics(getEnsquaredEnergy=self.getEnsquaredEnergy,\
                        getEncircledEnergy=self.getEncircledEnergy,getFWHM=self.getFWHM)

                # DISPLAYING THE PSFS
                if self.display:
                    self.displayResults(displayContour=self.displayContour)

            # COMPUTE THE ERROR BREAKDOWN
            if self.getErrorBreakDown:
                self.errorBreakDown(verbose=self.verbose)

        # DEFINING BOUNDS
        self.bounds = self.define_bounds()

        self.t_init = 1000*(time.time()  - tstart)

        # DISPLAYING EXECUTION TIMES
        if self.verbose:
            self.displayExecutionTime()

    def __repr__(self):
        s = '\t\t\t\t________________________ FOURIER MODEL ________________________\n\n'
        s += self.ao.__repr__() + '\n'
        s += self.freq.__repr__() + '\n'
        s +=  '\n'
        return s

#%% BOUNDS FOR PSF-FITTING
    def define_bounds(self):
        """
            Defines the bounds for the PSF model parameters :
                Cn2/r0, jitterX, jitterY, jitterXY, dx, dy, bg, stat
        """

        # Photometry
        bounds_down = [-nnp.inf,-nnp.inf,-nnp.inf]
        bounds_up = [nnp.inf,nnp.inf,nnp.inf]
        # Photometry
        bounds_down += nnp.zeros(self.ao.src.nSrc).tolist()
        bounds_up += (nnp.inf*nnp.ones(self.ao.src.nSrc)).tolist()
        # Astrometry
        bounds_down += (-self.freq.nPix//2 * np.ones(2*self.ao.src.nSrc)).tolist()
        bounds_up += ( self.freq.nPix//2 * np.ones(2*self.ao.src.nSrc)).tolist()
        # Background
        bounds_down += [-nnp.inf]
        bounds_up += [nnp.inf]

        return (bounds_down,bounds_up)

#%% RECONSTRUCTOR DEFINITION
    def spatialReconstructor(self, MV=0):
        """
        Computes the WFS spatial reconstructor and the tomographic reconstructor
        for tomographic AO systems.
        """

        tstart = time.time()
        if self.nGs<2:
            # SINGLE AO SYSTEM
            self.reconstructionFilter(MV=MV)
        else:
            # TOMOGRAPHIC SYSTEM
            self.Wtomo = self.tomographicReconstructor()
            self.Popt = self.optimalProjector()
            self.W = np.matmul(self.Popt, self.Wtomo)
            if self.reduce_memory:
                self.Popt = None
                self.Wtomo = None

            # Computation of the Pbeta^DM matrix
            k = np.sqrt(self.freq.k2AO_)
            h_dm = self.ao.dms.heights
            nDm = len(h_dm)
            nK = self.freq.resAO
            i = complex(0, 1)
            nH = self.ao.atm.nL
            Hs = self.ao.atm.heights * self.strechFactor
            #d = self.freq.pitch[0]
            d_sub = [self.ao.wfs.optics[j].dsub for j in range(self.nGs)]   #sub-aperture size
            clock_rate = [self.ao.wfs.detector[j].clock_rate for j in range(self.nGs)]
            sampTime = 1/self.ao.rtc.holoop['rate']

            self.PbetaDM = []
            for s in range(self.ao.src.nSrc):
                fx = self.ao.src.direction[0, s]*self.freq.kxAO_
                fy = self.ao.src.direction[1, s]*self.freq.kyAO_
                PbetaDM = np.zeros([nK, nK, 1, nDm], dtype=complex)
                for j in range(nDm): #loop on DMs
                    index = k<=self.freq.kc_[j] # note : circular masking
                    PbetaDM[index, 0, j] = np.exp(2*i*np.pi*h_dm[j]*(fx[index] + fy[index]))
                self.PbetaDM.append(PbetaDM)

            # Computation of the Malpha matrix
            wDir_x = nnp.cos(self.ao.atm.wDir*np.pi/180)
            wDir_y = nnp.sin(self.ao.atm.wDir*np.pi/180)
            self.MPalphaL = np.zeros([nK, nK, self.nGs, nH], dtype=complex)
            for h in range(nH):
                freq_t = wDir_x[h]*self.freq.kxAO_ + wDir_y[h]*self.freq.kyAO_
                for g in range(self.nGs):
                    Alpha = [self.gs.direction[0, g],self.gs.direction[1, g]]
                    fx = Alpha[0]*self.freq.kxAO_
                    fy = Alpha[1]*self.freq.kyAO_
                    www = 2*i*np.pi*k * np.sinc(sampTime*clock_rate[g]*self.ao.atm.wSpeed[h]*freq_t)
                    self.MPalphaL[:, :, g, h] = www*np.sinc(d_sub[g]*self.freq.kxAO_)\
                                                   *np.sinc(d_sub[g]*self.freq.kyAO_)\
                                                   *np.exp(i*2*np.pi*Hs[h]*(fx+fy))

            self.Walpha = np.matmul(self.W,self.MPalphaL)
        if self.reduce_memory:
            self.MPalphaL = None
        self.t_finalReconstructor = 1000*(time.time() - tstart)

    def reconstructionFilter(self, MV=0):
        """
        Reconstructs the WFS spatial filters. If MV = 1, uses the Minimum Variance
        reconstructor by accounting for the noise variance.
        """
        tstart = time.time()
        # reconstructor derivation
        i = complex(0,1)
        d = self.ao.wfs.optics[0].dsub

        if self.ao.wfs.optics[0].wfstype.upper()=='SHACK-HARTMANN':
            Sx = 2*i*np.pi*self.freq.kxAO_*d
            Sy = 2*i*np.pi*self.freq.kyAO_*d
            Av = np.sinc(d*self.freq.kxAO_)*np.sinc(d*self.freq.kyAO_)\
                 *np.exp(i*np.pi*d*(self.freq.kxAO_ + self.freq.kyAO_))

        elif self.ao.wfs.optics[0].wfstype.upper()=='PYRAMID':
            # forward pyramid filter (continuous) from Conan
            umod = 1/(2*d)/(self.ao.wfs.optics[0].nL/2)*self.ao.wfs.optics[0].modulation
            Sx = np.zeros((self.freq.resAO,self.freq.resAO), dtype=complex)
            idx = abs(self.freq.kxAO_) > umod
            Sx[idx] = i*np.sign(self.freq.kxAO_[idx])
            idx = abs(self.freq.kxAO_) <= umod
            Sx[idx] = 2*i/np.pi*np.arcsin(self.freq.kxAO_[idx]/umod)
            Av = np.sinc(self.ao.wfs.detector[0].binning*d*self.freq.kxAO_)\
                * np.sinc(self.ao.wfs.detector[0].binning*d*self.freq.kxAO_).T
            Sy = Sx.T
        else:
            raise ValueError("The WFS type is not supported; must be Shack-Hartmann or Pyramid.")
        self.SxAv = Sx*Av
        self.SyAv = Sy*Av

        # Reconstructor
        wvlGs = self.gs.wvl[0]
        Watm = self.ao.atm.spectrum(np.sqrt(self.freq.k2AO_))*(self.ao.atm.wvl/wvlGs)**2
        gPSD = abs(self.SxAv)**2 + abs(self.SyAv)**2 + MV*self.Wn/Watm
        self.Rx = np.conj(self.SxAv)/gPSD
        self.Ry = np.conj(self.SyAv)/gPSD

        # Set central point (i.e. kx=0,ky=0) to zero
        self.Rx[self.freq.resAO//2, self.freq.resAO//2] = 0
        self.Ry[self.freq.resAO//2, self.freq.resAO//2] = 0

        self.t_reconstructor = 1000*(time.time()  - tstart)

    def tomographicReconstructor(self):
        """
        Computes the tomographic reconstructor based on Neichel+09.
        """
        tstart = time.time()
        k = np.sqrt(self.freq.k2AO_)
        nK = self.freq.resAO
        nL = len(self.ao.atm.heights)
        h_mod = self.atm_mod.heights*self.strechFactor_mod
        nL_mod = len(h_mod)
        nGs = self.nGs
        i = complex(0,1)
        d = [self.ao.wfs.optics[j].dsub for j in range(nGs)]   #sub-aperture size

        # WFS operator and projection matrices
        M = np.zeros([nK, nK, nGs, nGs], dtype=complex)
        P = np.zeros([nK, nK, nGs, nL_mod], dtype=complex)
        for j in range(nGs):
            M[:, :, j, j] = 2*i*np.pi*k*np.sinc(d[j]*self.freq.kxAO_)\
                            *np.sinc(d[j]*self.freq.kyAO_)
            for n in range(nL_mod):
                P[:, :, j, n] = np.exp(i*2*np.pi*h_mod[n]\
                                        *(self.freq.kxAO_*self.gs.direction[0, j]
                                        + self.freq.kyAO_*self.gs.direction[1, j]))
        self.M = M
        MP = np.matmul(self.M,P)
        if self.reduce_memory:
            self.M = None
        
        MP_t = np.conj(MP.transpose(0, 1, 3, 2))

        # Noise covariance matrix
        self.Cb = np.ones((nK,nK,nGs,nGs))*np.diag(self.ao.wfs.processing.noiseVar)

        # Atmospheric PSD with the true atmosphere
        self.Cphi = np.zeros([nK,nK,nL,nL],dtype=complex)
        cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
        kernel = self.ao.atm.r0**(-5/3)*cte*(self.freq.k2AO_ + 1/self.ao.atm.L0**2)**(-11/6)*self.freq.pistonFilterAO_
        self.Cphi = kernel.repeat(nL**2, axis=1)
        self.Cphi = self.Cphi.reshape((nK, nK, nL, nL))*np.diag(self.ao.atm.weights)

        # Atmospheric PSD with the modelled atmosphere
        if nL_mod == nL:
            self.Cphi_mod = self.Cphi
        else:
            self.Cphi_mod = kernel.repeat(nL_mod**2, axis=1)
            self.Cphi_mod = self.Cphi_mod.reshape((nK, nK, nL_mod, nL_mod))*np.diag(self.atm_mod.weights)
        to_inv  = np.matmul(np.matmul(MP, self.Cphi_mod), MP_t) + self.Cb

        # Wtomo
        inv = np.linalg.pinv(to_inv.astype(np.complex64),rcond=np.finfo(np.float32).eps)
        Wtomo = np.matmul(np.matmul(self.Cphi_mod, MP_t), inv)
        self.t_tomo = 1000*(time.time() - tstart)

        return Wtomo

    def optimalProjector(self):
        """
        Computes the projector from layers to DM from Neichel+09.
        """
        tstart = time.time()
        k = np.sqrt(self.freq.k2AO_)
        h_dm = self.ao.dms.heights
        nDm = len(h_dm)
        nDir = len(self.ao.dms.opt_dir[0])
        h_mod = self.atm_mod.heights * cpuArray(self.strechFactor_mod)
        nL = len(h_mod)
        nK = self.freq.resAO
        i = complex(0,1)

        mat1 = np.zeros([nK, nK, nDm, nL], dtype=complex)
        to_inv = np.zeros([nK, nK, nDm, nDm], dtype=complex)
        theta_x = self.ao.dms.opt_dir[0]/206264.8 * nnp.cos(self.ao.dms.opt_dir[1]*np.pi/180)
        theta_y = self.ao.dms.opt_dir[0]/206264.8 * nnp.sin(self.ao.dms.opt_dir[1]*np.pi/180)

        for d_o in range(nDir):                 #loop on optimization directions
            Pdm = np.zeros([nK, nK, 1, nDm], dtype=complex)
            Pl = np.zeros([nK, nK, 1, nL], dtype=complex)
            fx = theta_x[d_o]*self.freq.kxAO_
            fy = theta_y[d_o]*self.freq.kyAO_
            for j in range(nDm):                # loop on DM
                index = k <= self.freq.kc_[j] # note : circular masking here
                Pdm[index, 0, j] = np.exp(i*2*np.pi*h_dm[j]*(fx[index]+fy[index]))
            Pdm_t = np.conj(Pdm.transpose(0, 1, 3, 2))
            for l in range(nL):                 #loop on atmosphere layers
                Pl[:, :, 0, l] = np.exp(i*2*np.pi*h_mod[l]*(fx + fy))

            mat1 += np.matmul(Pdm_t, Pl)*self.ao.dms.opt_weights[d_o]
            to_inv += np.matmul(Pdm_t, Pdm)*self.ao.dms.opt_weights[d_o]

        # Popt
        if nDir == 1:
            mat2 = np.linalg.pinv(to_inv.astype(np.complex64),rcond=1/self.ao.dms.opt_cond)
        else:
            # Tikhonov regularization
            #transpose of 3rd and 4th dimensions
            to_inv_t = to_inv.transpose(0,1,3,2)
            lambda_tikhonov = 1/self.ao.dms.opt_cond
            identity_4d = np.eye(nDm)[np.newaxis, np.newaxis, :, :] # shape: (1, 1, nDm, nDm)
            identity_4d = np.broadcast_to(identity_4d, (nK, nK, nDm, nDm)) # shape: (nK, nK, nDm, nDm)
            A = to_inv_t.astype(np.complex64) @ to_inv.astype(np.complex64) \
                + lambda_tikhonov * identity_4d.astype(np.complex64)
            b = to_inv_t.astype(np.complex64)
            mat2 = np.linalg.solve(A, b) # aka W, 5x(2*nstars)

        Popt = np.matmul(mat2, mat1)

        self.t_opt = 1000*(time.time() - tstart)
        return Popt


#%% CONTROLLER DEFINITION
    def controller(self, nTh=1, nF=1000, display=False):
        """
        Define the AO loop controller and compute the spatialized filter thanks
        to the Taylor hypothesis : f = k.v
        """
        tstart  = time.time()

        if self.ao.rtc.holoop['gain']:

            i = complex(0,1)
            vx = self.ao.atm.wSpeed*nnp.cos(self.ao.atm.wDir*np.pi/180)
            vy = self.ao.atm.wSpeed*nnp.sin(self.ao.atm.wDir*np.pi/180)
            nPts = self.freq.resAO
            thetaWind = np.linspace(0, 2*np.pi-2*np.pi/nTh, nTh)
            costh = np.cos(thetaWind)
            weights = self.ao.atm.weights
            Ts = 1.0/self.ao.rtc.holoop['rate']#samplingTime
            delay = self.ao.rtc.holoop['delay']#latency
            loopGain = self.ao.rtc.holoop['gain']

            if Ts <= 0:
                raise ValueError('Error : the AO loop rate must be positive\n')

            # Instantiation
            h1 = np.zeros((nPts,nPts),dtype=complex)
            h2 = np.zeros((nPts,nPts))
            hn = np.zeros((nPts,nPts))

            # Get the noise propagation factor
            # Start from a small positive frequency to avoid f=0
            f = np.logspace(-3, np.log10(0.5/Ts), nF)
            z = np.exp(-2*i*np.pi*f*Ts)

            # Compute transfer functions with safe division
            # Add small epsilon to denominator to avoid division by zero
            eps = np.finfo(float).eps
            denom = 1.0 - z**(-1.0)
            # Set small values to eps to avoid division by zero
            denom = np.where(np.abs(denom) < eps, eps, denom)

            self.hInt = loopGain / denom

            denom2 = 1.0 + self.hInt * z**(-delay)
            denom2 = np.where(np.abs(denom2) < eps, eps, denom2)
            self.rtfInt = 1.0 / denom2

            self.atfInt = self.hInt * z**(-delay) * self.rtfInt

            if loopGain == 0:
                self.ntfInt = 1
            else:
                self.ntfInt = self.atfInt/z

            self.noiseGain = trapz(abs(self.ntfInt)**2, f)*2*Ts

            # Get transfer functions
            for l in range(self.ao.atm.nL):
                h1buf = np.zeros((nPts, nPts, nTh), dtype=complex)
                h2buf = np.zeros((nPts, nPts, nTh))
                hnbuf = np.zeros((nPts, nPts, nTh))
                for iTheta in range(nTh):
                    fi = -vx[l]*self.freq.kxAO_*costh[iTheta] - vy[l]*self.freq.kyAO_*costh[iTheta]
                    z  = np.exp(-2*i*np.pi*fi*Ts)

                    # Safe division for spatially varying transfer functions
                    denom = 1.0 - z**(-1.0)
                    denom = np.where(np.abs(denom) < eps, eps, denom)
                    hInt = loopGain / denom

                    denom2 = 1.0 + hInt * z**(-delay)
                    denom2 = np.where(np.abs(denom2) < eps, eps, denom2)
                    rtfInt = 1.0 / denom2

                    atfInt = hInt * z**(-delay) * rtfInt

                    # AO transfer function
                    h2buf[:, :, iTheta] = abs(atfInt)**2
                    h1buf[:, :, iTheta] = atfInt
                    # noise transfer function
                    if loopGain == 0:
                        ntfInt = 1
                    else:
                        ntfInt = atfInt/z
                    hnbuf[:, :, iTheta] = abs(ntfInt)**2

                h1 += weights[l]*np.sum(h1buf, axis=2)/nTh
                h2 += weights[l]*np.sum(h2buf, axis=2)/nTh
                hn += weights[l]*np.sum(hnbuf, axis=2)/nTh

            self.h1 = h1
            self.h2 = h2
            self.hn = hn

            if display:
                plt.figure()
                plt.semilogx(f, 10*np.log10(abs(self.rtfInt)**2), label='Rejection transfer function')
                plt.semilogx(f, 10*np.log10(abs(self.ntfInt)**2), label='Noise transfer function')
                plt.semilogx(f, 10*np.log10(abs(self.atfInt)**2), label='Aliasing transfer function')
                plt.xlabel('Temporal frequency (Hz)')
                plt.ylabel('Magnitude (dB)')
                plt.grid('on')
                plt.legend()

        self.t_controller = 1000*(time.time() - tstart)

#%% PSD DEFINTIONS
    def powerSpectrumDensity(self,wfe=None):
        """ Total power spectrum density in nm^2.m^2
        """
        tstart  = time.time()

        dk     = 2*self.freq.kcMax_/self.freq.resAO
        rad2nm = self.freq.wvlRef*1e9/2/np.pi

        if self.ao.rtc.holoop['gain']==0:
            # OPEN-LOOP
            k = np.sqrt(self.freq.k2_)
            pf = FourierUtils.pistonFilter(self.ao.tel.D, k)
            psd = self.ao.atm.spectrum(k) * pf
            psd = psd[:, :, np.newaxis]
        else:
            # CLOSED-LOOP
            psd = np.zeros((self.freq.nOtf,self.freq.nOtf,self.ao.src.nSrc))

            # AO correction area
            id1 = np.ceil(self.freq.nOtf/2 - self.freq.resAO/2).astype(int)
            id2 = np.ceil(self.freq.nOtf/2 + self.freq.resAO/2).astype(int)
            # Noise
            self.psdNoise = np.real(self.noisePSD())
            if self.nGs == 1:
                psd[id1:id2,id1:id2,:] = np.repeat(self.psdNoise[:, :, np.newaxis], self.ao.src.nSrc, axis=2)
            else:
                psd[id1:id2,id1:id2,:] = self.psdNoise

            # --- free memory
            if self.reduce_memory and not self.getErrorBreakDown:
                self.psdNoise = None

            # Aliasing
            self.psdAlias = np.real(self.aliasingPSD())
            psd[id1:id2,id1:id2,:] = psd[id1:id2,id1:id2,:]\
            + np.repeat(self.psdAlias[:, :, np.newaxis], self.ao.src.nSrc, axis=2)

            # --- free memory
            if self.reduce_memory and not self.getErrorBreakDown:
                self.psdAlias = None
                
            # Differential refractive anisoplanatism
            self.psdDiffRef = self.differentialRefractionPSD()
            psd[id1:id2,id1:id2,:] = psd[id1:id2,id1:id2,:] + self.psdDiffRef

            # --- free memory
            if self.reduce_memory and not self.getErrorBreakDown:
                self.psdDiffRef = None

            # Chromatism
            self.psdChromatism = self.chromatismPSD()
            psd[id1:id2,id1:id2,:] = psd[id1:id2,id1:id2,:] + self.psdChromatism

            # --- free memory
            if self.reduce_memory and not self.getErrorBreakDown:
                self.psdChromatism = None

            # Add the noise and spatioTemporal PSD
            self.psdSpatioTemporal = np.real(self.spatioTemporalPSD())
            psd[id1:id2,id1:id2,:] = psd[id1:id2,id1:id2,:] + self.psdSpatioTemporal

            # --- free memory
            if self.reduce_memory and not self.getErrorBreakDown:
                self.psdSpatioTemporal = None

            # Cone effect
            if self.nGs == 1 and self.gs.height[0] != 0:
                if self.verbose:
                    print('SLAO case adding cone effect')
                self.psdCone = self.focalAnisoplanatismPSD()
                psd += np.repeat(self.psdCone[:, :, np.newaxis], self.ao.src.nSrc, axis=2)

                # --- free memory
                if self.reduce_memory and not self.getErrorBreakDown:
                    self.psdCone = None

            # additional error for MCAO system with laser GS:
            # reduced volume for WF sensing due to the cone effect
            if self.ao.addMcaoWFsensConeError and self.nGs != 1 and self.gs.height[0] != 0:
                if self.verbose:
                    print('MCAO and laser case: adding error due to reduced volume for WF sensing')
                self.psdMcaoWFsensCone = self.mcaoWFsensConePSD(psd)
                psd += self.psdMcaoWFsensCone

                # --- free memory
                if self.reduce_memory and not self.getErrorBreakDown:
                    self.psdMcaoWFsensCone = None

            # NORMALIZATION
            if wfe != None:
                psd *= (dk * rad2nm)**2
                psd *= wfe**2/psd.sum()

            # Fitting
            self.psdFit = np.real(self.fittingPSD())
            psd += np.repeat(self.psdFit[:, :, np.newaxis], self.ao.src.nSrc, axis=2)

            # --- free memory
            if self.reduce_memory and not self.getErrorBreakDown:
                self.psdFit = None

            # wind shake / vibrations
            if self.applyTiltFilter is False and self.ao.windPsdFile != 0:
                self.psdVib = self.windShakePSD()
                psd[id1:id2,id1:id2,:] = psd[id1:id2,id1:id2,:] + np.repeat(self.psdVib[:, :, np.newaxis], self.ao.src.nSrc, axis=2)

                # --- free memory
                if self.reduce_memory and not self.getErrorBreakDown:
                    self.psdVib = None

            # Tilt filter
            if self.applyTiltFilter == True:
                tiltFilter = self.TiltFilter()                      
                for i in range(self.ao.src.nSrc):
                    psd[:,:,i] *= tiltFilter

            # Extra error
            if self.verbose:
                print('extra error in nm RMS: ',self.ao.tel.extraErrorNm)
                print('extra error spatial frequency exponent: ',self.ao.tel.extraErrorExp)
                print('extra error in nm RMS (LO): ',self.ao.tel.extraErrorLoNm)
                print('extra error spatial frequency exponent (LO): ',self.ao.tel.extraErrorLoExp)
            if self.ao.tel.extraErrorNm > 0:
                self.psdExtra = np.real(self.extraErrorPSD())
            errLOsum = nnp.sum(self.ao.tel.extraErrorLoNm)
            if self.ao.getPSDatNGSpositions and errLOsum >= 0:
                nLO = len(self.ao.azimuthGsLO)
                self.psdExtraLo = self.extraErrorLoPSD()
            else:
                nLO = 0
            for i in range(self.ao.src.nSrc):
                if nLO > 0 and errLOsum >= 0 and self.ao.src.nSrc-i <= nLO:
                    if isinstance(self.psdExtraLo, list):
                        psd[:,:,i] += self.psdExtraLo[i-self.ao.src.nSrc+nLO]
                    else:
                        psd[:,:,i] += self.psdExtraLo
                elif self.ao.tel.extraErrorNm > 0:
                    psd[:,:,i] += self.psdExtra

            # --- free memory
            if self.reduce_memory and not self.getErrorBreakDown:
                self.psdExtra = None
                self.psdExtraLo = None
                    
        self.t_powerSpectrumDensity = 1000*(time.time() - tstart)

        # Return the 3D PSD array in nm^2
        return psd * (dk * rad2nm)**2

    def fittingPSD(self):
        """ Fitting error power spectrum density """
        tstart  = time.time()
        #Instantiate the function output
        psd = np.zeros((self.freq.nOtf,self.freq.nOtf))
        psd[self.freq.mskOut_]  = self.ao.atm.spectrum(np.sqrt(self.freq.k2_[self.freq.mskOut_]))
        self.t_fittingPSD = 1000*(time.time() - tstart)
        return psd

    def aliasingPSD(self):
        """
        Aliasing error power spectrum density
        TO BE REVIEWED IN THE CASE OF A PYRAMID WFSs
        """

        tstart  = time.time()
        psd = np.zeros((self.freq.resAO,self.freq.resAO))
        i = complex(0,1)
        d = self.ao.wfs.optics[0].dsub
        clock_rate = np.array([self.ao.wfs.detector[j].clock_rate for j in range(self.nGs)])
        T = np.mean(clock_rate/self.ao.rtc.holoop['rate'])
        td = T * self.ao.rtc.holoop['delay']
        vx = self.ao.atm.wSpeed*nnp.cos(self.ao.atm.wDir*np.pi/180)
        vy = self.ao.atm.wSpeed*nnp.sin(self.ao.atm.wDir*np.pi/180)
        weights = self.ao.atm.weights
        w = 2*i*np.pi*d

        if hasattr(self, 'Rx') == False:
            self.reconstructionFilter()
        Rx = self.Rx*w
        Ry = self.Ry*w

        if self.ao.rtc.holoop['gain'] == 0:
            tf = 1
        else:
            tf = self.h1

        '''        
        # old, non vectorized computation, as a refernece
        # loops on frequency shifts
        psd = np.zeros((self.freq.resAO,self.freq.resAO))
        for mi in range(-self.freq.nTimes,self.freq.nTimes):
            for ni in range(-self.freq.nTimes,self.freq.nTimes):
                if (mi!=0) | (ni!=0):
                    km   = self.freq.kxAO_ - mi/d
                    kn   = self.freq.kyAO_ - ni/d
                    print('km', km.shape)
                    PR   = FourierUtils.pistonFilter(self.ao.tel.D,np.hypot(km,kn),fm=mi/d,fn=ni/d)
                    W_mn = (km**2 + kn**2 + 1/self.ao.atm.L0**2)**(-11/6)     
                    Q    = (Rx*km + Ry*kn) * (np.sinc(d*km)*np.sinc(d*kn))
                    avr  = 0
                        
                    for l in range(self.ao.atm.nL):
                        avr +=  weights[l] * (np.sinc(km*vx[l]*T) * np.sinc(kn*vy[l]*T)
                        * np.exp(2*i*np.pi*km*vx[l]*td)*np.exp(2*i*np.pi*kn*vy[l]*td)*tf)
                                                          
                    psd0 += PR*W_mn * abs(Q*avr)**2
        '''
        # Create grid of frequency shifts
        mi, ni = np.meshgrid(
            np.arange(-self.freq.nTimes, self.freq.nTimes),
            np.arange(-self.freq.nTimes, self.freq.nTimes),
            indexing="ij"
        )
        # Mask to exclude (0,0) shift
        mask = (mi != 0) | (ni != 0)
        # Reshape mi and ni for broadcasting
        mi = mi[:, :, None]  # Shape (N, N, 1)
        ni = ni[:, :, None]  # Shape (N, N, 1)
        # Ensure kxAO_ and kyAO_ are 1D
        kxAO = self.freq.kxAO_.ravel()  # Shape (K,)
        kyAO = self.freq.kyAO_.ravel()  # Shape (K,)
        # Broadcast km and kn correctly
        km = kxAO[None, None, :] - mi / d  # Shape (N, N, K)
        kn = kyAO[None, None, :] - ni / d  # Shape (N, N, K)
        NN = Rx.shape
        # Compute PR using vectorized function
        PR = FourierUtils.pistonFilter(self.ao.tel.D, np.hypot(km, kn), fm=mi[:, :, None] / d, fn=ni[:, :, None] / d)
        # Compute W_mn
        W_mn = (km**2 + kn**2 + 1 / self.ao.atm.L0**2) ** (-11 / 6)
        # Ensure Rx and Ry are 1D (flatten if needed)
        Rx = Rx.ravel()  # Shape (K,)
        Ry = Ry.ravel()  # Shape (K,)
        # Ensure Rx and Ry are reshaped correctly for broadcasting
        Rx = Rx[None, None, :]  # Shape (1, 1, K)
        Ry = Ry[None, None, :]  # Shape (1, 1, K)
        # Compute Q factor
        Q = (Rx * km + Ry * kn) * (np.sinc(d * km) * np.sinc(d * kn))
        # Compute avr using broadcasting
        vx_exp = vx.ravel()
        vy_exp = vy.ravel()
        vx_exp = vx_exp[:, None, None, None]  # Shape (nL, 1, 1, 1)
        vy_exp = vy_exp[:, None, None, None]  # Shape (nL, 1, 1, 1)
        vx_exp = np.asarray(vx_exp)
        vy_exp = np.asarray(vy_exp)
        f1 = km * vx_exp * T
        f2 = km * vx_exp * T
        f3 = 2j * np.pi * km * vx_exp * td
        f4 = 2j * np.pi * kn * vy_exp * td
        # Ensure tf is correctly shaped for broadcasting
        tf = np.asarray(tf.ravel()[None, None, None, :]) # Shape (1, 1, K)
        avr = np.sum( np.asarray(weights[:, None, None, None]) * (np.sinc(f1) * np.sinc(f2) * np.exp(f3) * np.exp(f4) * tf), axis=0 )
        # Compute final PSD
        psd = np.sum(PR * W_mn * np.abs(Q * avr) ** 2 * mask[:, :, None], axis=(0, 1))
        psd = np.reshape(psd, NN)
#        test agaist old, non-vectorize computation
#        print("Is PSD the same as old computation?", np.allclose(psd0, psd2,  rtol=1e-04, atol=1e-04))
        self.t_aliasingPSD = 1000*(time.time() - tstart)
        return self.freq.mskInAO_ * psd * self.ao.atm.r0**(-5/3)*0.0229

    def noisePSD(self):
        """Noise error power spectrum density
        """
        tstart = time.time()
        psd = np.zeros((self.freq.resAO,self.freq.resAO))
        if self.ao.wfs.processing.noiseVar[0] > 0:
            if self.nGs < 2:
                psd = abs(self.Rx**2 + self.Ry**2)
                psd = psd/(2*self.freq.kcMax_)**2
                psd = self.freq.mskInAO_ * psd * self.freq.pistonFilterAO_
            else:
                psd = np.zeros((self.freq.resAO,self.freq.resAO,self.ao.src.nSrc),dtype=complex)
                #where is the noise level ?
                for j in range(self.ao.src.nSrc):
                    PW = np.matmul(self.PbetaDM[j],self.W)
                    PW_t = np.conj(PW.transpose(0,1,3,2))
                    tmp = np.matmul(PW,np.matmul(self.Cb,PW_t))
                    psd[:,:,j] = self.freq.mskInAO_ * tmp[:, :, 0, 0]*self.freq.pistonFilterAO_

        self.t_noisePSD = 1000*(time.time() - tstart)
        # NOTE: the noise variance is the same for all WFS
        return  psd*self.noiseGain * np.mean(self.ao.wfs.processing.noiseVar)

    def reconstructionPSD(self):
        """ Power spectrum density of the wavefront reconstruction error
        """
        tstart = time.time()
        psd = np.zeros((self.freq.resAO,self.freq.resAO))
        if not hasattr(self, 'Rx'):
            self.reconstructionFilter()

        F = self.Rx*self.SxAv + self.Ry*self.SyAv
        psd = abs(1-F)**2 * self.freq.mskInAO_ * self.Wphi * self.freq.pistonFilterAO_

        self.t_recPSD = 1000*(time.time() - tstart)
        return  psd

    def servoLagPSD(self):
        """ Servo-lag power spectrum density.
        Note : sometimes the sum becomes negative, a further analysis is needed
        """
        tstart = time.time()
        psd = np.zeros((self.freq.resAO,self.freq.resAO))
        if hasattr(self, 'Rx') == False:
            self.reconstructionFilter()

        F = self.Rx*self.SxAv + self.Ry*self.SyAv
        Watm = self.Wphi * self.freq.pistonFilterAO_
        if (self.ao.rtc.holoop['gain'] == 0):
            psd = abs(1-F)**2 * Watm
        else:
            psd = (1.0 + abs(F)**2*self.h2 - 2*np.real(F*self.h1))*Watm

        self.t_servoLagPSD = 1000*(time.time() - tstart)
        return self.freq.mskInAO_ * abs(psd)

    def windShakePSD(self):
        """ wind shake / vibrations power spectrum density.
        """
        tstart  = time.time()    
        psd = np.zeros((self.freq.resAO,self.freq.resAO))

        Wtilt1 = 1-self.TiltFilter()
        # AO correction area
        id1 = np.ceil(self.freq.nOtf/2 - self.freq.resAO/2).astype(int)
        id2 = np.ceil(self.freq.nOtf/2 + self.freq.resAO/2).astype(int)
        Wtilt1 = Wtilt1[id1:id2,id1:id2] * self.freq.pistonFilterAO_
        Wtilt1 *= 1/np.sum(Wtilt1)

        # wind-shake PSD
        from astropy.io import fits
        hdul = fits.open(self.ao.windPsdFile)
        psd_data = np.asarray(hdul[0].data, np.float32)
        hdul.close()
        psd_freq = np.asarray(np.linspace(0.1, 0.5*self.ao.rtc.holoop['rate'], int(5*self.ao.rtc.holoop['rate'])))
        psd_tip_wind = np.interp(psd_freq, psd_data[0,:], psd_data[1,:],left=0,right=0)
        psd_tilt_wind = np.interp(psd_freq, psd_data[0,:], psd_data[2,:],left=0,right=0)

        #rejection transfer function
        ic      = complex(0,1)
        z       = np.exp(-2*ic*np.pi/self.ao.rtc.holoop['rate']*psd_freq)
        hInt    = self.ao.rtc.holoop['gain']/(1.0 - z**(-1.0))
        rtfInt  = 1.0/(1.0 + hInt * z**(-self.ao.rtc.holoop['delay']))

        #fig, _ = plt.subplots()
        #plt.loglog(psd_freq,np.abs(rtfInt))
        #fig, _ = plt.subplots()
        #plt.loglog(psd_freq,psd_tip_wind)
        #plt.loglog(psd_freq,np.abs(rtfInt**2*psd_tip_wind))

        power = np.abs(np.sum(rtfInt**2*(psd_tip_wind+psd_tilt_wind))*(psd_freq[1]-psd_freq[0]))       
        rad2nm = (2*self.freq.kcMax_/self.freq.resAO) * self.freq.wvlRef*1e9/2/np.pi
        power *= 1/rad2nm**2

        psd[:,:] = power*Wtilt1

        self.t_windShakePSD = 1000*(time.time() - tstart)
        return self.freq.mskInAO_ * abs(psd)

    def spatioTemporalPSD(self):
        """%% Power spectrum density including reconstruction, field variations and temporal effects
        """
        tstart = time.time()
        nK = self.freq.resAO
        psd = np.zeros((nK,nK,self.ao.src.nSrc),dtype=complex)
        i = complex(0,1)
        nH = self.ao.atm.nL
        Hs = self.ao.atm.heights * self.strechFactor
        Ws = self.ao.atm.weights
        deltaT = self.ao.rtc.holoop['delay']/self.ao.rtc.holoop['rate']
        wDir_x = nnp.cos(self.ao.atm.wDir*np.pi/180)
        wDir_y = nnp.sin(self.ao.atm.wDir*np.pi/180)
        Watm = self.Wphi * self.freq.pistonFilterAO_
        F = self.Rx*self.SxAv + self.Ry*self.SyAv

        for s in range(self.ao.src.nSrc):
            if self.nGs<2:
                th = self.ao.src.direction[:, s] - self.gs.direction[:, 0]
                if np.any(np.asarray(th)):
                    A = np.zeros((nK, nK))
                    for l in range(self.ao.atm.nL):
                        A = A + Ws[l]*np.exp(2*i*np.pi*Hs[l]*(self.freq.kxAO_*th[1] + self.freq.kyAO_*th[0]))
                else:
                    A = np.ones((self.freq.resAO, self.freq.resAO))

                if (self.ao.rtc.holoop['gain'] == 0):
                    psd[:, :, s] = abs(1-F)**2 * Watm
                else:
                    psd[:, :, s] = self.freq.mskInAO_ * (1 + abs(F)**2*self.h2 - 2*np.real(F*self.h1*A))*Watm
            else:
                # tomographic case
                Beta = [self.ao.src.direction[0,s],self.ao.src.direction[1,s]]
                PbetaL = np.zeros([nK, nK, 1, nH], dtype=complex)
                fx = Beta[0]*self.freq.kxAO_
                fy = Beta[1]*self.freq.kyAO_
                for j in range(nH):
                    freq_t = wDir_x[j]*self.freq.kxAO_+ wDir_y[j]*self.freq.kyAO_
                    delta_h = Hs[j]*(fx+fy) - deltaT*self.ao.atm.wSpeed[j]*freq_t
                    PbetaL[: , :, 0, j] = np.exp(i*2*np.pi*delta_h)

                proj = PbetaL - np.matmul(self.PbetaDM[s], self.Walpha)
                proj_t = np.conj(proj.transpose(0, 1, 3, 2))
                tmp = np.matmul(proj,np.matmul(self.Cphi, proj_t))
                psd[:, :, s] = self.freq.mskInAO_ * tmp[:, :, 0, 0]*self.freq.pistonFilterAO_
        if self.reduce_memory:
            self.Walpha = None
        self.t_spatioTemporalPSD = 1000*(time.time() - tstart)
        return psd

    def anisoplanatismPSD(self):
        """%% Anisoplanatism power spectrum density
        """
        tstart  = time.time()
        psd = np.zeros((self.freq.resAO,self.freq.resAO,self.ao.src.nSrc))
        Hs = self.ao.atm.heights * self.strechFactor
        Ws = self.ao.atm.weights
        Watm = self.Wphi * self.freq.pistonFilterAO_

        for s in range(self.ao.src.nSrc):
            th  = self.ao.src.direction[:,s] - self.gs.direction[:,0]
            if any(th):
                A = np.zeros((self.freq.resAO,self.freq.resAO))
                for l in range(self.ao.atm.nL):
                    A   = A + 2*Ws[l]*(1 - np.cos(2*np.pi*Hs[l]*(self.freq.kxAO_*th[1] + self.freq.kyAO_*th[0])))
                psd[:,:,s] = self.freq.mskInAO_ * A*Watm
        self.t_anisoplanatismPSD = 1000*(time.time() - tstart)
        return np.real(psd)
    
    def differentialRefractionPSD(self):
        def refractionIndex(wvl,nargout=1):
            ''' Refraction index -1 as a fonction of the wavelength.
            Valid for lambda between 0.2 and 4µm with 1 atm of pressure and 15 degrees Celsius
                Inputs : wavelength in meters
                Outputs : n-1 and dn/dwvl
            '''
            c1 = 64.328
            c2 = 29498.1
            c3 = 146.0
            c4 = 255.4
            c5 = 41.0
            wvlRef = wvl*1e6

            nm1 = 1e-6 * (c1 +  c2/(c3-1.0/wvlRef**2) + c4/(c5 - 1.0/wvlRef**2) )
            dndw= -2e-6 * (c1 +  c2/(c3-1.0/wvlRef**2)**2 + c4/(c5 - 1.0/wvlRef**2)**2 )/wvlRef**3
            if nargout == 1:
                return nm1
            else:
                return (nm1,dndw)

        def refractiveAnisoplanatism(zenithAngle,wvl):
            ''' Calculate the angular shift due to the atmospheric refraction at wvl
            and for a zenith angle zenithAngle in rad
            '''
            return refractionIndex(wvl) * np.tan(zenithAngle)

        def differentialRefractiveAnisoplanatism(zenithAngle,wvlGs,wvlSrc):
            return (refractionIndex(wvlSrc) - refractionIndex(wvlGs)) * np.tan(zenithAngle)

        tstart  = time.time()

        psd= np.zeros((self.freq.resAO,self.freq.resAO,self.ao.src.nSrc))
        if self.ao.tel.zenith_angle != 0:
            Hs   = self.ao.atm.heights * self.strechFactor
            Ws   = self.ao.atm.weights
            Watm = self.Wphi * self.freq.pistonFilterAO_
            k    = np.sqrt(self.freq.k2AO_)
            arg_k= np.arctan2(self.freq.kyAO_,self.freq.kxAO_)
            azimuth = self.ao.src.azimuth

            theta = differentialRefractiveAnisoplanatism(self.ao.tel.zenith_angle*np.pi/180,self.gs.wvl[0], self.freq.wvlRef)
            for s in range(self.ao.src.nSrc):
                A    = 0
                for l in range(self.ao.atm.nL):
                    A   = A + 2*Ws[l]*(1 - np.cos(2*np.pi*Hs[l]*k*np.tan(theta)*np.cos(arg_k-azimuth[s])))
                psd[:,:,s] = self.freq.mskInAO_ *A*Watm

        self.t_differentialRefractionPSD = 1000*(time.time() - tstart)
        return  psd

    def chromatismPSD(self):
        """ PSD of the chromatic effects"""
        tstart  = time.time()
        Watm = self.Wphi * self.freq.pistonFilterAO_
        psd= np.zeros((self.freq.resAO,self.freq.resAO,self.ao.src.nSrc))
        n2 =  23.7+6839.4/(130-(self.gs.wvl[0]*1.e6)**(-2))+45.47/(38.9-(self.gs.wvl[0]*1.e6)**(-2))
        for s in range(self.ao.src.nSrc):
            n1 =  23.7+6839.4/(130-(self.freq.wvlRef*1.e6)**(-2))+45.47/(38.9-(self.freq.wvlRef*1.e6)**(-2))
            psd[:,:,s] = ((n2-n1)/n2)**2 * Watm

        self.t_chromatismPSD = 1000*(time.time() - tstart)
        return psd

    def phaseStructureFunction(self):
       '''
           GET THE AO RESIDUAL PHASE STRUCTURE FUNCTION
       '''
       cov = fft.fftshift(fft.fftn(fft.fftshift(self.PSD,axes=(0,1)),axes=(0,1)),axes=(0,1))
       return 2*np.real(cov.max(axis=(0,1)) - cov)


    def focalAnisoplanatismPSD(self):
        """%% Focal Anisoplanatism power spectrum density
        """
        tstart  = time.time()

        #Instantiate the function output
        psd      = np.zeros((self.freq.nOtf,self.freq.nOtf))
        # atmo PSD
        psd_atmo = self.ao.atm.spectrum(np.sqrt(self.freq.k2_))

        nPoints = 1001
        nPhase = 5 # number of phase shift cases
        x = self.ao.tel.D*np.linspace(-0.5, 0.5, nPoints, endpoint=1)
        h = self.ao.atm.heights
        h_laser = self.gs.height[0]
        ratio = np.array((h_laser-h)/h_laser)
        nCn2 = len(h)
        freqs = self.freq.kx_[int(np.ceil(self.freq.nOtf/2)-1):,0]
        if freqs[0] < 0:
            freqs = freqs[1:]

        # We create grids to avoid explicit loops
        freqs_matrix = freqs[:, np.newaxis]  # len(freqs) x 1 (for broadcasting)
        ratio_matrix = ratio[np.newaxis, :]  # 1 x nCn2       (for broadcasting)

        x4D = x[np.newaxis, np.newaxis, :, np.newaxis]
        freqs_4Dmatrix = (freqs_matrix * (np.ones(nCn2))[np.newaxis, :])[:, :, np.newaxis, np.newaxis ]
        freqs_ratio_matrix = freqs_matrix * ratio_matrix
        freqs_ratio_4Dmatrix = freqs_ratio_matrix[:, :, np.newaxis, np.newaxis]

        # We vector-initialise sin_ref and sin_temp over all combinations of i, j and k
        k_values = np.arange(nPhase)
        phase_4Dmatrix = (2 * np.pi * k_values / nPhase)[np.newaxis, np.newaxis, np.newaxis, :]

        # Calculation of sinusoids, their std dev and differences for each phase
        sin_ref = np.sin(2 * np.pi * freqs_4Dmatrix * x4D + phase_4Dmatrix)
        sin_temp = np.sin(2 * np.pi * freqs_ratio_4Dmatrix * x4D + phase_4Dmatrix)

        sin_ref_std = np.std(sin_ref, axis=2)
        sin_temp_std = np.std(sin_temp, axis=2)
        std_ratio = sin_ref_std/sin_temp_std
        sin_res = sin_ref - std_ratio[:,:,np.newaxis,:] * sin_temp

        # We calculate the coefficients using the average on the phase
        coeff = np.mean(np.std(sin_res, axis=2) / sin_ref_std, axis=2)

        # Now we calculate where the conditions are not satisfied and we put the coefficients to 0
        condition1 = freqs_matrix * ratio_matrix > self.freq.kc_
        condition2 = freqs_matrix < 1e-5
        non_valid_mask = (condition1 | condition2)
        coeff[non_valid_mask] = 0

        # We calculate 2D coefficients and PSD
        coeff_tot = []
        for j in range(nCn2):
            coeff_tot = np.interp(np.sqrt(self.freq.k2_), freqs, coeff[:,j])**2

            #fig, ax1 = plt.subplots(1,1)
            #im = ax1.plot(coeff[:,j])
            #fig, ax2 = plt.subplots(1,1)
            #im = ax2.imshow(coeff_tot, cmap='hot')
            #ax2.set_title('cone effect filter coefficients', color='black')

            psd += coeff_tot*psd_atmo*self.ao.atm.weights[j]

        self.t_focalAnisoplanatism = 1000*(time.time() - tstart)

        return np.real(psd)

    def mcaoWFsensConePSD(self, psdRes):
        """%% power spectrum density related to the reduced volume sensed
            by the LGS WFS due to cone effect in MCAO systems.
            This effect is related to the cone effect and it depends on
            the LGS geometry and the uncorrected part of the input PSD.
        """
        tstart = time.time()

        # Instantiate the function output
        psd = np.zeros((self.freq.nOtf, self.freq.nOtf, self.ao.src.nSrc))

        # atmo PSD
        psd_atmo = self.ao.atm.spectrum(np.sqrt(self.freq.k2_))

        # AO correction area
        id1 = np.ceil(self.freq.nOtf/2 - self.freq.resAO/2).astype(int)
        id2 = np.ceil(self.freq.nOtf/2 + self.freq.resAO/2).astype(int)

        # geometry
        lfov = 2 * np.max(self.gs.zenith)

        # effective FoV
        eFoV = (lfov * (1/rad2arc) - self.ao.tel.D * (1/self.gs.height[0])) * rad2arc

        # filter considering the maximum cut off frequency
        k = np.sqrt(self.freq.k2_)
        fs = np.max(k) * 2.
        x = k / (fs/2.) * np.pi
        xc = 1j * x
        z = np.exp(xc)

        # piston filter
        pf = FourierUtils.pistonFilter(self.ao.tel.D, np.sqrt(self.freq.k2_))
        pf = pf[id1:id2, id1:id2]

        # Vectorize source-dependent calculations
        src_zenith = np.array(self.ao.src.zenith)  # Shape: (nSrc,)
        max_gs_zenith = np.max(self.gs.zenith)

        # Calculate deltaAngleE and deltaAngleL for all sources at once
        if eFoV > 0:
            deltaAngleE = np.minimum(src_zenith, max_gs_zenith) - eFoV/2
        else:
            deltaAngleE = np.minimum(src_zenith, max_gs_zenith) - eFoV

        deltaAngleL = src_zenith - lfov/2
        deltaAngleL = np.maximum(deltaAngleL, 0)  # Equivalent to: deltaAngleL[deltaAngleL < 0] = 0

        # Pre-compute deltaPsd for all sources
        deltaPsd = psd_atmo[id1:id2, id1:id2, np.newaxis] - psdRes[id1:id2, id1:id2, :]  # Shape: (resAO, resAO, nSrc)
        deltaPsd = np.maximum(deltaPsd, 0)  # Equivalent to: deltaPsd[deltaPsd < 0] = 0
        deltaPsdPf = deltaPsd * pf[:, :, np.newaxis]  # Broadcasting piston filter

        # Vectorize layer calculations
        atm_heights = np.array(self.ao.atm.heights)  # Shape: (nCn2,)
        atm_weights = np.array(self.ao.atm.weights)  # Shape: (nCn2,)

        # Create masks for valid layers and sources
        valid_layers = atm_heights > 0  # Shape: (nCn2,)
        valid_sources = deltaAngleE > 0  # Shape: (nSrc,)

        # Only process valid combinations
        if np.any(valid_layers) and np.any(valid_sources):
            # Get indices of valid layers and sources
            valid_layer_idx = np.where(valid_layers)[0]
            valid_source_idx = np.where(valid_sources)[0]

            # Extract valid data
            valid_heights = atm_heights[valid_layer_idx]  # Shape: (nValidLayers,)
            valid_weights = atm_weights[valid_layer_idx]  # Shape: (nValidLayers,)
            valid_deltaAngleE = deltaAngleE[valid_source_idx]  # Shape: (nValidSources,)
            valid_deltaAngleL = deltaAngleL[valid_source_idx]  # Shape: (nValidSources,)

            # Broadcast for vectorized computation
            # Create grids: (nValidLayers, nValidSources)
            heights_grid, deltaAngleE_grid = np.meshgrid(valid_heights, valid_deltaAngleE, indexing='ij')
            weights_grid, deltaAngleL_grid = np.meshgrid(valid_weights, valid_deltaAngleL, indexing='ij')

            # Calculate fCut and eqD for all valid combinations
            fCut = rad2arc / (deltaAngleE_grid * heights_grid)  # Shape: (nValidLayers, nValidSources)
            eqD = self.ao.tel.D - deltaAngleL_grid * heights_grid * (1/rad2arc)  # Shape: (nValidLayers, nValidSources)
            eqD = np.minimum(eqD, self.ao.tel.D)

            # Create mask for valid frequency cuts
            valid_fcut_mask = (fCut < self.freq.kcMax_) & (eqD > 0)  # Shape: (nValidLayers, nValidSources)

            if np.any(valid_fcut_mask):
                # Extract z values for the AO correction area only
                z_ao = z[id1:id2, id1:id2]  # Shape: (resAO, resAO)

                # Process only valid combinations
                for i, layer_idx in enumerate(valid_layer_idx):
                    for j, source_idx in enumerate(valid_source_idx):
                        if valid_fcut_mask[i, j]:
                            fCut_val = fCut[i, j]
                            eqD_val = eqD[i, j]
                            weight_val = weights_grid[i, j]

                            sPole = 2 * np.pi * fCut_val
                            zPole = np.exp(sPole/fs)
                            lpFilter = z_ao * (1 - zPole) / (z_ao - zPole)
                            lpFilter2 = (1 - np.abs(lpFilter)**2) * (eqD_val/self.ao.tel.D)**2
                            lpFilter2 = np.maximum(lpFilter2, 0)  # Equivalent to: lpFilter2[lpFilter2 < 0] = 0

                            psd[id1:id2, id1:id2, source_idx] += weight_val * lpFilter2 * deltaPsdPf[:, :, source_idx]

        self.t_mcaoWFsensCone = 1000 * (time.time() - tstart)

        return np.real(psd)

    def extraErrorPSD(self):
        """%% extra error
        """

        tstart  = time.time()

        k   = np.sqrt(self.freq.k2_)
        psd = k**self.ao.tel.extraErrorExp
        pf  = FourierUtils.pistonFilter(self.ao.tel.D,k)
        psd = psd * pf
        if self.ao.tel.extraErrorMin>0:
            psd[np.where(k<self.ao.tel.extraErrorMin)] = 0
        if self.ao.tel.extraErrorMax>0:
            psd[np.where(k>self.ao.tel.extraErrorMax)] = 0

        psd = psd * self.ao.tel.extraErrorNm**2/np.sum(psd)

        #fig, ax1 = plt.subplots(1,1)
        #im = ax1.imshow(np.log(np.abs(psd)), cmap='hot')
        #ax1.set_title('extra error PSD', color='black')

        # Derives wavefront error
        rad2nm = (2*self.freq.kcMax_/self.freq.resAO) * self.freq.wvlRef*1e9/2/np.pi

        psd = psd * 1/rad2nm**2

        self.t_extra = 1000*(time.time() - tstart)

        return np.real(psd)

    def extraErrorLoPSD(self):
        """%% extra error for LO
        """
        tstart = time.time()

        k = np.sqrt(self.freq.k2_)
        pf = FourierUtils.pistonFilter(self.ao.tel.D, k)
        rad2nm = (2 * self.freq.kcMax_ / self.freq.resAO) * self.freq.wvlRef * 1e9 / (2 * np.pi)

        psd = k**self.ao.tel.extraErrorLoExp * pf
        if self.ao.tel.extraErrorLoMin > 0:
            psd[k < self.ao.tel.extraErrorLoMin] = 0
        if self.ao.tel.extraErrorLoMax > 0:
            psd[k > self.ao.tel.extraErrorLoMax] = 0
        psd *= 1/np.sum(psd)

        # Check if extraErrorLoExp is a list
        if isinstance(self.ao.tel.extraErrorLoNm, list):
            psd_list = []
            for ii in range(len(self.ao.zenithGsLO)):
                x = nnp.array([0,self.ao.TechnicalFoV/2])
                sqrtpower = nnp.interp(self.ao.zenithGsLO[ii], x, nnp.array(self.ao.tel.extraErrorLoNm))

                psdI = psd*sqrtpower**2

                # Derives wavefront error in rad
                psdI *= 1 / rad2nm**2

                psd_list.append(np.real(psdI))

            result = psd_list  # return a list of 2d arrays

        else:
            psd *= self.ao.tel.extraErrorLoNm**2

            #fig, ax1 = plt.subplots(1,1)
            #im = ax1.imshow(np.log(np.abs(psd)), cmap='hot')
            #ax1.set_title('extra error PSD', color='black')

            # Derives wavefront error in rad
            psd *= 1 / rad2nm**2

            result = np.real(psd)  # a single 2d array

        self.t_extraLo = 1000 * (time.time() - tstart)

        return result

    def TiltFilter(self):
        """%% Spatial filter to remove tilt related errors
        """

        tstart  = time.time()

        # from Sasiela 93
        x = 0.5*self.ao.tel.D*2*np.pi*np.sqrt(self.freq.k2_)
        coeff_tot = 1 - (2*spc.j1(x)/(x))**2 - (4*besselj__n(2,x)/(x))**2

        #fig, ax1 = plt.subplots(1,1)
        #from matplotlib import colors
        #im = ax1.imshow(cpuArray(coeff_tot), cmap='hot', norm=colors.LogNorm())
        #ax1.set_title('tilt filter coefficients', color='black')

        self.t_tiltFilter = 1000*(time.time() - tstart)
 
        return np.real(coeff_tot)

    def FocusFilter(self):
        """%% Spatial filter to remove focus related errors
        """

        tstart  = time.time()

        # from Sasiela 93
        x = 0.5*self.ao.tel.D*2*np.pi*np.sqrt(self.freq.k2_)
        coeff_tot = 1 - 3*(2*besselj__n(3,x)/(x))**2

        #fig, ax1 = plt.subplots(1,1)
        #from matplotlib import colors
        #im = ax1.imshow(cpuArray(coeff_tot), cmap='hot', norm=colors.LogNorm())
        #ax1.set_title('focus filter coefficients', color='black')

        self.t_focusFilter = 1000*(time.time() - tstart)

        return np.real(coeff_tot)


#%% AO ERROR BREAKDOWN
    def errorBreakDown(self,verbose=True):
        """ AO error breakdown from the PSD integrals
        """
        tstart  = time.time()

        if self.ao.rtc.holoop['gain'] != 0:
            # Derives wavefront error
            rad2nm      = (2*self.freq.kcMax_/self.freq.resAO) * self.freq.wvlRef*1e9/2/np.pi

            if not self.ao.tel.opdMap_on is None:
                self.wfeNCPA= np.std(self.ao.tel.opdMap_on[self.ao.tel.pupil!=0])
            else:
                self.wfeNCPA = 0.0

            self.wfeFit    = np.sqrt(self.psdFit.sum()) * rad2nm
            self.wfeAl     = np.sqrt(self.psdAlias.sum()) * rad2nm
            self.wfeN      = np.sqrt(self.psdNoise.sum(axis=(0,1)))* rad2nm
            self.wfeST     = np.sqrt(self.psdSpatioTemporal.sum(axis=(0,1)))* rad2nm
            self.wfeDiffRef= np.sqrt(self.psdDiffRef.sum(axis=(0,1)))* rad2nm
            self.wfeChrom  = np.sqrt(self.psdChromatism.sum(axis=(0,1)))* rad2nm
            self.wfeJitter = 1e9*self.ao.tel.D*nnp.mean(self.ao.cam.spotFWHM[0][0:2])/rad2mas/4
            if self.ao.addMcaoWFsensConeError:
                self.wfeMcaoCone = np.sqrt(self.psdMcaoWFsensCone[:,:,0].sum())* rad2nm
            else:
                self.wfeMcaoCone = 0
            if self.applyTiltFilter is False and self.ao.windPsdFile != 0:
                self.wfeWindShake = np.sqrt(self.psdVib.sum())* rad2nm
            else:
                self.wfeWindShake = 0
            self.wfeExtra  = self.ao.tel.extraErrorNm

            if self.reduce_memory:
                self.psdAlias = None
                self.psdFit = None
                self.psdNoise = None
                self.psdSpatioTemporal = None
                self.psdDiffRef = None
                self.psdChromatism = None
                self.psdMcaoWFsensCone = None
                self.psdVib = None
                self.psdExtra = None
                self.psdExtraLo = None

            # Total wavefront error
            self.wfeTot = np.sqrt(self.wfeNCPA**2 + self.wfeFit**2 + self.wfeAl**2\
                                  + self.wfeST**2 + self.wfeN**2 + self.wfeDiffRef**2\
                                  + self.wfeChrom**2 + self.wfeJitter**2 + self.wfeMcaoCone**2\
                                  + self.wfeWindShake**2 + self.wfeExtra**2)

            # Maréchal appoximation to get the Strehl-ratio
            self.SRmar  = 100*np.exp(-(self.wfeTot*2*np.pi*1e-9/self.freq.wvlRef)**2)

            # bonus
            self.psdS = self.servoLagPSD()
            self.wfeS = np.sqrt(self.psdS.sum()) * rad2nm
            self.wfeR = np.sqrt(max(0,self.reconstructionPSD().sum()))* rad2nm
            if self.nGs == 1:
                self.psdAni = self.anisoplanatismPSD()
                self.wfeAni = np.sqrt(self.psdAni.sum(axis=(0,1))) * rad2nm
            else:
                self.wfeTomo = np.sqrt(self.wfeST**2 - self.wfeS**2)
                
            if self.reduce_memory:
                self.psdS = None
                self.psdAni = None

            # Print
            if verbose == True:
                print('\n_____ ERROR BREAKDOWN  ON-AXIS_____')
                print('------------------------------------------')
                idCenter = self.ao.src.zenith.argmin()
                if hasattr(self,'SR'):
                    print('.Image Strehl at %4.2fmicron:\t%4.2f%s'%(self.freq.wvlRef*1e6,self.SR[idCenter,0],'%'))
                print('.Maréchal Strehl at %4.2fmicron:\t%4.2f%s'%(self.ao.atm.wvl*1e6,self.SRmar[idCenter],'%'))
                print('.Residual wavefront error:\t%4.2fnm'%self.wfeTot[idCenter])
                print('.NCPA residual:\t\t\t%4.2fnm'%self.wfeNCPA)
                print('.Fitting error:\t\t\t%4.2fnm'%self.wfeFit)
                print('.Differential refraction:\t%4.2fnm'%self.wfeDiffRef[idCenter])
                print('.Chromatic error:\t\t%4.2fnm'%self.wfeChrom[idCenter])
                print('.Aliasing error:\t\t%4.2fnm'%self.wfeAl)
                if self.nGs == 1:
                    print('.Noise error:\t\t\t%4.2fnm'%self.wfeN)
                else:
                    print('.Noise error:\t\t\t%4.2fnm'%self.wfeN[idCenter])
                print('.Spatio-temporal error:\t\t%4.2fnm'%self.wfeST[idCenter])
                print('.Wind-shake error:\t\t%4.2fnm'%self.wfeWindShake)
                print('.Additionnal jitter:\t\t%4.2fmas / %4.2fnm'%(nnp.mean(self.ao.cam.spotFWHM[0][0:2]),self.wfeJitter))
                if self.ao.addMcaoWFsensConeError:
                    print('.Mcao Cone:\t\t\t%4.2fnm'%self.wfeMcaoCone)
                print('.Extra error:\t\t\t%4.2fnm'%self.wfeExtra)
                print('-------------------------------------------')
                print('.Sole servoLag error:\t\t%4.2fnm'%self.wfeS)
                print('.Sole reconstruction error:\t%4.2fnm'%self.wfeR)
                print('-------------------------------------------')
                if self.nGs == 1:
                    print('.Sole anisoplanatism error:\t%4.2fnm'%self.wfeAni[idCenter])
                else:
                    print('.Sole tomographic error:\t%4.2fnm'%self.wfeTomo[idCenter])
                print('-------------------------------------------')

        self.t_errorBreakDown = 1000*(time.time() - tstart)

  #%% PSF COMPUTATION
    def point_spread_function(self, x0=[], nPix=None, verbose=False,
                            fftphasor=False, addOtfPixel=False):
        """
          Computation of the 4D PSF from the 3D cube of phase structure function
          If x0 kept empty, the residual jitter is included from the values given
          in the .ini file.
        """

        tstart  = time.time()

        # ----------------- GETTING THE PARAMETERS
        (Cn2, r0, x0_dphi, x0_jitter, x0_stellar, x0_stat) = FourierUtils.sort_params_from_labels(self,x0)

        # ----------------- MANAGING THE PIXEL OTF
        otfPixel=1
        if addOtfPixel:
            otfPixel = np.sinc(self.freq.U_)* np.sinc(self.freq.V_)

        # ----------------- COMPUTING THE PSF
        PSF, SR = FourierUtils.sf_3D_to_psf_4D(self.SF,
                                               self.freq,
                                               self.ao,
                                               x_jitter = x0_jitter,
                                               x_stat = x0_stat,
                                               x_stellar = x0_stellar,
                                               nPix = nPix,
                                               otfPixel = otfPixel)

        self.t_getPSF = 1000*(time.time() - tstart)

        return PSF, SR

    def __call__(self, x0, nPix=None):

        psf,_ = self.point_spread_function(x0 = x0, nPix = nPix,
                                           verbose = False,
                                           fftphasor = True,
                                           addOtfPixel = self.addOtfPixel)
        return psf


  #%% METRICS COMPUTATION
    def getPsfMetrics(self, getEnsquaredEnergy=False, getEncircledEnergy=False, getFWHM=False):
        tstart  = time.time()
        self.FWHM = np.zeros((2,self.ao.src.nSrc,self.freq.nWvl))

        if getEnsquaredEnergy==True:
            self.EnsqE   = np.zeros((int(self.freq.nOtf/2)+1,self.ao.src.nSrc,self.freq.nWvl))
        if getEncircledEnergy==True:
            rr,radialprofile = FourierUtils.radial_profile(self.PSF[:,:,0,0])
            self.EncE   = np.zeros((len(radialprofile),self.ao.src.nSrc,self.freq.nWvl))
        for n in range(self.ao.src.nSrc):
            for j in range(self.freq.nWvl):
                if getFWHM == True:
                    self.FWHM[:,n,j]  = FourierUtils.getFWHM(self.PSF[:,:,n,j],self.freq.psInMas[j],rebin=1,method='contour',nargout=2)
                if getEnsquaredEnergy == True:
                    self.EnsqE[:,n,j] = 1e2*FourierUtils.getEnsquaredEnergy(self.PSF[:,:,n,j])
                if getEncircledEnergy == True:
                    self.EncE[:,n,j]  = 1e2*FourierUtils.getEncircledEnergy(self.PSF[:,:,n,j])

        self.t_getPsfMetrics = 1000*(time.time() - tstart)

#%% DISPLAY

    def displayResults(self,eeRadiusInMas=75,displayContour=False):
        """
        """
        tstart  = time.time()
        # GEOMETRY
        plt.figure()
        plt.polar(self.ao.src.azimuth*deg2rad,self.ao.src.zenith,'ro',markersize=7,label='PSF evaluation (arcsec)')
        plt.polar(self.gs.azimuth*deg2rad,self.gs.zenith,'bs',markersize=7,label='GS position')
        plt.polar(self.ao.dms.opt_dir[1]*deg2rad,self.ao.dms.opt_dir[0],'kx',markersize=10,label='Optimization directions')
        plt.legend(bbox_to_anchor=(1.05, 1))

        if hasattr(self,'PSF'):
            if (self.PSF.ndim == 2):
                plt.figure()
                plt.imshow(np.log10(np.abs(self.PSF)))

            else:
                # PSFs
                if np.any(self.PSF):
                    nmin = self.ao.src.zenith.argmin()
                    nmax = self.ao.src.zenith.argmax()
                    plt.figure()
                    if self.PSF.shape[2] >1 and self.PSF.shape[3] == 1:
                        plt.title("PSFs at {:.1f} and {:.1f} arcsec from center".format(self.ao.src.zenith[nmin],self.ao.src.zenith[nmax]))
                        P = np.concatenate((self.PSF[:,:,nmin,0],self.PSF[:,:,nmax,0]),axis=1)
                    elif self.PSF.shape[2] >1 and self.PSF.shape[3] >1:
                        plt.title("PSFs at {:.0f} and {:.0f} arcsec from center\n - Top: {:.0f}nm - Bottom:{:.0f} nm".format(self.ao.src.zenith[0],self.ao.src.zenith[-1],1e9*self.wvl[0],1e9*self.wvl[-1]))
                        P1 = np.concatenate((self.PSF[:,:,nmin,0],self.PSF[:,:,nmax,0]),axis=1)
                        P2 = np.concatenate((self.PSF[:,:,nmin,-1],self.PSF[:,:,nmax,-1]),axis=1)
                        P  = np.concatenate((P1,P2),axis=0)
                    else:
                        plt.title('PSF')
                        P = self.PSF[:,:,nmin,0]
                    plt.imshow(np.log10(np.abs(P)))


                if displayContour == True and np.any(self.SR) and self.SR.size > 1:
                    self.displayPsfMetricsContours(eeRadiusInMas=eeRadiusInMas)
                else:
                    # STREHL-RATIO
                    if hasattr(self,'SR') and np.any(self.SR) and self.SR.size > 1:
                        plt.figure()
                        plt.plot(self.ao.src.zenith,self.SR[:,0],'bo',markersize=10)
                        plt.xlabel("Off-axis distance")
                        plt.ylabel("Strehl-ratio at {:.1f} nm (percents)".format(self.freq.wvlRef*1e9))
                        plt.show()

                    # FWHM
                    if hasattr(self,'FWHM') and np.any(self.FWHM) and self.FWHM.size > 1:
                        plt.figure()
                        plt.plot(self.ao.src.zenith,0.5*(self.FWHM[0,:,0]+self.FWHM[1,:,0]),'bo',markersize=10)
                        plt.xlabel("Off-axis distance")
                        plt.ylabel("Mean FWHM at {:.1f} nm (mas)".format(self.freq.wvlRef*1e9))
                        plt.show()

                    # Ensquared energy
                    if hasattr(self,'EnsqE') and np.any(self.EnsqE):
                        nntrue      = eeRadiusInMas/self.freq.psInMas[0]
                        nn2         = int(nntrue)
                        EEmin       = self.EnsqE[nn2,:,0]
                        EEmax       = self.EnsqE[nn2+1,:,0]
                        EEtrue      = (nntrue - nn2)*EEmax + (nn2+1-nntrue)*EEmin
                        plt.figure()
                        plt.plot(self.ao.src.zenith,EEtrue,'bo',markersize=10)
                        plt.xlabel("Off-axis distance")
                        plt.ylabel("{:.1f}-mas-side Ensquared energy at {:.1f} nm (percents)".format(eeRadiusInMas,self.freq.wvlRef*1e9))
                        plt.show()

                    if hasattr(self,'EncE') and np.any(self.EncE):
                        nntrue      = eeRadiusInMas/self.freq.psInMas[0]
                        nn2         = int(nntrue)
                        EEmin       = self.EncE[nn2,:,0]
                        EEmax       = self.EncE[nn2+1,:,0]
                        EEtrue      = (nntrue - nn2)*EEmax + (nn2+1-nntrue)*EEmin
                        plt.figure()
                        plt.plot(self.ao.src.zenith,EEtrue,'bo',markersize=10)
                        plt.xlabel("Off-axis distance")
                        plt.ylabel("{:.1f}-mas-diameter Encircled energy at {:.1f} nm (percents)".format(eeRadiusInMas*2,self.freq.wvlRef*1e9))
                        plt.show()

        self.t_displayResults = 1000*(time.time() - tstart)

    def displayPsfMetricsContours(self,eeRadiusInMas=75,wvlIndex=0):

        tstart  = time.time()
        # Polar to cartesian
        x = self.ao.src.zenith * np.cos(np.pi/180*self.ao.src.azimuth)
        y = self.ao.src.zenith * np.sin(np.pi/180*self.ao.src.azimuth)


        nn = int(np.sqrt(self.SR.shape[0]))

        if nn**2 == self.SR.shape[0]:
            nIntervals  = nn
            X           = np.reshape(x,(nn,nn))
            Y           = np.reshape(y,(nn,nn))

            # Strehl-ratio
            if hasattr(self,'SR') and  np.any(self.SR):
                SR = np.reshape(self.SR[:,wvlIndex],(nn,nn))
                plt.figure()
                contours = plt.contour(X, Y, SR, nIntervals, colors='black')
                plt.clabel(contours, inline=True,fmt='%1.1f')
                plt.contourf(X,Y,SR)
                plt.title("Strehl-ratio at {:.1f} nm (percents)".format(self.freq.wvl[wvlIndex]*1e9))
                plt.colorbar()

            # FWHM
            if hasattr(self,'FWHM') and np.any(self.FWHM) and self.FWHM.size > 1:
                FWHM = np.reshape(0.5*(self.FWHM[0,:,wvlIndex] + self.FWHM[1,:,wvlIndex]),(nn,nn))
                plt.figure()
                contours = plt.contour(X, Y, FWHM, nIntervals, colors='black')
                plt.clabel(contours, inline=True,fmt='%1.1f')
                plt.contourf(X,Y,FWHM)
                plt.title("Mean FWHM at {:.1f} nm (mas)".format(self.freq.wvl[wvlIndex]*1e9))
                plt.colorbar()

            # Ensquared Enery
            if hasattr(self,'EnsqE') and np.any(self.EnsqE) and self.EnsqE.shape[1] > 1:
                nntrue      = eeRadiusInMas/self.freq.psInMas[0]
                nn2         = int(nntrue)
                EEmin       = self.EnsqE[nn2,:,wvlIndex]
                EEmax       = self.EnsqE[nn2+1,:,wvlIndex]
                EEtrue      = (nntrue - nn2)*EEmax + (nn2+1-nntrue)*EEmin
                EE          = np.reshape(EEtrue,(nn,nn))
                plt.figure()
                contours = plt.contour(X, Y, EE, nIntervals, colors='black')
                plt.clabel(contours, inline=True,fmt='%1.1f')
                plt.contourf(X,Y,EE)
                plt.title("{:.1f}-mas-side Ensquared energy at {:.1f} nm (percents)".format(eeRadiusInMas*2,self.freq.wvl[wvlIndex]*1e9))
                plt.colorbar()

            # Encircled Enery
            if hasattr(self,'EncE') and np.any(self.EncE) and self.EncE.shape[1] > 1:
                nntrue      = eeRadiusInMas/self.freq.psInMas[wvlIndex]
                nn2         = int(nntrue)
                EEmin       = self.EncE[nn2,:,wvlIndex]
                EEmax       = self.EncE[nn2+1,:,wvlIndex]
                EEtrue      = (nntrue - nn2)*EEmax + (nn2+1-nntrue)*EEmin
                EE          = np.reshape(EEtrue,(nn,nn))
                plt.figure()
                contours = plt.contour(X, Y, EE, nIntervals, colors='black')
                plt.clabel(contours, inline=True,fmt='%1.1f')
                plt.contourf(X,Y,EE)
                plt.title("{:.1f}-mas-diameter Encircled energy at {:.1f} nm (percents)".format(eeRadiusInMas*2,self.freq.wvl[wvlIndex]*1e9))
                plt.colorbar()
        else:
            print('You must define a square grid for PSF evaluations directions - no contours plots avalaible')

        self.t_displayPsfMetricsContours = 1000*(time.time() - tstart)

    def displayExecutionTime(self):

        # total
        if self.t_init > 0: print("Required time for total calculation (ms)\t : {:f}".format(self.t_init))
        if self.t_initAO > 0: print("Required time for AO system model init (ms)\t : {:f}".format(self.t_initAO))
        if self.ao.error == False:
            if self.t_initFreq > 0: print("Required time for frequency domain init (ms)\t : {:f}".format(self.t_initFreq))
            if self.t_atmo > 0: print("Required time for atmosphere nodel init (ms)\t : {:f}".format(self.t_atmo))

            if self.t_powerSpectrumDensity > 0: print("Required time for final PSD calculation (ms)\t : {:f}".format(self.t_powerSpectrumDensity))

            # Reconstructors
            if self.ao.rtc.holoop['gain'] > 0:
                if self.t_reconstructor > 0:
                    print("Required time for WFS reconstructors init (ms)\t : {:f}".format(self.t_reconstructor))
                if self.nGs > 1:
                    if self.t_finalReconstructor > 0:
                        print("Required time for optimization init (ms)\t : {:f}".format(self.t_finalReconstructor))
                    if self.t_tomo > 0:
                        print("Required time for tomography init (ms)\t\t : {:f}".format(self.t_tomo))
                    if self.t_opt > 0:
                        print("Required time for optimization init (ms)\t : {:f}".format(self.t_opt))
                # Controller
                if self.t_controller > 0:
                    print("Required time for controller instantiation (ms)\t : {:f}".format(self.t_controller))
                # PSD
                if self.t_fittingPSD > 0:
                    print("Required time for fitting PSD calculation (ms)\t : {:f}".format(self.t_fittingPSD))
                if self.t_aliasingPSD > 0:
                    print("Required time for aliasing PSD calculation (ms)\t : {:f}".format(self.t_aliasingPSD))
                if self.t_noisePSD > 0:
                    print("Required time for noise PSD calculation (ms)\t : {:f}".format(self.t_noisePSD))
                if self.t_spatioTemporalPSD > 0:
                    print("Required time for ST PSD calculation (ms)\t : {:f}".format(self.t_spatioTemporalPSD))
                if self.t_windShakePSD > 0:
                    print("Required time for wind shake calculation (ms)\t : {:f}".format(self.t_windShakePSD))
                if self.t_focalAnisoplanatism > 0:
                    print("Required time for focal Aniso PSD calc. (ms)\t : {:f}".format(self.t_focalAnisoplanatism))
                if self.t_mcaoWFsensCone > 0:
                    print("Required time for MCAO WFs cone PSD calc. (ms)\t : {:f}".format(self.t_mcaoWFsensCone))
                if self.t_extra > 0:
                    print("Required time for extra PSD calculation (ms)\t : {:f}".format(self.t_extra))
                if self.t_extraLo > 0:
                    print("Required time for extra PSD (LO) calc. (ms)\t : {:f}".format(self.t_extraLo))
                if self.t_tiltFilter > 0:
                    print("Required time for tilt filter calculation (ms)\t : {:f}".format(self.t_tiltFilter))
                if self.t_focusFilter > 0:
                    print("Required time for focus filter calculation (ms)\t : {:f}".format(self.t_focusFilter))

                # Error breakdown
                if self.t_errorBreakDown > 0:
                    print("Required time for error calculation (ms)\t : {:f}".format(self.t_errorBreakDown))

                # PSF metrics
                if self.t_getPsfMetrics >0:
                    print("Required time for get PSF metrics (ms)\t\t : {:f}".format(self.t_getPsfMetrics))

                # Display
                if self.display and self.calcPSF:
                    print("Required time for displaying figures (ms)\t : {:f}".format(self.t_displayResults))
                    
            if self.calcPSF:
                print("Required time for all PSFs calculation (ms)\t : {:f}".format(self.t_getPSF))

import pathlib
file_ini0 = str(pathlib.Path(__file__).parent.parent.absolute()) + '/dummy.ini'
faoDummy = fourierModel(path_ini=file_ini0, calcPSF=False, verbose=False, display=False, path_root="", doComputations=True)
