#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:50:20 2021

@author: omartin
"""

#%% MANAGE PYTHON LIBRAIRIES
import time
import numpy as np
import numpy.fft as fft
from maoppy.psfmodel import Psfao
from maoppy.instrument import Instrument
import p3.aoSystem.FourierUtils as FourierUtils
from p3.aoSystem.aoSystem import aoSystem as aoSys
from p3.aoSystem.frequencyDomain import frequencyDomain

#%%
RAD2MAS = 3600 * 180 * 1000 / np.pi

class psfao21:
    # INIT
    def __init__(self, path_ini, path_root='', otfPixel=1, coo_stars=None):
        '''
            Instantiating the psfao21 model :
                - creating the spatial frequency domain,
                  including the computation of the static OTF and anisoplanatism
                - instantiating the psfao19 model
                - instantiating the bounds
        '''
        tstart = time.time()
        self.tag = 'PSFAO21 MODEL'
        self.param_labels = ['Cn2', 'C', 'A', 'ax', 'p', 'theta', 'beta',
                             'jitterX', 'jitterY', 'jitterXY',
                             'F', 'dx', 'dy', 'bkg', 'stat']

        # INSTANTIATING THE AOSYSTEM CLASS
        self.file = path_ini
        self.ao = aoSys(path_ini, path_root=path_root, coo_stars=coo_stars)

        # GETTING INPUTS
        self.isStatic = self.ao.tel.nModes>0
        self.otfPixel = otfPixel

        # INSTANTIATING THE MAOPPY MODEL
        self.npix = self.ao.cam.fovInPix
        system = Instrument(D=self.ao.tel.D,
                            occ=self.ao.tel.obsRatio,
                            res=self.ao.tel.resolution,
                            Nact=self.ao.dms.nActu1D,
                            gain=self.ao.cam.gain,
                            ron=self.ao.cam.ron)

        self.wvl, self.nwvl = FourierUtils.create_wavelength_vector(self.ao)
        samp = RAD2MAS * self.wvl/(self.ao.cam.psInMas*self.ao.tel.D)

        if self.nwvl > 1:
            self.psfao_19 = []
            self.freq = []
            src_wvl = self.wvl
            for n in range(self.nwvl):
                # CREATING INSTANCES OF THE MAOPPY MODEL
                self.psfao_19.append(Psfao((self.npix, self.npix),
                                    system=system, samp=samp[n]))

                # DEFINING THE FREQUENCY DOMAIN
                self.ao.cam.nWvl = 1
                self.ao.src.wvl = [src_wvl[n]]
                self.ao.src.nWvl = 1
                self.ao.cam.bandwidth = 0
                self.freq.append(frequencyDomain(self.ao))
        else:
            self.psfao_19 = Psfao((self.npix, self.npix),
                                  system=system, samp=samp)
            self.freq = frequencyDomain(self.ao)

        # DEFINING THE NUMBER OF PSF PARAMETERS
        self.n_param_atm = min(self.ao.atm.nL, self.ao.dms.nRecLayers)
        self.n_param_dphi = 6

        # DEFINING BOUNDS
        self.bounds = self.define_bounds()

        self.t_init = 1000*(time.time()  - tstart)

    def __repr__(self):
        s = '---------------------------------------------' \
            + self.tag  + '--------------------------------------------- \n\n'
        s += self.ao.__repr__() + '\n'
        s += self.freq.__repr__() + '\n'
        return s

    def define_bounds(self):
        """
        Defining bounds on the PSFAO21 parameters based physics-based a priori
        """
        #Cn2/r0 , C , A , ax , p , theta , beta , sx , sy , sxy , F , dx , dy , bg , stat
        # Bounds on r0
        nL = min(self.ao.atm.nL, self.ao.dms.nRecLayers)
        bounds_down = list(np.ones(nL)*1e-3)
        bounds_up = list(np.inf * np.ones(nL))
        # PSD Parameters
        bounds_down += [0, 0, 1e-3, 1e-2, -np.pi, 1.01]
        bounds_up += [np.inf, np.inf, np.inf, 1e2, np.pi, 5]
        # Jitter
        bounds_down += [0, 0, -np.inf]
        bounds_up += [np.inf, np.inf, np.inf]
        # Photometry
        bounds_down += list(np.zeros(self.ao.src.nSrc * self.nwvl))
        bounds_up += list(np.inf*np.ones(self.ao.src.nSrc * self.nwvl))
        # Astrometry
        bounds_down += list(-self.npix//2 * np.ones(2*self.ao.src.nSrc * self.nwvl))
        bounds_up += list(self.npix//2 * np.ones(2*self.ao.src.nSrc * self.nwvl))
        # Background
        bounds_down += list(-np.inf * np.ones(self.nwvl))
        bounds_up += list(np.inf * np.ones(self.nwvl))
        # Static aberrations
        bounds_down += list(-max(self.wvl)/2*1e9 * np.ones(self.ao.tel.nModes))
        bounds_up += list(max(self.wvl)/2 *1e9 * np.ones(self.ao.tel.nModes))

        return (bounds_down, bounds_up)

    def update_bounds(self, xfinal, xerr, fixed=None, sig=5):
        '''
            Defining bounds on the PSFAO21 parameters based on the first step
            of the split fitting.
        '''
        bounds = np.array(self.bounds)
        # lower bounds
        bounds_low_psd = list(np.maximum(xfinal[:7] - sig/3*xerr[:7], bounds[0, :7]))
        bounds_low = bounds_low_psd\
                    + [-np.inf, -np.inf, -np.inf, 0, -np.inf, -np.inf, -np.inf]\
                    + [-max(self.wvl)*1e9/2,]*self.ao.tel.nModes

        #upper bounds
        bounds_up_psd = list(np.minimum(xfinal[:7] + sig/3*xerr[:7], bounds[1, :7]))
        bounds_up = bounds_up_psd\
                    + [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]\
                    + [max(self.wvl)*1e9/2,]*self.ao.tel.nModes

        self.psfao_19.bounds = (bounds_low_psd, bounds_up_psd)
        return (bounds_low, bounds_up)

    def get_power_spectrum_density(self, x0, psd_model, freq, grad=False):
        '''
            Define the model of the AO-corrected power spectrum density of the
            electric field phase in the pupil plan.
        '''
        if grad:
            psd, self.wfe, grad, integral_grad = psd_model.psd(x0, grad=True)
        else:
            psd, self.wfe = psd_model.psd(x0, grad=False)

        # wavefront errors
        cte = freq.wvl*1e9/2/np.pi
        self.wfe = np.sqrt(self.wfe) * cte
        self.wfe_fit = np.sqrt(x0[0]**(-5/3)) * freq.wfe_fit_norm  * cte

        return psd

    def get_opd_structure_function(self, freq, Cn2=None):
        '''
            Define the spatially-variable phase structure function from the
            psd model and the anisoplanatism model
        '''

        #covariance map
        real_fft = fft.rfft2(fft.fftshift(self.psd)) / (self.ao.tel.D * freq.sampRef)**2
        Bphi = fft.fftshift(FourierUtils._rfft2_to_fft2(self.psd.shape, real_fft))
        #Bphi = fft.fft2(fft.fftshift(self.psd)) / (self.ao.tel.D * freq.sampRef)**2
        # On-axis phase structure function
        SF = np.real(2*(Bphi.max() - Bphi))

        # Anisoplanatism phase structure function
        if freq.isAniso and Cn2 is not None and (len(Cn2) == freq.dani_ang.shape[1]):
            SF = SF[:, :, np.newaxis] + (freq.dphi_ani * Cn2).sum(axis=2)
        else:
            SF = np.repeat(SF[:, :, np.newaxis], self.ao.src.nSrc, axis=2)

        return SF

    def __call__(self, x0, nPix=None):
        '''
            Returns the 4D array containing 2D psf for each wavelength
            and field direction
        '''

        # ----------------- GETTING THE PARAMETERS
        (Cn2, r0, x0_dphi, x0_jitter, x0_stellar, x0_stat) = \
        FourierUtils.sort_params_from_labels(self, x0)

        if self.nwvl == 1:

            # ----------------- SELECTING THE OBJECT PARAMETERS
            x0_stellar_1D = [x0_stellar[0][:, 0].reshape(-1),
                             x0_stellar[1][:, 0].reshape(-1),
                             x0_stellar[2][:, 0].reshape(-1),
                             x0_stellar[3][0]]

            # ----------------- GETTING THE PHASE STRUCTURE FUNCTION
            self.psd = self.get_power_spectrum_density([r0]+ x0_dphi,
                                                       self.psfao_19,
                                                       self.freq)
            self.SF = self.get_opd_structure_function(self.freq, Cn2=Cn2)

            # ----------------- COMPUTING THE PSF
            PSF, self.SR = FourierUtils.sf_3D_to_psf_3D(self.SF,
                                                        self.freq,
                                                        self.ao,
                                                        x_jitter=x0_jitter,
                                                        x_stat=x0_stat,
                                                        x_stellar=x0_stellar_1D,
                                                        nPix=nPix,
                                                        otfPixel=self.otfPixel)
        else:
            PSF = []
            for n in range(self.nwvl):
                # ----------------- SCALING FACTOR
                wvl_ratio = (self.wvl[0]/self.wvl[n])**2

                # ----------------- SELECTING THE OBJECT PARAMETERS
                if len(self.ao.src.wvl)>1:
                    x0_stellar_n = [x0_stellar[0][:, n], x0_stellar[1][:, n],
                                    x0_stellar[2][:, n], x0_stellar[3][n]]

                else:
                    x0_stellar_n = [x0_stellar[0][:, n], x0_stellar[1][:, n],
                                    x0_stellar[2][:, n], x0_stellar[3]]

                # ----------------- GETTING THE PHASE STRUCTURE FUNCTION
                self.psd = self.get_power_spectrum_density([r0]+ x0_dphi,
                                                           self.psfao_19[n],
                                                           self.freq[n])
                self.SF = self.get_opd_structure_function(self.freq[n], Cn2=Cn2)

                # ----------------- COMPUTING THE PSF
                psf, self.SR = FourierUtils.sf_3D_to_psf_3D(self.SF * wvl_ratio,
                                                            self.freq[n],
                                                            self.ao,
                                                            x_jitter=x0_jitter,
                                                            x_stat=x0_stat,
                                                            x_stellar=x0_stellar_n,
                                                            nPix=nPix,
                                                            otfPixel=self.otfPixel)
                PSF.append(psf)
            PSF = np.array(PSF).transpose((1, 2, 3, 0))
        return PSF
