#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 13:57:21 2021

@author: omartin
"""

import numpy as np
from p3.aoSystem.optics import optics
from p3.aoSystem.detector import detector
from p3.aoSystem.processing import processing

class sensor:
    """
    Wavefront sensor class. This class is instantiated through three sub-classes:\
    optics, detector and processing.
    """

    def __init__(self, pixel_scale, fov, binning=1, spotFWHM=[[0.0, 0.0, 0.0]],
                 nph=[np.inf], bandwidth=0.0, transmittance=[1.0], dispersion=[[0.0],[0.0]],
                 ron=0.0, gain=1.0, dark=0.0, sky=0.0, excess=1.0, wfstype='Shack-Hartmann',
                 nL=[1], dsub=[1], nSides=4, modulation=None, clock_rate=[1],
                 algorithm='wcog', algo_param=[5,0,0], noiseVar=[None], tag='WAVEFRONT SENSOR'):

        self.wfstype = wfstype
        self.nWfs = len(nL)
        self.tag = tag

        # optics class
        self.optics = [optics(nL=nL[k],
                              dsub=dsub[k],
                              nSides=nSides,
                              wfstype=wfstype,
                              modulation=modulation) for k in range(self.nWfs)]

        # detector class
        if type(nph)==float:
            nph = [nph]
        elif nph is None:
            nph = [np.inf]

        if len(clock_rate)==1:
            clock_rate = [clock_rate[0] for k in range(self.nWfs)]

        if len(nph) < self.nWfs:
            nph.append([0] * (self.nWfs - len(nph)))

        self.detector = [detector(pixel_scale, fov, binning=binning, spotFWHM=spotFWHM,
                                 nph=nph[k], bandwidth=bandwidth, transmittance=transmittance,
                                 dispersion=dispersion, ron=ron, gain=gain,
                                 dark=dark, sky=sky, excess=excess,
                                 clock_rate=clock_rate[k]) for k in range(self.nWfs)]

        # processing class
        self.processing = processing(algorithm=algorithm, settings=algo_param,
                                     noiseVar=noiseVar)

    def __repr__(self):

        if self.nWfs==1:
            s = '__ ' + self.tag + ' __\n' + \
                '--------------------------------------------- \n'
        else:
            s = '__ '+str(self.nWfs) + ' WAVEFRONT SENSORS __\n' \
                + '--------------------------------------------- \n'
        s += self.optics[0].__repr__() + '\n'
        s += self.detector.__repr__() + '\n'
        s += self.processing.__repr__() +'\n'
        s += '---------------------------------------------\n'
        return s

    def NoiseVariance(self,r0,wvl):

        rad2arcsec = 3600 * 180 / np.pi

        # parsing inputs
        varNoise = np.zeros(self.nWfs)

        for k in range(self.nWfs):
            pixelScale  = self.detector[k].psInMas/1e3 # in arcsec
            ron = self.detector[k].ron
            nph = np.array(self.detector[k].nph)
            # pixel per subaperture, N_s in Thomas et al. 2006
            # nPix**2 is the total number of pixels used in the CoG calculation
            nPix = self.detector[k].fovInPix
            dsub = self.optics[k].dsub

            if self.wfstype.upper() == 'SHACK-HARTMANN':
                # spot FWHM in pixels and without turbulence, N_samp in Thomas et al. 2006
                # The condition nD = 2 corresponds to the Nyquist sampling of the spots
                nD = max(1,rad2arcsec * wvl/dsub /pixelScale)

                # The full width at half-maximum (FWHM) of the spot, N_T in Thomas et al. 2006
                # For diffraction-limited spots nT = nD = 2
                nT = max(1,
                         np.hypot(max(self.detector[k].spotFWHM[0][0:2])/1e3,
                                  rad2arcsec*wvl/r0)/pixelScale)

                # for WCoG, Nw is the weighting function FWHM in pixel
                nW = self.processing.settings[0]
                if nW < nT:
                    print(f'WARNING: weighting function FWHM of the WCoG, {nW} is smaller than'
                          f' the FWHM of the spot {nT}, forcing it to have the same size.')
                    nW = nT

                # tCoG parameters
                th = self.processing.settings[1]
                new_val_th = self.processing.settings[2]

                # read-out noise calculation & photo-noise calculation
                # from Thomas et al. 2006
                if self.processing.algorithm == 'cog':
                    var_ron  = np.pi**2/3 * (ron**2 /nph**2) * (nPix**2/nD)**2
                    var_shot  = np.pi**2/(2*np.log(2)*nph) * (nT/nD)**2
                if self.processing.algorithm == 'tcog':
                    # Here we consider that the pixel used in the computation
                    # are the ones where the PSF is above the 0.5 w.r.t. the maximum value,
                    # so, nPix**2 is subsituted by np.ceil(nT**2*np.pi/4)
                    var_ron  = np.pi**2/3 * (ron**2 /nph**2) * (np.ceil(nT**2*np.pi/4)/nD)**2
                    var_shot  = np.pi**2/(2*np.log(2)*nph) * (nT/nD)**2
                if self.processing.algorithm == 'wcog':
                    var_ron  = np.pi**3/(32*np.log(2)**2) * (ron**2 /nph**2) \
                        * (nT**2+nW**2)**4/(nD**2*nW**4)
                    var_shot  = np.pi**2/(2*np.log(2)*nph) * (nT/nD)**2 \
                        * (nT**2+nW**2)**4/((2*nT**2+nW**2)**2*nW**4)
                if self.processing.algorithm == 'qc':
                    if nT > nD:
                        k_factor = np.sqrt(2*np.pi) * (nT/(2*np.sqrt(2*np.log(2))) / nD)
                    else:
                        k_factor = 1
                    var_ron  = k_factor *  4*np.pi**2 * (ron/nph)**2
                    var_shot  = k_factor * np.pi**2/nph

                if np.any(var_ron > 5):
                    print(f'The read-out noise variance is very high ({var_ron:.1f} >5 rd^2),'
                          ' there is certainly smth wrong with your inputs, set to 5')
                    var_ron = 5

                if np.any(var_shot > 5):
                    print(f'The shot noise variance is very high ({var_shot:.1f} >5 rd^2),'
                          ' there is certainly smth wrong with your inputs, set to 5')
                    var_shot = 5

            if self.wfstype.upper() == 'PYRAMID':
                var_ron  = 4*ron**2/nph**2
                var_shot = nph/nph**2

            varNoise[k] = var_ron + self.detector[k].excess * var_shot

        return varNoise

    def computeNoiseVarianceAtWavelength(self, wvl_science, wvl_wfs, r0_at_500nm):
        """
        Compute noise variance scaled to science wavelength.
        
        Parameters:
        -----------
        wvl_science : float
            Science wavelength in meters
        wvl_wfs : float
            WFS wavelength in meters
        r0_at_500nm : float
            Fried parameter at 500 nm in meters.
        Returns:
        --------
        noise_var_scaled : array
            Noise variance scaled to science wavelength
        """

        # Calculate r0 at WFS wavelength
        r0_wfs = r0_at_500nm * (wvl_wfs / 500e-9)**(6/5)

        # Calculate noise at WFS wavelength
        noise_var_wfs = self.NoiseVariance(r0_wfs, wvl_wfs)

        # Scale to science wavelength
        wvl_scale_factor = wvl_wfs / wvl_science
        noise_var_scaled = noise_var_wfs * wvl_scale_factor**2

        return noise_var_scaled
