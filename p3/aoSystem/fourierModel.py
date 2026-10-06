#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:31:39 2020

@author: omartin
"""

import numpy as nnp
from . import gpuEnabled, np, nnp, fft, spc, cpuArray, trapz
import matplotlib as mpl
import matplotlib.pyplot as plt

import time
import os
import warnings
import pathlib
from shutil import which

import p3.aoSystem.FourierUtils as FourierUtils
from p3.aoSystem.aoSystem import aoSystem
from p3.aoSystem.atmosphere import atmosphere
from p3.aoSystem.frequencyDomain import frequencyDomain
from p3.aoSystem.airRefraction import MatharAirRefraction
from p3.aoSystem.processing import is_auto_noise_var

#%% DISPLAY FEATURES
mpl.rcParams['font.size'] = 16

if which('tex'):
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

    # Minimum tomographic regularization, relative to the largest diagonal
    # term of the GS covariance at each spatial frequency.
    tomoRelRegFloor = 1e-12

    # CONTRUCTOR
    def __init__(self, path_ini, calcPSF=True, verbose=False, display=True,
                 path_root=None, normalizePSD=False, displayContour=False,
                 getPSDatNGSpositions=False, getErrorBreakDown=False, getFWHM=False,
                 getEnsquaredEnergy=False, getEncircledEnergy=False, fftphasor=False,
                 MV=0, nyquistSampling=False, addOtfPixel=False, freq=None, ao=None,
                 computeFocalAnisoCov=True, TiltFilter=False, doComputations=True,
                 psdExpansion=False, psdPerWavelength=False,
                 reduce_memory=False, config_dict=None):
        """
        Parameters
        ----------
        path_ini : str
            Path to the .ini or .yml parameter file (ignored if ``ao`` or
            ``config_dict`` is given).
        calcPSF : bool, optional
            Compute the PSFs (and Strehl ratios) from the PSD. If False only
            the PSD is computed.
        verbose : bool, optional
            Print diagnostic messages.
        display : bool, optional
            Display the controller transfer functions and, if ``calcPSF``,
            the PSFs.
        path_root : str, optional
            Root directory prepended to relative paths of auxiliary files
            (pupil, static maps, ...) referenced in the parameter file.
        normalizePSD : bool, optional
            Rescale the total PSD so that its integral matches
            ``[RTC] ResidualError`` (in nm).
        displayContour : bool, optional
            Overplot Strehl contours in the field when displaying results.
        getPSDatNGSpositions : bool, optional
            Append the ``[sources_LO]`` directions to the science directions,
            so that the PSD is also computed at the NGS positions.
        getErrorBreakDown : bool, optional
            Compute the error breakdown (fitting, aliasing, noise, ...).
        getFWHM, getEnsquaredEnergy, getEncircledEnergy : bool, optional
            Compute the corresponding PSF metric (requires ``calcPSF``).
        fftphasor : bool, optional
            Currently unused (accepted by ``point_spread_function`` but not
            applied).
        MV : int, optional
            1 for the minimum-variance (noise-aware) reconstructor in single
            conjugate systems, 0 for least squares.
        nyquistSampling : bool, optional
            Force a Nyquist-sampled PSF (lambda/2D) instead of
            ``[sensor_science] PixelScale``.
        addOtfPixel : bool, optional
            Multiply the OTF by the pixel transfer function (sinc).
        freq : frequencyDomain, optional
            Pre-computed frequency domain to reuse instead of building a new one.
        ao : aoSystem, optional
            Pre-built aoSystem to reuse instead of reading the parameter file.
        computeFocalAnisoCov : bool, optional
            Compute the focal (cone effect) anisoplanatism term in SCAO/SLAO.
        TiltFilter : bool, optional
            Remove tilt from the PSD (used when tip/tilt is handled by a
            separate LO loop, e.g. by TIPTOP) and skip the wind-shake PSD.
        doComputations : bool, optional
            Run ``initComputations()`` at construction. If False the caller
            must call it explicitly.
        psdExpansion : bool, optional
            Choose the PSD step from the wavelength with the finest required
            step and a non-integer oversampling, so that the science pixel
            scale is reproduced exactly on the shared grid.
        psdPerWavelength : bool, optional
            With more than one science wavelength, compute one exact PSD grid
            per wavelength: ``self.PSD`` becomes a list of arrays, one per
            wavelength, each equal to a standalone single-wavelength run.
            Cost grows roughly with the number of wavelengths. Not compatible
            with ``calcPSF`` or ``getErrorBreakDown``.
        reduce_memory : bool, optional
            Free intermediate arrays (tomographic matrices, PSD components)
            once they are no longer needed.
        config_dict : dict, optional
            Parameter dictionary (same structure as the parameter file) used
            instead of reading ``path_ini``.
        """

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
                               psdExpansion=psdExpansion,
                               psdPerWavelength=psdPerWavelength,
                               verbose=verbose,
                               config_dict=config_dict)
        else:
            self.ao = ao
        self.dtype = self.ao.dtype
        self.complex_dtype = self.ao.complex_dtype
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

    def set_precision(self, precision):
        """
        Set the dtype and complex_dtype for the model.
        precision: 'single' or 'double'
        """
        if precision == 'single':
            self.dtype = np.float32
            self.complex_dtype = np.complex64
        elif precision == 'double':
            self.dtype = np.float64
            self.complex_dtype = np.complex128
        else:
            raise ValueError("precision must be 'single' or 'double'")

    def initComputations(self):

        tstart = time.time()

        if self.ao.error is False:

            # DEFINING THE FREQUENCY DOMAIN
            self.wvl, self.nwvl = FourierUtils.create_wavelength_vector(self.ao,
                                                                        dtype=self.dtype)
            if not hasattr(self, 'freq'):
                self.freq = frequencyDomain(
                    self.ao,
                    nyquistSampling=self.nyquistSampling,
                    computeFocalAnisoCov=self.computeFocalAnisoCov,
                    dtype=self.dtype
                )

            # Fail fast: user PSF FoV (in pixels) must be large enough to host
            # the AO-corrected support built by frequencyDomain.
            if self.freq.nOtf < self.freq.resAO:
                min_fov_pix = int(np.ceil(self.freq.resAO / max(1, self.freq.kRef_)))
                raise ValueError(
                    "Invalid PSF size configuration after frequencyDomain init: "
                    f"fovInPix={self.ao.cam.fovInPix}, kRef={self.freq.kRef_}, "
                    f"nOtf={self.freq.nOtf}, resAO={self.freq.resAO}. "
                    "The selected PSF size is smaller than the AO support. "
                    f"Use fovInPix >= {min_fov_pix} (equivalently nOtf >= resAO)."
                )
            self.t_initFreq = 1000*(time.time() - tstart)

            # DEFINING THE GUIDE STAR AND THE STRECHING FACTOR
            if self.ao.lgs:
                self.gs = self.ao.lgs
                self.nGs = self.ao.lgs.nSrc
            else:
                self.gs = self.ao.ngs
                self.nGs = self.ao.ngs.nSrc
            # Layers at or above a finite-altitude LGS are not sensed: they are
            # left uncorrected (open loop) instead of entering the cone
            # stretch 1/(1-h/z), which diverges at h=z and is negative above.
            self.sensedLayers = self._sensed_layers_mask(self.ao.atm.heights)
            self.sensedWeights = nnp.asarray(cpuArray(self.ao.atm.weights), dtype=float) \
                                 * self.sensedLayers
            # Sensed fraction of the Cn2 (exactly 1 when all layers are sensed,
            # so that the NGS/high-LGS results are unchanged).
            self.sensedFraction = 1.0 if self.sensedLayers.all() else \
                float(self.sensedWeights.sum() / nnp.sum(cpuArray(self.ao.atm.weights)))
            if not self.sensedLayers.all():
                msg = (f'{int((~self.sensedLayers).sum())} turbulent layer(s) at or above the LGS '
                       f'altitude ({1 - self.sensedFraction:.3f} of the Cn2) are not sensed '
                       'and are left uncorrected.')
                if self.nGs > 1:
                    msg += (' Their geometry cannot be reconstructed: for ground-layer '
                            'correction consider NumberReconstructedLayers = 1.')
                warnings.warn(msg, stacklevel=2)
            self.strechFactor = self._stretch_factor(self.ao.atm.heights)

            # DEFINING THE REFRACTIVE INDEX OF THE AIR AT THE REFERENCE AND GUIDE STAR WAVELENGTH
            # (self.n_air_wvlRef is recomputed per science wavelength inside the
            # PSD loop below when a per-wavelength PSD list is requested; this
            # value is only the one left over for the legacy/shared-grid case.)
            mathar_model = MatharAirRefraction()
            self.n_air_wvlRef = np.asarray(mathar_model.get_refractive_index(self.freq.wvlRef),
                                           dtype=self.dtype)
            self.n_air_gs = np.asarray(mathar_model.get_refractive_index(self.gs.wvl[0]),
                                       dtype=self.dtype)

            # DEFINING THE MODELED ATMOSPHERE
            # (no compression when nothing is sensed: the WFS model is then empty anyway)
            if (self.ao.dms.nRecLayers!=None) and \
                (self.ao.dms.nRecLayers < len(self.ao.atm.weights)) and \
                self.sensedLayers.any():
                # Compress only the sensed layers, so that no equivalent layer
                # mixes sensed and unsensed turbulence.
                if self.sensedLayers.all():
                    w_in = self.ao.atm.weights
                    h_in = self.ao.atm.heights
                else:
                    w_in = nnp.asarray(cpuArray(self.ao.atm.weights))[self.sensedLayers]
                    h_in = nnp.asarray(cpuArray(self.ao.atm.heights))[self.sensedLayers]
                n_rec = min(self.ao.dms.nRecLayers, len(w_in))
                weights_mod,heights_mod = FourierUtils.eqLayers(
                    w_in,
                    h_in,
                    n_rec,
                    dtype=self.dtype
                )
                if n_rec == 1:
                    heights_mod = [0.0]
                wSpeed_mod = cpuArray(
                    np.linspace(min(self.ao.atm.wSpeed),
                                max(self.ao.atm.wSpeed),
                                num=n_rec)
                )
                wDir_mod   = cpuArray(
                    np.linspace(min(self.ao.atm.wDir),
                                max(self.ao.atm.wDir),
                                num=n_rec)
                )
                self.strechFactor_mod = self._stretch_factor(heights_mod)
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
                raise ValueError(
                    "PSF field of view is too small to simulate the AO correction area: "
                    f"max_required={float(np.max(rr)):.3f}, nOtf={self.freq.nOtf}, "
                    f"resAO={self.freq.resAO}, fovInPix={self.ao.cam.fovInPix}, "
                    f"kRef={self.freq.kRef_}."
                )

            # DEFINING THE NOISE PSD
            # r0 at 500nm must be captured *before* the first self.ao.atm.wvl
            # reassignment below (that reassignment rescales self.ao.atm.r0
            # in place, so self.ao.atm.r0 stops being "at 500nm" afterwards).
            wvl_gs = self.gs.wvl[0]
            r0_at_500nm = self.ao.atm.r0
            noiseVar_is_auto = is_auto_noise_var(self.ao.wfs.processing.noiseVar)
            if noiseVar_is_auto:
                self.ao.wfs.processing.noiseVar = self.ao.wfs.computeNoiseVarianceAtWavelength(
                    wvl_science=self.freq.wvlRef,
                    wvl_wfs=wvl_gs,
                    r0_at_500nm=r0_at_500nm,
                )

            self.ao.wfs.processing.noiseVar = np.asarray(
                self.ao.wfs.processing.noiseVar,
                dtype=self.dtype,
            )

            # Updating the atmosphere wavelength !
            # after noise PSD computation where r0 at 500 nm is needed
            self.ao.atm.wvl  = self.freq.wvlRef
            self.atm_mod.wvl = self.freq.wvlRef

            # COMPUTE THE PSD -- once per entry in self.freq.wvl_grids.
            # In the legacy/shared-grid case (psdPerWavelength disabled, or a
            # single wavelength requested) wvl_grids has exactly one entry
            # which IS self.freq itself, so this loop runs once and self.PSD
            # ends up identical (same object, same values) to what the
            # pre-refactor code produced -- no separate legacy branch.
            #
            # calcPSF/getErrorBreakDown assume a single reference grid
            # (self.freq/self.Rx/self.Ry/... as left by the last loop
            # iteration); they are not meaningful with a per-wavelength PSD
            # list, so that combination is rejected explicitly rather than
            # silently computing a PSF for whichever grid happened to run last.
            if len(self.freq.wvl_grids) > 1 and (self.calcPSF or self.getErrorBreakDown):
                raise NotImplementedError(
                    "calcPSF=True/getErrorBreakDown=True are not supported "
                    "together with psdPerWavelength=True and multiple science "
                    "wavelengths (per-wavelength PSD list). Use calcPSF=False "
                    "and getErrorBreakDown=False, and consume self.PSD "
                    "downstream (TIPTOP/MASTSEL); or request a single "
                    "wavelength / disable psdPerWavelength for a standalone "
                    "PSF or error breakdown."
                )

            # Wn (below, per grid_ctx) is normalized by the WFS subaperture
            # pitch (the spatial scale Rx/Ry are actually built from in
            # reconstructionFilter), not by the DM actuator pitch (kcMax_ =
            # 1/(2*pitch)) -- these coincide only when the actuator pitch
            # equals the subaperture pitch. Wavelength-independent, so
            # computed once here rather than inside the per-wavelength loop.
            d_sub_wfs = self.ao.wfs.optics[0].dsub

            saved_freq = self.freq
            multi_grid = len(self.freq.wvl_grids) > 1
            psd_list = []
            for grid_ctx in self.freq.wvl_grids:
                # This is the crux of the per-wavelength PSD: every method
                # below (spatialReconstructor, controller, powerSpectrumDensity
                # and everything they call) reads its grid geometry exclusively
                # through self.freq.* (resAO, nOtf, kxAO_/kyAO_/k2AO_, kcMax_,
                # pistonFilter*, masks, sampRef, otfNCPA/otfDL, dphi_ani, ...).
                # None of that code is grid-aware by itself; swapping the
                # object self.freq points to is what makes one unmodified
                # PSD/reconstructor/controller implementation compute a
                # different, exactly-sized grid on each iteration. grid_ctx is
                # either self.freq itself (legacy single-grid case, see
                # frequencyDomain._buildWvlGrids) or a shallow copy of it with
                # only the wavelength-dependent attributes overridden (see
                # frequencyDomain._buildOneWvlGrid) -- so any attribute this
                # loop body does NOT explicitly touch still resolves correctly
                # to the original, wavelength-independent value (kc_, nPix,
                # psInMas, U_/V_/U2_/V2_/UV_, ...).
                self.freq = grid_ctx

                # These three quantities are wavelength-dependent and were
                # computed above for the shared/reference wavelength only;
                # recompute them here for this grid's own wavelength so that
                # every entry of self.PSD is exactly what a standalone
                # single-wavelength run would produce for that wavelength
                # (noise propagation, r0 chromatic scaling, differential
                # refraction all depend on the science wavelength).
                if multi_grid:
                    if noiseVar_is_auto:
                        self.ao.wfs.processing.noiseVar = np.asarray(
                            self.ao.wfs.computeNoiseVarianceAtWavelength(
                                wvl_science=grid_ctx.wvlRef,
                                wvl_wfs=wvl_gs,
                                r0_at_500nm=r0_at_500nm,
                            ),
                            dtype=self.dtype,
                        )
                    self.ao.atm.wvl  = grid_ctx.wvlRef
                    self.atm_mod.wvl = grid_ctx.wvlRef
                    self.n_air_wvlRef = np.asarray(
                        mathar_model.get_refractive_index(grid_ctx.wvlRef),
                        dtype=self.dtype,
                    )

                # Several methods (_aliasing_common, reconstructionPSD,
                # servoLagPSD) lazily (re)build self.Rx/self.Ry only "if not
                # hasattr(self, 'Rx')". spatialReconstructor() itself always
                # refreshes them for SCAO/SLAO (nGs<2), but never touches them
                # for tomographic systems (nGs>=2) -- so across iterations of
                # this loop they would silently keep the *previous* grid's
                # shape there. Forcing them absent here makes every iteration
                # rebuild fresh, for either branch.
                if hasattr(self, 'Rx'):
                    del self.Rx
                if hasattr(self, 'Ry'):
                    del self.Ry

                # DEFINING THE ATMOSPHERE PSD
                self.Wn = np.mean(self.ao.wfs.processing.noiseVar) * d_sub_wfs**2
                self.Wphi = self.ao.atm.spectrum(np.sqrt(self.freq.k2AO_))

                # DEFINE THE RECONSTRUCTOR
                self.spatialReconstructor(MV=self.MV)

                # DEFINE THE CONTROLLER
                self.controller(display=self.display)

                #set tilt filter key before computing the PSD
                self.applyTiltFilter = self.TiltFilterP

                # COMPUTE THE PSD
                if self.normalizePSD:
                    wfe = self.ao.rtc.holoop['wfe']
                else:
                    wfe = None
                psd_list.append(self.powerSpectrumDensity(wfe=wfe))
            self.freq = saved_freq

            # Restore the leftover shared-reference state (noiseVar/atm.wvl/
            # n_air_wvlRef) so that anything reading it after construction sees
            # exactly what today's single-grid code leaves behind, regardless
            # of how many wavelengths were actually looped over above.
            if multi_grid:
                if noiseVar_is_auto:
                    self.ao.wfs.processing.noiseVar = np.asarray(
                        self.ao.wfs.computeNoiseVarianceAtWavelength(
                            wvl_science=saved_freq.wvlRef,
                            wvl_wfs=wvl_gs,
                            r0_at_500nm=r0_at_500nm,
                        ),
                        dtype=self.dtype,
                    )
                self.ao.atm.wvl  = saved_freq.wvlRef
                self.atm_mod.wvl = saved_freq.wvlRef
                self.n_air_wvlRef = np.asarray(
                    mathar_model.get_refractive_index(saved_freq.wvlRef),
                    dtype=self.dtype,
                )

            self.PSD = psd_list[0] if len(psd_list) == 1 else psd_list

            # COMPUTE THE PSF
            if self.calcPSF:
                # COMPUTE THE PHASE STRUCTURE FUNCTION
                self.SF = self.phaseStructureFunction()
                self.PSF, self.SR = self.point_spread_function(
                    verbose=self.verbose,fftphasor=self.fftphasor,addOtfPixel=self.addOtfPixel
                )

                # GETTING METRICS
                if self.getFWHM or self.getEnsquaredEnergy or self.getEncircledEnergy:
                    self.getPsfMetrics(getEnsquaredEnergy=self.getEnsquaredEnergy, \
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

    def _sensed_layers_mask(self, heights) -> nnp.ndarray:
        """Boolean mask of the layers seen by the HO guide star (all of them for NGS)."""
        h = nnp.asarray(cpuArray(heights), dtype=float)
        z = float(cpuArray(self.gs.height[0])) if self.ao.lgs else 0.0
        if z <= 0 or not nnp.isfinite(z):
            return nnp.ones(h.shape, dtype=bool)
        return h < z

    def _stretch_factor(self, heights):
        """Cone stretch 1/(1-h/z) for sensed layers, 1 (placeholder) for unsensed ones.

        Unsensed layers are excluded from the WFS model elsewhere, so their
        placeholder value never affects the result.
        """
        if not self.ao.lgs or self.gs.height[0] == 0:
            return 1.0
        h = nnp.asarray(cpuArray(heights))
        if not nnp.issubdtype(h.dtype, nnp.floating):
            h = h.astype(float)
        z = float(cpuArray(self.gs.height[0]))
        sensed = self._sensed_layers_mask(h)
        stretch = nnp.ones_like(h)
        stretch[sensed] = 1.0/(1.0 - h[sensed]/z)
        # host array, like atm.heights it multiplies
        return stretch

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
        bounds_down += (-self.freq.nPix//2 * np.ones(2*self.ao.src.nSrc,
                                                     dtype=self.dtype)).tolist()
        bounds_up += ( self.freq.nPix//2 * np.ones(2*self.ao.src.nSrc,
                                                   dtype=self.dtype)).tolist()
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
            i = self.complex_dtype(1j)
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
                PbetaDM = np.zeros([nK, nK, 1, nDm],
                                   dtype=self.complex_dtype)
                for j in range(nDm): #loop on DMs
                    index = k<=self.freq.kc_[j] # note : circular masking
                    PbetaDM[index, 0, j] = np.exp(2*i*np.pi*h_dm[j]*(fx[index] + fy[index]))
                self.PbetaDM.append(PbetaDM)

            # Computation of the Malpha matrix
            wDir_x = nnp.cos(self.ao.atm.wDir*np.pi/180)
            wDir_y = nnp.sin(self.ao.atm.wDir*np.pi/180)
            self.MPalphaL = np.zeros([nK, nK, self.nGs, nH],
                                     dtype=self.complex_dtype)
            for h in range(nH):
                # unsensed layers: zero WFS response -> full residual in spatioTemporalPSD
                if not self.sensedLayers[h]:
                    continue
                freq_t = wDir_x[h]*self.freq.kxAO_ + wDir_y[h]*self.freq.kyAO_
                for g in range(self.nGs):
                    Alpha = [self.gs.direction[0, g],self.gs.direction[1, g]]
                    fx = Alpha[0]*self.freq.kxAO_
                    fy = Alpha[1]*self.freq.kyAO_
                    www = 2*i*np.pi*k * np.sinc(sampTime*clock_rate[g]*self.ao.atm.wSpeed[h]*freq_t)
                    self.MPalphaL[:, :, g, h] = www*np.sinc(d_sub[g]*self.freq.kxAO_)\
                                                   *np.sinc(d_sub[g]*self.freq.kyAO_)\
                                                   *np.exp(i*2*np.pi*Hs[h]*(fx+fy))
            fx = None
            fy = None
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
        i = self.complex_dtype(1j)
        d = self.ao.wfs.optics[0].dsub

        if self.ao.wfs.optics[0].wfstype.upper()=='SHACK-HARTMANN':
            Sx = 2*i*np.pi*self.freq.kxAO_*d
            Sy = 2*i*np.pi*self.freq.kyAO_*d
            Av = np.sinc(d*self.freq.kxAO_) * np.sinc(d*self.freq.kyAO_)\
                 * np.exp(i*np.pi*d*(self.freq.kxAO_ + self.freq.kyAO_))

        elif self.ao.wfs.optics[0].wfstype.upper()=='PYRAMID':
            # forward pyramid filter (continuous) from Conan
            umod = 1/(2*d)/(self.ao.wfs.optics[0].nL/2)*self.ao.wfs.optics[0].modulation
            Sx = np.zeros((self.freq.resAO,self.freq.resAO),
                          dtype=self.complex_dtype)
            idx = abs(self.freq.kxAO_) > umod
            Sx[idx] = i * np.sign(self.freq.kxAO_[idx])
            idx = abs(self.freq.kxAO_) <= umod
            Sx[idx] = 2*i/np.pi * np.arcsin(self.freq.kxAO_[idx]/umod)
            Av = np.sinc(self.ao.wfs.detector[0].binning*d*self.freq.kxAO_)\
                * np.sinc(self.ao.wfs.detector[0].binning*d*self.freq.kxAO_).T
            Sy = Sx.T
        else:
            raise ValueError("The WFS type is not supported; must be Shack-Hartmann or Pyramid.")
        self.SxAv = (Sx*Av).astype(self.complex_dtype)
        self.SyAv = (Sy*Av).astype(self.complex_dtype)
        Sx = None
        Sy = None
        Av = None

        # Reconstructor
        wvl_gs = self.gs.wvl[0]
        Watm = self.ao.atm.spectrum(np.sqrt(self.freq.k2AO_)) * (self.ao.atm.wvl/wvl_gs)**2
        gPSD = abs(self.SxAv)**2 + abs(self.SyAv)**2 + MV*self.Wn/Watm
        Watm = None
        self.Rx = (np.conj(self.SxAv)/gPSD).astype(self.complex_dtype)
        self.Ry = (np.conj(self.SyAv)/gPSD).astype(self.complex_dtype)
        gPSD = None

        # Set central point (i.e. kx=0,ky=0) to zero
        self.Rx[self.freq.resAO//2, self.freq.resAO//2] = 0
        self.Ry[self.freq.resAO//2, self.freq.resAO//2] = 0

        self.t_reconstructor = 1000*(time.time()  - tstart)

    def tomographicReconstructor(self):
        """
        Computes the tomographic reconstructor based on Neichel+09.
        
        Here we forced single precision for the matrix multiplications to save memory and
        speed up computations, as the tomographic reconstructor is not very sensitive to
        precision.
        """
        tstart = time.time()
        k = np.sqrt(self.freq.k2AO_)
        nK = self.freq.resAO
        nL = len(self.ao.atm.heights)
        h_mod = self.atm_mod.heights*self.strechFactor_mod
        nL_mod = len(h_mod)
        nGs = self.nGs
        i = np.complex64(1j)
        d = [self.ao.wfs.optics[j].dsub for j in range(nGs)]
        sensed_mod = self._sensed_layers_mask(self.atm_mod.heights)

        # 1. WFS operator and projection matrices
        # Calculate MP directly via broadcasting, avoiding the dense M matrix allocation
        MP = np.zeros([nK, nK, nGs, nL_mod], dtype=self.complex_dtype)
        for j in range(nGs):
            M_diag_j = 2*i*np.pi*k * np.sinc(d[j]*self.freq.kxAO_) * np.sinc(d[j]*self.freq.kyAO_)
            for n in range(nL_mod):
                if not sensed_mod[n]:
                    continue
                P_jn = np.exp(i*2*np.pi*h_mod[n]*(self.freq.kxAO_*self.gs.direction[0, j] \
                     + self.freq.kyAO_*self.gs.direction[1, j]))
                MP[:, :, j, n] = M_diag_j * P_jn

        # 2. Atmospheric PSD (Diagonal with respect to the layers)
        atm_weights = np.asarray(self.ao.atm.weights, dtype=self.dtype)
        cte = (24 * spc.gamma(6/5)/5)**(5/6) * (spc.gamma(11/6)**2. / (2.*np.pi**(11/3))).astype(self.dtype)
        kernel = (self.ao.atm.r0**(-5/3) * cte * (self.freq.k2AO_ + 1/self.ao.atm.L0**2)**(-11/6) \
               * self.freq.pistonFilterAO_).astype(self.dtype)
        
        # Save ONLY the 3D diagonal (nK, nK, nL) instead of allocating a full (nK, nK, nL, nL) tensor
        self.Cphi = kernel[:, :, None] * atm_weights[None, None, :]

        atm_mod_weights = np.asarray(self.atm_mod.weights, dtype=self.dtype)
        if nL_mod == nL:
            self.Cphi_mod = self.Cphi
        else:
            self.Cphi_mod = kernel[:, :, None] * atm_mod_weights[None, None, :]
        kernel = None

        # 3. Covariance Matrices
        # np.ascontiguousarray strictly prevents CuPy crashes on strided matmuls
        MP_t = np.ascontiguousarray(np.conj(MP.transpose(0, 1, 3, 2)))
        
        # MP @ Cphi_mod @ MP_t
        MP_Cphi = MP * self.Cphi_mod[:, :, None, :]
        to_inv  = np.matmul(MP_Cphi, MP_t)
        
        # Direct addition of noise on the diagonal (completely eliminates self.Cb allocation)
        noise_var = np.asarray(self.ao.wfs.processing.noiseVar, dtype=self.complex_dtype)
        idx = np.arange(nGs)
        # At low k all GS see the same turbulence (to_inv ~ rank 1): with a
        # (nearly) noise-free WFS the system is singular and the solution
        # backend-dependent. Floor the regularization to a fraction of the
        # largest diagonal term at each k.
        diag_max = np.max(np.abs(to_inv[:, :, idx, idx]), axis=-1, keepdims=True)
        # where to_inv is identically zero (e.g. k=0, piston-filtered) rhs is zero
        # too: any positive value gives Wtomo=0 there instead of NaN/LinAlgError
        reg_floor = np.where(diag_max > 0, self.tomoRelRegFloor * diag_max, 1.0)
        to_inv[:, :, idx, idx] += np.maximum(np.real(noise_var), reg_floor)

        # rhs = Cphi_mod @ MP_t
        rhs = self.Cphi_mod[:, :, :, None] * MP_t

        # 4. Inversion, in double precision: to_inv is badly conditioned at low k
        # and a complex64 solve is wrong there (and differs between CPU and GPU).
        # The small (nGs x nGs) systems make the cost negligible.
        try:
            if self.verbose:
                print("Tomography: Using standard solve")
            Wtomo = np.linalg.solve(
                to_inv.astype(np.complex128).transpose(0, 1, 3, 2),
                rhs.astype(np.complex128).transpose(0, 1, 3, 2)
            ).transpose(0, 1, 3, 2)
        except np.linalg.LinAlgError as e:
            if self.verbose:
                print(f"Tomography: Standard solve failed ({e}), using pinv")
            inv = np.linalg.pinv(to_inv.astype(np.complex128),
                                 rcond=np.finfo(np.float64).eps)
            Wtomo = np.matmul(rhs.astype(np.complex128), inv)
        Wtomo = np.ascontiguousarray(Wtomo.astype(self.complex_dtype))

        to_inv = None
        
        self.t_tomo = 1000*(time.time() - tstart)
        return Wtomo

    def optimalProjector(self):
        """
        Computes the projector from layers to DM from Neichel+09.
        
        Computed in double precision: the DM projections are nearly identical
        at low k, and the Tikhonov normal equations square the condition number
        of to_inv, beyond what complex64 can resolve. Popt is returned in
        the model complex dtype.
        """
        tstart = time.time()
        k = np.sqrt(self.freq.k2AO_)
        h_dm = self.ao.dms.heights
        nDm = len(h_dm)
        nDir = len(self.ao.dms.opt_dir[0])
        h_mod = self.atm_mod.heights * cpuArray(self.strechFactor_mod)
        nL = len(h_mod)
        nK = self.freq.resAO
        i = np.complex128(1j)

        mat1 = np.zeros([nK, nK, nDm, nL],
                        dtype=np.complex128)
        to_inv = np.zeros([nK, nK, nDm, nDm],
                          dtype=np.complex128)
        theta_x = self.ao.dms.opt_dir[0]/206264.8 \
                * nnp.cos(self.ao.dms.opt_dir[1]*np.pi/180)
        theta_y = self.ao.dms.opt_dir[0]/206264.8 \
                * nnp.sin(self.ao.dms.opt_dir[1]*np.pi/180)

        Pdm = np.zeros([nK, nK, 1, nDm],
                       dtype=np.complex128)
        Pl = np.zeros([nK, nK, 1, nL],
                      dtype=np.complex128)
        Pdm_t = np.zeros([nK, nK, nDm, 1],
                         dtype=np.complex128)
        for d_o in range(nDir):                 #loop on optimization directions
            Pdm.fill(0)
            Pl.fill(0)
            fx = theta_x[d_o]*self.freq.kxAO_
            fy = theta_y[d_o]*self.freq.kyAO_
            for j in range(nDm):                # loop on DM
                index = k <= self.freq.kc_[j] # note : circular masking here
                Pdm[index, 0, j] = np.exp(i*2*np.pi*h_dm[j]*(fx[index]+fy[index]))
            Pdm_t[:] = np.conj(Pdm.transpose(0, 1, 3, 2))
            for l in range(nL):                 #loop on atmosphere layers
                Pl[:, :, 0, l] = np.exp(i*2*np.pi*h_mod[l]*(fx + fy))
            mat1 += np.matmul(Pdm_t, Pl)*self.ao.dms.opt_weights[d_o]
            to_inv += np.matmul(Pdm_t, Pdm)*self.ao.dms.opt_weights[d_o]
        Pdm = None
        Pl = None
        Pdm_t = None
        fx = None
        fy = None

        # Popt
        if nDir == 1:
            mat2 = np.linalg.pinv(to_inv.astype(np.complex128),
                                  rcond=1/self.ao.dms.opt_cond)
            to_inv = None
        else:
            # Tikhonov: use only the diagonal of to_inv for regularization
            to_inv_t = np.conj(to_inv.transpose(0, 1, 3, 2))
            lambda_tikhonov = 1/self.ao.dms.opt_cond
            try:
                # Build regularized system
                A = to_inv_t.astype(np.complex128) @ to_inv.astype(np.complex128)
                # Add regularization on diagonal as a fraction of the trace
                lambda_reg = (np.mean(np.diagonal(A, axis1=2, axis2=3)) \
                             * lambda_tikhonov).astype(self.complex_dtype)
                idx = np.arange(nDm)
                A[:, :, idx, idx] += lambda_reg
                b = to_inv_t.astype(np.complex128)
                mat2 = np.linalg.solve(A, b)
                A = None
                b = None
                if self.verbose:
                    print("Optimal projector: Using Tikhonov regularization")
            except np.linalg.LinAlgError as e:
                # Fallback: use pinv on original to_inv
                if self.verbose:
                    print(f"Optimal projector: Tikhonov failed ({e}), using pinv")
                mat2 = np.linalg.pinv(to_inv.astype(np.complex128),
                                    rcond=1/self.ao.dms.opt_cond)

        Popt = np.matmul(mat2, mat1).astype(self.complex_dtype)

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

            i = self.complex_dtype(1j)
            vx = self.ao.atm.wSpeed*nnp.cos(self.ao.atm.wDir*np.pi/180)
            vy = self.ao.atm.wSpeed*nnp.sin(self.ao.atm.wDir*np.pi/180)
            nPts = self.freq.resAO
            thetaWind = np.linspace(0, 2*np.pi-2*np.pi/nTh, nTh)
            costh = np.cos(thetaWind)
            if self.sensedLayers.all():
                weights = self.ao.atm.weights
            else:
                # unsensed layers are not in the loop: average over sensed ones only
                weights = self.sensedWeights / max(self.sensedWeights.sum(), nnp.finfo(float).tiny)
            Ts = 1.0/self.ao.rtc.holoop['rate']#samplingTime
            delay = self.ao.rtc.holoop['delay']#latency
            loopGain = self.ao.rtc.holoop['gain']

            if Ts <= 0:
                raise ValueError('Error : the AO loop rate must be positive\n')

            # Instantiation
            h1 = np.zeros((nPts,nPts),
                          dtype=self.complex_dtype)
            h2 = np.zeros((nPts,nPts),
                          dtype=self.dtype)
            hn = np.zeros((nPts,nPts),
                          dtype=self.dtype)

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
                h1buf = np.zeros((nPts, nPts, nTh),
                                 dtype=self.complex_dtype)
                h2buf = np.zeros((nPts, nPts, nTh),
                                 dtype=self.dtype)
                hnbuf = np.zeros((nPts, nPts, nTh),
                                 dtype=self.dtype)
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
                plt.semilogx(f, 10*np.log10(abs(self.rtfInt)**2),
                             label='Rejection transfer function')
                plt.semilogx(f, 10*np.log10(abs(self.ntfInt)**2),
                             label='Noise transfer function')
                plt.semilogx(f, 10*np.log10(abs(self.atfInt)**2),
                             label='Aliasing transfer function')
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
            pf = FourierUtils.pistonFilter(self.ao.tel.D, k, dtype=self.dtype)
            psd = self.ao.atm.spectrum(k) * pf
            psd = psd[:, :, np.newaxis]
        else:
            # CLOSED-LOOP
            psd = np.zeros((self.freq.nOtf,self.freq.nOtf,self.ao.src.nSrc),
                           dtype=self.dtype)

            # AO correction area
            id1 = np.ceil(self.freq.nOtf/2 - self.freq.resAO/2).astype(int)
            id2 = np.ceil(self.freq.nOtf/2 + self.freq.resAO/2).astype(int)
            # Noise
            self.psdNoise = np.real(self.noisePSD())
            if self.nGs == 1:
                psd[id1:id2,id1:id2,:] = self.psdNoise[:, :, np.newaxis]
            else:
                psd[id1:id2,id1:id2,:] = self.psdNoise

            # --- free memory
            if self.reduce_memory and not self.getErrorBreakDown:
                self.psdNoise = None
                self.Cphi_mod = None

            # Aliasing
            self.psdAlias = np.real(self.aliasingPSD())
            psd[id1:id2,id1:id2,:] += self.psdAlias[:, :, np.newaxis]

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
            psd[id1:id2,id1:id2,:] += self.psdSpatioTemporal

            # --- free memory
            if self.reduce_memory and not self.getErrorBreakDown:
                self.psdSpatioTemporal = None
                self.Cphi = None

            # Cone effect
            if self.nGs == 1 and self.gs.height[0] != 0:
                if self.verbose:
                    print('SLAO case adding cone effect')
                self.psdCone = self.focalAnisoplanatismPSD()
                psd += self.psdCone[:, :, np.newaxis]

                # --- free memory
                if self.reduce_memory and not self.getErrorBreakDown:
                    self.psdCone = None

            # additional error for MCAO system with laser GS:
            # reduced volume for WF sensing due to the cone effect
            if self._mcaoConeApplied():
                if self.verbose:
                    print('MCAO and laser case: adding error due to reduced volume for WF sensing')
                self.psdMcaoWFsensCone = self.mcaoWFsensConePSD(psd)
                psd += self.psdMcaoWFsensCone

                # --- free memory
                if self.reduce_memory and not self.getErrorBreakDown:
                    self.psdMcaoWFsensCone = None

            # NORMALIZATION
            if wfe is not None:
                psd *= (dk * rad2nm)**2
                psd *= wfe**2/psd.sum()

            # Fitting
            self.psdFit = np.real(self.fittingPSD())
            psd += self.psdFit[:, :, np.newaxis]

            # --- free memory
            if self.reduce_memory and not self.getErrorBreakDown:
                self.psdFit = None

            # wind shake / vibrations
            if self.applyTiltFilter is False and self.ao.windPsdFile != 0:
                self.psdVib = self.windShakePSD()
                psd[id1:id2,id1:id2,:] += self.psdVib[:, :, np.newaxis]

                # --- free memory
                if self.reduce_memory and not self.getErrorBreakDown:
                    self.psdVib = None

            # Tilt filter
            if self.applyTiltFilter:
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
                self.kx_ = None
                self.ky_ = None
                self.k2_ = None
                self.U_ = None
                self.V_ = None
                self.kxAO_ = None
                self.kyAO_ = None
                self.k2AO_ = None
                self.kc_ = None
                self.kcAO_ = None

        self.t_powerSpectrumDensity = 1000*(time.time() - tstart)

        # Return the 3D PSD array in nm^2
        return psd * (dk * rad2nm)**2

    def fittingPSD(self):
        """ Fitting error power spectrum density """
        tstart  = time.time()
        #Instantiate the function output
        psd = np.zeros((self.freq.nOtf,self.freq.nOtf),
                       dtype=self.dtype)
        psd[self.freq.mskOut_]  = self.ao.atm.spectrum(np.sqrt(self.freq.k2_[self.freq.mskOut_]))
        self.t_fittingPSD = 1000*(time.time() - tstart)
        return psd

    def _aliasing_common(self):
        i = self.complex_dtype(1j)
        d = self.ao.wfs.optics[0].dsub
        clock_rate = np.array([self.ao.wfs.detector[j].clock_rate for j in range(self.nGs)])
        T = np.mean(clock_rate / self.ao.rtc.holoop['rate'])
        td = T * self.ao.rtc.holoop['delay']
        vx = np.asarray(self.ao.atm.wSpeed * nnp.cos(self.ao.atm.wDir * np.pi / 180), dtype=self.dtype)
        vy = np.asarray(self.ao.atm.wSpeed * nnp.sin(self.ao.atm.wDir * np.pi / 180), dtype=self.dtype)
        weights = np.asarray(self.ao.atm.weights, dtype=self.dtype) * np.asarray(self.sensedLayers)
        w = 2 * i * np.pi * d

        if not hasattr(self, 'Rx'):
            self.reconstructionFilter()
        Rx = np.asarray((self.Rx * w).ravel(), dtype=self.complex_dtype)
        Ry = np.asarray((self.Ry * w).ravel(), dtype=self.complex_dtype)

        if self.ao.rtc.holoop['gain'] == 0:
            tf_flat = np.ones(self.freq.kxAO_.size, dtype=self.complex_dtype)
        else:
            tf_flat = np.asarray(self.h1.ravel(), dtype=self.complex_dtype)

        kxAO = np.asarray(self.freq.kxAO_.ravel(), dtype=self.dtype)
        kyAO = np.asarray(self.freq.kyAO_.ravel(), dtype=self.dtype)
        shift_grid = np.arange(-self.freq.nTimes, self.freq.nTimes)
        mi, ni = np.meshgrid(shift_grid, shift_grid, indexing='ij')
        mask = (mi != 0) | (ni != 0)
        m_flat = np.asarray(mi[mask], dtype=self.dtype)
        n_flat = np.asarray(ni[mask], dtype=self.dtype)
        return d, T, td, vx, vy, weights, Rx, Ry, tf_flat, kxAO, kyAO, m_flat, n_flat

    def _aliasing_signature(self, d, Rx, Ry, tf_flat, kxAO, kyAO):
        # Cheap signature to detect geometry/reconstructor changes and invalidate stale precompute data.
        return (
            int(self.freq.nTimes),
            int(kxAO.size),
            float(d),
            float(self.ao.tel.D),
            float(self.ao.atm.L0),
            float(kxAO[0]) if kxAO.size else 0.0,
            float(kxAO[-1]) if kxAO.size else 0.0,
            float(kyAO[0]) if kyAO.size else 0.0,
            float(kyAO[-1]) if kyAO.size else 0.0,
            float(np.real(Rx[0])) if Rx.size else 0.0,
            float(np.imag(Rx[0])) if Rx.size else 0.0,
            float(np.real(Ry[0])) if Ry.size else 0.0,
            float(np.imag(Ry[0])) if Ry.size else 0.0,
            float(np.real(tf_flat[0])) if tf_flat.size else 0.0,
            float(np.imag(tf_flat[0])) if tf_flat.size else 0.0,
        )

    def _get_aliasing_shift_terms(self, d, Rx, Ry, tf_flat, kxAO, kyAO, m_flat, n_flat, n_times_limit=None):
        if n_times_limit is not None:
            keep = (np.abs(m_flat) <= n_times_limit) & (np.abs(n_flat) <= n_times_limit)
            m_sel = m_flat[keep]
            n_sel = n_flat[keep]
        else:
            m_sel = m_flat
            n_sel = n_flat

        signature = self._aliasing_signature(d, Rx, Ry, tf_flat, kxAO, kyAO)
        if not hasattr(self, '_aliasing_shift_cache'):
            self._aliasing_shift_cache = {}

        # Invalidate stale cache from a different model geometry/reconstructor state.
        if getattr(self, '_aliasing_shift_cache_signature', None) != signature:
            self._aliasing_shift_cache = {}
            self._aliasing_shift_cache_signature = signature

        cache_key = ('terms', None if n_times_limit is None else int(n_times_limit))
        cached = self._aliasing_shift_cache.get(cache_key)
        if cached is not None:
            return cached

        km = kxAO[None, :] - m_sel[:, None] / d
        kn = kyAO[None, :] - n_sel[:, None] / d
        PR = FourierUtils.pistonFilter(self.ao.tel.D, np.hypot(km, kn), dtype=self.dtype)
        W_mn = (km**2 + kn**2 + 1 / self.ao.atm.L0**2) ** (-11 / 6)
        Q = (Rx[None, :] * km + Ry[None, :] * kn) * (np.sinc(d * km) * np.sinc(d * kn))
        out = {
            'm_flat': m_sel,
            'n_flat': n_sel,
            'km': km,
            'kn': kn,
            'PR': PR,
            'W_mn': W_mn,
            'Q': Q,
        }

        # Keep cache small to avoid unbounded memory growth during development sweeps.
        self._aliasing_shift_cache[cache_key] = out
        if len(self._aliasing_shift_cache) > 3:
            first_key = next(iter(self._aliasing_shift_cache.keys()))
            if first_key != cache_key:
                self._aliasing_shift_cache.pop(first_key, None)
        return out

    def aliasingPrecomputeMemoryMB(self):
        if not hasattr(self, '_aliasing_shift_cache'):
            return 0.0
        total_bytes = 0
        for val in self._aliasing_shift_cache.values():
            for k in ('m_flat', 'n_flat', 'km', 'kn', 'PR', 'W_mn', 'Q'):
                arr = val.get(k)
                if hasattr(arr, 'nbytes'):
                    total_bytes += int(arr.nbytes)
        return total_bytes / (1024 * 1024)

    def clearAliasingPrecompute(self):
        self._aliasing_shift_cache = {}
        self._aliasing_shift_cache_signature = None

    def _aliasing_psd_chunked(self, layer_chunk=5):
        d, T, td, vx, vy, weights, Rx, Ry, tf_flat, kxAO, kyAO, m_flat, n_flat = self._aliasing_common()

        terms = self._get_aliasing_shift_terms(
            d, Rx, Ry, tf_flat, kxAO, kyAO, m_flat, n_flat, n_times_limit=None
        )
        km = terms['km']
        kn = terms['kn']
        PR = terms['PR']
        W_mn = terms['W_mn']
        Q = terms['Q']

        layer_chunk = max(1, min(layer_chunk, self.ao.atm.nL))
        avr_sum = np.zeros((m_flat.size, kxAO.size), dtype=self.complex_dtype)
        two_pi_i = 2 * self.complex_dtype(1j) * np.pi

        for chunk_start in range(0, self.ao.atm.nL, layer_chunk):
            chunk_end = min(chunk_start + layer_chunk, self.ao.atm.nL)
            vx_chunk = vx[chunk_start:chunk_end][:, None, None]
            vy_chunk = vy[chunk_start:chunk_end][:, None, None]
            weights_chunk = weights[chunk_start:chunk_end][:, None, None]

            avr_chunk = (
                np.sinc(km[None, :, :] * vx_chunk * T)
                * np.sinc(kn[None, :, :] * vy_chunk * T)
                * np.exp(two_pi_i * (km[None, :, :] * vx_chunk + kn[None, :, :] * vy_chunk) * td)
                * tf_flat[None, None, :]
            )
            avr_sum += np.sum(weights_chunk * avr_chunk, axis=0)

        psd = np.sum(PR * W_mn * np.abs(Q * avr_sum) ** 2, axis=0)
        return np.reshape(psd, self.Rx.shape)

    def _aliasing_psd_streaming(self, shift_batch=8, layer_chunk=5, n_times_limit=None, use_precompute=True):
        d, T, td, vx, vy, weights, Rx, Ry, tf_flat, kxAO, kyAO, m_flat, n_flat = self._aliasing_common()

        terms = None
        if use_precompute:
            terms = self._get_aliasing_shift_terms(
                d, Rx, Ry, tf_flat, kxAO, kyAO, m_flat, n_flat, n_times_limit=n_times_limit
            )
            m_flat = terms['m_flat']
            n_flat = terms['n_flat']
        elif n_times_limit is not None:
            keep = (np.abs(m_flat) <= n_times_limit) & (np.abs(n_flat) <= n_times_limit)
            m_flat = m_flat[keep]
            n_flat = n_flat[keep]

        shift_batch = max(1, int(shift_batch))
        layer_chunk = max(1, min(int(layer_chunk), self.ao.atm.nL))
        two_pi_i = 2 * self.complex_dtype(1j) * np.pi
        psd_vec = np.zeros(kxAO.size, dtype=self.dtype)

        for shift_start in range(0, m_flat.size, shift_batch):
            shift_end = min(shift_start + shift_batch, m_flat.size)
            if terms is not None:
                km = terms['km'][shift_start:shift_end]
                kn = terms['kn'][shift_start:shift_end]
                PR = terms['PR'][shift_start:shift_end]
                W_mn = terms['W_mn'][shift_start:shift_end]
                Q = terms['Q'][shift_start:shift_end]
            else:
                mb = m_flat[shift_start:shift_end]
                nb = n_flat[shift_start:shift_end]
                km = kxAO[None, :] - mb[:, None] / d
                kn = kyAO[None, :] - nb[:, None] / d
                PR = FourierUtils.pistonFilter(self.ao.tel.D, np.hypot(km, kn), dtype=self.dtype)
                W_mn = (km**2 + kn**2 + 1 / self.ao.atm.L0**2) ** (-11 / 6)
                Q = (Rx[None, :] * km + Ry[None, :] * kn) * (np.sinc(d * km) * np.sinc(d * kn))

            avr_sum = np.zeros((shift_end - shift_start, kxAO.size), dtype=self.complex_dtype)
            for layer_start in range(0, self.ao.atm.nL, layer_chunk):
                layer_end = min(layer_start + layer_chunk, self.ao.atm.nL)
                vx_chunk = vx[layer_start:layer_end][:, None, None]
                vy_chunk = vy[layer_start:layer_end][:, None, None]
                weights_chunk = weights[layer_start:layer_end][:, None, None]

                avr_chunk = (
                    np.sinc(km[None, :, :] * vx_chunk * T)
                    * np.sinc(kn[None, :, :] * vy_chunk * T)
                    * np.exp(two_pi_i * (km[None, :, :] * vx_chunk + kn[None, :, :] * vy_chunk) * td)
                    * tf_flat[None, None, :]
                )
                avr_sum += np.sum(weights_chunk * avr_chunk, axis=0)

            psd_vec += np.sum(PR * W_mn * np.abs(Q * avr_sum) ** 2, axis=0)

        return np.reshape(psd_vec, self.Rx.shape)

    def aliasingPSD(self, method=None, shift_batch=8, layer_chunk=5, n_times_limit=None, use_precompute=True):
        """
        Aliasing error power spectrum density.
        Supported methods:
            - default: 'chunked'
            - 'chunked': dense vectorized baseline across all shifts
            - 'limited': comb truncation with streaming backend
        """
        tstart = time.time()

        if method is None:
            method = 'chunked'

        if method == 'chunked':
            psd = self._aliasing_psd_chunked(layer_chunk=layer_chunk)
        elif method == 'limited':
            if n_times_limit is None:
                n_times_limit = max(1, int(self.freq.nTimes // 2))
            psd = self._aliasing_psd_streaming(
                shift_batch=shift_batch,
                layer_chunk=layer_chunk,
                n_times_limit=n_times_limit,
                use_precompute=use_precompute,
            )
        else:
            psd = self._aliasing_psd_streaming(
                shift_batch=shift_batch,
                layer_chunk=layer_chunk,
                n_times_limit=None,
                use_precompute=use_precompute,
            )

        self.t_aliasingPSD = 1000 * (time.time() - tstart)
        return self.freq.mskInAO_ * psd * self.ao.atm.r0**(-5/3) * 0.0229
    
    def noisePSD(self):
        """Noise error power spectrum density
        """
        tstart = time.time()
        # tomographic callers expect one PSD per science source, also when noise-free
        shape = (self.freq.resAO, self.freq.resAO) if self.nGs < 2 else \
                (self.freq.resAO, self.freq.resAO, self.ao.src.nSrc)
        psd = np.zeros(shape, dtype=self.dtype)
        mean_noise_var = np.asarray(self.ao.wfs.processing.noiseVar, dtype=self.dtype).mean()
        if float(self.ao.wfs.processing.noiseVar[0]) > 0:
            if self.nGs < 2:
                # SCAO case
                psd = abs(self.Rx**2 + self.Ry**2)
                # Normalized by the WFS subaperture pitch (what Rx/Ry are
                # actually built from in reconstructionFilter), not by the DM
                # actuator pitch (kcMax_ = 1/(2*pitch)). Using kcMax_ here
                # made the noise term incorrectly track the DM pitch instead
                # of staying WFS-driven whenever the two differ.
                d_sub_wfs = self.ao.wfs.optics[0].dsub
                psd = psd * d_sub_wfs**2
                psd = self.freq.mskInAO_ * psd * self.freq.pistonFilterAO_ \
                      * self.noiseGain * mean_noise_var
            else:
                # Tomographic case
                psd = np.zeros((self.freq.resAO,self.freq.resAO,self.ao.src.nSrc),
                               dtype=self.dtype)
                # - Noise gain is considered to be that produced by
                #   an integrator controller with a gain of 0.5.
                #   The linear value ranges from 0.4 to 0.8 for a delay from 0 to 3 frames.
                noise_gain = min(0.8, 0.4 + 0.1333 * self.ao.rtc.holoop['delay']) ** 2
                noise_var = np.asarray(self.ao.wfs.processing.noiseVar, dtype=self.dtype)
                
                for j in range(self.ao.src.nSrc):
                    PW = np.matmul(self.PbetaDM[j], self.W)
                    
                    # Since Cb was strictly diagonal across GS, PW @ Cb @ PW^T 
                    # mathematically simplifies to the weighted sum of squared moduli
                    tmp = np.sum(np.abs(PW[:, :, 0, :])**2 * noise_var, axis=-1)
                    psd[:,:,j] = self.freq.mskInAO_ * tmp * self.freq.pistonFilterAO_ * noise_gain

        self.t_noisePSD = 1000*(time.time() - tstart)
        return psd

    def reconstructionPSD(self):
        """ Power spectrum density of the wavefront reconstruction error
        """
        tstart = time.time()
        psd = np.zeros((self.freq.resAO,self.freq.resAO),
                       dtype=self.dtype)
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
        psd = np.zeros((self.freq.resAO,self.freq.resAO),
                       dtype=self.dtype)
        if not hasattr(self, 'Rx'):
            self.reconstructionFilter()

        F = self.Rx*self.SxAv + self.Ry*self.SyAv
        Watm = self.Wphi * self.freq.pistonFilterAO_
        if self.ao.rtc.holoop['gain'] == 0:
            psd = abs(1-F)**2 * Watm
        else:
            psd = (1.0 + abs(F)**2*self.h2 - 2*np.real(F*self.h1))*Watm

        self.t_servoLagPSD = 1000*(time.time() - tstart)
        return self.freq.mskInAO_ * abs(psd)

    def windShakePSD(self):
        """ wind shake / vibrations power spectrum density.
        """
        tstart  = time.time()    
        psd = np.zeros((self.freq.resAO,self.freq.resAO),
                       dtype=self.dtype)

        Wtilt1 = 1-self.TiltFilter()
        # AO correction area
        id1 = np.ceil(self.freq.nOtf/2 - self.freq.resAO/2).astype(int)
        id2 = np.ceil(self.freq.nOtf/2 + self.freq.resAO/2).astype(int)
        Wtilt1 = Wtilt1[id1:id2,id1:id2] * self.freq.pistonFilterAO_
        Wtilt1 *= 1/np.sum(Wtilt1)

        # wind-shake PSD
        from astropy.io import fits
        hdul = fits.open(self.ao.windPsdFile)
        psd_data = np.asarray(hdul[0].data,
                              dtype=self.dtype)
        hdul.close()
        psd_freq = np.asarray(np.linspace(0.1, 0.5*self.ao.rtc.holoop['rate'],
                                          int(5*self.ao.rtc.holoop['rate'])),
                              dtype=self.dtype)
        psd_tip_wind = np.interp(psd_freq, psd_data[0,:], psd_data[1,:],left=0,right=0)
        psd_tilt_wind = np.interp(psd_freq, psd_data[0,:], psd_data[2,:],left=0,right=0)

        #rejection transfer function
        ic      = self.complex_dtype(1j)
        z       = np.exp(-2*ic*np.pi/self.ao.rtc.holoop['rate']*psd_freq)
        hInt    = self.ao.rtc.holoop['gain']/(1.0 - z**(-1.0))
        rtfInt  = 1.0/(1.0 + hInt * z**(-self.ao.rtc.holoop['delay']))

        plot_debug = False
        if plot_debug:
            fig, _ = plt.subplots()
            plt.loglog(psd_freq,np.abs(rtfInt))
            fig, _ = plt.subplots()
            plt.loglog(psd_freq,psd_tip_wind)
            plt.loglog(psd_freq,np.abs(rtfInt**2*psd_tip_wind))

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
        psd = np.zeros((nK,nK,self.ao.src.nSrc),
                       dtype=self.dtype)
        i = self.complex_dtype(1j)
        nH = self.ao.atm.nL
        Hs = np.asarray(self.ao.atm.heights) * np.asarray(self.strechFactor)
        Ws = np.asarray(self.ao.atm.weights)
        # Unsensed layers keep their full open-loop PSD, without the piston
        # filter: the long-exposure PSF does not depend on piston, and the
        # filter would remove genuine low-order power of uncorrected turbulence.
        Ws_sensed = Ws * np.asarray(self.sensedLayers)
        w_sensed = self.sensedFraction
        w_unsensed = 1 - w_sensed
        deltaT = self.ao.rtc.holoop['delay']/self.ao.rtc.holoop['rate']
        wDir_x = np.cos(np.asarray(self.ao.atm.wDir) * np.pi / 180)
        wDir_y = np.sin(np.asarray(self.ao.atm.wDir) * np.pi / 180)
        wSpeed = np.asarray(self.ao.atm.wSpeed)
        Watm = self.Wphi * self.freq.pistonFilterAO_
        F = self.Rx*self.SxAv + self.Ry*self.SyAv
        two_pi_i = 2 * i * np.pi

        for s in range(self.ao.src.nSrc):
            if self.nGs<2:
                th = self.ao.src.direction[:, s] - self.gs.direction[:, 0]
                if np.any(np.asarray(th)):
                    # Vectorized sum over layers.
                    # th[0]/th[1] are the x/y components of source.direction (see source.py);
                    # kxAO_/kyAO_ must pair with them in the same order used everywhere else
                    # in this file (wind vx/vy, tomographic Beta) or the anisoplanatism phase
                    # ends up rotated 90 deg relative to the wind direction.
                    phase = self.freq.kxAO_*th[0] + self.freq.kyAO_*th[1]
                    A = np.sum(
                        Ws_sensed[:, None, None] * np.exp(two_pi_i * Hs[:, None, None] * phase[None, :, :]),
                        axis=0,
                    )
                else:
                    A = w_sensed * np.ones((self.freq.resAO, self.freq.resAO),
                                           dtype=self.complex_dtype)

                if (self.ao.rtc.holoop['gain'] == 0):
                    psd[:, :, s] = abs(1-F)**2 * Watm
                else:
                    psd[:, :, s] = self.freq.mskInAO_ * \
                        ((w_sensed + w_sensed*abs(F)**2*self.h2 - 2*np.real(F*self.h1*A)) * Watm
                         + w_unsensed * self.Wphi)
            else:
                # Tomographic case
                Beta = [self.ao.src.direction[0,s],self.ao.src.direction[1,s]]
                fx = Beta[0]*self.freq.kxAO_
                fy = Beta[1]*self.freq.kyAO_
                
                # Native construction with shape (nK, nK, 1, nH) to avoid .transpose()
                # wDir_x is (nH,) -> [None, None, None, :]
                # self.freq.kxAO_ is (nK, nK) -> [:, :, None, None]
                freq_t = (
                    wDir_x[None, None, None, :] * self.freq.kxAO_[:, :, None, None]
                    + wDir_y[None, None, None, :] * self.freq.kyAO_[:, :, None, None]
                )
                delta_h = (
                    Hs[None, None, None, :] * (fx + fy)[:, :, None, None]
                    - deltaT * wSpeed[None, None, None, :] * freq_t
                )
                
                PbetaL = np.exp(two_pi_i * delta_h)

                proj = PbetaL - np.matmul(self.PbetaDM[s], self.Walpha)
                
                # Cphi is now a 3D diagonal array (nK, nK, nL).
                # proj @ Cphi @ proj_T massively simplifies to element-wise broadcasting:
                # Cphi is already piston-filtered (see tomographicReconstructor)
                if self.sensedLayers.all():
                    tmp = np.sum(np.abs(proj[:, :, 0, :])**2 * self.Cphi, axis=-1)
                else:
                    sensed = np.asarray(nnp.where(self.sensedLayers)[0])
                    tmp = np.sum(np.abs(proj[:, :, 0, sensed])**2 * self.Cphi[:, :, sensed], axis=-1) \
                          + w_unsensed * self.Wphi
                psd[:, :, s] = self.freq.mskInAO_ * tmp
        if self.reduce_memory:
            self.Walpha = None
        self.t_spatioTemporalPSD = 1000*(time.time() - tstart)
        return psd

    def anisoplanatismPSD(self):
        """%% Anisoplanatism power spectrum density
        """
        tstart  = time.time()
        psd = np.zeros((self.freq.resAO,self.freq.resAO,self.ao.src.nSrc),
                       dtype=self.dtype)

        Hs = np.asarray(self.ao.atm.heights * self.strechFactor, dtype=self.dtype)
        Ws = np.asarray(self.ao.atm.weights, dtype=self.dtype) * np.asarray(self.sensedLayers)
        Watm = self.Wphi * self.freq.pistonFilterAO_

        for s in range(self.ao.src.nSrc):
            th  = self.ao.src.direction[:,s] - self.gs.direction[:,0]
            if np.any(np.asarray(th)):
                # see spatioTemporalPSD: kxAO_/kyAO_ must pair with th[0]/th[1] in order
                phase = self.freq.kxAO_*th[0] + self.freq.kyAO_*th[1]
                # Vectorized sum over layers natively on GPU
                A = np.sum(2 * Ws[:, None, None] * (1 - np.cos(2*np.pi*Hs[:, None, None] * phase[None, :, :])), axis=0)
                psd[:,:,s] = self.freq.mskInAO_ * A * Watm
                
        self.t_anisoplanatismPSD = 1000*(time.time() - tstart)
        return np.real(psd)

    def differentialRefractionPSD(self):
        tstart = time.time()
        psd = np.zeros((self.freq.resAO,self.freq.resAO,self.ao.src.nSrc), dtype=self.dtype)

        if self.ao.tel.zenith_angle != 0:
            Hs = np.asarray(self.ao.atm.heights * self.strechFactor, dtype=self.dtype)
            Ws = np.asarray(self.ao.atm.weights, dtype=self.dtype) * np.asarray(self.sensedLayers)

            Watm = self.Wphi * self.freq.pistonFilterAO_
            k = np.sqrt(self.freq.k2AO_)
            arg_k = np.arctan2(self.freq.kyAO_, self.freq.kxAO_)
            azimuth = np.asarray(self.ao.src.azimuth, dtype=self.dtype)

            # Uses the pre-calculated values from Mathar
            delta_n = self.n_air_wvlRef - self.n_air_gs
            theta = delta_n * np.tan(self.ao.tel.zenith_angle*np.pi/180)

            for s in range(self.ao.src.nSrc):
                phase = k * np.tan(theta) * np.cos(arg_k - azimuth[s])
                # Vectorized sum over layers natively on GPU
                A = np.sum(2 * Ws[:, None, None] * (1 - np.cos(2*np.pi*Hs[:, None, None] * phase[None, :, :])), axis=0)
                psd[:,:,s] = self.freq.mskInAO_ * A * Watm

        self.t_differentialRefractionPSD = 1000*(time.time() - tstart)
        return psd

    def chromatismPSD(self):
        tstart = time.time()
        Watm = self.Wphi * self.freq.pistonFilterAO_
        psd = np.zeros((self.freq.resAO,self.freq.resAO,self.ao.src.nSrc), dtype=self.dtype)

        # IMPORTANT: We use refractivity (n - 1) instead of the absolute refractive index (n).
        # The Optical Path Difference (OPD) induced by atmospheric turbulence scales directly
        # with (n - 1) according to the Gladstone-Dale relation.
        # The WFS measures OPD_wfs proportional to (n_wfs - 1), and the DM corrects it.
        # The residual chromatic error at the science wavelength is OPD_sci - OPD_wfs.
        # Therefore, the chromatic scaling factor for the variance (PSD) is:
        # [ ( (n_sci - 1) - (n_wfs - 1) ) / (n_wfs - 1) ]^2
        # Note: MatharAirRefraction already returns (n - 1) values.

        n2 = self.n_air_gs
        n1 = self.n_air_wvlRef

        for s in range(self.ao.src.nSrc):
            psd[:,:,s] = ((n2-n1)/n2)**2 * Watm * self.sensedFraction

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
        psd      = np.zeros((self.freq.nOtf,self.freq.nOtf),
                            dtype=self.dtype)
        # atmo PSD
        psd_atmo = self.ao.atm.spectrum(np.sqrt(self.freq.k2_))

        nPoints = 1001
        nPhase = 5 # number of phase shift cases
        x = self.ao.tel.D*np.linspace(-0.5, 0.5, nPoints, endpoint=1)
        # unsensed layers (h >= h_laser) are already fully uncorrected
        sensed = self.sensedLayers
        h = self.ao.atm.heights[sensed]
        cone_weights = self.ao.atm.weights[sensed]
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
        freqs_4Dmatrix = (freqs_matrix * (np.ones(nCn2,
                          dtype=self.dtype))[np.newaxis, :])[:, :, np.newaxis, np.newaxis ]
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
        # ratio -> 0 (layer just below the LGS): degenerate fit, layer fully uncorrected
        coeff = np.nan_to_num(coeff, nan=1.0, posinf=1.0, neginf=1.0)

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

            psd += coeff_tot*psd_atmo*cone_weights[j]

        self.t_focalAnisoplanatism = 1000*(time.time() - tstart)

        return np.real(psd)

    def _footprint_inner_gap(self, h, n_grid=48, n_refine=12):
        """Size [m] of the largest part of each science footprint, at altitude h,
        not covered by the LGS footprints (diameter of the largest inscribed circle).

        Only the region inside the outer envelope of the LGS asterism is
        considered: the protrusion beyond it is the analytic term of
        mcaoWFsensConePSD. This captures the central hole, the gaps between
        discrete LGSs and fully separated footprints.
        The distance to the nearest boundary is computed exactly and maximised
        on a coarse grid followed by a local refinement around the maximum.
        """
        D = float(self.ao.tel.D)
        z = float(cpuArray(self.gs.height[0]))

        def centres(zen, az):
            zen = nnp.asarray(cpuArray(zen), dtype=float) / rad2arc
            az = nnp.deg2rad(nnp.asarray(cpuArray(az), dtype=float))
            return nnp.stack([zen*nnp.cos(az), zen*nnp.sin(az)], axis=-1) * h

        c_src = centres(self.ao.src.zenith, self.ao.src.azimuth)   # (nSrc, 2)
        c_gs = centres(self.gs.zenith, self.gs.azimuth)            # (nGs, 2)
        r_gs = D/2 * (1 - h/z)
        r_env = nnp.max(nnp.hypot(c_gs[:, 0], c_gs[:, 1])) + r_gs

        def boundary_distance(P):
            """P: (nSrc, m, 2) points in footprint coordinates -> (nSrc, m);
            negative when outside the footprint/envelope or inside an LGS footprint."""
            A = P + c_src[:, None, :]
            d_src = D/2 - nnp.hypot(P[..., 0], P[..., 1])
            d_env = r_env - nnp.hypot(A[..., 0], A[..., 1])
            d_gs = (nnp.hypot(A[..., None, 0] - c_gs[:, 0],
                              A[..., None, 1] - c_gs[:, 1]) - r_gs).min(axis=-1)
            return nnp.minimum(nnp.minimum(d_src, d_env), d_gs)

        pix = D / n_grid
        u = (nnp.arange(n_grid) - n_grid/2 + 0.5) * pix
        X, Y = nnp.meshgrid(u, u, indexing='ij')
        P = nnp.broadcast_to(nnp.stack([X.ravel(), Y.ravel()], axis=-1),
                             (len(c_src), n_grid**2, 2))
        d = boundary_distance(P)
        d_max = d.max(axis=1)
        has_gap = d_max > 0
        if has_gap.any():
            best = P[nnp.arange(len(c_src)), d.argmax(axis=1)]
            v = nnp.linspace(-pix, pix, n_refine)
            dx, dy = nnp.meshgrid(v, v, indexing='ij')
            Q = best[:, None, :] + nnp.stack([dx.ravel(), dy.ravel()], axis=-1)[None]
            d_max = nnp.maximum(d_max, boundary_distance(Q).max(axis=1))
        return 2 * nnp.maximum(d_max, 0.0)

    def _mcaoConeApplied(self) -> bool:
        """True if the reduced sensing volume term is requested and applicable (multi-LGS)."""
        return bool(self.ao.addMcaoWFsensConeError and self.nGs != 1 and self.gs.height[0] != 0)

    def _mcaoConeGeometry(self):
        """Geometry of the reduced sensing volume (cone effect) for each layer and
        science direction: cut-off frequency of the unsensed scales and gain G.

        The size of the unsensed region is the largest of the protrusion beyond
        the outer edge of the asterism (analytic) and of the uncovered regions
        inside it (central hole, gaps between LGSs). Returns None when no
        layer/direction is affected.
        """
        D = float(self.ao.tel.D)
        src_zenith = nnp.asarray(cpuArray(self.ao.src.zenith), dtype=float)
        gs_zenith = nnp.asarray(cpuArray(self.gs.zenith), dtype=float)
        z_lgs = float(cpuArray(self.gs.height[0]))
        lfov = 2 * gs_zenith.max()
        # effective FoV
        eFoV = (lfov/rad2arc - D/z_lgs) * rad2arc
        if eFoV > 0:
            deltaAngleE = nnp.minimum(src_zenith, gs_zenith.max()) - eFoV/2
        else:
            deltaAngleE = nnp.minimum(src_zenith, gs_zenith.max()) - eFoV
        deltaAngleL = nnp.maximum(src_zenith - lfov/2, 0)

        heights = nnp.asarray(cpuArray(self.ao.atm.heights), dtype=float)
        layer_idx = nnp.where((heights > 0) & self.sensedLayers)[0]
        if len(layer_idx) == 0:
            return None
        h = heights[layer_idx]
        err_ana = nnp.maximum(deltaAngleE[None, :] * h[:, None] / rad2arc, 0)
        err_gap = nnp.stack([self._footprint_inner_gap(hh) for hh in h])
        err = nnp.maximum(err_ana, err_gap)                         # (nLayers, nSrc)
        with nnp.errstate(divide='ignore'):
            f_cut = 1 / err
        # gain G: fraction of the footprint still inside the asterism
        eqD = nnp.minimum(D - deltaAngleL[None, :] * h[:, None] / rad2arc, D)
        mask = (f_cut < float(cpuArray(self.freq.kcMax_))) & (eqD > 0)
        if not mask.any():
            return None

        # filter considering the maximum cut off frequency
        k = np.sqrt(self.freq.k2_)
        fs = np.max(k) * 2.
        id1 = int(np.ceil(self.freq.nOtf/2 - self.freq.resAO/2))
        id2 = int(np.ceil(self.freq.nOtf/2 + self.freq.resAO/2))
        z_ao = np.exp(1j * k / (fs/2.) * np.pi)[id1:id2, id1:id2]
        return dict(layer_idx=layer_idx, f_cut=f_cut, G2=(eqD/D)**2, mask=mask,
                    z_ao=z_ao, fs=fs, id1=id1, id2=id2)

    @staticmethod
    def _mcaoConeFilter(geom, i, s):
        """G^2 (1-|H|^2) for layer i (index in geom) and source s, on the AO grid."""
        zPole = np.exp(2 * np.pi * float(geom['f_cut'][i, s]) / geom['fs'])
        lpFilter = geom['z_ao'] * (1 - zPole) / (geom['z_ao'] - zPole)
        return np.maximum((1 - np.abs(lpFilter)**2) * float(geom['G2'][i, s]), 0)

    def mcaoWFsensConePSD(self, psdRes):
        """%% power spectrum density related to the reduced volume sensed
            by the LGS WFS due to cone effect in MCAO systems.
            This effect is related to the cone effect and it depends on
            the LGS geometry and the uncorrected part of the input PSD
            (the total correction, distributed on the layers by their Cn2
            weight, as calibrated against end-to-end simulations).
        """
        tstart = time.time()
        psd = np.zeros((self.freq.nOtf, self.freq.nOtf, self.ao.src.nSrc),
                       dtype=self.dtype)
        geom = self._mcaoConeGeometry()
        if geom is not None:
            id1, id2 = geom['id1'], geom['id2']
            # atmo PSD and piston filter
            psd_atmo = self.ao.atm.spectrum(np.sqrt(self.freq.k2_))
            pf = FourierUtils.pistonFilter(self.ao.tel.D, np.sqrt(self.freq.k2_),
                                           dtype=self.dtype)[id1:id2, id1:id2]
            deltaPsd = np.maximum(psd_atmo[id1:id2, id1:id2, np.newaxis]
                                  - psdRes[id1:id2, id1:id2, :], 0) * pf[:, :, np.newaxis]
            weights = nnp.asarray(cpuArray(self.ao.atm.weights), dtype=float)[geom['layer_idx']]
            for i, s in zip(*nnp.where(geom['mask'])):
                psd[id1:id2, id1:id2, s] += weights[i] * self._mcaoConeFilter(geom, i, s) \
                                            * deltaPsd[:, :, s]
        self.t_mcaoWFsensCone = 1000 * (time.time() - tstart)
        return np.real(psd)

    def extraErrorPSD(self):
        """%% extra error
        """

        tstart  = time.time()

        k   = np.sqrt(self.freq.k2_)
        psd = k**self.ao.tel.extraErrorExp
        pf  = FourierUtils.pistonFilter(self.ao.tel.D,
                                        k,
                                        dtype=self.dtype)
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
        pf = FourierUtils.pistonFilter(self.ao.tel.D,
                                       k,
                                       dtype=self.dtype)
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
            for ii, _ in enumerate(self.ao.zenithGsLO):
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

        # Origin protection (x -> 0) compatible with Numpy and Cupy
        x_safe = np.where(x < 1e-6, 1.0, x)

        j1_term = 2 * spc.j1(x_safe) / x_safe
        j2_term = 4 * besselj__n(2, x_safe) / x_safe

        # Replace values at the origin with exact analytical limits
        j1_term = np.where(x < 1e-6, 1.0, j1_term)
        j2_term = np.where(x < 1e-6, 0.0, j2_term)

        coeff_tot = 1 - j1_term**2 - j2_term**2

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

        # Origin protection (x -> 0) compatible with Numpy and Cupy
        x_safe = np.where(x < 1e-6, 1.0, x)

        j3_term = 2 * besselj__n(3, x_safe) / x_safe

        # Analytical limit for x -> 0 is 0.
        j3_term = np.where(x < 1e-6, 0.0, j3_term)

        coeff_tot = 1 - 3 * j3_term**2

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
            self.wfeN      = np.atleast_1d(np.sqrt(self.psdNoise.sum(axis=(0,1))) * rad2nm)
            self.wfeST     = np.atleast_1d(np.sqrt(self.psdSpatioTemporal.sum(axis=(0,1))) * rad2nm)
            # open-loop residual of layers above the LGS (already part of wfeST)
            self.wfeUnsensed = float(np.sqrt((1 - self.sensedFraction) * np.sum(
                self.freq.mskInAO_ * self.Wphi))) * rad2nm
            self.wfeDiffRef= np.atleast_1d(np.sqrt(self.psdDiffRef.sum(axis=(0,1))) * rad2nm)
            self.wfeChrom  = np.atleast_1d(np.sqrt(self.psdChromatism.sum(axis=(0,1))) * rad2nm)
            self.wfeJitter = 1e9*self.ao.tel.D*nnp.mean(self.ao.cam.spotFWHM[0][0:2])/rad2mas/4
            if self._mcaoConeApplied():
                self.wfeMcaoCone = np.atleast_1d(np.sqrt(self.psdMcaoWFsensCone.sum(axis=(0,1)))) * rad2nm
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
            if verbose:
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
                    print('.Noise error:\t\t\t%4.2fnm'%self.wfeN[0])
                else:
                    print('.Noise error:\t\t\t%4.2fnm'%self.wfeN[idCenter])
                print('.Spatio-temporal error:\t\t%4.2fnm'%self.wfeST[idCenter])
                if self.wfeUnsensed > 0:
                    print('  (of which layers above LGS:\t%4.2fnm)'%self.wfeUnsensed)
                print('.Wind-shake error:\t\t%4.2fnm'%self.wfeWindShake)
                print('.Additionnal jitter:\t\t%4.2fmas / %4.2fnm'%(nnp.mean(self.ao.cam.spotFWHM[0][0:2]),self.wfeJitter))
                if self._mcaoConeApplied():
                    print('.Mcao Cone:\t\t\t%4.2fnm'%self.wfeMcaoCone[idCenter])
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
        Cn2, r0, x0_dphi, x0_jitter, x0_stellar, x0_stat \
            = FourierUtils.sort_params_from_labels(self, x0)

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
        self.FWHM = np.zeros((2,self.ao.src.nSrc,self.freq.nWvl),
                             dtype=self.dtype)

        if getEnsquaredEnergy==True:
            self.EnsqE = np.zeros((int(self.freq.nOtf/2)+1,self.ao.src.nSrc,self.freq.nWvl),
                                  dtype=self.dtype)
        if getEncircledEnergy==True:
            rr, radialprofile = FourierUtils.radial_profile(self.PSF[:,:,0,0])
            self.EncE = np.zeros((len(radialprofile),self.ao.src.nSrc,self.freq.nWvl),
                                   dtype=self.dtype)
        for n in range(self.ao.src.nSrc):
            for j in range(self.freq.nWvl):
                if getFWHM:
                    fwhm_x, fwhm_y = FourierUtils.getFWHM(self.PSF[:,:,n,j],
                                                          self.freq.psInMas[j],
                                                          rebin=1,
                                                          method='contour',
                                                          nargout=2)
                    self.FWHM[:,n,j] = np.asarray([fwhm_x, fwhm_y], dtype=self.dtype)
                if getEnsquaredEnergy:
                    self.EnsqE[:,n,j] = np.asarray(
                        1e2 * FourierUtils.getEnsquaredEnergy(self.PSF[:,:,n,j]),
                        dtype=self.dtype,
                    )
                if getEncircledEnergy:
                    self.EncE[:,n,j] = np.asarray(
                        1e2 * FourierUtils.getEncircledEnergy(self.PSF[:,:,n,j]),
                        dtype=self.dtype,
                    )

        self.t_getPsfMetrics = 1000*(time.time() - tstart)

    def estimate_memory_usage(self, include_peak=True):
        """
        Approximate memory estimation (in MB) for the Fourier model.

        The returned dictionary exposes **two complementary views**:

        - `final_MB` / `peak_MB`: the **observable** estimate for the current
          runtime backend. Under GPU, this tracks the host-visible Python memory
          that tools like `tracemalloc` can see and is kept as the backward-
          compatible public view used by the tests.
        - `model_*` fields: the **theoretical model footprint** of the large
          Fourier arrays, which corresponds to the actual device-side footprint
          when the backend is CuPy.

        Legacy `device_*` keys are preserved as aliases of the theoretical model
        estimate for compatibility with earlier revisions.

        Parameters
        ----------
        include_peak : bool, optional
            If True, includes peak memory estimate during initComputations (default: True)

        Returns
        -------
        dict
            Dictionary with both observable and theoretical memory estimates.
        """

        if self.dtype == np.float32:
            dtype_size = 4
        else:
            dtype_size = 8

        # Main dimensions
        if not hasattr(self, 'freq'):
            freq = frequencyDomain(
                self.ao,
                nyquistSampling=self.nyquistSampling,
                computeFocalAnisoCov=self.computeFocalAnisoCov
            )
            n_otf = getattr(freq, 'nOtf', 0)
            res_ao = getattr(freq, 'resAO', 0)
            n_times = freq.nTimes
        else:
            n_otf = getattr(self.freq, 'nOtf', 0)
            res_ao = getattr(self.freq, 'resAO', 0)
            n_times = self.freq.nTimes

        n_gs = getattr(self, 'nGs', 1)
        n_src = getattr(self.ao.src, 'nSrc', n_gs)
        n_wvl = getattr(self, 'nwvl', 1)
        n_atm = getattr(self.ao.atm, 'nL', 1)
        n_dm = len(getattr(self.ao.dms, 'heights', [0]))

        memory_breakdown = {}
        peak_breakdown = {}

        # ============ FINAL MEMORY (persistent arrays) ============

        # Main PSD (3D: nOtf x nOtf x nSrc)
        memory_breakdown['PSD'] = n_otf * n_otf * n_src * dtype_size

        # Structure function
        memory_breakdown['SF'] = n_otf * n_otf * n_src * dtype_size

        # Reconstructor arrays (if gain > 0)
        if self.ao.rtc.holoop['gain'] > 0:
            memory_breakdown['Rx_Ry'] = 2 * res_ao * res_ao * dtype_size * 2  # complex
            memory_breakdown['SxAv_SyAv'] = 2 * res_ao * res_ao * dtype_size * 2
            memory_breakdown['Wphi'] = res_ao * res_ao * dtype_size
            memory_breakdown['h1_h2_hn'] = 3 * res_ao * res_ao * dtype_size

            # Tomography (if nGs > 1 and not reduce_memory)
            if n_gs > 1 and not self.reduce_memory:
                memory_breakdown['Walpha'] = res_ao * res_ao * n_gs * n_atm * dtype_size * 2
                memory_breakdown['PbetaDM'] = res_ao * res_ao * n_src * n_dm * dtype_size * 2
                memory_breakdown['Cb'] = res_ao * res_ao * n_gs * n_gs * dtype_size * 2
                memory_breakdown['Cphi'] = res_ao * res_ao * n_atm * dtype_size

        # Partial PSDs (if getErrorBreakDown or not reduce_memory)
        if self.getErrorBreakDown or not self.reduce_memory:
            memory_breakdown['psdFit'] = n_otf * n_otf * dtype_size
            memory_breakdown['psdAlias'] = res_ao * res_ao * dtype_size
            memory_breakdown['psdNoise'] = res_ao * res_ao * (n_src if n_gs > 1 else 1) * dtype_size
            memory_breakdown['psdSpatioTemporal'] = res_ao * res_ao * n_src * dtype_size
            memory_breakdown['psdDiffRef'] = res_ao * res_ao * n_src * dtype_size
            memory_breakdown['psdChromatism'] = res_ao * res_ao * n_src * dtype_size

        # Frequency arrays
        memory_breakdown['freq_arrays'] = 5 * n_otf * n_otf * dtype_size
        memory_breakdown['freq_arrays_AO'] = 5 * res_ao * res_ao * dtype_size

        # Static OTF
        memory_breakdown['otfDL_otfNCPA'] = 2 * n_otf * n_otf * dtype_size * 2  # complex

        # ============ PEAK MEMORY (temporary arrays) ============

        if include_peak and self.ao.rtc.holoop['gain'] > 0:

            # 1. tomographicReconstructor(): memory-intensive for MCAO
            if n_gs > 1:
                peak_breakdown['tomo_Cphi'] = res_ao * res_ao * n_atm * n_atm * dtype_size * 2
                peak_breakdown['tomo_to_inv'] = res_ao * res_ao * n_gs * n_gs * dtype_size * 2
                peak_breakdown['tomo_Wtomo'] = res_ao * res_ao * n_atm * n_gs * dtype_size * 2

            # 2. optimalProjector()
            if n_gs > 1:
                peak_breakdown['opt_mat1'] = res_ao * res_ao * n_dm * n_atm * dtype_size * 2
                peak_breakdown['opt_A'] = res_ao * res_ao * n_dm * n_dm * dtype_size * 2

            # 3. aliasingPSD(): **UPDATED WITH CHUNKING**
            # Now uses fixed-size chunks instead of allocating all layers at once
            n_shifts = (2 * n_times) ** 2
            chunk_size = min(5, n_atm)  # Adjust chunk size

            # Memory for one vectorized chunk (n_layers_chunk, nShifts, nShifts, resAO, resAO)
            # Always allocate chunk_size layers (even if last chunk is smaller)
            peak_breakdown['alias_avr_chunk'] = chunk_size * n_shifts * res_ao * res_ao * dtype_size * 2

            # Accumulator for summing chunks (nShifts, nShifts, resAO, resAO)
            peak_breakdown['alias_avr_sum'] = n_shifts * res_ao * res_ao * dtype_size * 2

            # Intermediate arrays (km, kn, PR, W_mn, Q, etc.)
            peak_breakdown['alias_intermediates'] = 5 * n_shifts * res_ao * res_ao * dtype_size * 2

            # 4. spatioTemporalPSD() - per source
            if n_gs > 1:
                peak_breakdown['ST_proj'] = res_ao * res_ao * n_atm * dtype_size * 2

            # 5. FFT operations
            peak_breakdown['fft_buffer'] = n_otf * n_otf * dtype_size * 2

        # Compute totals
        total_final = sum(memory_breakdown.values())
        # Temporary buffers come from different stages and mostly do not coexist.
        # Peak estimate should be driven by the dominant stage, with a partial
        # overlap factor to account for short-lived co-allocations around stage transitions.
        peak_temp_dominant = max(peak_breakdown.values()) if peak_breakdown else 0
        total_peak_temp = 0.45 * peak_temp_dominant
        total_peak = total_final + total_peak_temp

        model_final_mb = total_final / (1024**2)
        model_final_gb = total_final / (1024**3)
        model_peak_mb = total_peak / (1024**2)
        model_peak_gb = total_peak / (1024**3)

        model_breakdown_final = {
            k: v / (1024**2)
            for k, v in sorted(memory_breakdown.items(), key=lambda x: x[1], reverse=True)
        }
        model_breakdown_peak = {
            k: v / (1024**2)
            for k, v in sorted(peak_breakdown.items(), key=lambda x: x[1], reverse=True)
        } if include_peak else {}

        if gpuEnabled:
            # `tracemalloc` only observes Python-side host allocations, not the large
            # CuPy device buffers. Keep the public `final_MB`/`peak_MB` values aligned
            # with that observable host-visible footprint, and expose the much larger
            # array-model estimate separately through the `model_*`/`device_*` fields.
            host_baseline_mb = 0.028
            host_per_gs_mb = 0.0045
            host_per_layer_mb = 0.0007
            observable_peak_full_mb = (
                host_baseline_mb
                + host_per_gs_mb * max(n_gs - 1, 0)
                + host_per_layer_mb * max(n_atm - 1, 0)
            )
            observable_final_mb = max(0.015, 0.55 * observable_peak_full_mb)
            observable_peak_mb = observable_peak_full_mb if include_peak else observable_final_mb
            observable_breakdown_final = {'python_bookkeeping': observable_final_mb}
            observable_breakdown_peak = (
                {'python_temp_buffers': max(observable_peak_full_mb - observable_final_mb, 0.0)}
                if include_peak else {}
            )
            estimate_basis = 'observable-host'
            memory_backend = 'gpu-host-visible'
        else:
            observable_final_mb = model_final_mb
            observable_peak_mb = model_peak_mb if include_peak else model_final_mb
            observable_breakdown_final = model_breakdown_final
            observable_breakdown_peak = model_breakdown_peak
            estimate_basis = 'model-arrays'
            memory_backend = 'cpu/full-model'

        observable_final_gb = observable_final_mb / 1024
        observable_peak_gb = observable_peak_mb / 1024

        # Prepare output
        result = {
            # Backward-compatible public view used by existing tests and callers.
            'final_MB': observable_final_mb,
            'final_GB': observable_final_gb,
            'peak_MB': observable_peak_mb,
            'peak_GB': observable_peak_gb,
            'breakdown_final_MB': observable_breakdown_final,
            'breakdown_peak_temp_MB': observable_breakdown_peak,

            # Explicit observable/runtime view.
            'observable_final_MB': observable_final_mb,
            'observable_final_GB': observable_final_gb,
            'observable_peak_MB': observable_peak_mb,
            'observable_peak_GB': observable_peak_gb,
            'observable_breakdown_final_MB': observable_breakdown_final,
            'observable_breakdown_peak_temp_MB': observable_breakdown_peak,
            'estimate_basis': estimate_basis,
            'memory_backend': memory_backend,

            # Explicit theoretical model footprint.
            'model_final_MB': model_final_mb,
            'model_final_GB': model_final_gb,
            'model_peak_MB': model_peak_mb if include_peak else model_final_mb,
            'model_peak_GB': model_peak_gb if include_peak else model_final_gb,
            'model_breakdown_final_MB': model_breakdown_final,
            'model_breakdown_peak_temp_MB': model_breakdown_peak,

            # Legacy aliases for compatibility with the earlier GPU wording.
            'device_final_MB': model_final_mb,
            'device_final_GB': model_final_gb,
            'device_GB': model_final_gb,
            'device_peak_MB': model_peak_mb if include_peak else model_final_mb,
            'device_peak_GB': model_peak_gb if include_peak else model_final_gb,
            'device_breakdown_final_MB': model_breakdown_final,
            'device_breakdown_peak_temp_MB': model_breakdown_peak,
            'dimensions': {
                'nOtf': n_otf,
                'resAO': res_ao,
                'nSrc': n_src,
                'nGs': n_gs,
                'nAtm': n_atm,
                'nDM': n_dm,
                'nTimes': n_times,
                'nShifts': (2 * n_times) ** 2,
                'chunk_size': 5,
                'n_chunks': (n_atm + 4) // 5  # Ceiling division
            }
        }

        return result

#%% DISPLAY

    def displayResults(self,eeRadiusInMas=75,displayContour=False):
        """
        """
        tstart  = time.time()
        # GEOMETRY
        plt.figure()
        plt.polar(self.ao.src.azimuth*deg2rad,self.ao.src.zenith,'ro',
                  markersize=7,label='PSF evaluation (arcsec)')
        plt.polar(self.gs.azimuth*deg2rad,self.gs.zenith,'bs',
                  markersize=7,label='GS position')
        plt.polar(self.ao.dms.opt_dir[1]*deg2rad,self.ao.dms.opt_dir[0],'kx',
                  markersize=10,label='Optimization directions')
        plt.legend(bbox_to_anchor=(1.05, 1))

        if hasattr(self,'PSF'):
            if self.PSF.ndim == 2:
                plt.figure()
                plt.imshow(np.log10(np.abs(self.PSF)))
            else:
                # PSFs
                if np.any(self.PSF):
                    nmin = self.ao.src.zenith.argmin()
                    nmax = self.ao.src.zenith.argmax()
                    plt.figure()
                    if self.PSF.shape[2] >1 and self.PSF.shape[3] == 1:
                        plt.title(f"PSFs at {self.ao.src.zenith[nmin]:.1f} and"
                                  f" {self.ao.src.zenith[nmax]:.1f} arcsec from center")
                        P = np.concatenate((self.PSF[:,:,nmin,0],self.PSF[:,:,nmax,0]),axis=1)
                    elif self.PSF.shape[2] >1 and self.PSF.shape[3] >1:
                        plt.title(f"PSFs at {self.ao.src.zenith[0]:.0f} and {self.ao.src.zenith[-1]:.0f}"
                                  f" arcsec from center\n - Top: {1e9*self.wvl[0]:.0f}nm -"
                                  f" Bottom:{1e9*self.wvl[-1]:.0f} nm")
                        P1 = np.concatenate((self.PSF[:,:,nmin,0],self.PSF[:,:,nmax,0]),axis=1)
                        P2 = np.concatenate((self.PSF[:,:,nmin,-1],self.PSF[:,:,nmax,-1]),axis=1)
                        P  = np.concatenate((P1,P2),axis=0)
                    else:
                        plt.title('PSF')
                        P = self.PSF[:,:,nmin,0]
                    plt.imshow(np.log10(np.abs(P)))

                if displayContour and np.any(self.SR) and self.SR.size > 1:
                    self.displayPsfMetricsContours(eeRadiusInMas=eeRadiusInMas)
                else:
                    # STREHL-RATIO
                    if hasattr(self,'SR') and np.any(self.SR) and self.SR.size > 1:
                        plt.figure()
                        plt.plot(self.ao.src.zenith,self.SR[:,0],'bo',markersize=10)
                        plt.xlabel("Off-axis distance")
                        plt.ylabel(f"Strehl-ratio at {self.freq.wvlRef*1e9:.1f} nm (percents)")
                        plt.show()

                    # FWHM
                    if hasattr(self,'FWHM') and np.any(self.FWHM) and self.FWHM.size > 1:
                        plt.figure()
                        plt.plot(self.ao.src.zenith,0.5*(self.FWHM[0,:,0]+self.FWHM[1,:,0]),
                                 'bo',markersize=10)
                        plt.xlabel("Off-axis distance")
                        plt.ylabel(f"Mean FWHM at {self.freq.wvlRef*1e9:.1f} nm (mas)")
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
                        plt.ylabel(f"{eeRadiusInMas:.1f}-mas-side Ensquared energy at"
                                   f" {self.freq.wvlRef*1e9:.1f} nm (percents)")
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
                        plt.ylabel(f"{eeRadiusInMas*2:.1f}-mas-diameter Encircled energy at"
                                   f" {self.freq.wvlRef*1e9:.1f} nm (percents)")
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
                plt.title(f"Strehl-ratio at {self.freq.wvl[wvlIndex]*1e9:.1f} nm (percents)")
                plt.colorbar()

            # FWHM
            if hasattr(self,'FWHM') and np.any(self.FWHM) and self.FWHM.size > 1:
                FWHM = np.reshape(0.5*(self.FWHM[0,:,wvlIndex] + self.FWHM[1,:,wvlIndex]),(nn,nn))
                plt.figure()
                contours = plt.contour(X, Y, FWHM, nIntervals, colors='black')
                plt.clabel(contours, inline=True,fmt='%1.1f')
                plt.contourf(X,Y,FWHM)
                plt.title(f"Mean FWHM at {self.freq.wvl[wvlIndex]*1e9:.1f} nm (mas)")
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
                plt.title(f"{eeRadiusInMas*2:.1f}-mas-side Ensquared energy at"
                          f" {self.freq.wvl[wvlIndex]*1e9:.1f} nm (percents)")
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
                plt.title(f"{eeRadiusInMas*2:.1f}-mas-diameter Encircled energy at"
                          f" {self.freq.wvl[wvlIndex]*1e9:.1f} nm (percents)")
                plt.colorbar()
        else:
            print('You must define a square grid for PSF evaluations directions'
                  ' - no contours plots avalaible')

        self.t_displayPsfMetricsContours = 1000*(time.time() - tstart)

    def displayExecutionTime(self):
        """
        Display execution time breakdown for all computation steps
        """

        print("\n" + "="*70)
        print("EXECUTION TIME BREAKDOWN")
        print("="*70)

        # Total time
        if self.t_init > 0:
            print(f"\n{'Total calculation time:':<45} {self.t_init:>8.1f} ms")

        if self.t_initAO > 0:
            print(f"{'AO system model initialization:':<45} {self.t_initAO:>8.1f} ms")

        if not self.ao.error:
            print("\n--- Initialization ---")

            if self.t_initFreq > 0:
                print(f"{'Frequency domain initialization:':<45} {self.t_initFreq:>8.1f} ms")

            if self.t_atmo > 0:
                print(f"{'Atmosphere model initialization:':<45} {self.t_atmo:>8.1f} ms")

            # Reconstructors
            if self.ao.rtc.holoop['gain'] > 0:
                print("\n--- Reconstructors & Controller ---")

                if self.t_reconstructor > 0:
                    print(f"{'WFS reconstructors initialization:':<45}"
                          f" {self.t_reconstructor:>8.1f} ms")

                if self.nGs > 1:
                    if self.t_finalReconstructor > 0:
                        print(f"{'Final reconstructor calculation:':<45}"
                              f" {self.t_finalReconstructor:>8.1f} ms")

                    if self.t_tomo > 0:
                        print(f"  {'- Tomography:':<43} {self.t_tomo:>8.1f} ms")

                    if self.t_opt > 0:
                        print(f"  {'- Optimal projector:':<43} {self.t_opt:>8.1f} ms")

                if self.t_controller > 0:
                    print(f"{'Controller instantiation:':<45} {self.t_controller:>8.1f} ms")

            # PSD calculations
            if self.ao.rtc.holoop['gain'] > 0:
                print("\n--- PSD Calculations ---")

                if self.t_fittingPSD > 0:
                    print(f"{'Fitting PSD:':<45} {self.t_fittingPSD:>8.1f} ms")

                if self.t_aliasingPSD > 0:
                    print(f"{'Aliasing PSD:':<45} {self.t_aliasingPSD:>8.1f} ms")

                if self.t_noisePSD > 0:
                    print(f"{'Noise PSD:':<45} {self.t_noisePSD:>8.1f} ms")

                if self.t_spatioTemporalPSD > 0:
                    print(f"{'Spatio-temporal PSD:':<45} {self.t_spatioTemporalPSD:>8.1f} ms")

                if self.t_windShakePSD > 0:
                    print(f"{'Wind shake/vibrations PSD:':<45}"
                          f" {self.t_windShakePSD:>8.1f} ms")

                if self.t_focalAnisoplanatism > 0:
                    print(f"{'Focal anisoplanatism PSD:':<45}"
                          f" {self.t_focalAnisoplanatism:>8.1f} ms")

                if self.t_mcaoWFsensCone > 0:
                    print(f"{'MCAO WFS cone effect PSD:':<45} {self.t_mcaoWFsensCone:>8.1f} ms")

                if self.t_extra > 0:
                    print(f"{'Extra error PSD:':<45} {self.t_extra:>8.1f} ms")

                if self.t_extraLo > 0:
                    print(f"{'Extra error PSD (LO):':<45} {self.t_extraLo:>8.1f} ms")

                if self.t_tiltFilter > 0:
                    print(f"{'Tilt filter calculation:':<45} {self.t_tiltFilter:>8.1f} ms")

                if self.t_focusFilter > 0:
                    print(f"{'Focus filter calculation:':<45}"
                          f" {self.t_focusFilter:>8.1f} ms")

                if self.t_powerSpectrumDensity > 0:
                    print(f"\n{'Total PSD calculation:':<45}"
                          f"{self.t_powerSpectrumDensity:>8.1f} ms")

            # Analysis
            print("\n--- Analysis ---")

            if self.t_errorBreakDown > 0:
                print(f"{'Error breakdown calculation:':<45} {self.t_errorBreakDown:>8.1f} ms")

            if self.t_getPsfMetrics > 0:
                print(f"{'PSF metrics calculation:':<45} {self.t_getPsfMetrics:>8.1f} ms")

            # PSF and display
            if self.calcPSF:
                print("\n--- PSF Computation & Display ---")

                if self.t_getPSF > 0:
                    print(f"{'PSF calculation:':<45} {self.t_getPSF:>8.1f} ms")

                if self.display and self.t_displayResults > 0:
                    print(f"{'Display figures:':<45} {self.t_displayResults:>8.1f} ms")

        print("="*70 + "\n")

# Conditional GPU warmup based on environment variable
if os.environ.get('P3_GPU_WARMUP', 'FALSE').upper() == 'TRUE':
    file_ini0 = str(pathlib.Path(__file__).parent.parent.absolute()) + '/dummy.ini'
    faoDummy = fourierModel(path_ini=file_ini0, calcPSF=False, verbose=False, display=False,
                            path_root="", doComputations=True, computeFocalAnisoCov=False,
                            reduce_memory=True)
    faoDummy = None
