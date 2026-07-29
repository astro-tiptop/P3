#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:34:44 2021

@author: omartin
"""

# IMPORTING PYTHON LIBRAIRIES
import numpy as nnp
from . import np, nnp, trapz
import copy

from p3.aoSystem.FourierUtils import *
from p3.aoSystem.anisoplanatismModel import anisoplanatism_structure_function
import time

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000

class frequencyDomain():

    # CUT-OFF FREQUENCY
    @property
    def pitch(self):
        return self.__pitch
    @pitch.setter
    def pitch(self,val):
        if isinstance(val, np.ndarray):
            self.__pitch = np.asarray(val, dtype=self.dtype)
        else:
            self.__pitch = nnp.asarray(val, dtype=self.ndtype)

        # redefining the ao-corrected area
        if not self.kcExt is None and np.all(self.kcExt):
            self.kc_= self.kcExt
        else:
            self.kc_ =  1/(2*self.__pitch)

        self.kcMax_ = np.max(self.kc_)
        self.kc_    = np.asarray(self.kc_, dtype=self.dtype)
        self.resAO  = int(np.max(2*self.kc_/self.PSDstep))

        # ---- SPATIAL FREQUENCY DOMAIN OF THE AO-CORRECTED AREA
        self.kxAO_, self.kyAO_ = freq_array(self.resAO,
                                            offset=1e-10,
                                            L=self.PSDstep,
                                            dtype=self.dtype)
        self.k2AO_ = self.kxAO_**2 + self.kyAO_**2
        self.pistonFilterAO_ = pistonFilter(self.ao.tel.D, np.sqrt(self.k2AO_),
                                            dtype=self.dtype)
        self.pistonFilterAO_[self.resAO//2, self.resAO//2] = 0

        # ---- DEFINING MASKS
        if self.ao.dms.AoArea == 'circle':
            self.mskOut_  = (self.k2_ >= self.kcMax_**2)
            self.mskIn_   = (self.k2_ < self.kcMax_**2)
            self.mskOutAO_= self.k2AO_ >= self.kcMax_**2
            self.mskInAO_ = self.k2AO_ < self.kcMax_**2
        else:
            self.mskIn_    = np.logical_and(abs(self.kx_) < self.kcMax_,
                                            abs(self.ky_) < self.kcMax_)
            self.mskOut_   = np.logical_or(abs(self.kx_) >= self.kcMax_,
                                           abs(self.ky_) >= self.kcMax_)
            self.mskInAO_  = np.logical_and(abs(self.kxAO_) < self.kcMax_,
                                            abs(self.kyAO_) < self.kcMax_)
            self.mskOutAO_ = np.logical_or(abs(self.kxAO_) >= self.kcMax_,
                                           abs(self.kyAO_) >= self.kcMax_)

        self.psdKolmo_ = 0.0229 * self.mskOut_* ((1.0 /self.ao.atm.L0**2) + self.k2_) ** (-11.0/6.0)
        self.wfe_fit_norm  = np.sqrt(trapz(trapz(self.psdKolmo_,
                                                 self.kx_[:,0]), self.kx_[:,0]))
    @property
    def kcInMas(self):
        """DM cut-of frequency"""
        radian2mas = 180*3600*1e3/np.pi
        return self.kc_*self.ao.atm.wvl*radian2mas

    @property
    def nTimes(self):
        """"""
        return min(4,max(2,int(np.ceil(self.nOtf/self.resAO/2))))

    def __init__(self, aoSys, kcExt=None, nyquistSampling=False,
                 Hfilter=1, computeFocalAnisoCov=True, dtype=np.float64):

        self.dtype = dtype
        if self.dtype == np.float32:
            self.ndtype = nnp.float32
        else:
            self.ndtype = nnp.float64

        # PARSING INPUTS TO GET THE SAMPLING VALUES
        self.ao = aoSys

        # cast the cut-off frequency to the right type
        if kcExt is not None:
            if isinstance(kcExt, np.ndarray):
                self.kcExt = np.asarray(kcExt, dtype=self.dtype)
            else:
                self.kcExt = nnp.asarray(kcExt, dtype=self.ndtype)
        else:
            self.kcExt = None

        self.Hfilter = Hfilter

        # MANAGING THE WAVELENGTH
        self.nBin = self.ao.cam.nWvl # number of spectral bins for polychromatic PSFs
        self.nWvlCen = len(nnp.unique(self.ao.src.wvl))
        self.nWvl = self.nBin * self.nWvlCen #central wavelengths
        wvlCen_ = nnp.unique(self.ao.src.wvl)
        bw = self.ao.cam.bandwidth
        self.wvl_ = nnp.zeros(self.nWvl, dtype=self.ndtype)
        for j in range(self.nWvlCen):
            self.wvl_[j:(j+1)*self.nBin] = nnp.linspace(wvlCen_[j] - bw/2,wvlCen_[j] + bw/2,
                                                        num=self.nBin, dtype=self.ndtype)

        # MANAGING THE PIXEL SCALE
        t0 = time.time()

        self.nPix   = self.ao.cam.fovInPix

        self.nyquistSampling = nyquistSampling

        self.wvl    = np.asarray(self.wvl_, dtype=self.dtype)

        self.wvlCen = np.asarray(wvlCen_, dtype=self.dtype)
        if self.wvl_.shape[0] > 1:
            idxWmin = nnp.argmin(self.wvl_)
        else:
            idxWmin = 0
        self.wvlRef = self.wvl_[idxWmin]

        if self.nyquistSampling == True:
            self.psInMas    = rad2mas*self.wvl/self.ao.tel.D/2
            self.psInMasCen = rad2mas*wvlCen_/self.ao.tel.D/2
            samp  = 2.0 * np.ones_like(self.psInMas)
            sampCen  = 2.0 * np.ones(len(self.wvlCen), dtype=self.dtype)
            sampRef  = 2.0 * np.ones(len(self.wvlCen), dtype=self.dtype)

        else:
            self.psInMas    = self.ao.cam.psInMas * np.ones(self.nWvl, dtype=self.dtype)
            self.psInMasCen = self.ao.cam.psInMas * np.ones(self.nWvlCen, dtype=self.dtype)
            samp  = self.wvl* rad2mas / (self.psInMas*self.ao.tel.D)
            sampCen  = self.wvlCen * rad2mas / (self.psInMasCen*self.ao.tel.D)
            sampRef  = np.asarray(self.wvlRef * rad2mas, dtype=self.dtype) \
                       / np.asarray(self.psInMas[0]*self.ao.tel.D, dtype=self.dtype)

        self.k_      = np.ceil(2.0/samp).astype('int') # works for oversampling
        self.samp    = self.k_ * samp

        self.kCen_   = np.ceil(2.0/sampCen).astype('int') # works for oversampling
        self.sampCen = self.kCen_ * sampCen

        psdSteps = self.psInMas/(self.wvl*rad2mas*self.k_)
        if psdSteps.shape[0] > 1:
            idxPmin = nnp.argmin(psdSteps)
        else:
            idxPmin = 0
        self.PSDstep = np.asarray(psdSteps[idxPmin], dtype=self.dtype)

        if self.ao.psdExpansion:
            # kRef must compensate for PSDstep coming from wvl[idxPmin] (not wvl_min),
            # so that target_ps = wvl_min * PSDstep * kRef_float = psInMas/rad2mas exactly.
            # int(idxPmin) converts a potential CuPy scalar index to a Python int,
            # which is required to index self.wvl_ (CPU numpy) without triggering __array__().
            _idxPmin    = int(idxPmin)
            _kGrid      = int(self.k_[_idxPmin])
            _kRef_float = _kGrid * (float(self.wvl_[_idxPmin]) / float(self.wvl_[idxWmin]))
        else:
            _idxPmin    = int(idxWmin)
            _kGrid      = int(self.k_[_idxPmin])
            _kRef_float = _kGrid
        self.kRef_      = int(nnp.ceil(_kRef_float))   # integer, for nOtf grid size
        self.kRef_float = _kRef_float                  # float, for exact science PSF pixel scale
        self.kGrid_     = _kGrid                       # k_[idxPmin]: native grid oversampling
        self.sampRef    = _kRef_float * sampRef

        self.nOtf    = self.nPix * self.kRef_

        #  ---- FULL DOMAIN OF FREQUENCY
        self.kx_,self.ky_ = freq_array(self.nOtf, offset=1e-10,
                                       L=self.PSDstep, dtype=self.dtype)
        self.k2_          = self.kx_**2 + self.ky_**2
        #piston filtering
        self.pistonFilter_ = pistonFilter(self.ao.tel.D, np.sqrt(self.k2_), dtype=self.dtype)
        self.pistonFilter_[self.nOtf//2,self.nOtf//2] = 0

        self.pitch  = self.ao.dms.pitch

        # Derived interface quantities for TIPTOP/MASTSEL
        self.dk_        = 1e9 * self.kcMax_ / self.resAO
        self.nPupilPix_ = int(2 * np.round(self.ao.tel.D / 2.0 * self.resAO * float(self.PSDstep)))

        # MANAGING THE PIXEL SCALE
        self.tfreq = 1000*(time.time()-t0)

        # DEFINING THE DOMAIN ANGULAR FREQUENCIES
        t0 = time.time()
        self.U_, self.V_, self.U2_, self.V2_, self.UV_= instantiateAngularFrequencies(
                                                                            self.nOtf,
                                                                            fact=2,
                                                                            dtype=self.dtype)

        # COMPUTING THE STATIC OTF IF A PHASE MAP IS GIVEN
        self.otfNCPA, self.otfDL, self.phaseMap = getStaticOTF(self.ao.tel,
                                                               self.nOtf,
                                                               self.sampRef,
                                                               self.wvlRef,
                                                               dtype=self.dtype)
        self.totf = 1000*(time.time()-t0)

        # ANISOPLANATISM PHASE STRUCTURE FUNCTION
        t0 = time.time()
        if (self.ao.aoMode == 'SCAO') or (self.ao.aoMode == 'SLAO'):
            self.dphi_ani = self.anisoplanatismPhaseStructureFunction(
                                 computeFocalAnisoCov=computeFocalAnisoCov)
        else:
            self.isAniso = False
            self.dphi_ani = None
        self.tani = 1000*(time.time()-t0)

        # ---- PER-WAVELENGTH GRIDS
        # One grid context (PSDstep/resAO/nOtf/...) per requested science
        # wavelength, each exactly reproducing what a standalone single-wavelength
        # run would compute for that wavelength. Legacy behaviour (psdPerWavelength
        # disabled, or a single wavelength requested) collapses to a one-element
        # list wrapping this shared grid -- there is no separate legacy code path,
        # only the len(wvl_grids) == 1 case of this same interface.
        #
        # psdPerWavelength is independent from psdExpansion: psdExpansion only
        # picks the exact PSDstep/kRef_float for *this* shared grid (see the
        # kRef_float block above); psdPerWavelength controls whether that shared
        # grid is used at all, or replaced by one exact grid per wavelength.
        self.wvl_grids = self._buildWvlGrids(computeFocalAnisoCov=computeFocalAnisoCov)

    def _buildWvlGrids(self, computeFocalAnisoCov=True):
        """
        Legacy/shared case (psdPerWavelength disabled, or a single wavelength
        requested): the list wraps this very instance -- self.freq.wvl_grids[0]
        IS self.freq, not a copy, so there is zero overhead and zero risk of
        drift between the two.
        """
        if not (self.ao.psdPerWavelength and self.nWvl > 1):
            return [self]

        return [self._buildOneWvlGrid(i, computeFocalAnisoCov=computeFocalAnisoCov)
                for i in range(self.nWvl)]

    def _buildOneWvlGrid(self, i, computeFocalAnisoCov=True):
        """
        Exact per-wavelength grid context for science wavelength index i:
        same PSDstep/resAO/nOtf a standalone single-wavelength run would use
        for self.wvl_[i], reusing the per-element quantities (k_[i], psInMas[i])
        already computed above for the shared-grid selection.

        Returned as a *shallow copy* of this frequencyDomain instance with only
        the grid-dependent attributes overridden, rather than a hand-picked
        subset of attributes: this guarantees that any attribute read on
        self.freq elsewhere in fourierModel (including ones not anticipated
        here) resolves to a sensible value -- either the per-wavelength
        override, or the correct wavelength-independent shared value (e.g.
        kc_, nPix, psInMas, U_/V_/U2_/V2_/UV_) -- instead of raising
        AttributeError or silently reading stale shared-grid data.
        """
        wvl_i = float(self.wvl_[i])
        k_i   = int(self.k_[i])

        PSDstep_i = np.asarray(self.psInMas[i] / (self.wvl[i] * rad2mas * self.k_[i]),
                               dtype=self.dtype)
        resAO_i = int(np.max(2 * self.kc_ / PSDstep_i))
        nOtf_i  = int(self.nPix) * k_i

        kxAO_i, kyAO_i = freq_array(resAO_i, offset=1e-10, L=PSDstep_i, dtype=self.dtype)
        k2AO_i = kxAO_i**2 + kyAO_i**2
        pistonFilterAO_i = pistonFilter(self.ao.tel.D, np.sqrt(k2AO_i), dtype=self.dtype)
        pistonFilterAO_i[resAO_i//2, resAO_i//2] = 0

        kx_i, ky_i = freq_array(nOtf_i, offset=1e-10, L=PSDstep_i, dtype=self.dtype)
        k2_i = kx_i**2 + ky_i**2
        pistonFilter_i = pistonFilter(self.ao.tel.D, np.sqrt(k2_i), dtype=self.dtype)
        pistonFilter_i[nOtf_i//2, nOtf_i//2] = 0

        if self.ao.dms.AoArea == 'circle':
            mskOut_i   = (k2_i >= self.kcMax_**2)
            mskIn_i    = (k2_i < self.kcMax_**2)
            mskOutAO_i = k2AO_i >= self.kcMax_**2
            mskInAO_i  = k2AO_i < self.kcMax_**2
        else:
            mskIn_i    = np.logical_and(abs(kx_i) < self.kcMax_, abs(ky_i) < self.kcMax_)
            mskOut_i   = np.logical_or(abs(kx_i) >= self.kcMax_, abs(ky_i) >= self.kcMax_)
            mskInAO_i  = np.logical_and(abs(kxAO_i) < self.kcMax_, abs(kyAO_i) < self.kcMax_)
            mskOutAO_i = np.logical_or(abs(kxAO_i) >= self.kcMax_, abs(kyAO_i) >= self.kcMax_)

        # self.samp[i] == k_i * (wvl_i*rad2mas)/(psInMas[i]*D): the oversampling
        # factor that makes this wavelength's native FFT pixel scale hit
        # psInMas[i] exactly (see self.samp definition above).
        sampRef_i = self.samp[i]

        otfNCPA_i, otfDL_i, phaseMap_i = getStaticOTF(self.ao.tel, nOtf_i, sampRef_i,
                                                      wvl_i, dtype=self.dtype)

        grid_ctx = copy.copy(self)
        grid_ctx.resAO = resAO_i
        grid_ctx.nOtf = nOtf_i
        grid_ctx.PSDstep = PSDstep_i
        # Derived interface quantities for TIPTOP/MASTSEL (see their definition
        # above): both depend on resAO/PSDstep, so they must be recomputed per
        # grid too, not just inherited from the shared-grid shallow copy.
        grid_ctx.dk_        = 1e9 * self.kcMax_ / resAO_i
        grid_ctx.nPupilPix_ = int(2 * np.round(self.ao.tel.D / 2.0 * resAO_i * float(PSDstep_i)))
        grid_ctx.kxAO_ = kxAO_i
        grid_ctx.kyAO_ = kyAO_i
        grid_ctx.k2AO_ = k2AO_i
        grid_ctx.pistonFilterAO_ = pistonFilterAO_i
        grid_ctx.mskInAO_ = mskInAO_i
        grid_ctx.mskOutAO_ = mskOutAO_i
        grid_ctx.kx_ = kx_i
        grid_ctx.ky_ = ky_i
        grid_ctx.k2_ = k2_i
        grid_ctx.pistonFilter_ = pistonFilter_i
        grid_ctx.mskIn_ = mskIn_i
        grid_ctx.mskOut_ = mskOut_i
        grid_ctx.sampRef = sampRef_i
        grid_ctx.otfNCPA = otfNCPA_i
        grid_ctx.otfDL = otfDL_i
        grid_ctx.phaseMap = phaseMap_i
        grid_ctx.wvlRef = wvl_i
        # Avoid keeping nWvl deep (recursive) copies of the whole grid list
        # alive from every single per-wavelength copy.
        grid_ctx.wvl_grids = None

        if (self.ao.aoMode == 'SCAO') or (self.ao.aoMode == 'SLAO'):
            grid_ctx.dphi_ani = self._wvlGridAnisoplanatismDphi(
                nOtf_i, sampRef_i, mskIn_i, computeFocalAnisoCov=computeFocalAnisoCov)
        else:
            grid_ctx.dphi_ani = None

        return grid_ctx

    def _wvlGridAnisoplanatismDphi(self, nOtf, samp, msk_in, computeFocalAnisoCov=True):
        """
        Standalone (side-effect-free) copy of the branching logic in
        anisoplanatismPhaseStructureFunction, parametrised on nOtf/samp/msk_in
        so it can be called once per per-wavelength grid without touching the
        state (self.dani_ang, self.isAniso, ...) used by the legacy/shared path.
        """
        if computeFocalAnisoCov == False:
            return None

        Cn2 = self.ao.atm.weights * self.ao.atm.r0**(-5/3)

        if self.ao.aoMode == 'SCAO':
            if nnp.all(nnp.equal(nnp.asarray(self.ao.src.direction),
                                 nnp.asarray(self.ao.ngs.direction))):
                return None
            dani_ang = anisoplanatism_structure_function(self.ao.tel, self.ao.atm,
                                                          self.ao.src, self.ao.lgs,
                                                          self.ao.ngs, nOtf, samp,
                                                          self.ao.dms.nActu1D,
                                                          msk_in=msk_in,
                                                          Hfilter=self.Hfilter)
            return (dani_ang * Cn2[np.newaxis, :, np.newaxis, np.newaxis]).sum(axis=1)

        elif self.ao.aoMode == 'SLAO':
            dani_focang, dani_ang, dani_tt = anisoplanatism_structure_function(
                self.ao.tel, self.ao.atm, self.ao.src, self.ao.lgs, self.ao.ngs,
                nOtf, samp, self.ao.dms.nActu1D)
            return ((dani_focang + dani_tt)
                    * Cn2[np.newaxis, :, np.newaxis, np.newaxis]).sum(axis=1)

        return None

    def __repr__(self):

        s = '__ FREQUENCY DOMAIN __\n' + '--------------------------------------------- \n'
        s += '. Reference wavelength : %.2f µm\n'%(self.wvlRef*1e6)
        s += '. Oversampling factor at the reference wavelength : %.2f\n'%(self.sampRef)
        s += '. Size of the frequency domain : %d pixels\n'%(self.nOtf)
        s += '. Pixel scale at the reference wavelength : %.4f m^-1\n'%(self.PSDstep)
        s += '. Instantiantion of the anisoplanatism model : %s\n'%(str(self.isAniso))
        s += '. Include a static aberrations map : %s\n'%(str(np.any(self.otfNCPA != self.otfDL)))
        s += '---------------------------------------------\n'
        return s


    def anisoplanatismPhaseStructureFunction(self,
                                             computeFocalAnisoCov=True):

        # compute th Cn2 profile in m^(-5/3)
        Cn2 = self.ao.atm.weights * self.ao.atm.r0**(-5/3)

        if computeFocalAnisoCov == False:
            self.isAniso = False
            return None
        elif self.ao.aoMode == 'SCAO':
            # NGS case : angular-anisoplanatism only              
            if nnp.all(nnp.equal(nnp.asarray(self.ao.src.direction),
                                 nnp.asarray(self.ao.ngs.direction))):
                self.isAniso = False
                return None
            else:
                self.isAniso = True
                self.dani_ang = anisoplanatism_structure_function(self.ao.tel,
                                                                  self.ao.atm,
                                                                  self.ao.src,
                                                                  self.ao.lgs,
                                                                  self.ao.ngs,
                                                                  self.nOtf,
                                                                  self.sampRef,
                                                                  self.ao.dms.nActu1D,
                                                                  msk_in = self.mskIn_,
                                                                  Hfilter=self.Hfilter)

                return (self.dani_ang *Cn2[np.newaxis,:,np.newaxis,np.newaxis]).sum(axis=1)

        elif self.ao.aoMode == 'SLAO':
            # LGS case : focal-angular  + anisokinetism
            self.isAniso = True
            self.dani_focang, self.dani_ang, self.dani_tt = \
            anisoplanatism_structure_function(self.ao.tel, self.ao.atm, self.ao.src,
                                              self.ao.lgs, self.ao.ngs, self.nOtf,
                                              self.sampRef, self.ao.dms.nActu1D)

            return ( (self.dani_focang + self.dani_tt) \
                     * Cn2[np.newaxis,:,np.newaxis,np.newaxis]).sum(axis=1)
        else:
            # LTAO, GLAO or MCAO case
            self.isAniso = False
            return None
