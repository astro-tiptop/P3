#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:34:44 2021

@author: omartin
"""

# IMPORTING PYTHON LIBRAIRIES
import numpy as nnp
from . import gpuEnabled, cp, np, nnp, trapz

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
        self.__pitch = val
        # redefining the ao-corrected area
        if not self.kcExt is None and np.all(self.kcExt):
            self.kc_= self.kcExt
        else:
            self.kc_ =  1/(2*val)
        #self.kc_= (val-1)/(2.0*self.ao.tel.D)
        self.kcMax_ =  np.max(self.kc_)
        #kc2         = self.kc_**2
        self.kc_    = np.asarray(self.kc_)
        self.resAO  = int(np.max(2*self.kc_/self.PSDstep))

        # ---- SPATIAL FREQUENCY DOMAIN OF THE AO-CORRECTED AREA
        self.kxAO_, self.kyAO_ = freq_array(self.resAO,
                                            offset=1e-10,
                                            L=self.PSDstep)
        self.k2AO_ = self.kxAO_**2 + self.kyAO_**2
        self.pistonFilterAO_ = pistonFilter(self.ao.tel.D, np.sqrt(self.k2AO_))
        self.pistonFilterAO_[self.resAO//2, self.resAO//2] = 0

        # ---- DEFINING MASKS
        if self.ao.dms.AoArea == 'circle':
            self.mskOut_  = (self.k2_ >= self.kcMax_**2)
            self.mskIn_   = (self.k2_ < self.kcMax_**2)
            self.mskOutAO_= self.k2AO_ >= self.kcMax_**2
            self.mskInAO_ = self.k2AO_ < self.kcMax_**2
            
        else:
            self.mskIn_    = np.logical_and(abs(self.kx_) < self.kcMax_, abs(self.ky_) < self.kcMax_)
            self.mskOut_   = np.logical_or(abs(self.kx_) >= self.kcMax_, abs(self.ky_) >= self.kcMax_)
            self.mskInAO_  = np.logical_and(abs(self.kxAO_) < self.kcMax_, abs(self.kyAO_) < self.kcMax_)
            self.mskOutAO_ = np.logical_or(abs(self.kxAO_) >= self.kcMax_, abs(self.kyAO_) >= self.kcMax_)

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
                 Hfilter=1, computeFocalAnisoCov=True, multiWvlPSD=False):

        # PARSING INPUTS TO GET THE SAMPLING VALUES
        self.ao = aoSys

        self.kcExt  = kcExt
        self.Hfilter = Hfilter
        self.multiWvlPSD = multiWvlPSD

        # MANAGING THE WAVELENGTH
        self.nBin = self.ao.cam.nWvl # number of spectral bins for polychromatic PSFs
        self.nWvlCen = len(nnp.unique(self.ao.src.wvl))
        self.nWvl = self.nBin * self.nWvlCen #central wavelengths
        wvlCen_ = nnp.unique(self.ao.src.wvl)
        bw = self.ao.cam.bandwidth
        self.wvl_ = nnp.zeros(self.nWvl)
        for j in range(self.nWvlCen):
            self.wvl_[j:(j+1)*self.nBin] = \
                nnp.linspace(wvlCen_[j] - bw/2,wvlCen_[j] + bw/2,num=self.nBin)

        # MANAGING THE PIXEL SCALE
        t0 = time.time()

        self.nPix   = self.ao.cam.fovInPix

        self.nyquistSampling = nyquistSampling

        self.wvl    = np.asarray(self.wvl_)

        self.wvlCen = np.asarray(wvlCen_)
        if self.wvl_.shape[0] > 1:
            idxWmin = nnp.argmin(self.wvl_)
        else:
            idxWmin = 0
        self.wvlRef = self.wvl_[idxWmin]

        if self.nyquistSampling == True:
            self.psInMas    = rad2mas*self.wvl/self.ao.tel.D/2
            self.psInMasCen = rad2mas*wvlCen_/self.ao.tel.D/2
            samp  = 2.0 * np.ones_like(self.psInMas)
            sampCen  = 2.0 * np.ones(len(self.wvlCen))
            sampRef  = 2.0 * np.ones(len(self.wvlCen))

        else:
            self.psInMas    = self.ao.cam.psInMas * np.ones(self.nWvl)
            self.psInMasCen = self.ao.cam.psInMas * np.ones(self.nWvlCen)
            samp  = self.wvl* rad2mas / (self.psInMas*self.ao.tel.D)
            sampCen  = self.wvlCen * rad2mas / (self.psInMasCen*self.ao.tel.D)
            sampRef  = np.asarray(self.wvlRef * rad2mas) / np.asarray(self.psInMas[0]*self.ao.tel.D)

        self.k_      = np.ceil(2.0/samp).astype('int') # works for oversampling
        self.samp    = self.k_ * samp

        self.kCen_   = np.ceil(2.0/sampCen).astype('int') # works for oversampling
        self.sampCen = self.kCen_ * sampCen

        psdSteps = self.psInMas/(self.wvl*rad2mas*self.k_)
        if psdSteps.shape[0] > 1:
            idxPmin = nnp.argmin(psdSteps)
        else:
            idxPmin = 0
        self.PSDstep = np.asarray(psdSteps[idxPmin])

        if self.ao.psdExpansion:
            self.kRef_   = int(self.k_[idxPmin]) # works for oversampling
        else:
            self.kRef_   = int(self.k_[idxWmin]) # works for oversampling
        self.sampRef = self.kRef_ * sampRef

        self.nOtf    = self.nPix * self.kRef_

        #  ---- FULL DOMAIN OF FREQUENCY
        self.kx_,self.ky_ = freq_array(self.nOtf,offset=1e-10,L=self.PSDstep)
        self.k2_          = self.kx_**2 + self.ky_**2
        #piston filtering
        self.pistonFilter_ = pistonFilter(self.ao.tel.D,np.sqrt(self.k2_))
        self.pistonFilter_[self.nOtf//2,self.nOtf//2] = 0

        self.pitch  = self.ao.dms.pitch

        if self.multiWvlPSD:
            # Pre-compute frequency domains for each wavelength
            # Each wavelength can have a different resAO
            self.resAO_wvl = np.zeros(self.nWvl, dtype=int)
            self.PSDstep_wvl = np.zeros(self.nWvl)

            # Full OTF grids per wavelength (same size nOtf, different step)
            self.kx_wvl = []
            self.ky_wvl = []
            self.k2_wvl = []
            self.pistonFilter_wvl = []
            self.mskOut_wvl = []
            self.mskIn_wvl = []
            
            # AO grids per wavelength (different size resAO_wvl)
            self.kxAO_wvl = []
            self.kyAO_wvl = []
            self.k2AO_wvl = []
            self.pistonFilterAO_wvl = []
            self.mskInAO_wvl = []
            self.mskOutAO_wvl = []

            for idx_wvl in range(self.nWvl):
                # Scale spatial frequencies based on wavelength ratio
                wvl_ratio = self.wvl[idx_wvl] / self.wvlRef
                PSDstep_wvl = self.PSDstep * wvl_ratio
                self.PSDstep_wvl[idx_wvl] = PSDstep_wvl

                # Full OTF grid with scaled step (same size nOtf)
                kx_w, ky_w = freq_array(self.nOtf, offset=1e-10, L=PSDstep_wvl)
                k2_w = kx_w**2 + ky_w**2
                
                self.kx_wvl.append(kx_w)
                self.ky_wvl.append(ky_w)
                self.k2_wvl.append(k2_w)
                
                # Piston filter for full OTF
                pf = pistonFilter(self.ao.tel.D, np.sqrt(k2_w))
                pf[self.nOtf//2, self.nOtf//2] = 0
                self.pistonFilter_wvl.append(pf)
                
                # Calculate resAO for this wavelength
                # Note: kc_ (DM cutoff) is independent of wavelength
                resAO_wvl = int(np.max(2*self.kc_/PSDstep_wvl))
                self.resAO_wvl[idx_wvl] = resAO_wvl

                # Generate AO frequency arrays with scaled step and possibly different size
                kxAO_w, kyAO_w = freq_array(resAO_wvl, offset=1e-10, L=PSDstep_wvl)
                k2AO_w = kxAO_w**2 + kyAO_w**2

                self.kxAO_wvl.append(kxAO_w)
                self.kyAO_wvl.append(kyAO_w)
                self.k2AO_wvl.append(k2AO_w)

                # Piston filter per wavelength (AO area)
                pf_w = pistonFilter(self.ao.tel.D, np.sqrt(k2AO_w))
                pf_w[resAO_wvl//2, resAO_wvl//2] = 0
                self.pistonFilterAO_wvl.append(pf_w)

                # Masks per wavelength
                if self.ao.dms.AoArea == 'circle':
                    mskOut_w = k2_w >= self.kcMax_**2
                    mskIn_w = k2_w < self.kcMax_**2
                    mskInAO_w = k2AO_w < self.kcMax_**2
                    mskOutAO_w = k2AO_w >= self.kcMax_**2
                else:
                    mskOut_w = np.logical_or(abs(kx_w) >= self.kcMax_, abs(ky_w) >= self.kcMax_)
                    mskIn_w = np.logical_and(abs(kx_w) < self.kcMax_, abs(ky_w) < self.kcMax_)
                    mskInAO_w = np.logical_and(abs(kxAO_w) < self.kcMax_,
                                            abs(kyAO_w) < self.kcMax_)
                    mskOutAO_w = np.logical_or(abs(kxAO_w) >= self.kcMax_,
                                            abs(kyAO_w) >= self.kcMax_)
                
                self.mskOut_wvl.append(mskOut_w)
                self.mskIn_wvl.append(mskIn_w)
                self.mskInAO_wvl.append(mskInAO_w)
                self.mskOutAO_wvl.append(mskOutAO_w)

            # Note: These remain as lists (not arrays) because some elements 
            # (AO grids) can have different shapes (resAO_wvl[i], resAO_wvl[i])
            # while OTF grids all have shape (nOtf, nOtf)

        # MANAGING THE PIXEL SCALE
        self.tfreq = 1000*(time.time()-t0)

        # DEFINING THE DOMAIN ANGULAR FREQUENCIES
        t0 = time.time()
        self.U_, self.V_, self.U2_, self.V2_, self.UV_= \
            instantiateAngularFrequencies(self.nOtf, fact=2)

        # COMPUTING THE STATIC OTF IF A PHASE MAP IS GIVEN
        self.otfNCPA, self.otfDL, self.phaseMap = \
            getStaticOTF(self.ao.tel, self.nOtf, self.sampRef, self.wvlRef)
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


    def anisoplanatismPhaseStructureFunction(self,computeFocalAnisoCov=True):

        # compute th Cn2 profile in m^(-5/3)
        Cn2 = self.ao.atm.weights * self.ao.atm.r0**(-5/3)
       
        if computeFocalAnisoCov == False:
            self.isAniso = False
            return None
        elif self.ao.aoMode == 'SCAO':
            # NGS case : angular-anisoplanatism only              
            if nnp.all(nnp.equal(nnp.asarray(self.ao.src.direction), nnp.asarray(self.ao.ngs.direction))):
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

            return ( (self.dani_focang + self.dani_tt) *Cn2[np.newaxis,:,np.newaxis,np.newaxis]).sum(axis=1)
        else:
            # LTAO, GLAO or MCAO case
            self.isAniso = False
            return None
