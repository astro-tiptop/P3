#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 11:34:44 2021

@author: omartin
"""

# IMPORTING PYTHON LIBRAIRIES
import time
import numpy as np
import aoSystem.FourierUtils as FourierUtils
from aoSystem.anisoplanatismModel import anisoplanatism_structure_function

#%%
rad2mas = 3600 * 180 * 1000 / np.pi
rad2arc = rad2mas / 1000

class frequencyDomain():
    """
    Object that carries the definition of the spatial frequency domain
    """

    # WAVELENGTH
    @property
    def wvl(self):
        return self.__wvl
    @wvl.setter
    def wvl(self,val):
        self.__wvl = val
        if self.nyquistSampling == True:
            self.samp  = 2.0 * np.ones_like(self.psInMas)
        else:
            self.samp  = val* rad2mas/(self.psInMas*self.ao.tel.D)

    @property
    def wvlCen(self):
        return self.__wvlCen
    @wvlCen.setter
    def wvlCen(self,val):
        self.__wvlCen = val
        if self.nyquistSampling == True:
            self.sampCen  = 2.0 * np.ones(len(val))
        else:
            self.sampCen  = val* rad2mas/(self.psInMasCen*self.ao.tel.D)

    @property
    def wvlRef(self):
        return self.__wvlRef
    @wvlRef.setter
    def wvlRef(self,val):
        self.__wvlRef = val
        if self.nyquistSampling == True:
            self.sampRef  = 2.0
        else:
            self.sampRef  = val* rad2mas/(self.psInMas[0]*self.ao.tel.D)
    # SAMPLING
    @property
    def samp(self):
        return self.__samp
    @samp.setter
    def samp(self,val):
        self.k_      = np.ceil(2.0/val).astype('int') # works for oversampling
        self.__samp  = self.k_ * val
        if np.any(self.k_ > 2):
            self.PSDstep= np.min(1/self.ao.tel.D/self.__samp)
        else:
            self.PSDstep= np.min(self.psInMas/self.wvl_/rad2mas)
            
    @property
    def sampCen(self):
        return self.__sampCen
    @sampCen.setter
    def sampCen(self,val):
        self.kCen_      = np.ceil(2.0/val).astype(int)# works for oversampling
        self.__sampCen  = self.kCen_ * val
    @property
    def sampRef(self):
        return self.__sampRef
    @sampRef.setter
    def sampRef(self,val):
        self.kRef_ = int(np.ceil(2.0/val)) # works for oversampling
        self.__sampRef = self.kRef_ * val
        self.nOtf = self.nPix * self.kRef_
        #  ---- FULL DOMAIN OF FREQUENCY
        self.kx_,self.ky_ = FourierUtils.freq_array(self.nOtf,offset=1e-10,L=self.PSDstep)
        self.k2_ = self.kx_**2 + self.ky_**2
        #piston filtering
        self.pistonFilter_ = FourierUtils.pistonFilter(self.ao.tel.D,np.sqrt(self.k2_))
        self.pistonFilter_[self.nOtf//2,self.nOtf//2] = 0

    # CUT-OFF FREQUENCY
    @property
    def pitch(self):
        return self.__pitch
    @pitch.setter
    def pitch(self,val):
        self.__pitch = val
        # redefining the ao-corrected area
        if np.all(self.kcExt !=None):
            self.kc_= self.kcExt
        else:
            self.kc_ =  1/(2*val)
        #self.kc_= (val-1)/(2.0*self.ao.tel.D)
        self.kcMax_ =  np.max(self.kc_)
        #kc2         = self.kc_**2
        self.resAO  = int(np.max(2*self.kc_/self.PSDstep))

        # ---- SPATIAL FREQUENCY DOMAIN OF THE AO-CORRECTED AREA
        self.kxAO_, self.kyAO_ = FourierUtils.freq_array(self.resAO,
                                                         offset=1e-10,
                                                         L=self.PSDstep)
        self.k2AO_ = self.kxAO_**2 + self.kyAO_**2
        self.pistonFilterAO_ = FourierUtils.pistonFilter(self.ao.tel.D, np.sqrt(self.k2AO_))
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
        self.wfe_fit_norm  = np.sqrt(np.trapz(np.trapz(self.psdKolmo_,
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

    def __init__(self, aoSys, kcExt=None, nyquistSampling=False, Hfilter=1):

        # PARSING INPUTS TO GET THE SAMPLING VALUES
        self.ao = aoSys

        # MANAGING THE WAVELENGTH
        self.nBin = self.ao.cam.nWvl # number of spectral bins for polychromatic PSFs
        self.nWvlCen = len(np.unique(self.ao.src.wvl))
        self.nWvl = self.nBin * self.nWvlCen #central wavelengths
        wvlCen_ = np.unique(self.ao.src.wvl)
        bw = self.ao.cam.bandwidth
        self.wvl_ = np.zeros(self.nWvl)
        for j in range(self.nWvlCen):
            self.wvl_[j:(j+1)*self.nBin] = np.linspace(wvlCen_[j] - bw/2,wvlCen_[j] + bw/2,num=self.nBin)

        # MANAGING THE PIXEL SCALE
        t0 = time.time()
        if nyquistSampling:
            self.nyquistSampling = True
            self.psInMas = rad2mas*self.wvl_/self.ao.tel.D/2
            self.psInMasCen = rad2mas*wvlCen_/self.ao.tel.D/2
        else:
            self.psInMas = self.ao.cam.psInMas * np.ones(self.nWvl)
            self.psInMasCen = self.ao.cam.psInMas * np.ones(self.nWvlCen)
            self.nyquistSampling = False

        self.kcExt = kcExt
        self.nPix = self.ao.cam.fovInPix
        self.wvl = self.wvl_
        self.wvlCen = wvlCen_
        self.wvlRef = np.min(self.wvl_)
        self.pitch = self.ao.dms.pitch
        self.Hfilter = Hfilter
        self.tfreq = 1000*(time.time()-t0)

        # DEFINING THE DOMAIN ANGULAR FREQUENCIES
        t0 = time.time()
        self.U_, self.V_, self.U2_, self.V2_, self.UV_=  FourierUtils.instantiateAngularFrequencies(self.nOtf, fact=2)

        # COMPUTING THE STATIC OTF IF A PHASE MAP IS GIVEN
        self.otfNCPA, self.otfDL, self.phaseMap =\
        FourierUtils.getStaticOTF(self.ao.tel, self.nOtf, self.sampRef, self.wvlRef)
        self.totf = 1000*(time.time()-t0)

        # ANISOPLANATISM PHASE STRUCTURE FUNCTION
        t0 = time.time()
        if (self.ao.aoMode == 'SCAO') or (self.ao.aoMode == 'SLAO'):
            self.dphi_ani = self.anisoplanatismPhaseStructureFunction()
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


    def anisoplanatismPhaseStructureFunction(self):

        # compute th Cn2 profile in m^(-5/3)
        Cn2 = self.ao.atm.weights * self.ao.atm.r0**(-5/3)

        if self.ao.aoMode == 'SCAO':
            # NGS case : angular-anisoplanatism only
            if np.all(self.ao.src.direction == self.ao.ngs.direction):
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
            self.isAniso = False
            return None
