#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:34:02 2018

@author: omartin
"""
import os.path as ospath
import re

import numpy as nnp
from . import gpuEnabled, cp, np, nnp, rotate

from astropy.io import fits
import p3.aoSystem.FourierUtils as FourierUtils
from p3.aoSystem.zernike import zernike

class Attribute(object):
    pass

class telescope:
    """ Telescope class that defines the telescope characteristics.
    Inputs are:
        - D: telescope diameter
        - elevation: telescope elevation in degree
        - obsRatio: central obstruction ratio from 0 to 1
        - resolution: pupil pixels resolution
    """

    # DEPENDANT VARIABLES DEFINITION
    def get_pupil(self):
        return self.p_pupil
    def set_pupil(self,val):
        self.p_pupil = val
        self.resolution = val.shape[0]
    pupil = property(get_pupil,set_pupil)

    @property
    def R(self):
        """Telescope radius in meter"""
        return self.D/2
    @property
    def area(self):
        """Telescope area in meter square"""
        return np.pi*self.R**2*(1 - self.obsRatio**2)
    @property
    def pupilLogical(self):
        """Telescope area in meter square"""
        return self.pupil.astype(bool)
    @property
    def airmass(self):
        return 1/nnp.cos(self.zenith_angle*np.pi/180)

    # CONSTRUCTOR
    def __init__(self, D, resolution, zenith_angle=0.0, obsRatio=0.0, pupilAngle=0.0,
                 path_pupil=None, path_static_on=None, zcoef_static_on=[], path_static_off=None,
                 path_static_pos=None, path_apodizer=None, path_statModes=None,
                 extraErrorNm=0.0, extraErrorExp=-2, extraErrorMin=0.0, extraErrorMax=0.0,
                 extraErrorLoNm=0.0, extraErrorLoExp=-2, extraErrorLoMin=0.0, extraErrorLoMax=0.0,
                 verbose=True):

        # PARSING INPUTS
        self.D = D # primary mirror diameter in meters
        self.zenith_angle = zenith_angle # telescope zenith angle in degrees
        self.obsRatio = obsRatio # secondary mirror diameter in D units
        self.resolution = resolution # pupil resolution in pixels
        self.verbose = verbose
        self.pupilAngle = pupilAngle
        self.path_pupil = path_pupil
        self.path_static_on = path_static_on
        self.zcoef_static_on = zcoef_static_on
        self.path_static_off = path_static_off
        self.path_static_pos = path_static_pos
        self.path_apodizer = path_apodizer
        self.path_statModes = path_statModes
        self.extraErrorNm      = extraErrorNm
        self.extraErrorExp     = extraErrorExp
        self.extraErrorMin     = extraErrorMin
        self.extraErrorMax     = extraErrorMax
        self.extraErrorLoNm    = extraErrorLoNm
        self.extraErrorLoExp   = extraErrorLoExp
        self.extraErrorLoMin   = extraErrorLoMin
        self.extraErrorLoMax   = extraErrorLoMax

        def check_path(path):
            is_ok = path is not None\
                    and ospath.isfile(path)\
                    and re.search(".fits", path) is not None
            return is_ok

        def load_and_process_data(path_data):
            data = np.asarray(fits.getdata(path_data))
            data[data!=data] = 0
            if self.pupilAngle != 0.0:
                data = rotate(data, self.pupilAngle, reshape=False)
            if data.shape[0] != resolution:
                data = FourierUtils.interpolateSupport(data, resolution, kind='linear')
            return data

        #----- PUPIL DEFINITION
        if check_path(path_pupil):
            self.verb = True
            self.pupil = load_and_process_data(path_pupil)
        else:
            # build an annular pupil model
            th = self.pupilAngle*nnp.pi/180
            x = nnp.linspace(-D/2, D/2, resolution)
            X,Y = nnp.meshgrid(x,x)
            Xr  = X*nnp.cos(th) + Y*nnp.sin(th)
            Yr  = Y*nnp.cos(th) - X*nnp.sin(th)
            R   = nnp.hypot(Xr,Yr)
            P   = ((R <= self.R) * (R > self.R * self.obsRatio)).astype(int)
            self.pupil = np.asarray(P)
            self.verb = False

        #----- APODIZER
        self.apodizer = 1.0
        if check_path(path_apodizer):
            self.apodizer = load_and_process_data(path_apodizer)

        #----- NCPA
        self.opdMap_on = None
        if check_path(path_static_on):
            self.opdMap_on = load_and_process_data(path_static_on)

        # 2D map from a vector of Zernike coefficients
        if self.zcoef_static_on and len(self.zcoef_static_on) > 0:
            n_zern = len(zcoef_static_on)
            zern_idx = list(np.linspace(2,n_zern+1,n_zern).astype(int))
            zern_obj = zernike(zern_idx,resolution,D=None,pupil=[],unitNorm=False,radius=[],angle=[],cobs=0)
            zern_base = zern_obj.modes
            #
            # mask and orthonormalize the zernike modes
            for i in range(zern_base.shape[0]):
                zern_base[i] *= np.array(self.pupil)
            zern_base = zern_base.reshape((zern_base.shape[0],resolution**2))
            size_base = np.shape(zern_base)
            if size_base[1] > size_base[0]:
                Q, R = np.linalg.qr(zern_base.T)
                Q = Q.T
            else:
                Q, R = np.linalg.qr(zern_base)
            zern_base = zern_base.reshape((zern_base.shape[0],resolution,resolution))
            # Normalize Zernike modes
            for ii in range(zern_base.shape[0]):
                zern_base[ii, :] -= np.mean(zern_base[ii, :])
                zern_base[ii, :] /= np.sqrt(np.mean(zern_base[ii, :]**2))
            #
            # build the phase map
            if self.opdMap_on is None:
                self.opdMap_on = np.zeros((resolution,resolution))
            for i in range(zern_base.shape[0]):
                self.opdMap_on += zcoef_static_on[i] * zern_base[i]

        #----- FIELD-DEPENDANT STATIC ABERRATIONS
        self.opdMap_off = None
        self.opdMap_pos = None
        if check_path(path_static_off):
            if check_path(path_static_pos):
                opdMap_off = fits.getdata(path_static_off)
                opdMap_pos = fits.getdata(path_static_pos)
                if opdMap_pos.shape[0] != opdMap_off[0]:
                    raise ValueError('You must provide as many positions values as maps')
                else:
                    self.opdMap_off = opdMap_off
                    self.opdMap_pos = opdMap_pos
            else:
                raise ValueError(("Positions (zenith in arcsec, azimuth in radian)"
                                  "of the field-dependent aberrations must be"
                                  "provided as well as the maps"))

        #----- MODAL BASIS FOR TELESCOPE ABERRATIONS
        self.statModes = None
        self.nModes = 0
        if check_path(path_statModes):
            statModes = np.asarray(fits.getdata(path_statModes))
            s1, s2, s3 = statModes.shape
            if s1 != s2: # mode on first dimension
                tmp = np.transpose(statModes, (1, 2, 0))
            else:
                tmp = statModes

            self.nModes = tmp.shape[-1]
            self.statModes = np.zeros((resolution, resolution, self.nModes))
            for k in range(self.nModes):
                mode = FourierUtils.interpolateSupport(tmp[:, :, k],
                                                       resolution,
                                                       kind='linear')
                if pupilAngle !=0:
                    mode = rotate(mode,pupilAngle, reshape=False)
                self.statModes[:, :, k] = mode

        if self.verbose:
            self

    def __repr__(self):
        s = "___TELESCOPE___\n -------------------------------------------------------------------------------- \n"
        s += '. Aperture diameter \t:%.2fm \n'%(self.D)
        s += '. Central obstruction \t:%.2f%s \n'%(self.obsRatio*1e2, '%')
        s += '. Collecting area \t:%.2fm^2\n'%(self.area)
        s += '. Pupil resolution \t:%dX%d pixels'%(self.resolution, self.resolution)
        if self.path_pupil is not None:
            s += '\n. User-defined pupil at :\n ' + self.path_pupil
        if self.path_apodizer is not None:
            s += '\n. User-defined amplitude apodizer at :\n ' + self.path_apodizer
        if self.path_static_on is not None:
            s += '\n. User-defined on-axis static aberrations map at :\n ' + self.path_static_on
        if self.path_static_off is not None:
            s += '\n. User-defined off-axis static aberrations maps at :\n ' + self.path_static_off
        if self.path_statModes is not None:
            s += '\n. User-defined modes of static aberrations at :\n ' + self.path_statModes
        s = s +"\n--------------------------------------------------------------------------------\n"
        return s

    def displayPupil(self):
        """
        Display the telescope pupil
        """

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(self.pupil)
        plt.xlabel("Pixels")
        plt.ylabel("Pixels")
        plt.title("Telescope pupil")

