#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests derived from p3/testing/tests_p3.py.
Each test verifies that the corresponding code runs without errors.
Display is disabled (matplotlib Agg backend).
"""

import unittest
import os
import pathlib
import tempfile
import importlib.util
from configparser import ConfigParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import p3.aoSystem as aoSystemMain
from p3.aoSystem.aoSystem import aoSystem
from p3.aoSystem.pupil import pupil
from p3.aoSystem.segment import segment
from p3.aoSystem.spiders import spiders
from p3.aoSystem.frequencyDomain import frequencyDomain
from p3.aoSystem.fourierModel import fourierModel
from p3.psfFitting.psfFitting import psfFitting

HAS_MAOPPY = importlib.util.find_spec('maoppy') is not None
MAOPPY_SKIP_REASON = "" if HAS_MAOPPY else "maoppy not installed"

if HAS_MAOPPY:
    from p3.psfao21.psfao21 import psfao21
else:
    psfao21 = None


def _p3_path():
    return str(pathlib.Path(aoSystemMain.__file__).parent.parent.absolute())


def _ao_path():
    return str(pathlib.Path(aoSystemMain.__file__).parent.absolute())


class TestFourierModelFitting(unittest.TestCase):
    """Tests for the Fourier spatial-frequency AO model and PSF fitting."""

    @classmethod
    def setUpClass(cls):
        cls.path_p3 = _p3_path()
        cls.path_ini = cls.path_p3 + '/aoSystem/parFiles/KECKII_NIRC2_20130801_12_00_19.254.ini'
        cls.path_img = cls.path_p3 + '/aoSystem/data/20130801_n0004.fits'
        cls.im_nirc2 = fits.getdata(cls.path_img, ignore_missing_simple=True)
        os.chdir(cls.path_p3)

    def tearDown(self):
        plt.close('all')

    def test_fourier_model_instantiation(self):
        """fourierModel instantiates without errors for KECKII/NIRC2 config."""
        fao = fourierModel(self.path_ini, path_root=self.path_p3,
                           calcPSF=False, verbose=False, display=False)
        self.assertIsNotNone(fao)

    def test_science_field_of_view_auto_from_minus_one(self):
        """aoSystem auto-computes science FieldOfView when the config value is `-1`."""
        template_ini = os.path.join(self.path_p3, 'dummy.ini')
        rad2mas = 180 * 3600 * 1e3 / np.pi

        config = ConfigParser()
        config.optionxform = str
        config.read(template_ini)
        config.set('sensor_science', 'FieldOfView', '-1')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as temp_file:
            config.write(temp_file)
            temp_name = temp_file.name

        try:
            ao = aoSystem(temp_name, path_root=self.path_p3, verbose=False)
            wvl_min = float(np.min(np.atleast_1d(ao.wvlGs)))
            pitch_min = float(np.min(np.atleast_1d(ao.dms.pitch)))
            expected_fov = int(np.ceil(rad2mas * wvl_min / (pitch_min * ao.cam.psInMas)))
            self.assertEqual(ao.cam.fovInPix, expected_fov)
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)

    def test_fourier_fitting(self):
        """psfFitting with fourierModel runs without errors."""
        fao = fourierModel(self.path_ini, path_root=self.path_p3,
                           calcPSF=True, verbose=False, display=False)
        x0 = fao.ao.cam.spotFWHM[0] + [1, 0, 0, 0]
        fixed = (True,) * 3 + (False,) * 4
        res = psfFitting(self.im_nirc2, fao, x0, verbose=0, fixed=fixed)
        self.assertIsNotNone(res)


@unittest.skipUnless(HAS_MAOPPY, MAOPPY_SKIP_REASON)
class TestPSFAO21(unittest.TestCase):
    """Tests for the PSFAO21 model (instantiation, PSF sweeps, fitting)."""

    @classmethod
    def setUpClass(cls):
        cls.path_p3 = _p3_path()
        cls.path_ini = cls.path_p3 + '/aoSystem/parFiles/KECKII_NIRC2_20130801_12_00_19.254.ini'
        cls.path_img = cls.path_p3 + '/aoSystem/data/20130801_n0004.fits'
        cls.im_nirc2 = fits.getdata(cls.path_img, ignore_missing_simple=True)
        os.chdir(cls.path_p3)

    def tearDown(self):
        plt.close('all')

    def test_psfao21_instantiation(self):
        """psfao21 instantiates without errors."""
        psfao = psfao21(self.path_ini, path_root=self.path_p3)
        self.assertIsNotNone(psfao)

    def test_psfao21_psf_sweep_r0(self):
        """PSF computation for five r0 values produces finite, non-NaN arrays."""
        psfao = psfao21(self.path_ini, path_root=self.path_p3)
        for r0 in [0.2, 0.3, 0.4, 0.5, 0.8]:
            x0 = [r0, 4e-2, 0.5, 1e-2, 1, 0, 1.5, 0, 0, 0, 1.0, 0, 0, 0]
            psf = np.squeeze(psfao(x0))
            self.assertFalse(np.any(np.isnan(psf)), f"NaN in PSF for r0={r0}")
            self.assertFalse(np.any(np.isinf(psf)), f"Inf in PSF for r0={r0}")

    def test_psfao21_psf_sweep_sigma2(self):
        """PSF computation for five sigma^2 values produces finite, non-NaN arrays."""
        psfao = psfao21(self.path_ini, path_root=self.path_p3)
        for sigma2 in [0.05, 0.1, 0.25, 0.5, 1.0]:
            x0 = [0.4, 4e-2, sigma2, 1e-2, 1, 0, 1.5, 0, 0, 0, 1.0, 0, 0, 0]
            psf = np.squeeze(psfao(x0))
            self.assertFalse(np.any(np.isnan(psf)), f"NaN in PSF for sigma2={sigma2}")
            self.assertFalse(np.any(np.isinf(psf)), f"Inf in PSF for sigma2={sigma2}")

    def test_psfao21_psf_sweep_alpha(self):
        """PSF computation for five alpha values produces finite, non-NaN arrays."""
        psfao = psfao21(self.path_ini, path_root=self.path_p3)
        for alpha in [1e-2, 1e-1, 2.5e-1, 5e-1, 1e0]:
            x0 = [0.4, 4e-2, 0.5, alpha, 1, 0, 1.5, 0, 0, 0, 1.0, 0, 0, 0]
            psf = np.squeeze(psfao(x0))
            self.assertFalse(np.any(np.isnan(psf)), f"NaN in PSF for alpha={alpha}")
            self.assertFalse(np.any(np.isinf(psf)), f"Inf in PSF for alpha={alpha}")

    def test_psfao21_psf_sweep_beta(self):
        """PSF computation for five beta values produces finite, non-NaN arrays."""
        psfao = psfao21(self.path_ini, path_root=self.path_p3)
        for beta in [1.1, 1.5, 1.8, 2.8, 3.8]:
            x0 = [0.4, 4e-2, 0.5, 1e-2, 1, 0, beta, 0, 0, 0, 1.0, 0, 0, 0]
            psf = np.squeeze(psfao(x0))
            self.assertFalse(np.any(np.isnan(psf)), f"NaN in PSF for beta={beta}")
            self.assertFalse(np.any(np.isinf(psf)), f"Inf in PSF for beta={beta}")

    def test_psfao21_fitting(self):
        """psfFitting with psfao21 runs without errors."""
        psfao = psfao21(self.path_ini, path_root=self.path_p3)
        n_modes = psfao.ao.tel.nModes
        x0 = [0.7, 4e-2, 0.5, 1e-2, 1, 0, 1.8, 0, 0, 0, 1.0, 0, 0, 0] + [0] * n_modes
        fixed = (False,) * 7 + (True,) * 3 + (False,) * 4 + (True,) * n_modes
        tol = 1e-8
        res = psfFitting(self.im_nirc2, psfao, x0, verbose=0, fixed=fixed,
                         ftol=tol, gtol=tol, xtol=tol, weights=None, normType=1)
        self.assertIsNotNone(res)


class TestPSFReconstruction(unittest.TestCase):
    """Tests for PSF reconstruction from Keck telemetry data.

    The telemetry .sav file is downloaded automatically if not present.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from p3.telemetry.telemetryKeck import telemetryKeck
            from p3.telemetry.systemDiagnosis import systemDiagnosis
            from p3.telemetry.configFile import configFile
            from p3.psfr.psfR import psfR
            import p3.psfr.psfrUtils as psfrUtils
        except ImportError as e:
            raise unittest.SkipTest(f"Telemetry dependencies not available: {e}")

        cls.telemetryKeck = telemetryKeck
        cls.systemDiagnosis = systemDiagnosis
        cls.configFile = configFile
        cls.psfR = psfR
        cls.psfrUtils = psfrUtils

        cls.path_p3 = _p3_path()
        cls.path_root = cls.path_p3 + '/'
        cls.path_img = cls.path_p3 + '/aoSystem/data/20130801_n0004.fits'
        cls.path_calib = cls.path_p3 + '/aoSystem/data/KECK_CALIBRATION/'
        cls.im_nirc2 = fits.getdata(cls.path_img, ignore_missing_simple=True)
        cls.filename = 'n0004_fullNGS_trs.sav'
        os.chdir(cls.path_p3)

        path_sav = cls.path_root + cls.filename
        if not os.path.isfile(path_sav):
            try:
                cls.psfrUtils.get_data_file(cls.path_root, cls.filename)
            except Exception as e:
                raise unittest.SkipTest(
                    f"Telemetry file not available and download failed: {e}"
                )

        cls.trs = cls.telemetryKeck(path_sav, cls.path_img, cls.path_calib,
                                    path_save=cls.path_root, nLayer=1)
        cls.sd = cls.systemDiagnosis(cls.trs)
        cls.configFile(cls.sd)

    def tearDown(self):
        plt.close('all')

    def test_telemetry_instantiation(self):
        """telemetryKeck object is created successfully."""
        self.assertIsNotNone(self.trs)

    def test_system_diagnosis(self):
        """systemDiagnosis object is created successfully."""
        self.assertIsNotNone(self.sd)

    def test_config_file(self):
        """configFile runs without errors."""
        cfg = self.configFile(self.sd)
        self.assertIsNotNone(cfg)

    def test_psfr_instantiation(self):
        """psfR instantiates without errors."""
        psfr = self.psfR(self.sd.trs)
        self.assertIsNotNone(psfr)

    def test_psf_reconstruction_fitting(self):
        """PSF reconstruction + psfFitting runs without errors."""
        psfr = self.psfR(self.sd.trs)
        r0 = psfr.ao.atm.r0
        x0 = [r0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        fixed = (True,) * 3 + (False,) * 4
        tol = 1e-5
        res = psfFitting(self.im_nirc2, psfr, x0, verbose=0, fixed=fixed,
                         ftol=tol, xtol=tol, gtol=tol)
        self.assertIsNotNone(res)

    def test_prime_fitting(self):
        """PRIME fitting (r0 + optical gain) runs without errors."""
        psfr = self.psfR(self.sd.trs)
        n_modes = psfr.ao.tel.nModes
        x0 = [0.7, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0] + [0] * n_modes
        fixed = (False,) * 2 + (True,) + (False,) * 4 + (True,) * n_modes
        tol = 1e-5
        res = psfFitting(self.im_nirc2, psfr, x0, verbose=0,
                         fixed=fixed, xtol=tol, ftol=tol, gtol=tol)
        self.assertIsNotNone(res)


class TestAOSystemExtracted(unittest.TestCase):
    """Subset of tests extracted from p3/aoSystem/testing/tests_aoSystem.py."""

    @classmethod
    def setUpClass(cls):
        cls.path_p3 = _p3_path()
        cls.path_ao = _ao_path()
        os.chdir(cls.path_p3)

    def tearDown(self):
        plt.close('all')

    def test_make_keck_pupil_no_display(self):
        """Build Keck pupil geometry without calling display."""
        path_txt = self.path_ao + '/_txtFile/Keck_segmentVertices.txt'
        spi_ref = spiders([0, 60, 120], 0.0254, symetric=True, D=10.5)
        keck_pup = pupil(segClass=segment(6, 0.9, 200), segCoord=path_txt,
                         D=10.5, cobs=0.2311, spiderClass=spi_ref,
                         fill_gap=False)
        self.assertIsNotNone(keck_pup)

    def test_make_elt_pupil_no_display(self):
        """Build ELT pupil geometry without calling display."""
        path_txt = self.path_ao + '/_txtFile/ELT_segmentVertices.txt'
        spi_ref = spiders([0, 60, 120], 0.5, symetric=True, D=39)
        elt_pup = pupil(segClass=segment(6, 1.42 / 2, 50), segCoord=path_txt,
                        D=39, cobs=0.2375, spiderClass=spi_ref,
                        fill_gap=True)
        self.assertIsNotNone(elt_pup)

    def test_init_selected_systems_and_frequency_domain(self):
        """Instantiate aoSystem and frequencyDomain for selected configs."""
        for sys_name in ['nirc2', 'irdis', 'eris']:
            with self.subTest(system=sys_name):
                path_ini = self.path_ao + '/parFiles/' + sys_name + '.ini'
                ao = aoSystem(path_ini, path_root=self.path_p3)
                freq = frequencyDomain(ao)
                self.assertIsNotNone(ao)
                self.assertIsNotNone(freq)

    def test_fourier_psd_selected_systems(self):
        """Run Fourier PSD model (no PSF) for selected configs."""
        for sys_name in ['nirc2', 'irdis', 'eris']:
            with self.subTest(system=sys_name):
                path_ini = self.path_ao + '/parFiles/' + sys_name + '.ini'
                fao = fourierModel(path_ini, path_root=self.path_p3,
                                   calcPSF=False, verbose=False, display=False,
                                   getErrorBreakDown=True,
                                   getFWHM=False,
                                   getEncircledEnergy=False,
                                   getEnsquaredEnergy=False,
                                   displayContour=False)
                self.assertIsNotNone(fao)

    def test_fourier_psf_nirc2(self):
        """Run Fourier PSF model for NIRC2 with display disabled."""
        path_ini = self.path_ao + '/parFiles/nirc2.ini'
        fao = fourierModel(path_ini, path_root=self.path_p3,
                           calcPSF=True, verbose=False, display=False,
                           getErrorBreakDown=True,
                           getFWHM=True,
                           getEncircledEnergy=True,
                           getEnsquaredEnergy=True,
                           displayContour=False)
        self.assertIsNotNone(fao)


if __name__ == '__main__':
    unittest.main(verbosity=2)
