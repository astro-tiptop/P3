#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAVIS regression test inspired by TIPTOP test style.

Compares current P3 MAVIS outputs against stored numpy baselines.
Set P3_UPDATE_MAVIS_BASELINE=1 to (re)generate baseline files.

The command to run this test is:
P3_UPDATE_MAVIS_BASELINE=1 python -m unittest tests/test_p3_mavis.py

Normal execution (without updating baseline) is:
python -m unittest tests/test_p3_mavis.py
"""

import os
import pathlib
import unittest

import numpy as np

import p3.aoSystem as aoSystemMain
from p3.aoSystem.fourierModel import fourierModel


def _cpu_array(value):
    if isinstance(value, (np.ndarray, list, tuple)):
        return value
    if hasattr(value, 'get'):
        return value.get()
    return value


class TestMavisRegression(unittest.TestCase):
    """Regression check for MAVIS outputs against .npy baseline files."""

    @classmethod
    def setUpClass(cls):
        cls.path_p3 = str(pathlib.Path(aoSystemMain.__file__).parent.parent.absolute())
        cls.path_ao = str(pathlib.Path(aoSystemMain.__file__).parent.absolute())
        cls.tests_dir = pathlib.Path(__file__).parent.absolute()
        cls.update_baseline = os.environ.get('P3_UPDATE_MAVIS_BASELINE', '0') == '1'

        cls.path_ini = cls.tests_dir / 'MAVIStest.ini'
        if not cls.path_ini.is_file():
            cls.path_ini = pathlib.Path(cls.path_ao) / 'parFiles' / 'MavisMCAO.ini'

        cls.ref_psd = cls.tests_dir / 'mavisResult0.npy'
        cls.ref_psf = cls.tests_dir / 'mavisResult1.npy'

    def _compute_outputs(self):
        os.chdir(self.path_p3)
        fao = fourierModel(
            str(self.path_ini),
            path_root=self.path_p3,
            calcPSF=True,
            verbose=False,
            display=False,
            getPSDatNGSpositions=True,
        )

        psd = np.asarray(_cpu_array(fao.powerSpectrumDensity().transpose()), dtype=np.float64)
        psf = np.asarray(_cpu_array(fao.PSF), dtype=np.float64)
        return psd, psf

    def test_mavis_against_npy_baseline(self):
        """Compare current MAVIS PSD/PSF with stored .npy references."""
        psd, psf = self._compute_outputs()

        if self.update_baseline:
            np.save(self.ref_psd, psd)
            np.save(self.ref_psf, psf)

        if not self.ref_psd.is_file() or not self.ref_psf.is_file():
            self.skipTest(
                'Missing MAVIS baseline files.'
                ' Run with P3_UPDATE_MAVIS_BASELINE=1 once to generate.'
            )

        stored_psd = np.load(self.ref_psd)
        stored_psf = np.load(self.ref_psf)

        self.assertEqual(psd.shape, stored_psd.shape)
        self.assertEqual(psf.shape, stored_psf.shape)

        self.assertTrue(
            np.testing.assert_allclose(psd, stored_psd, rtol=1e-3, atol=1e-5) is None
        )
        self.assertTrue(
            np.testing.assert_allclose(psf, stored_psf, rtol=1e-3, atol=1e-5) is None
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
