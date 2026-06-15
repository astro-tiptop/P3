#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for frequencyDomain.kRef_float — the exact (float) oversampling factor
introduced to give the correct target pixel scale in multi-wavelength mode.

The key invariant is:
    wvl_min * PSDstep * kRef_float * rad2mas == psInMas   (within floating-point)
and for multi-λ:
    kRef_float = k_[idxPmin] * wvl[idxPmin] / wvl_min   (may be non-integer)
    kRef_       = ceil(kRef_float)                        (integer, for grid size)
"""

import math
import pathlib
import unittest

import numpy as nnp

import p3.aoSystem as aoSystemMain
from p3.aoSystem.fourierModel import fourierModel

_RAD2MAS = 3600 * 180 * 1000 / nnp.pi


def _p3_path():
    return str(pathlib.Path(aoSystemMain.__file__).parent.parent.parent.absolute())


def _build_model(ini_path, path_p3, psdExpansion=False):
    return fourierModel(
        ini_path,
        path_root=path_p3,
        calcPSF=False,
        verbose=False,
        display=False,
        psdExpansion=psdExpansion,
    )


class TestKRefFloatAttribute(unittest.TestCase):
    """kRef_float must exist on frequencyDomain after our change."""

    @classmethod
    def setUpClass(cls):
        cls.path_p3 = _p3_path()
        cls.mono_ini  = cls.path_p3 + '/tests/scao_test_wvl1100nm.ini'
        cls.multi_ini = cls.path_p3 + '/tests/scao_test_multi_wvl.ini'

    def test_kref_float_exists_mono(self):
        fao = _build_model(self.mono_ini, self.path_p3)
        self.assertTrue(hasattr(fao.freq, 'kRef_float'),
                        "frequencyDomain is missing kRef_float attribute (mono)")

    def test_kref_float_exists_multi(self):
        fao = _build_model(self.multi_ini, self.path_p3)
        self.assertTrue(hasattr(fao.freq, 'kRef_float'),
                        "frequencyDomain is missing kRef_float attribute (multi-λ)")


class TestKRefFloatMono(unittest.TestCase):
    """For a single science wavelength kRef_float must equal kRef_ (integer)."""

    @classmethod
    def setUpClass(cls):
        path_p3 = _p3_path()
        cls.fao = _build_model(path_p3 + '/tests/scao_test_wvl1100nm.ini', path_p3)

    def test_kref_float_equals_kref_for_mono(self):
        freq = self.fao.freq
        self.assertAlmostEqual(float(freq.kRef_float), float(freq.kRef_), places=9,
                               msg="Mono: kRef_float should equal kRef_ (integer value)")

    def test_target_pixel_scale_correct_mono(self):
        freq = self.fao.freq
        psInMas = float(freq.psInMas[0])
        wvl_min = float(freq.wvlRef)
        PSDstep = float(freq.PSDstep)
        target_ps_mas = wvl_min * PSDstep * float(freq.kRef_float) * _RAD2MAS
        self.assertAlmostEqual(target_ps_mas, psInMas, places=4,
                               msg=f"Mono: target_ps={target_ps_mas:.6f} ≠ psInMas={psInMas}")


class TestKRefFloatMulti(unittest.TestCase):
    """
    For three science wavelengths [1.2, 1.66, 2.2] µm with psdExpansion=True
    (as TIPTOP sets it in baseSimulation.py):
    - kRef_float must be non-integer (because idxPmin is wvl_max, not wvl_min)
    - kRef_  = ceil(kRef_float)
    - target pixel scale = psInMas  (the core correctness property)
    - kRef_float formula: k_[idxPmin] * wvl[idxPmin] / wvl_min
    """

    @classmethod
    def setUpClass(cls):
        path_p3 = _p3_path()
        cls.fao = _build_model(path_p3 + '/tests/scao_test_multi_wvl.ini', path_p3,
                               psdExpansion=True)

    def test_kref_float_is_non_integer_for_multi(self):
        kRef_float = float(self.fao.freq.kRef_float)
        kRef_int   = int(self.fao.freq.kRef_)
        self.assertNotAlmostEqual(kRef_float, float(kRef_int), places=3,
                                  msg="Multi-λ: kRef_float should differ from kRef_ (non-integer)")

    def test_kref_is_ceil_of_kref_float(self):
        kRef_float = float(self.fao.freq.kRef_float)
        kRef_int   = int(self.fao.freq.kRef_)
        self.assertEqual(kRef_int, math.ceil(kRef_float),
                         msg=f"kRef_={kRef_int} ≠ ceil(kRef_float={kRef_float:.4f})")

    def test_target_pixel_scale_correct_multi(self):
        freq = self.fao.freq
        psInMas  = float(freq.psInMas[0])
        wvl_min  = float(freq.wvlRef)
        PSDstep  = float(freq.PSDstep)
        kRef_float = float(freq.kRef_float)
        target_ps_mas = wvl_min * PSDstep * kRef_float * _RAD2MAS
        self.assertAlmostEqual(target_ps_mas, psInMas, places=4,
                               msg=f"Multi-λ: target_ps={target_ps_mas:.6f} ≠ psInMas={psInMas}")

    def test_kref_float_matches_formula(self):
        """kRef_float = k_[idxPmin] * wvl[idxPmin] / wvl_min."""
        from p3.aoSystem import asnumpy
        freq = self.fao.freq

        wvl_ = nnp.asarray(asnumpy(freq.wvl_), dtype=nnp.float64)
        k_   = nnp.asarray(asnumpy(freq.k_),   dtype=nnp.float64)
        psInMas_ = nnp.asarray(asnumpy(freq.psInMas), dtype=nnp.float64)

        psdSteps = psInMas_ / (wvl_ * _RAD2MAS * k_)
        idxPmin  = int(nnp.argmin(psdSteps))
        idxWmin  = int(nnp.argmin(wvl_))

        expected = float(k_[idxPmin]) * (float(wvl_[idxPmin]) / float(wvl_[idxWmin]))
        actual   = float(freq.kRef_float)
        self.assertAlmostEqual(actual, expected, places=9,
                               msg=f"kRef_float={actual:.6f} ≠ formula value {expected:.6f}")

    def test_nOtf_uses_kref_integer(self):
        freq = self.fao.freq
        expected_nOtf = freq.nPix * int(freq.kRef_)
        self.assertEqual(int(freq.nOtf), expected_nOtf,
                         msg=f"nOtf={freq.nOtf} ≠ nPix*kRef_={expected_nOtf}")


if __name__ == '__main__':
    unittest.main()
