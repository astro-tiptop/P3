#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct regression tests for the per-wavelength PSD equivalence bug that
motivated frequencyDomain.wvl_grids / fourierModel's per-wavelength PSD loop:

Before this change, a single shared frequency grid (sized for the most
demanding requested wavelength) was used to compute self.PSD for *every*
requested science wavelength, so a wavelength's slice inside a multi-wavelength
run did not numerically match a standalone single-wavelength run for that same
wavelength (see frequencyDomain.py / fourierModel.py for the underlying
PSDstep/resAO/nOtf construction).

These tests assert bit-for-bit equivalence between:
  - each wavelength's slice of self.PSD from a multi-wavelength run
    (psdExpansion=True, several entries in [sources_science] Wavelength), and
  - self.PSD from a standalone run of an otherwise-identical config with only
    that one wavelength requested,
for both a SCAO system and a tomographic (MCAO, nGs>=2) system -- the
tomographic case additionally exercises the reconstructor/controller
recomputation per wavelength (spatialReconstructor/controller), not just the
PSD term methods.
"""

import pathlib
import tempfile
import os
import re
import unittest

import numpy as nnp

import p3.aoSystem as aoSystemMain
from p3.aoSystem.fourierModel import fourierModel


def _p3_path():
    return str(pathlib.Path(aoSystemMain.__file__).parent.parent.parent.absolute())


def _build_model(ini_path, path_p3, psdExpansion=True, psdPerWavelength=True):
    return fourierModel(
        ini_path,
        path_root=path_p3,
        calcPSF=False,
        verbose=False,
        display=False,
        psdExpansion=psdExpansion,
        psdPerWavelength=psdPerWavelength,
    )


def _single_wavelength_variant(multi_ini_path, wavelengths_line, single_wvl_expr, tmp_dir):
    """
    Writes a copy of multi_ini_path with its [sources_science] Wavelength list
    replaced by a single wavelength, returns the new path.
    """
    with open(multi_ini_path) as f:
        content = f.read()
    new_content = content.replace(wavelengths_line, f"Wavelength = [{single_wvl_expr}]")
    assert new_content != content, "Wavelength line not found for substitution"
    fd, path = tempfile.mkstemp(suffix='.ini', dir=tmp_dir)
    with os.fdopen(fd, 'w') as f:
        f.write(new_content)
    return path


class TestScaoMultiWavelengthEquivalence(unittest.TestCase):
    """
    scao_test_multi_wvl.ini requests [1200e-9, 1660e-9, 2200e-9] with
    psdExpansion=True; each slice of the resulting self.PSD list must be
    bit-identical to a standalone run requesting only that one wavelength.
    """

    @classmethod
    def setUpClass(cls):
        cls.path_p3 = _p3_path()
        cls.multi_ini = cls.path_p3 + '/tests/scao_test_multi_wvl.ini'
        cls.wvls_nm = [1200, 1660, 2200]
        cls.tmp_dir = tempfile.mkdtemp(prefix='p3_multi_wvl_equiv_')
        cls.multi_model = _build_model(cls.multi_ini, cls.path_p3)

    def test_psd_is_a_list_with_one_entry_per_wavelength(self):
        self.assertIsInstance(self.multi_model.PSD, list)
        self.assertEqual(len(self.multi_model.PSD), len(self.wvls_nm))

    def test_each_wavelength_slice_matches_standalone_run(self):
        for idx, wvl_nm in enumerate(self.wvls_nm):
            with self.subTest(wvl_nm=wvl_nm):
                mono_ini = _single_wavelength_variant(
                    self.multi_ini,
                    'Wavelength = [1200e-9, 1660e-9, 2200e-9]',
                    f'{wvl_nm}e-9',
                    self.tmp_dir,
                )
                mono_model = _build_model(mono_ini, self.path_p3)

                multi_psd = nnp.asarray(self.multi_model.PSD[idx])
                mono_psd = nnp.asarray(mono_model.PSD)

                self.assertEqual(multi_psd.shape, mono_psd.shape,
                                 f"shape mismatch at {wvl_nm}nm")
                nnp.testing.assert_array_equal(
                    multi_psd, mono_psd,
                    err_msg=f"PSD slice for {wvl_nm}nm differs from standalone run"
                )


class TestMcaoMultiWavelengthEquivalence(unittest.TestCase):
    """
    Same equivalence property for a tomographic (MCAO, nGs>=2) system, which
    additionally exercises spatialReconstructor()/controller() recomputation
    per wavelength (tomographicReconstructor/optimalProjector), not just the
    PSD term methods.
    """

    @classmethod
    def setUpClass(cls):
        cls.path_p3 = _p3_path()
        cls.mcao_ini = cls.path_p3 + '/p3/aoSystem/parFiles/MavisMCAO.ini'
        cls.wvls_nm = [550, 750]
        cls.tmp_dir = tempfile.mkdtemp(prefix='p3_mcao_multi_wvl_equiv_')

        with open(cls.mcao_ini) as f:
            content = f.read()
        multi_content = content.replace(
            'Wavelength = [640e-9]',
            f'Wavelength = [{cls.wvls_nm[0]}e-9, {cls.wvls_nm[1]}e-9]',
        )
        assert multi_content != content
        fd, cls.multi_ini = tempfile.mkstemp(suffix='.ini', dir=cls.tmp_dir)
        with os.fdopen(fd, 'w') as f:
            f.write(multi_content)

        cls.multi_model = _build_model(cls.multi_ini, cls.path_p3)

    def test_is_tomographic(self):
        self.assertGreaterEqual(self.multi_model.nGs, 2)

    def test_each_wavelength_slice_matches_standalone_run(self):
        for idx, wvl_nm in enumerate(self.wvls_nm):
            with self.subTest(wvl_nm=wvl_nm):
                mono_ini = _single_wavelength_variant(
                    self.mcao_ini,
                    'Wavelength = [640e-9]',
                    f'{wvl_nm}e-9',
                    self.tmp_dir,
                )
                mono_model = _build_model(mono_ini, self.path_p3)

                multi_psd = nnp.asarray(self.multi_model.PSD[idx])
                mono_psd = nnp.asarray(mono_model.PSD)

                self.assertEqual(multi_psd.shape, mono_psd.shape,
                                 f"shape mismatch at {wvl_nm}nm")
                nnp.testing.assert_array_equal(
                    multi_psd, mono_psd,
                    err_msg=f"PSD slice for {wvl_nm}nm differs from standalone run"
                )


class TestLegacySingleWavelengthUnchanged(unittest.TestCase):
    """
    psdPerWavelength=False (the default), or a single requested wavelength,
    must produce self.PSD as a plain array (not a list) -- the pre-refactor
    interface, with no change in shape or values.

    psdPerWavelength is a *separate* flag from psdExpansion: TIPTOP's
    baseSimulation.py always constructs fourierModel with psdExpansion=True
    (unconditionally, regardless of how many science wavelengths are
    requested) -- see tiptop/baseSimulation.py. If the new per-wavelength PSD
    list behaviour had been wired to psdExpansion instead of a new flag, every
    existing multi-wavelength TIPTOP run would have silently started
    receiving a list instead of an array for self.PSD. This class is the
    regression guard for that specific compatibility requirement.
    """

    def test_single_wavelength_psd_is_plain_array_not_list(self):
        path_p3 = _p3_path()
        fao = _build_model(path_p3 + '/tests/scao_test_wvl1100nm.ini', path_p3,
                           psdExpansion=True, psdPerWavelength=True)
        self.assertNotIsInstance(fao.PSD, list)
        self.assertEqual(len(fao.freq.wvl_grids), 1)

    def test_tiptop_call_pattern_stays_single_array_for_multi_wavelength(self):
        """
        Exact parameter pattern TIPTOP uses today (psdExpansion=True,
        psdPerWavelength not passed -> defaults to False) on a
        multi-wavelength config: self.PSD must stay a plain array.
        """
        path_p3 = _p3_path()
        fao = fourierModel(
            path_p3 + '/tests/scao_test_multi_wvl.ini',
            path_root=path_p3, calcPSF=False, verbose=False, display=False,
            psdExpansion=True,
        )
        self.assertNotIsInstance(fao.PSD, list)
        self.assertEqual(len(fao.freq.wvl_grids), 1)

    def test_psd_per_wavelength_false_with_multi_wavelength_config_stays_single_grid(self):
        path_p3 = _p3_path()
        fao = _build_model(path_p3 + '/tests/scao_test_multi_wvl.ini', path_p3,
                           psdExpansion=False, psdPerWavelength=False)
        self.assertNotIsInstance(fao.PSD, list)
        self.assertEqual(len(fao.freq.wvl_grids), 1)


if __name__ == '__main__':
    unittest.main()
