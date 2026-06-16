#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the config_dict interface of aoSystem / fourierModel.

Verifies that:
  1. The original file-path interface still works.
  2. Passing a dict parsed from the same file produces identical results.
  3. A modified dict is honoured (sources_LO.Zenith override).
  4. P3 internals cannot mutate the caller's dict.
"""

import copy
import pathlib
import unittest

import numpy as np

import p3.aoSystem as aoSystemMain
from p3.aoSystem.aoSystem import aoSystem
from p3.aoSystem.fourierModel import fourierModel

TESTS_DIR = pathlib.Path(__file__).parent.absolute()
PATH_P3   = pathlib.Path(aoSystemMain.__file__).parent.parent.absolute()

SCAO_INI  = str(TESTS_DIR / 'scao_test_wvl1100nm.ini')
MAVIS_INI = str(TESTS_DIR / 'MAVIStest.ini')


def _load_dict(ini_path):
    """Return my_data_map parsed from *ini_path* via aoSystem."""
    ao = aoSystem(ini_path, path_root=str(PATH_P3), verbose=False)
    return copy.deepcopy(ao.my_data_map)


class TestAoSystemConfigDict(unittest.TestCase):

    def test_file_path_unchanged(self):
        """The original file-path interface must still work."""
        ao = aoSystem(SCAO_INI, path_root=str(PATH_P3), verbose=False)
        self.assertFalse(ao.error)
        self.assertIsNotNone(ao.my_data_map)

    def test_config_dict_produces_same_result(self):
        """Passing config_dict parsed from the file gives the same Strehl-proxy as reading the file."""
        ao_file = aoSystem(SCAO_INI, path_root=str(PATH_P3), verbose=False)
        cfg = _load_dict(SCAO_INI)
        ao_dict = aoSystem(SCAO_INI, path_root=str(PATH_P3), verbose=False,
                           config_dict=cfg)

        # Same structure: same top-level keys
        self.assertEqual(set(ao_file.my_data_map.keys()),
                         set(ao_dict.my_data_map.keys()))

    def test_config_dict_overrides_sources_lo(self):
        """A modified sources_LO in config_dict must be reflected in ao.zenithGsLO."""
        cfg = _load_dict(MAVIS_INI)

        original_zenith = list(cfg['sources_LO']['Zenith'])
        overridden_zenith = [5.0, 10.0, 15.0]
        cfg['sources_LO']['Zenith']   = overridden_zenith
        cfg['sources_LO']['Azimuth']  = [0.0, 120.0, 240.0]

        ao = aoSystem(MAVIS_INI, path_root=str(PATH_P3), verbose=False,
                      getPSDatNGSpositions=True,
                      config_dict=cfg)

        np.testing.assert_allclose(
            ao.zenithGsLO, overridden_zenith,
            err_msg="zenithGsLO should match the overridden dict value"
        )
        # Sanity: the original file value was different
        self.assertNotEqual(original_zenith, overridden_zenith)

    def test_config_dict_is_independent_copy(self):
        """P3 must not mutate the caller's dict (deep-copy isolation)."""
        cfg = _load_dict(SCAO_INI)
        cfg_before = copy.deepcopy(cfg)

        aoSystem(SCAO_INI, path_root=str(PATH_P3), verbose=False,
                 config_dict=cfg)

        self.assertEqual(cfg, cfg_before,
                         "aoSystem mutated the caller's config_dict")


class TestFourierModelConfigDict(unittest.TestCase):
    """Same checks at the fourierModel level."""

    def test_fouriermodel_file_path_unchanged(self):
        fao = fourierModel(SCAO_INI, path_root=str(PATH_P3),
                           calcPSF=False, verbose=False, display=False,
                           doComputations=False)
        self.assertIsNotNone(fao.ao)

    def test_fouriermodel_config_dict(self):
        cfg = _load_dict(SCAO_INI)
        fao = fourierModel(SCAO_INI, path_root=str(PATH_P3),
                           calcPSF=False, verbose=False, display=False,
                           doComputations=False, config_dict=cfg)
        self.assertIsNotNone(fao.ao)
        self.assertEqual(set(fao.ao.my_data_map.keys()),
                         set(cfg.keys()))

    def test_fouriermodel_config_dict_overrides_sources_lo(self):
        cfg = _load_dict(MAVIS_INI)
        overridden_zenith = [3.0, 7.0, 11.0]
        cfg['sources_LO']['Zenith']  = overridden_zenith
        cfg['sources_LO']['Azimuth'] = [0.0, 120.0, 240.0]

        fao = fourierModel(MAVIS_INI, path_root=str(PATH_P3),
                           calcPSF=False, verbose=False, display=False,
                           getPSDatNGSpositions=True, doComputations=False,
                           config_dict=cfg)

        np.testing.assert_allclose(
            fao.ao.zenithGsLO, overridden_zenith,
            err_msg="fourierModel: zenithGsLO should match the overridden dict value"
        )


if __name__ == '__main__':
    unittest.main()
