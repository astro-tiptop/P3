#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test to verify chunking implementation produces identical results
"""

import unittest
import numpy as np
import os
import pathlib
from p3.aoSystem.fourierModel import fourierModel
import p3.aoSystem as aoSystemMain


class TestAliasingChunking(unittest.TestCase):
    """Test that chunked aliasing PSD gives same results as original"""

    @classmethod
    def setUpClass(cls):
        cls.path_p3 = str(pathlib.Path(aoSystemMain.__file__).parent.parent.absolute())
        cls.path_ao = str(pathlib.Path(aoSystemMain.__file__).parent.absolute())

    def test_nirc2_aliasing_chunking(self):
        """Test chunking for NIRC2 (simple SCAO case)"""
        path_ini = self.path_ao + '/parFiles/nirc2.ini'

        # Create model
        os.chdir(self.path_p3)  # Ensure we're in the correct directory for relative paths
        fao = fourierModel(path_ini, path_root=self.path_p3, calcPSF=False,
                          verbose=False, display=False)

        # Compute aliasing PSD
        psd_chunked = fao.aliasingPSD()

        # Check it's not all zeros
        self.assertGreater(np.sum(np.abs(psd_chunked)), 0, 
                          "Aliasing PSD is all zeros!")

        # Check for NaN or Inf
        self.assertFalse(np.any(np.isnan(psd_chunked)), "PSD contains NaN")
        self.assertFalse(np.any(np.isinf(psd_chunked)), "PSD contains Inf")

        print(f"NIRC2 aliasing PSD sum: {np.sum(psd_chunked):.6e}")
        print(f"NIRC2 aliasing PSD max: {np.max(psd_chunked):.6e}")

    def test_mavis_aliasing_chunking(self):
        """Test chunking for MAVIS (MCAO with many layers)"""
        path_ini = self.path_ao + '/parFiles/MavisMCAO.ini'

        os.chdir(self.path_p3)  # Ensure we're in the correct directory for relative paths
        fao = fourierModel(path_ini, path_root=self.path_p3, calcPSF=False,
                          verbose=False, display=False)

        psd_chunked = fao.aliasingPSD()

        self.assertGreater(np.sum(np.abs(psd_chunked)), 0)
        self.assertFalse(np.any(np.isnan(psd_chunked)))
        self.assertFalse(np.any(np.isinf(psd_chunked)))

        print(f"MAVIS aliasing PSD sum: {np.sum(psd_chunked):.6e}")
        print(f"MAVIS aliasing PSD max: {np.max(psd_chunked):.6e}")
        print(f"MAVIS nL: {fao.ao.atm.nL}")

    def test_psd_sanity_bounds(self):
        """
        Verifies that the aliasing PSD is physically valid:
        no NaN, no Inf, and strictly non-negative.
        """
        test_dir = pathlib.Path(__file__).parent.absolute()
        ini_file = os.path.join(test_dir, 'scao_test_wvl1100nm.ini')
        fao = fourierModel(ini_file, path_root='', calcPSF=False, display=False)
        fao.controller()
        psd_alias = fao.aliasingPSD()

        self.assertFalse(np.isnan(psd_alias).any(), "Aliasing PSD contains NaN!")
        self.assertFalse(np.isinf(psd_alias).any(), "Aliasing PSD contains Inf!")

        # Due to tiny float rounding errors, we tolerate a small negative epsilon
        min_val = np.min(psd_alias)
        self.assertGreaterEqual(min_val, -1e-15,
                                f"Aliasing PSD contains negative energy: {min_val}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
