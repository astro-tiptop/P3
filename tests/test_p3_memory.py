#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for memory usage estimation in fourierModel
"""

import unittest
import pathlib
import tracemalloc
import numpy as np
import os

from p3.aoSystem.fourierModel import fourierModel
import p3.aoSystem as aoSystemMain


class TestMemoryUsage(unittest.TestCase):
    """Test cases for memory usage estimation"""

    @classmethod
    def setUpClass(cls):
        """Set up test paths"""
        cls.path_p3 = str(pathlib.Path(aoSystemMain.__file__).parent.parent.absolute())
        cls.path_ao = str(pathlib.Path(aoSystemMain.__file__).parent.absolute())

    def _create_model(self, path_ini, calcPSF=False):
        """Helper to create fourierModel with error handling"""
        os.chdir(self.path_p3)  # Ensure we're in the correct directory for relative paths
        fao = fourierModel(path_ini, path_root=self.path_p3, calcPSF=calcPSF,
                            verbose=False, display=False)
        return fao

    def test_nirc2_memory_estimation(self):
        """Test memory estimation for NIRC2 system"""
        path_ini = self.path_ao + '/parFiles/nirc2.ini'

        fao = self._create_model(path_ini)
        estimated = fao.estimate_memory_usage()

        # Measure actual memory with PSD calculation
        tracemalloc.start()
        fao.powerSpectrumDensity()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        actual_mb = peak / (1024 * 1024)
        estimated_mb = estimated['peak_MB']

        # Estimation should be within 50% of actual
        ratio = estimated_mb / actual_mb
        self.assertGreater(ratio, 0.5,
                          f"Estimation too low: {estimated_mb:.1f}MB vs {actual_mb:.1f}MB")
        self.assertLess(ratio, 2.0,
                       f"Estimation too high: {estimated_mb:.1f}MB vs {actual_mb:.1f}MB")
        print(f"NIRC2: Estimated={estimated_mb:.1f}MB, Actual={actual_mb:.1f}MB, Ratio={ratio:.2f}")

    def test_irdis_memory_estimation(self):
        """Test memory estimation for IRDIS system"""
        path_ini = self.path_ao + '/parFiles/irdis.ini'

        fao = self._create_model(path_ini)
        estimated = fao.estimate_memory_usage()

        tracemalloc.start()
        fao.powerSpectrumDensity()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        actual_mb = peak / (1024 * 1024)
        estimated_mb = estimated['peak_MB']
        ratio = estimated_mb / actual_mb

        print(f"IRDIS: Estimated={estimated_mb:.1f}MB, Actual={actual_mb:.1f}MB, Ratio={ratio:.2f}")
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)

    def test_mavis_memory_estimation(self):
        """Test memory estimation for MAVIS MCAO system"""
        path_ini = self.path_ao + '/parFiles/MavisMCAO.ini'

        fao = self._create_model(path_ini)
        estimated = fao.estimate_memory_usage()

        tracemalloc.start()
        fao.powerSpectrumDensity()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        actual_mb = peak / (1024 * 1024)
        estimated_mb = estimated['peak_MB']
        ratio = estimated_mb / actual_mb

        print(f"MAVIS: Estimated={estimated_mb:.1f}MB, Actual={actual_mb:.1f}MB, Ratio={ratio:.2f}")
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)

    def test_memory_breakdown_structure(self):
        """Test that memory breakdown has expected structure"""
        path_ini = self.path_ao + '/parFiles/nirc2.ini'

        fao = self._create_model(path_ini)
        result = fao.estimate_memory_usage()

        # Check required keys
        self.assertIn('final_MB', result)
        self.assertIn('peak_MB', result)
        self.assertIn('breakdown_final_MB', result)
        self.assertIn('breakdown_peak_temp_MB', result)
        self.assertIn('dimensions', result)

        # Check dimensions
        dims = result['dimensions']
        self.assertIn('nOtf', dims)
        self.assertIn('resAO', dims)
        self.assertIn('nSrc', dims)
        self.assertGreater(dims['nOtf'], 0)
        self.assertGreater(dims['resAO'], 0)

    def test_positive_memory_estimation(self):
        """Test that memory estimation returns positive values"""
        path_ini = self.path_ao + '/parFiles/nirc2.ini'

        fao = self._create_model(path_ini)
        result = fao.estimate_memory_usage()

        self.assertGreater(result['final_MB'], 0)
        self.assertGreater(result['peak_MB'], 0)
        self.assertIsInstance(result['final_MB'], (int, float))
        self.assertIsInstance(result['peak_MB'], (int, float))

    def test_peak_includes_final(self):
        """Test that peak memory is greater than or equal to final memory"""
        path_ini = self.path_ao + '/parFiles/nirc2.ini'

        fao = self._create_model(path_ini)
        result = fao.estimate_memory_usage(include_peak=True)

        self.assertGreaterEqual(result['peak_MB'], result['final_MB'])

    def test_without_peak_estimation(self):
        """Test memory estimation without peak calculation"""
        path_ini = self.path_ao + '/parFiles/nirc2.ini'

        fao = self._create_model(path_ini)
        result = fao.estimate_memory_usage(include_peak=False)

        # Peak should equal final when not including peak
        self.assertEqual(result['peak_MB'], result['final_MB'])
        self.assertEqual(len(result['breakdown_peak_temp_MB']), 0)


if __name__ == '__main__':
    unittest.main()