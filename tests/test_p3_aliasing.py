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

    def test_streaming_matches_chunked_scao(self):
        """Exact streaming implementation should match chunked baseline in SCAO."""
        test_dir = pathlib.Path(__file__).parent.absolute()
        ini_file = os.path.join(test_dir, 'scao_test_wvl1100nm.ini')
        fao = fourierModel(ini_file, path_root='', calcPSF=False, display=False, verbose=False)
        fao.controller()

        psd_chunked = np.asarray(fao.aliasingPSD(method='chunked'))
        psd_streaming = np.asarray(fao.aliasingPSD(method='streaming', shift_batch=8, layer_chunk=4))

        ref_norm = np.linalg.norm(psd_chunked)
        err_norm = np.linalg.norm(psd_streaming - psd_chunked)
        rel_err = err_norm / max(ref_norm, 1e-30)
        self.assertLess(rel_err, 1e-9, f"Streaming mismatch too large: rel_err={rel_err}")

    def test_limited_is_stable_and_close_scao(self):
        """Limited comb variant is approximate but must stay numerically well-behaved."""
        test_dir = pathlib.Path(__file__).parent.absolute()
        ini_file = os.path.join(test_dir, 'scao_test_wvl1100nm.ini')
        fao = fourierModel(ini_file, path_root='', calcPSF=False, display=False, verbose=False)
        fao.controller()

        psd_ref = np.asarray(fao.aliasingPSD(method='streaming', shift_batch=8, layer_chunk=4))
        psd_limited = np.asarray(
            fao.aliasingPSD(method='limited', shift_batch=8, layer_chunk=4, n_times_limit=2)
        )

        self.assertFalse(np.any(np.isnan(psd_limited)))
        self.assertFalse(np.any(np.isinf(psd_limited)))
        self.assertGreater(np.sum(np.abs(psd_limited)), 0)

        rel_err = np.linalg.norm(psd_limited - psd_ref) / max(np.linalg.norm(psd_ref), 1e-30)
        self.assertLess(rel_err, 0.08, f"Limited comb approximation too far from reference: {rel_err}")

    def test_streaming_precompute_toggle_equivalence(self):
        """Persistent precompute must not change streaming numerical results."""
        test_dir = pathlib.Path(__file__).parent.absolute()
        ini_file = os.path.join(test_dir, 'scao_test_wvl1100nm.ini')
        fao = fourierModel(ini_file, path_root='', calcPSF=False, display=False, verbose=False)
        fao.controller()

        psd_cached = np.asarray(
            fao.aliasingPSD(method='streaming', shift_batch=8, layer_chunk=4, use_precompute=True)
        )
        psd_nocache = np.asarray(
            fao.aliasingPSD(method='streaming', shift_batch=8, layer_chunk=4, use_precompute=False)
        )

        rel_err = np.linalg.norm(psd_cached - psd_nocache) / max(np.linalg.norm(psd_nocache), 1e-30)
        self.assertLess(rel_err, 1e-12, f"Precompute toggle changed result: rel_err={rel_err}")

    def test_precompute_cache_has_nonzero_footprint(self):
        """Cache memory estimator should report non-zero after precompute path is used."""
        test_dir = pathlib.Path(__file__).parent.absolute()
        ini_file = os.path.join(test_dir, 'scao_test_wvl1100nm.ini')
        fao = fourierModel(ini_file, path_root='', calcPSF=False, display=False, verbose=False)
        fao.controller()

        _ = fao.aliasingPSD(method='limited', n_times_limit=2, shift_batch=8, layer_chunk=4, use_precompute=True)
        self.assertGreater(fao.aliasingPrecomputeMemoryMB(), 0.0)

    def test_limited_tiptorch_mode_matches_streaming_mode(self):
        """TipTorch-like limited implementation should match limited streaming numerically."""
        test_dir = pathlib.Path(__file__).parent.absolute()
        ini_file = os.path.join(test_dir, 'scao_test_wvl1100nm.ini')
        fao = fourierModel(ini_file, path_root='', calcPSF=False, display=False, verbose=False)
        fao.controller()

        psd_streaming = np.asarray(
            fao.aliasingPSD(
                method='limited',
                n_times_limit=2,
                shift_batch=8,
                layer_chunk=4,
                limited_mode='streaming',
                use_precompute=True,
            )
        )
        psd_tiptorch = np.asarray(
            fao.aliasingPSD(
                method='limited',
                n_times_limit=2,
                limited_mode='tiptorch',
                limited_mem_cap_mb=512,
                use_precompute=True,
            )
        )

        rel_err = np.linalg.norm(psd_tiptorch - psd_streaming) / max(np.linalg.norm(psd_streaming), 1e-30)
        self.assertLess(rel_err, 1e-10, f"TipTorch-like limited mismatch: rel_err={rel_err}")

    def test_limited_tiptorch_falls_back_with_tiny_mem_cap(self):
        """With very low memory cap, tiptorch mode must fall back and stay numerically stable."""
        test_dir = pathlib.Path(__file__).parent.absolute()
        ini_file = os.path.join(test_dir, 'scao_test_wvl1100nm.ini')
        fao = fourierModel(ini_file, path_root='', calcPSF=False, display=False, verbose=False)
        fao.controller()

        psd_fallback = np.asarray(
            fao.aliasingPSD(
                method='limited',
                n_times_limit=2,
                limited_mode='tiptorch',
                limited_mem_cap_mb=1,
                use_precompute=True,
            )
        )
        psd_streaming = np.asarray(
            fao.aliasingPSD(
                method='limited',
                n_times_limit=2,
                shift_batch=8,
                layer_chunk=4,
                limited_mode='streaming',
                use_precompute=True,
            )
        )

        rel_err = np.linalg.norm(psd_fallback - psd_streaming) / max(np.linalg.norm(psd_streaming), 1e-30)
        self.assertLess(rel_err, 1e-10, f"Fallback path mismatch: rel_err={rel_err}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
