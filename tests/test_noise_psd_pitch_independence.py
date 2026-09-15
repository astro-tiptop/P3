#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression test for GitHub issue #144 ("Incorrect noise scaling when changing
act pitch"): with a pyramid (or Shack-Hartmann) WFS and a single-GS (SCAO)
system, doubling the DM actuator pitch while keeping the number of
subapertures fixed used to *increase* the noise error term, whereas fitting,
aliasing and servo-lag all behaved as physically expected (see the issue for
the full report).

Root cause: fourierModel.noisePSD()'s SCAO branch normalized the
reconstruction-filter noise propagation by `1/(2*kcMax_)**2`, where kcMax_ is
derived from the DM actuator pitch (frequencyDomain.kc_ = 1/(2*pitch)). But
Rx/Ry (the reconstruction filter itself, see reconstructionFilter()) are built
entirely from the WFS subaperture pitch (self.ao.wfs.optics[0].dsub), never
from the DM pitch. The two coincide only when actuator pitch == subaperture
pitch (a common but not universal AO system design), which is why the bug
went unnoticed until the pitch/dsub ratio was changed deliberately.

The fix normalizes by the WFS subaperture pitch (dsub**2) instead, so the
per-mode noise level stays independent of the DM pitch; the *total* noise
error still decreases when the pitch doubles, because the AO-corrected area
(mskInAO_/resAO, correctly governed by kcMax_) shrinks -- exactly mirroring
the servo-lag term's behaviour, as expected in the issue.
"""

import os
import tempfile
import unittest

import p3.aoSystem as aoSystemMain
from p3.aoSystem.fourierModel import fourierModel


def _p3_path():
    return str(__import__('pathlib').Path(aoSystemMain.__file__).parent.parent.parent.absolute())


def _build_model(ini_path, path_p3):
    return fourierModel(
        ini_path,
        path_root=path_p3,
        calcPSF=False,
        verbose=False,
        display=False,
        getErrorBreakDown=True,
    )


class TestNoisePSDIndependentOfDmPitch(unittest.TestCase):
    """
    Uses tests/scao_test_wvl1100nm.ini (SCAO, pyramid WFS, NumberLenslets=[40],
    DmPitchs=[0.275] -- already a case where actuator pitch != subaperture
    pitch, D/NumberLenslets = 8.222/40 = 0.2056 m != 0.275 m).
    """

    @classmethod
    def setUpClass(cls):
        cls.path_p3 = _p3_path()
        base_ini = cls.path_p3 + '/tests/scao_test_wvl1100nm.ini'
        with open(base_ini) as f:
            cls._base_content = f.read()

        assert 'DmPitchs = [0.275]' in cls._base_content, \
            "fixture .ini changed, update this test's pitch substitution"

        cls._tmp_dir = tempfile.mkdtemp(prefix='p3_noise_pitch_')

        cls.standard_ini = os.path.join(cls._tmp_dir, 'standard_pitch.ini')
        with open(cls.standard_ini, 'w') as f:
            f.write(cls._base_content)

        cls.doubled_ini = os.path.join(cls._tmp_dir, 'doubled_pitch.ini')
        with open(cls.doubled_ini, 'w') as f:
            f.write(cls._base_content.replace('DmPitchs = [0.275]', 'DmPitchs = [0.550]'))

        cls.standard = _build_model(cls.standard_ini, cls.path_p3)
        cls.doubled = _build_model(cls.doubled_ini, cls.path_p3)

    def test_dsub_unchanged_between_configs(self):
        """Sanity check: the two configs really only differ by DM pitch."""
        self.assertEqual(
            float(self.standard.ao.wfs.optics[0].dsub),
            float(self.doubled.ao.wfs.optics[0].dsub),
        )
        self.assertAlmostEqual(
            float(self.doubled.freq.kcMax_), float(self.standard.freq.kcMax_) / 2, places=9,
        )

    def test_fitting_error_increases_with_pitch(self):
        self.assertGreater(float(self.doubled.wfeFit), float(self.standard.wfeFit))

    def test_aliasing_error_decreases_with_pitch(self):
        self.assertLess(float(self.doubled.wfeAl), float(self.standard.wfeAl))

    def test_servo_lag_error_decreases_with_pitch(self):
        self.assertLess(float(self.doubled.wfeST[0]), float(self.standard.wfeST[0]))

    def test_noise_error_decreases_with_pitch_not_increases(self):
        """
        The actual regression: before the fix, this quantity *increased* when
        only the DM pitch was doubled (issue #144). It must now decrease,
        consistently with fitting/aliasing/servo-lag all responding to the
        same shrinking AO-corrected area.
        """
        self.assertLess(float(self.doubled.wfeN[0]), float(self.standard.wfeN[0]))

    def test_noise_psd_peak_value_is_pitch_independent(self):
        """
        More targeted than the integrated WFE: the *per-mode* noise PSD level
        (before masking to the AO-corrected area) must not depend on the DM
        pitch at all, since it is governed entirely by the WFS subaperture
        pitch. Compare the noise PSD value at a spatial frequency shared by
        both grids (e.g. near the center, away from the zeroed DC point).
        """
        std_freq, dbl_freq = self.standard.freq, self.doubled.freq
        std_psd = self.standard.psdNoise
        dbl_psd = self.doubled.psdNoise

        # Both grids share the same PSDstep (same wavelength/pixel-scale
        # config), so a fixed pixel offset from center corresponds to the
        # same physical spatial frequency in both.
        self.assertAlmostEqual(float(std_freq.PSDstep), float(dbl_freq.PSDstep), places=12)

        cy_std, cx_std = std_psd.shape[0] // 2, std_psd.shape[1] // 2
        cy_dbl, cx_dbl = dbl_psd.shape[0] // 2, dbl_psd.shape[1] // 2
        offset = 3

        val_std = float(std_psd[cy_std + offset, cx_std + offset])
        val_dbl = float(dbl_psd[cy_dbl + offset, cx_dbl + offset])

        self.assertGreater(val_std, 0.0)
        self.assertAlmostEqual(val_dbl / val_std, 1.0, places=6,
                               msg=f"per-mode noise PSD changed with DM pitch alone: "
                                   f"standard={val_std:.6e}, doubled={val_dbl:.6e}")


if __name__ == '__main__':
    unittest.main()
