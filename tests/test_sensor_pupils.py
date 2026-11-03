#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the own pupils of the [sensor_LO] / [sensor_Focus] sections
(aoSystem.sensorPupil, attributes aoSystem.pupilLO and aoSystem.pupilFocus).

A section has its own pupil if it sets PathPupil (FITS over TelescopeDiameter, rotated by
PupilAngle, interpolated to [telescope] Resolution) or ObscurationRatio (annulus of diameter
TelescopeDiameter). Otherwise it uses the pupil of the previous stage:
telescope -> [sensor_LO] -> [sensor_Focus].

Predictable results checked here:
  - without keys the pupils are the very same objects (no copy);
  - the annulus area is pi/4 D^2 (1 - eps^2);
  - the pupils equal what the telescope class builds with the same inputs;
  - configuration errors are raised at aoSystem construction.
"""

import copy
import pathlib
import tempfile
import unittest

import numpy as np
from astropy.io import fits

import p3.aoSystem as aoSystemMain
from p3.aoSystem import cpuArray
from p3.aoSystem.aoSystem import aoSystem
from p3.aoSystem.telescope import telescope

TESTS_DIR = pathlib.Path(__file__).parent.absolute()
PATH_P3   = pathlib.Path(aoSystemMain.__file__).parent.parent.absolute()
SCAO_INI  = str(TESTS_DIR / 'scao_test_wvl1100nm.ini')

D   = 8.222   # TelescopeDiameter of SCAO_INI
RES = 256     # Resolution of SCAO_INI

# Minimum keys for the [sensor_LO] section to be valid
LO_BASE = {'PixelScale': 10.0, 'FieldOfView': 64}

_BASE_CFG = None


def _base_cfg():
    """Return a fresh deep copy of the config dict parsed from SCAO_INI."""
    global _BASE_CFG
    if _BASE_CFG is None:
        ao = aoSystem(SCAO_INI, path_root=str(PATH_P3), verbose=False)
        _BASE_CFG = copy.deepcopy(ao.my_data_map)
    return copy.deepcopy(_BASE_CFG)


def _make_ao(lo=None, focus=None, tel=None, precision=None):
    """Build an aoSystem from SCAO_INI with extra keys in [sensor_LO] / [sensor_Focus] / [telescope].

    Parameters
    ----------
    lo, focus, tel : dict, optional
        Keys to set in [sensor_LO] (added to the minimum valid ones), [sensor_Focus], [telescope].
        If *lo* is None the [sensor_LO] section is not created.
    precision : str, optional
        Value of [COMPUTATION] precision.
    """
    cfg = _base_cfg()
    if lo is not None:
        cfg['sensor_LO'] = {**LO_BASE, **lo}
    if focus is not None:
        cfg['sensor_Focus'] = dict(focus)
    if tel is not None:
        cfg['telescope'].update(tel)
    if precision is not None:
        cfg['COMPUTATION'] = {'precision': precision}
    return aoSystem(SCAO_INI, path_root=str(PATH_P3), verbose=False, config_dict=cfg)


def _expected(path=None, angle=0.0, obs=0.0, precision='double'):
    """Host-side pupil built directly with the telescope class."""
    return cpuArray(telescope(D, RES, obsRatio=obs, pupilAngle=angle, path_pupil=path,
                              precision=precision, verbose=False).pupil)


def _area(pupil):
    return float(cpuArray(pupil).sum()) * (D / RES) ** 2


def _offcentre_disc(n=RES):
    """Disc of diameter n/4 centred at (0.25 n, 0.5 n) from the origin: no rotational symmetry."""
    y, x = np.mgrid[:n, :n]
    return (np.hypot(x - 0.25 * n, y - 0.5 * n) <= n / 8).astype(float)


class TestSensorPupils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.disc = str(pathlib.Path(cls.tmp.name) / 'disc.fits')
        fits.writeto(cls.disc, _offcentre_disc(), overwrite=True)
        # file with a size different from Resolution: exercises the interpolation
        cls.disc128 = str(pathlib.Path(cls.tmp.name) / 'disc128.fits')
        fits.writeto(cls.disc128, _offcentre_disc(128), overwrite=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---- defaults
    def test_no_keys_give_default_objects(self):
        for ao in (_make_ao(), _make_ao(lo={}), _make_ao(lo={}, focus={})):
            self.assertIs(ao.pupilLO, ao.tel.pupil)
            self.assertIs(ao.pupilFocus, ao.pupilLO)

    def test_empty_path_is_absent(self):
        ao = _make_ao(lo={'PathPupil': ''}, focus={'PathPupil': ''})
        self.assertIs(ao.pupilLO, ao.tel.pupil)
        self.assertIs(ao.pupilFocus, ao.pupilLO)

    # ---- ObscurationRatio
    def test_lo_obscuration_ratio_annulus(self):
        eps = 0.4
        ao = _make_ao(lo={'ObscurationRatio': eps})
        pupil = cpuArray(ao.pupilLO)
        self.assertTrue(np.array_equal(pupil, _expected(obs=eps)))
        ratio = _area(pupil) / (np.pi / 4 * D ** 2 * (1 - eps ** 2))
        print(f"\n[annulus] area/analytic = {ratio:.4f}")
        self.assertLess(abs(ratio - 1.0), 0.01)
        self.assertIsNot(ao.pupilLO, ao.tel.pupil)
        self.assertFalse(np.array_equal(pupil, cpuArray(ao.tel.pupil)))  # telescope has eps = 0.111
        self.assertIs(ao.pupilFocus, ao.pupilLO)

    def test_focus_own_pupil_overrides_lo(self):
        ao = _make_ao(lo={'ObscurationRatio': 0.3}, focus={'ObscurationRatio': 0.5})
        lo, foc = cpuArray(ao.pupilLO), cpuArray(ao.pupilFocus)
        self.assertIsNot(ao.pupilFocus, ao.pupilLO)
        self.assertFalse(np.array_equal(lo, foc))
        self.assertTrue(np.array_equal(foc, _expected(obs=0.5)))
        self.assertLess(foc.sum(), lo.sum())

    def test_focus_own_pupil_without_lo_keys(self):
        ao = _make_ao(lo={}, focus={'ObscurationRatio': 0.5})
        self.assertIs(ao.pupilLO, ao.tel.pupil)
        self.assertTrue(np.array_equal(cpuArray(ao.pupilFocus), _expected(obs=0.5)))

    # ---- PathPupil
    def test_lo_path_pupil_matches_telescope_class(self):
        for path in (self.disc, self.disc128):
            ao = _make_ao(lo={'PathPupil': path})
            pupil = cpuArray(ao.pupilLO)
            self.assertEqual(pupil.shape, (RES, RES))
            self.assertTrue(np.array_equal(pupil, _expected(path=path)))
            self.assertIs(ao.pupilFocus, ao.pupilLO)
        # the interpolated map must keep the disc area (pi/4 (D/4)^2 of the unit-square map)
        area = _area(cpuArray(_make_ao(lo={'PathPupil': self.disc128}).pupilLO))
        print(f"\n[interp] area {area:.3f} m2, analytic {np.pi / 4 * (D / 4) ** 2:.3f} m2")
        self.assertLess(abs(area / (np.pi / 4 * (D / 4) ** 2) - 1.0), 0.05)

    def test_lo_pupil_equal_to_telescope_pupil(self):
        for angle in (0.0, 30.0):
            ao = _make_ao(lo={'PathPupil': self.disc, 'PupilAngle': angle},
                          tel={'PathPupil': self.disc, 'PupilAngle': angle})
            self.assertTrue(np.array_equal(cpuArray(ao.pupilLO), cpuArray(ao.tel.pupil)))
        self.assertIsNot(ao.pupilLO, ao.tel.pupil)

    def test_pupil_angle_rotates_asymmetric_map(self):
        angle = 90.0
        ao0 = _make_ao(lo={'PathPupil': self.disc})
        ao1 = _make_ao(lo={'PathPupil': self.disc, 'PupilAngle': angle})
        p0, p1 = cpuArray(ao0.pupilLO), cpuArray(ao1.pupilLO)
        self.assertTrue(np.array_equal(p1, _expected(path=self.disc, angle=angle)))
        self.assertFalse(np.array_equal(p0, p1))
        # rotation preserves the area to within the interpolation error
        print(f"\n[rotation] sum 0 deg {p0.sum():.1f}, {angle} deg {p1.sum():.1f}")
        self.assertLess(abs(p1.sum() / p0.sum() - 1.0), 0.02)
        # the disc centre moves: the centroid is no longer the same
        y, x = np.mgrid[:RES, :RES]
        c0 = np.array([(x * p0).sum(), (y * p0).sum()]) / p0.sum()
        c1 = np.array([(x * p1).sum(), (y * p1).sum()]) / p1.sum()
        self.assertGreater(np.hypot(*(c1 - c0)), 0.1 * RES)

    # ---- errors
    def test_both_keys_raise(self):
        with self.assertRaises(ValueError) as cm:
            _make_ao(lo={'PathPupil': self.disc, 'ObscurationRatio': 0.3})
        self.assertIn('not both', str(cm.exception))
        self.assertIn('sensor_LO', str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            _make_ao(lo={}, focus={'PathPupil': self.disc, 'ObscurationRatio': 0.3})
        self.assertIn('sensor_Focus', str(cm.exception))

    def test_pupil_angle_alone_raises(self):
        with self.assertRaises(ValueError) as cm:
            _make_ao(lo={'PupilAngle': 10.0})
        self.assertIn('PupilAngle needs PathPupil or ObscurationRatio', str(cm.exception))
        # an empty PathPupil does not count as set
        with self.assertRaises(ValueError):
            _make_ao(lo={'PathPupil': '', 'PupilAngle': 10.0})

    def test_missing_file_raises(self):
        missing = str(pathlib.Path(self.tmp.name) / 'nope.fits')
        with self.assertRaises(ValueError) as cm:
            _make_ao(lo={'PathPupil': missing})
        self.assertIn('not found', str(cm.exception))
        self.assertIn('nope.fits', str(cm.exception))

    # ---- precision
    def test_single_precision_dtype(self):
        ao = _make_ao(lo={'ObscurationRatio': 0.3}, focus={'PathPupil': self.disc},
                      precision='single')
        self.assertEqual(ao.tel.pupil.dtype, np.float32)
        self.assertEqual(ao.pupilLO.dtype, np.float32)
        self.assertEqual(ao.pupilFocus.dtype, np.float32)
        self.assertTrue(np.array_equal(cpuArray(ao.pupilLO),
                                       _expected(obs=0.3, precision='single')))


if __name__ == '__main__':
    unittest.main()
