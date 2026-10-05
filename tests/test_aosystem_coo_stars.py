#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the coo_stars override of the science sources in aoSystem.

coo_stars = [y, x] in arcsec, shape (2, nSrc). Source stores zenith in arcsec
and azimuth in DEGREES, with direction = tan(zenith)*[cos(az), sin(az)].
"""

import pathlib

import numpy as nnp
import pytest

import p3.aoSystem as aoSystemMain
from p3.aoSystem import cpuArray
from p3.aoSystem.aoSystem import aoSystem

ARCSEC2RAD = nnp.pi / 180 / 3600

_INI_TEMPLATE = """
[telescope]
TelescopeDiameter = 8.0
ObscurationRatio = 0.16
Resolution = 32
ZenithAngle = 0.0

[atmosphere]
Wavelength = 500e-9
L0 = 25.0
Seeing = 0.8
Cn2Weights = [1.0]
Cn2Heights = [0.0]
WindSpeed = [10.0]
WindDirection = [0.0]

[sources_science]
Wavelength = [1.65e-6]
Zenith = {zen}
Azimuth = {azi}

[sources_HO]
Wavelength = 589e-9
Zenith = [0.0]
Azimuth = [0.0]
Height = 0

[sensor_science]
PixelScale = 30.0
FieldOfView = 32

[sensor_HO]
WfsType = 'Shack-Hartmann'
Modulation = None
PixelScale = 800
FieldOfView = 6
NumberPhotons = [1e4]
SigmaRON = 0.0
ExcessNoiseFactor = 1.0
Algorithm = 'cog'
NumberLenslets = [10]
NoiseVariance = [None]

[DM]
NumberActuators = [10]
DmPitchs = [0.8]
InfModel = 'gaussian'
InfCoupling = [0.2]
DmHeights = [0.0]
OptimizationZenith = [0.0]
OptimizationAzimuth = [0.0]
OptimizationWeight = [1.0]
OptimizationConditioning = 1.0e2
NumberReconstructedLayers = 1
AoArea = 'circle'

[RTC]
LoopGain_HO = 0.5
SensorFrameRate_HO = 500.0
LoopDelaySteps_HO = 2
"""


def _p3_path() -> str:
    return str(pathlib.Path(aoSystemMain.__file__).parent.parent.parent.absolute())


def _build(tmp_path, name, zen=(0.0,), azi=(0.0,), coo_stars=None) -> aoSystem:
    path = tmp_path / name
    path.write_text(_INI_TEMPLATE.format(zen=list(zen), azi=list(azi)))
    return aoSystem(str(path), path_root=_p3_path(), coo_stars=coo_stars,
                    verbose=False)


def _src(ao):
    return (nnp.asarray(cpuArray(ao.src.zenith), dtype=float),
            nnp.asarray(cpuArray(ao.src.azimuth), dtype=float),
            nnp.asarray(cpuArray(ao.src.direction), dtype=float))


# (y, x) in arcsec -> expected zenith [arcsec], azimuth [deg]
_CASES = {
    'x_axis':   ([0.], [10.], [10.], [0.]),
    'y_axis':   ([10.], [0.], [10.], [90.]),
    'neg_y':    ([-10.], [0.], [10.], [-90.]),
    'diagonal': ([5.], [5.], [nnp.sqrt(50)], [45.]),
    'multi':    ([0., 10., -10., 5.], [10., 0., 0., 5.],
                 [10., 10., 10., nnp.sqrt(50)], [0., 90., -90., 45.]),
}


@pytest.mark.parametrize('case', list(_CASES))
def test_coo_stars_zenith_azimuth_degrees(tmp_path, case):
    y, x, zen_exp, azi_exp = _CASES[case]
    ao = _build(tmp_path, 'a.ini', coo_stars=nnp.array([y, x]))
    zen, azi, _ = _src(ao)
    nnp.testing.assert_allclose(zen, zen_exp, atol=1e-10)
    # With the old radian code, 90 deg would have been 1.5708
    nnp.testing.assert_allclose(azi, azi_exp, atol=1e-10)


@pytest.mark.parametrize('case', list(_CASES))
def test_coo_stars_direction_equals_config(tmp_path, case):
    y, x, zen_exp, azi_exp = _CASES[case]
    ao_coo = _build(tmp_path, 'coo.ini', coo_stars=nnp.array([y, x]))
    ao_cfg = _build(tmp_path, 'cfg.ini', zen=zen_exp, azi=azi_exp)
    _, _, d_coo = _src(ao_coo)
    _, _, d_cfg = _src(ao_cfg)
    assert d_coo.shape == d_cfg.shape == (2, len(y))
    nnp.testing.assert_allclose(d_coo, d_cfg, rtol=1e-10, atol=1e-15)


def test_coo_stars_direction_x_axis(tmp_path):
    ao = _build(tmp_path, 'a.ini', coo_stars=nnp.array([[0.], [10.]]))
    _, _, d = _src(ao)
    nnp.testing.assert_allclose(d[0], nnp.tan(10 * ARCSEC2RAD), rtol=1e-10)
    nnp.testing.assert_allclose(d[1], 0.0, atol=1e-15)


def test_coo_stars_direction_y_axis(tmp_path):
    ao = _build(tmp_path, 'a.ini', coo_stars=nnp.array([[10.], [0.]]))
    _, _, d = _src(ao)
    nnp.testing.assert_allclose(d[0], 0.0, atol=1e-15)
    nnp.testing.assert_allclose(d[1], nnp.tan(10 * ARCSEC2RAD), rtol=1e-10)


def test_coo_stars_overrides_config_sources(tmp_path):
    ao = _build(tmp_path, 'a.ini', zen=[3.0], azi=[30.0],
                coo_stars=nnp.array([[0.], [10.]]))
    zen, azi, _ = _src(ao)
    nnp.testing.assert_allclose(zen, [10.0])
    nnp.testing.assert_allclose(azi, [0.0], atol=1e-10)


def test_no_coo_stars_keeps_config(tmp_path):
    ao = _build(tmp_path, 'a.ini', zen=[3.0], azi=[30.0])
    zen, azi, _ = _src(ao)
    nnp.testing.assert_allclose(zen, [3.0])
    nnp.testing.assert_allclose(azi, [30.0])
