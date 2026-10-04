#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for aoSystem._validate_atmosphere and its use while parsing [atmosphere]."""

import pathlib
import warnings

import numpy as nnp
import pytest

import p3.aoSystem as aoSystemMain
from p3.aoSystem import cpuArray
from p3.aoSystem.aoSystem import aoSystem
from p3.aoSystem.fourierModel import fourierModel

validate = aoSystem._validate_atmosphere


def _p3_path() -> str:
    return str(pathlib.Path(aoSystemMain.__file__).parent.parent.parent.absolute())


_INI_TEMPLATE = """
[telescope]
TelescopeDiameter = 8.0
ObscurationRatio = 0.16
Resolution = 64
ZenithAngle = 0.0

[atmosphere]
Wavelength = 500e-9
L0 = {l0}
Seeing = 0.8
Cn2Weights = {w}
Cn2Heights = {h}
WindSpeed = {v}
WindDirection = {d}

[sources_science]
Wavelength = [1.65e-6]
Zenith = [0.0]
Azimuth = [0.0]

[sources_HO]
Wavelength = 589e-9
Zenith = [20.0, 20.0, 20.0]
Azimuth = [0.0, 120.0, 240.0]
Height = 90e3

[sensor_science]
PixelScale = 30.0
FieldOfView = 64

[sensor_HO]
WfsType = 'Shack-Hartmann'
Modulation = None
PixelScale = 800
FieldOfView = 6
NumberPhotons = [1e4, 1e4, 1e4]
SigmaRON = 0.0
ExcessNoiseFactor = 1.0
Algorithm = 'cog'
NumberLenslets = [20, 20, 20]
NoiseVariance = [None]

[DM]
NumberActuators = [20]
DmPitchs = [0.4]
InfModel = 'gaussian'
InfCoupling = [0.2]
DmHeights = [0.0]
OptimizationZenith = [0.0, 10.0, 10.0, 10.0]
OptimizationAzimuth = [0.0, 0.0, 120.0, 240.0]
OptimizationWeight = [1.0, 1.0, 1.0, 1.0]
OptimizationConditioning = 1.0e2
NumberReconstructedLayers = 2
AoArea = 'circle'

[RTC]
LoopGain_HO = 0.5
SensorFrameRate_HO = 500.0
LoopDelaySteps_HO = 2
"""

_DEFAULTS = dict(l0='25.0', w='[0.6, 0.0, 0.4]', h='[0, 4000, 10000]',
                 v='[8.0, 12.0, 20.0]', d='[0.0, 10.0, 20.0]')


def _write_ini(tmp_path, name: str = 'atm.ini', **kw) -> str:
    path = tmp_path / name
    path.write_text(_INI_TEMPLATE.format(**{**_DEFAULTS, **kw}))
    return str(path)


def _build(ini: str) -> fourierModel:
    return fourierModel(ini, path_root=_p3_path(), calcPSF=False, verbose=False,
                        display=False, reduce_memory=False,
                        computeFocalAnisoCov=False)


def _args(**kw) -> tuple:
    """Valid (wvl, r0, L0, weights, heights, wSpeed, wDir) with overrides."""
    a = dict(wvl=500e-9, r0=0.12, L0=25.0, weights=[0.5, 0.3, 0.2],
             heights=[0.0, 4000.0, 10000.0], wSpeed=[5.0, 10.0, 15.0],
             wDir=[0.0, 90.0, 180.0])
    a.update(kw)
    return tuple(a.values())


# ------------------------------------------------------------ unit: errors
class TestValidateErrors:
    @pytest.mark.parametrize('kw, name', [
        ({'weights': [1.2, -0.1, -0.1]}, 'Cn2Weights'),
        ({'weights': [0.5, nnp.nan, 0.5]}, 'Cn2Weights'),
        ({'heights': [0.0, -100.0, 10000.0]}, 'Cn2Heights'),
        ({'heights': [0.0, nnp.inf, 10000.0]}, 'Cn2Heights'),
        ({'wSpeed': [5.0, -1.0, 15.0]}, 'WindSpeed'),
        ({'wSpeed': [5.0, nnp.nan, 15.0]}, 'WindSpeed'),
        ({'wDir': [0.0, nnp.nan, 180.0]}, 'WindDirection'),
        ({'wDir': [0.0, nnp.inf, 180.0]}, 'WindDirection'),
        ({'r0': 0.0}, 'r0'),
        ({'r0': -0.1}, 'r0'),
        ({'r0': nnp.inf}, 'r0'),
        ({'r0': nnp.nan}, 'r0'),
        ({'L0': 0.0}, 'L0'),
        ({'L0': -25.0}, 'L0'),
        ({'L0': [25.0, -1.0, 25.0]}, 'L0'),
        ({'L0': nnp.nan}, 'L0'),
        ({'wvl': 0.0}, 'Wavelength'),
        ({'wvl': -500e-9}, 'Wavelength'),
        ({'wvl': nnp.nan}, 'Wavelength'),
    ])
    def test_invalid_raises_with_name(self, kw, name):
        with pytest.raises(ValueError, match=name):
            validate(*_args(**kw))

    def test_negative_wind_suggests_direction_flip(self):
        with pytest.raises(ValueError, match=r'WindDirection \+ 180'):
            validate(*_args(wSpeed=[5.0, -1.0, 15.0]))

    def test_valid_input_unchanged(self):
        args = _args()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            w, h, v, d, L0 = validate(*args)
        assert w == args[3] and h == args[4] and v == args[5] and d == args[6]
        assert L0 == args[2]

    def test_zero_height_and_zero_wind_are_valid(self):
        w, h, v, d, _ = validate(*_args(heights=[0.0, 0.0, 5000.0],
                                        wSpeed=[0.0, 0.0, 10.0]))
        assert h[0] == 0.0 and v[0] == 0.0 and len(w) == 3


# ------------------------------------------------------ unit: zero weights
class TestZeroWeightRemoval:
    def test_layer_removed_and_warns(self):
        with pytest.warns(UserWarning, match='Cn2Weights = 0'):
            w, h, v, d, L0 = validate(*_args(weights=[0.6, 0.0, 0.4]))
        assert list(w) == [0.6, 0.4]
        assert list(h) == [0.0, 10000.0]
        assert list(v) == [5.0, 15.0]
        assert list(d) == [0.0, 180.0]
        assert L0 == 25.0

    def test_per_layer_l0_trimmed(self):
        with pytest.warns(UserWarning):
            w, h, v, d, L0 = validate(*_args(weights=[0.0, 0.6, 0.4],
                                             L0=[10.0, 20.0, 30.0]))
        assert list(L0) == [20.0, 30.0]
        assert len(L0) == len(w) == len(h) == len(v) == len(d) == 2

    def test_scalar_l0_unchanged(self):
        with pytest.warns(UserWarning):
            *_, L0 = validate(*_args(weights=[0.6, 0.4, 0.0], L0=30.0))
        assert L0 == 30.0

    def test_l0_list_of_other_length_untouched(self):
        with pytest.warns(UserWarning):
            *_, L0 = validate(*_args(weights=[0.6, 0.0, 0.4], L0=[20.0, 30.0]))
        assert list(L0) == [20.0, 30.0]

    def test_multiple_zero_layers(self):
        with pytest.warns(UserWarning, match='2 layer'):
            w, h, *_ = validate(*_args(weights=[0.0, 1.0, 0.0]))
        assert list(w) == [1.0] and list(h) == [4000.0]


# ------------------------------------------------------------ integration
class TestIntegration:
    def test_zero_weight_layer_equals_reduced_profile(self, tmp_path):
        ini_zero = _write_ini(tmp_path, 'zero.ini')
        ini_red = _write_ini(tmp_path, 'red.ini', w='[0.6, 0.4]', h='[0, 10000]',
                             v='[8.0, 20.0]', d='[0.0, 20.0]')
        with pytest.warns(UserWarning, match='Cn2Weights = 0'):
            fao_zero = _build(ini_zero)
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            fao_red = _build(ini_red)
        assert fao_zero.ao.atm.nL == fao_red.ao.atm.nL == 2
        assert nnp.array_equal(cpuArray(fao_zero.PSD), cpuArray(fao_red.PSD))

    @pytest.mark.parametrize('kw, name', [
        ({'v': '[8.0, -12.0, 20.0]'}, 'WindSpeed'),
        ({'h': '[0, -4000, 10000]'}, 'Cn2Heights'),
        ({'w': '[0.7, -0.1, 0.4]'}, 'Cn2Weights'),
        ({'d': '[0.0, 1e999, 20.0]'}, 'WindDirection'),
        ({'l0': '-25.0'}, 'L0'),
    ])
    def test_invalid_ini_raises(self, tmp_path, kw, name):
        ini = _write_ini(tmp_path, **kw)
        with pytest.raises(ValueError, match=name):
            aoSystem(ini, path_root=_p3_path())

    @pytest.mark.parametrize('seeing', ['0.0', '-0.5'])
    def test_invalid_seeing_raises(self, tmp_path, seeing):
        # checked before the r0 conversion, which would divide by zero
        ini = _write_ini(tmp_path)
        with open(ini) as f:
            txt = f.read()
        with open(ini, 'w') as f:
            f.write(txt.replace('Seeing = 0.8', f'Seeing = {seeing}'))
        with pytest.raises(ValueError, match='Seeing'):
            aoSystem(ini, path_root=_p3_path())
