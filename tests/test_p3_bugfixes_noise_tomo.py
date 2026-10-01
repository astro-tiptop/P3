#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression tests for:
  A) is_auto_noise_var and multi-entry [sensor_HO] NoiseVariance
  B) fourierModel.tomoRelRegFloor (tomographic regularization floor)
"""

import os
import pathlib
import subprocess
import sys

import numpy as nnp
import pytest

import p3.aoSystem as aoSystemMain
from p3.aoSystem import cpuArray
from p3.aoSystem.aoSystem import aoSystem
from p3.aoSystem.fourierModel import fourierModel
from p3.aoSystem.processing import is_auto_noise_var

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None


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
L0 = 25.0
Seeing = 0.8
Cn2Weights = [0.6, 0.3, 0.1]
Cn2Heights = [0, 4000, 10000]
WindSpeed = [8.0, 12.0, 20.0]
WindDirection = [0.0, 0.0, 0.0]

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
NumberPhotons = {nph}
SigmaRON = 0.0
ExcessNoiseFactor = 1.0
Algorithm = 'cog'
NumberLenslets = [20, 20, 20]
NoiseVariance = {noisevar}

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
NumberReconstructedLayers = 3
AoArea = 'circle'

[RTC]
LoopGain_HO = 0.5
SensorFrameRate_HO = 500.0
LoopDelaySteps_HO = 2
"""


def _write_ini(tmp_path, nph: float = 1e4, noisevar: str = '[None]') -> str:
    path = tmp_path / 'tomo_reduced.ini'
    path.write_text(_INI_TEMPLATE.format(nph=[nph] * 3, noisevar=noisevar))
    return str(path)


def _build(ini: str) -> fourierModel:
    return fourierModel(ini, path_root=_p3_path(), calcPSF=False, verbose=False,
                        display=False, reduce_memory=False,
                        computeFocalAnisoCov=False)


def _residual(fao: fourierModel) -> float:
    """HO residual proxy: sqrt of the summed PSD (only ratios are compared)."""
    psd = nnp.asarray(cpuArray(fao.PSD), dtype=nnp.float64)
    assert nnp.all(nnp.isfinite(psd))
    return float(nnp.sqrt(psd.sum()))


# ---------------------------------------------------------------- Fix A
class TestIsAutoNoiseVar:
    @pytest.mark.parametrize('v', [
        [None],
        nnp.array([None]),
        nnp.array([None], dtype=object),
    ])
    def test_unset_is_auto(self, v):
        assert is_auto_noise_var(v) is True

    @pytest.mark.parametrize('v', [
        [0.1],
        [0.1, 0.2, 0.3],
        nnp.array([0.1]),
        nnp.array([0.1, 0.2, 0.3]),
        nnp.array([0.0]),
        nnp.array([0.1, 0.2, 0.3], dtype=nnp.float32),
    ])
    def test_set_is_not_auto(self, v):
        assert is_auto_noise_var(v) is False

    @pytest.mark.skipif(cp is None, reason='cupy not available')
    @pytest.mark.parametrize('v', [[0.1], [0.1, 0.2, 0.3]])
    def test_cupy_arrays(self, v):
        assert is_auto_noise_var(cp.asarray(v)) is False


class TestMultiEntryNoiseVariance:
    def test_aosystem_construction_does_not_raise(self, tmp_path):
        ini = _write_ini(tmp_path, noisevar='[0.01, 0.01, 0.01]')
        # errorBreakdown runs here (loop gain > 0)
        ao = aoSystem(ini, path_root=_p3_path())
        nv = nnp.asarray(cpuArray(ao.wfs.processing.noiseVar), dtype=float)
        assert nv.shape == (3,)

    def test_fourier_model_uses_given_noise_variance(self, tmp_path):
        ini = _write_ini(tmp_path, noisevar='[0.01, 0.01, 0.01]')
        fao = _build(ini)
        nv = nnp.asarray(cpuArray(fao.ao.wfs.processing.noiseVar), dtype=float)
        nnp.testing.assert_allclose(nv, [0.01] * 3, rtol=1e-6)
        assert _residual(fao) > 0


# ---------------------------------------------------------------- Fix B
class TestTomoRegFloor:
    def test_default_floor(self):
        assert fourierModel.tomoRelRegFloor == 1e-12

    def test_bright_wfs_residual_matches_nominal(self, tmp_path):
        r_nom = _residual(_build(_write_ini(tmp_path, nph=1e4)))
        r_bright = _residual(_build(_write_ini(tmp_path, nph=1e9)))
        assert r_bright == pytest.approx(r_nom, rel=2e-2)

    def test_floor_does_not_bias_nominal_case(self, tmp_path, monkeypatch):
        ini = _write_ini(tmp_path, nph=1e4)
        r_default = _residual(_build(ini))
        monkeypatch.setattr(fourierModel, 'tomoRelRegFloor', 0.0)
        r_nofloor = _residual(_build(ini))
        assert r_default == pytest.approx(r_nofloor, rel=1e-4)


# ---------------------------------------------------------------- Zero noise
class TestZeroNoise:
    def test_zero_noise_tomographic_builds(self, tmp_path):
        fao = _build(_write_ini(tmp_path, noisevar='[0.0, 0.0, 0.0]'))
        assert _residual(fao) > 0
        res, nsrc = fao.freq.resAO, fao.ao.src.nSrc
        psd_noise = nnp.asarray(cpuArray(fao.psdNoise))
        assert psd_noise.shape == (res, res, nsrc)
        assert not psd_noise.any()
        assert nnp.all(nnp.isfinite(cpuArray(fao.Wtomo)))

    def test_near_zero_noise_is_robust(self, tmp_path):
        r = [_residual(_build(_write_ini(tmp_path, noisevar=f'[{v}, {v}, {v}]')))
             for v in ('0.0', '1e-30', '1e-20')]
        assert r[1] == pytest.approx(r[0], rel=1e-6)
        assert r[2] == pytest.approx(r[0], rel=1e-6)


# ---------------------------------------------------------------- Popt
class TestOptimalProjector:
    def test_asymmetric_asterism_is_finite(self, tmp_path):
        ini = tmp_path / 'asym.ini'
        text = _INI_TEMPLATE.format(nph=[1e4] * 3, noisevar='[None]')
        text = (text
                .replace('OptimizationZenith = [0.0, 10.0, 10.0, 10.0]',
                         'OptimizationZenith = [0.0, 10.0, 20.0]')
                .replace('OptimizationAzimuth = [0.0, 0.0, 120.0, 240.0]',
                         'OptimizationAzimuth = [0.0, 30.0, 100.0]')
                .replace('OptimizationWeight = [1.0, 1.0, 1.0, 1.0]',
                         'OptimizationWeight = [1.0, 1.0, 1.0]')
                .replace('NumberActuators = [20]', 'NumberActuators = [20, 20]')
                .replace('DmPitchs = [0.4]', 'DmPitchs = [0.4, 0.4]')
                .replace('InfCoupling = [0.2]', 'InfCoupling = [0.2, 0.2]')
                .replace('DmHeights = [0.0]', 'DmHeights = [0.0, 6000.0]'))
        ini.write_text(text)
        fao = _build(str(ini))
        assert fao.Popt.shape[2] == 2
        assert nnp.all(nnp.isfinite(cpuArray(fao.Popt)))
        assert _residual(fao) > 0


# ---------------------------------------------------------------- CPU vs GPU
def _cuda_available() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _run_backend(ini: str, out: pathlib.Path, gpu: bool) -> dict:
    """Run tests/_backend_dump.py in a fresh process with the chosen backend."""
    here = pathlib.Path(__file__).parent
    env = {**os.environ, 'P3_DISABLE_GPU': 'FALSE' if gpu else 'TRUE',
           'PYTHONPATH': os.pathsep.join(
               [_p3_path()] + ([os.environ['PYTHONPATH']]
                               if os.environ.get('PYTHONPATH') else []))}
    subprocess.run([sys.executable, str(here / '_backend_dump.py'), ini, str(out)],
                   env=env, check=True, cwd=_p3_path(), timeout=600)
    with nnp.load(out) as d:
        res = {k: d[k] for k in d.files}
    assert bool(res['gpu']) == gpu
    return res


def _rel_fro(a, b) -> float:
    return float(nnp.linalg.norm(a - b) / nnp.linalg.norm(b))


@pytest.mark.skipif(not _cuda_available(), reason='no CUDA device / cupy')
class TestCpuGpuConsistency:
    def _compare(self, ini: str, tmp_path):
        cpu = _run_backend(ini, tmp_path / 'cpu.npz', gpu=False)
        gpu = _run_backend(ini, tmp_path / 'gpu.npz', gpu=True)
        for key in ('Popt', 'Wtomo', 'W'):
            assert _rel_fro(gpu[key], cpu[key]) < 1e-8, key
        mask = nnp.abs(cpu['PSD']) > 1e-3
        rel = nnp.abs(gpu['PSD'][mask] - cpu['PSD'][mask]) / nnp.abs(cpu['PSD'][mask])
        assert rel.max() < 1e-6

    def test_reduced_tomographic(self, tmp_path):
        self._compare(_write_ini(tmp_path), tmp_path)

    def test_mavis(self, tmp_path):
        self._compare(str(pathlib.Path(__file__).parent / 'MAVIStest.ini'), tmp_path)
