#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for LGS altitudes below some turbulent layers (e.g. Rayleigh LGS)."""

from types import SimpleNamespace
import warnings

import numpy as nnp
import pytest

from p3.aoSystem.FourierUtils import cpuArray
from p3.aoSystem.fourierModel import fourierModel
from p3.aoSystem import anisoplanatismModel as aniso

_INI_TEMPLATE = """
[telescope]
TelescopeDiameter = 2.2
ZenithAngle = {zenith}
ObscurationRatio = 0.4
Resolution = 96
TechnicalFoV = 120

[atmosphere]
Wavelength = 500e-9
Seeing = 0.8
L0 = 30.0
Cn2Weights = {weights}
Cn2Heights = {heights}
WindSpeed = {wspeed}
WindDirection = {wdir}

[sources_science]
Wavelength = [0.55e-06]
Zenith = [0.0, 14.0]
Azimuth = [0.0, 0.0]

[sources_HO]
Wavelength = [355e-9]
Zenith = {ho_zen}
Azimuth = {ho_az}
Height = {lgs_height}

[sources_LO]
Wavelength = [1250e-9]
Zenith = [0.0]
Azimuth = [0.0]

[sensor_science]
PixelScale = 23.4
FieldOfView = 128

[sensor_HO]
WfsType = 'Shack-Hartmann'
Modulation = None
PixelScale = 833
FieldOfView = 6
NumberPhotons = {phot}
SigmaRON = {ron}
ExcessNoiseFactor = 2.0
Algorithm = 'cog'
NumberLenslets = {nlens}
NoiseVariance = {noisevar}

[sensor_LO]
PixelScale = 30.0
FieldOfView = 100
Binning = 1
NumberPhotons = [1900, 1900, 1900]
SpotFWHM = [[0.0,0.0,0.0]]
SigmaRON = 0.5
Dark = 30.0
SkyBackground = 35.0
Gain = 1.0
ExcessNoiseFactor = 1.3
NumberLenslets = [1]
Algorithm = 'wcog'
WindowRadiusWCoG = 4
ThresholdWCoG = 0.0
NewValueThrPix = 0.0
noNoise = False

[DM]
NumberActuators = [17]
DmPitchs = [0.105]
InfModel = 'gaussian'
InfCoupling = [0.2]
DmHeights = [0]
OptimizationZenith = {opt_zen}
OptimizationAzimuth = {opt_az}
OptimizationWeight = {opt_w}
OptimizationConditioning = 1.0e4
NumberReconstructedLayers = {nrec}
AoArea = 'circle'

[RTC]
LoopGain_HO = 0.3
SensorFrameRate_HO = 1000.0
LoopDelaySteps_HO = 2
LoopGain_LO = 'optimize'
SensorFrameRate_LO = 1000.0
LoopDelaySteps_LO = 2
"""

# Default profile: ground + 3 layers
HEIGHTS = [0.0, 500.0, 2000.0, 8000.0]
WEIGHTS = [0.5, 0.2, 0.2, 0.1]


def _write_ini(tmp_path, lgs_height, multi=False, heights=HEIGHTS, weights=WEIGHTS,
               nrec=None, zenith=0.0, name='cfg.ini', noise_var=None, wspeed=8.0, photons=75, ron=0.2):
    """Write a reduced SLAO (single on-axis LGS) or GLAO (3 LGS) ini and return its path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    lgs_height : float
        LGS altitude in meters.
    multi : bool
        If True, 3 LGS at 420 arcsec (tomographic model), otherwise one on-axis LGS.
    heights, weights : list of float
        Cn2 profile.
    nrec : int or None
        NumberReconstructedLayers (defaults to the number of layers: no compression).
    zenith : float
        Telescope zenith angle in degrees.
    name : str
        File name.
    noise_var : float or None
        NoiseVariance of each HO WFS (0.0 for a noise-free WFS, None to compute it).
    wspeed : float
        Wind speed (m/s) of all layers.
    photons, ron : float
        HO WFS photon flux and read-out noise (use 1e9 and 0 to make the WFS ~noise-free
        in the multi-LGS case, where an explicit NoiseVariance list is not parsed).

    Returns
    -------
    str
        Path of the ini file.
    """
    nL = len(heights)
    nG = 3 if multi else 1
    if multi:
        opt_zen, opt_az, opt_w = '[0, 30, 30]', '[0, 0, 120]', '[1, 1, 1]'
        ho_zen, ho_az = '[420, 420, 420]', '[0, 120, 270]'
    else:
        opt_zen, opt_az, opt_w = '[0]', '[0]', '[1]'
        ho_zen, ho_az = '[0.0]', '[0.0]'
    cfg = _INI_TEMPLATE.format(
        zenith=zenith, weights=list(weights), heights=list(heights),
        wspeed=[wspeed] * nL, noisevar=[noise_var] * (nG if noise_var is not None else 1), wdir=[0.0] * nL,
        ho_zen=ho_zen, ho_az=ho_az, lgs_height=lgs_height,
        phot=[photons] * nG, ron=ron, nlens=[16] * nG,
        opt_zen=opt_zen, opt_az=opt_az, opt_w=opt_w,
        nrec=nrec if nrec is not None else nL)
    path = tmp_path / name
    path.write_text(cfg)
    return str(path)


def _build(tmp_path, lgs_height, breakdown=False, **kw):
    """Build a fourierModel the way TIPTOP does."""
    ini = _write_ini(tmp_path, lgs_height, **kw)
    return fourierModel(ini, calcPSF=False, verbose=False, display=False,
                        reduce_memory=False, computeFocalAnisoCov=False,
                        getErrorBreakDown=breakdown)


def _psd(fao):
    return nnp.asarray(cpuArray(fao.PSD), dtype=float)


def _residual(fao):
    """HO residual (rad rms, on-axis science source) from the PSD."""
    return float(nnp.sqrt(_psd(fao)[:, :, 0].sum()))


@pytest.fixture(scope='module', params=['slao', 'glao'])
def mode(request):
    return request.param == 'glao'


# ---------------------------------------------------------------------------
# fourierModel
# ---------------------------------------------------------------------------

def test_lgs_height_equal_to_layer_height(tmp_path, mode):
    fao = _build(tmp_path, 2000.0, multi=mode)
    assert nnp.all(nnp.isfinite(_psd(fao)))
    assert list(fao.sensedLayers) == [True, True, False, False]
    assert nnp.all(nnp.isfinite(fao.strechFactor))


def test_lgs_below_all_nonground_layers(tmp_path, mode):
    fao = _build(tmp_path, 100.0, multi=mode)
    psd = _psd(fao)
    assert nnp.all(nnp.isfinite(psd))
    assert list(fao.sensedLayers) == [True, False, False, False]
    stretch = nnp.asarray(fao.strechFactor)
    assert nnp.all(nnp.isfinite(stretch)) and nnp.all(stretch > 0)
    assert fao.sensedFraction == pytest.approx(WEIGHTS[0] / sum(WEIGHTS))


def test_unsensed_layers_warning(tmp_path, mode):
    """A warning reports the unsensed layers; the nrec hint only for multi-LGS systems."""
    with pytest.warns(UserWarning, match='not sensed') as rec:
        _build(tmp_path, 1000.0, multi=mode)
    hint = any('NumberReconstructedLayers = 1' in str(w.message) for w in rec)
    assert hint == mode


def test_no_warning_when_all_layers_sensed(tmp_path, mode):
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        # only our warning is promoted to an error
        warnings.filterwarnings('default', message='^(?!.*not sensed).*')
        _build(tmp_path, 90000.0, multi=mode)


def test_lgs_above_all_layers_is_unchanged(tmp_path, mode):
    fao = _build(tmp_path, 90000.0, multi=mode)
    assert fao.sensedLayers.all()
    h = nnp.asarray(cpuArray(fao.ao.atm.heights), dtype=float)
    z = float(cpuArray(fao.gs.height[0]))
    nnp.testing.assert_array_equal(nnp.asarray(fao.strechFactor), 1.0/(1.0 - h/z))
    assert fao.sensedFraction == 1.0
    assert nnp.all(nnp.isfinite(_psd(fao)))


def test_zenith_angle_scales_layers_and_lgs(tmp_path):
    """Heights and LGS altitude are both scaled by the airmass: the mask is zenith-invariant."""
    fao = _build(tmp_path, 1000.0, zenith=40.0, heights=[0.0, 500.0, 2000.0, 8000.0])
    assert list(fao.sensedLayers) == [True, True, False, False]
    assert nnp.all(nnp.isfinite(_psd(fao)))
    assert nnp.all(nnp.asarray(fao.strechFactor) > 0)


def test_psd_nonincreasing_unsensed_monotonic(tmp_path, mode):
    """Moving the LGS below a layer must not decrease the residual."""
    heights, weights = [0.0, 1000.0, 5000.0], [0.4, 0.3, 0.3]
    res = [_residual(_build(tmp_path, z, multi=mode, heights=heights, weights=weights))
           for z in (90000.0, 20000.0, 4000.0, 800.0)]
    # first two: all sensed; then one, then two layers lost
    assert res[2] >= res[1] * (1 - 1e-3)
    assert res[3] >= res[2] * (1 - 1e-3)
    assert res[3] > res[1]


def test_residual_lower_bound_open_loop_unsensed(tmp_path, mode):
    """Residual >= in-band open-loop variance of the unsensed Cn2 fraction."""
    fao = _build(tmp_path, 100.0, multi=mode)
    watm = nnp.asarray(cpuArray(fao.Wphi), dtype=float)
    msk = nnp.asarray(cpuArray(fao.freq.mskInAO_), dtype=float)
    lower = nnp.sqrt((1 - fao.sensedFraction) * (msk * watm).sum())
    assert _residual(fao) >= lower * 0.99


def test_error_breakdown_unsensed(tmp_path, mode):
    all_sensed = _build(tmp_path, 90000.0, breakdown=True, multi=mode)
    assert all_sensed.wfeUnsensed == 0
    low = _build(tmp_path, 1000.0, breakdown=True, multi=mode)
    assert nnp.isfinite(low.wfeUnsensed) and low.wfeUnsensed > 0


def test_unsensed_layers_do_not_enter_compression(tmp_path):
    """eqLayers only compresses sensed layers (NumberReconstructedLayers < nLayers)."""
    heights, weights = [0.0, 300.0, 600.0, 5000.0, 9000.0], [0.4, 0.2, 0.2, 0.1, 0.1]
    fao = _build(tmp_path, 1000.0, nrec=2, heights=heights, weights=weights)
    assert nnp.all(nnp.isfinite(_psd(fao)))
    assert fao.sensedLayers.sum() == 3
    assert nnp.all(nnp.asarray(fao.strechFactor_mod) > 0)


def test_fewer_sensed_layers_than_reconstructed_layers(tmp_path):
    """nRecLayers > nSensed: modelled atmosphere uses min(nRec, nSensed) consistently."""
    heights = [0.0, 300.0, 5000.0, 9000.0, 12000.0]
    weights = [0.4, 0.2, 0.2, 0.1, 0.1]
    fao = _build(tmp_path, 1000.0, multi=True, nrec=4, heights=heights, weights=weights)
    assert fao.sensedLayers.sum() == 2
    assert nnp.all(nnp.isfinite(_psd(fao)))
    atm_mod = fao.atm_mod
    n = len(nnp.atleast_1d(cpuArray(atm_mod.heights)))
    assert n == 2
    for attr in ('weights', 'wSpeed', 'wDir'):
        assert len(nnp.atleast_1d(cpuArray(getattr(atm_mod, attr)))) == n
    assert len(nnp.atleast_1d(fao.strechFactor_mod)) == n


# ---------------------------------------------------------------------------
# anisoplanatismModel
# ---------------------------------------------------------------------------

def _stand_ins(heights, weights, z):
    tel = SimpleNamespace(D=8.0)
    atm = SimpleNamespace(heights=nnp.asarray(heights, dtype=float),
                          weights=nnp.asarray(weights, dtype=float),
                          nL=len(heights), r0=0.15, wvl=500e-9)
    lgs = SimpleNamespace(height=[z])
    return tel, atm, lgs


def _old_focal(tel, atm, lgs, coef, norm):
    z = float(lgs.height[0])
    var = 0.0
    for h, w in zip(atm.heights, atm.weights):
        if h > 0:
            x = h/z
            var += w*(0.5*x**(5/3) - coef*x**2)/norm
    return nnp.sqrt(var*(tel.D/atm.r0)**(5/3))*atm.wvl*1e9/2/nnp.pi


_FUNCS = [
    (aniso.focal_anisoplanatism_variance, 0.425, 1.0),
    (aniso.focal_anisoplanatism_wfe, 0.452, 0.423),
]
_H = [0.0, 500.0, 2000.0, 8000.0, 20000.0]
_W = [0.4, 0.2, 0.2, 0.1, 0.1]


@pytest.mark.parametrize('func,coef,norm', _FUNCS)
def test_focal_aniso_nonnegative_finite(func, coef, norm):
    for z in (100.0, 500.0, 1000.0, 2000.0, 8000.0, 20000.0, 90000.0):
        val = func(*_stand_ins(_H, _W, z))
        assert nnp.isfinite(val) and val >= 0


@pytest.mark.parametrize('func,coef,norm', _FUNCS)
def test_focal_aniso_nonincreasing_with_height(func, coef, norm):
    # Restricted to z >= 1.3 max(h): the per-layer polynomial itself peaks at h/z ~ 0.8-0.95
    # (known non-monotonic bump just below a layer, then a jump to the full variance at h >= z).
    zs = nnp.geomspace(1.3*max(_H), 200000.0, 30)
    vals = nnp.array([func(*_stand_ins(_H, _W, z)) for z in zs])
    assert nnp.all(nnp.diff(vals) <= 1e-9*vals.max())


@pytest.mark.parametrize('func,coef,norm', _FUNCS)
def test_focal_aniso_variance_increases_when_lgs_drops_below_layer(func, coef, norm):
    z_unsensed = func(*_stand_ins(_H, _W, 0.99*_H[-1]))
    z_sensed = func(*_stand_ins(_H, _W, 1.01*_H[-1]))
    assert z_unsensed > z_sensed


@pytest.mark.parametrize('func,coef,norm', _FUNCS)
def test_focal_aniso_matches_old_formula_small_ratio(func, coef, norm):
    args = _stand_ins([0.0, 500.0, 2000.0], [0.5, 0.3, 0.2], 90000.0)
    assert func(*args) == pytest.approx(_old_focal(*args, coef, norm), rel=1e-12)


@pytest.mark.parametrize('func,coef,norm', _FUNCS)
def test_focal_aniso_layer_above_lgs_gives_full_variance(func, coef, norm):
    tel, atm, lgs = _stand_ins([0.0, 5000.0], [0.5, 0.5], 1000.0)
    expected = nnp.sqrt(0.5*1.0299*(tel.D/atm.r0)**(5/3))*atm.wvl*1e9/2/nnp.pi
    assert func(tel, atm, lgs) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Physics limit cases
# ---------------------------------------------------------------------------

def _quiet(multi: bool) -> dict:
    """Options making the HO WFS (almost) noise-free.

    NoiseVariance=[0.0] is only parsed for a single WFS; with several LGS the
    flux is raised instead (1e9 photons makes the tomographic inversion NaN).
    """
    return dict(photons=1e5, ron=0.0) if multi else dict(noise_var=0.0)


def _np(a) -> nnp.ndarray:
    return nnp.asarray(cpuArray(a))


def _inband_ol(fao) -> nnp.ndarray:
    """Open-loop in-band PSD of the whole Cn2 (resAO x resAO), no piston filter:
    the residual of unsensed layers is not piston-filtered."""
    return _np(fao.freq.mskInAO_) * _np(fao.Wphi)


def _nm2_scale(fao) -> float:
    """Factor converting the internal PSD units to the nm^2 returned in `fao.PSD`."""
    dk = 2*float(cpuArray(fao.freq.kcMax_))/fao.freq.resAO
    return (dk*float(cpuArray(fao.freq.wvlRef))*1e9/2/nnp.pi)**2


def _inband_slice(fao):
    n, r = fao.freq.nOtf, fao.freq.resAO
    id1 = int(nnp.ceil(n/2 - r/2))
    id2 = int(nnp.ceil(n/2 + r/2))
    return slice(id1, id2)


def test_single_ground_layer_independent_of_lgs_height(tmp_path, mode):
    """A: with only a ground layer the cone term vanishes and z_LGS is irrelevant."""
    psds = [_psd(_build(tmp_path, z, multi=mode, heights=[0.0], weights=[1.0],
                        **_quiet(mode))) for z in (500.0, 10000.0, 90000.0)]
    assert nnp.all(nnp.isfinite(psds[0]))
    for p in psds[1:]:
        nnp.testing.assert_allclose(p, psds[0], rtol=1e-6, atol=1e-6*psds[0].max())


def test_ground_plus_layer_above_lgs_scao_linear_combination(tmp_path):
    """B (SLAO): ground corrected as ground-only case, high layer fully open loop."""
    w0, w1 = 0.7, 0.3
    kw = dict(noise_var=0.0, breakdown=True, wspeed=8.0)
    two = _build(tmp_path, 1000.0, heights=[0.0, 5000.0], weights=[w0, w1], **kw)
    ground = _build(tmp_path, 1000.0, heights=[0.0], weights=[1.0], name='g.ini', **kw)
    assert list(two.sensedLayers) == [True, False]
    ref = w0*_np(ground.psdSpatioTemporal).real + w1*_inband_ol(two)[:, :, None]
    nnp.testing.assert_allclose(_np(two.psdSpatioTemporal).real, ref,
                                rtol=1e-4, atol=1e-6*abs(ref).max())
    # total in-band PSD cannot be below the open-loop part of the high layer
    inband = _psd(two)[_inband_slice(two), _inband_slice(two), :]
    assert nnp.all(inband >= w1*_inband_ol(two)[:, :, None]*_nm2_scale(two)*(1 - 1e-4))


def test_ground_plus_layer_above_lgs_tomographic(tmp_path):
    """B (GLAO): same identity, only approximate (MMSE prior differs: w0 vs 1)."""
    w0, w1 = 0.7, 0.3
    kw = dict(multi=True, breakdown=True, **_quiet(True))
    two = _build(tmp_path, 1000.0, heights=[0.0, 5000.0], weights=[w0, w1], **kw)
    ground = _build(tmp_path, 1000.0, heights=[0.0], weights=[1.0], name='g.ini', **kw)
    st = _np(two.psdSpatioTemporal).real
    ol1 = w1*_inband_ol(two)[:, :, None]
    ref = w0*_np(ground.psdSpatioTemporal).real + ol1
    assert nnp.all(nnp.isfinite(st))
    dev = abs(st.sum(axis=(0, 1)) - ref.sum(axis=(0, 1))) / ref.sum(axis=(0, 1))
    print(f'tomographic B: relative deviation of integrated PSD per source = {dev}')
    assert nnp.all(dev < 0.02)
    inband = _psd(two)[_inband_slice(two), _inband_slice(two), :]
    assert nnp.all(inband >= ol1*_nm2_scale(two)*(1 - 1e-4))


def test_single_layer_above_lgs_is_open_loop_scao(tmp_path):
    """C (SLAO): nothing sensed -> total PSD = fitting + open-loop in-band PSD."""
    fao = _build(tmp_path, 1000.0, heights=[5000.0], weights=[1.0], breakdown=True,
                 **_quiet(False))
    assert fao.sensedFraction == 0
    sl = _inband_slice(fao)
    psd = _psd(fao)
    expected = _np(fao.psdFit).real[:, :, None] * nnp.ones(psd.shape[2])
    expected = expected.copy()
    expected[sl, sl, :] += _inband_ol(fao)[:, :, None]
    expected *= _nm2_scale(fao)
    assert nnp.all(nnp.isfinite(psd))
    assert nnp.abs(_np(fao.psdAlias)).max() == 0
    nnp.testing.assert_allclose(psd, expected, rtol=1e-4, atol=1e-6*expected.max())


def test_single_layer_above_lgs_is_open_loop_tomographic(tmp_path):
    """C (GLAO): same limit; the zero-column WFS model leaves the full residual."""
    fao = _build(tmp_path, 1000.0, multi=True, heights=[5000.0], weights=[1.0],
                 breakdown=True, **_quiet(True))
    sl = _inband_slice(fao)
    psd = _psd(fao)
    assert nnp.all(nnp.isfinite(psd))
    expected = _np(fao.psdFit).real[:, :, None] * nnp.ones(psd.shape[2])
    expected = expected.copy()
    expected[sl, sl, :] += _inband_ol(fao)[:, :, None]
    expected *= _nm2_scale(fao)
    nnp.testing.assert_allclose(psd, expected, rtol=1e-3, atol=1e-5*expected.max())


@pytest.mark.parametrize('multi', [False, True])
def test_zero_sensed_layers_with_compression_is_finite(tmp_path, multi):
    """No sensed layer and NumberReconstructedLayers < nLayers (eqLayers on an empty set)."""
    fao = _build(tmp_path, 1000.0, multi=multi, nrec=1, heights=[3000.0, 5000.0, 8000.0],
                 weights=[0.5, 0.3, 0.2], breakdown=True, **_quiet(multi))
    assert fao.sensedFraction == 0
    assert nnp.all(nnp.isfinite(_psd(fao)))
    assert nnp.isfinite(fao.wfeUnsensed) and fao.wfeUnsensed > 0
