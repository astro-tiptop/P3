#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests of the fused / batched kernels of fourierModel (small matmul, DM phasors,
controller layer term, aliasing layer term, MCAO cone filter).

Convention: in p3.aoSystem `np` is the dispatched backend (cupy on GPU), `nnp` is real
numpy and `cpuArray` brings arrays to host. Reference values are computed with `nnp`."""

from types import SimpleNamespace

import numpy as nnp
import pytest

from p3.aoSystem import np, cpuArray
from p3.aoSystem import fourierModel as fm
from p3.aoSystem.fourierModel import fourierModel

from test_low_altitude_lgs import _build as _build_lgs, _write_ini
from test_mcao_cone_sensing import _build as _build_cone


def _host(a):
    return nnp.asarray(cpuArray(a))


def _rand_c(rng, shape):
    return rng.standard_normal(shape) + 1j*rng.standard_normal(shape)


# On the numpy backend _small_matmul is np.matmul and _controller_layer is
# _controller_layer_term: comparing them with themselves proves nothing.
_ON_CPU = not hasattr(np, 'fuse')
skip_on_cpu = pytest.mark.skipif(_ON_CPU, reason='numpy backend: the function under test is the reference itself')

# ---------------------------------------------------------------------------
# 1. _small_matmul
# ---------------------------------------------------------------------------

@skip_on_cpu
@pytest.mark.parametrize('K', [1, 3, 16])
@pytest.mark.parametrize('a_batch', [(5, 7), (5, 1), (1, 7)])
def test_small_matmul_matches_matmul(K, a_batch):
    rng = nnp.random.default_rng(K)
    m, n = 4, 3
    a = _rand_c(rng, a_batch + (m, K))
    b = _rand_c(rng, (5, 7, K, n))
    out = _host(fm._small_matmul(np.asarray(a), np.asarray(b)))
    ref = nnp.matmul(a, b)
    assert out.shape == ref.shape == (5, 7, m, n)
    assert nnp.max(nnp.abs(out - ref)) <= 1e-13*nnp.max(nnp.abs(ref))


@skip_on_cpu
def test_small_matmul_broadcast_b():
    rng = nnp.random.default_rng(0)
    a = _rand_c(rng, (5, 7, 2, 3))
    b = _rand_c(rng, (1, 7, 3, 4))
    out = _host(fm._small_matmul(np.asarray(a), np.asarray(b)))
    ref = nnp.matmul(a, b)
    assert nnp.max(nnp.abs(out - ref)) <= 1e-13*nnp.max(nnp.abs(ref))


@skip_on_cpu
def test_small_matmul_large_k_falls_back_exactly():
    rng = nnp.random.default_rng(1)
    a = _rand_c(rng, (5, 7, 3, 17))
    b = _rand_c(rng, (5, 7, 17, 2))
    ad, bd = np.asarray(a), np.asarray(b)
    assert nnp.array_equal(_host(fm._small_matmul(ad, bd)), _host(np.matmul(ad, bd)))


@skip_on_cpu
def test_small_matmul_max_k_argument():
    rng = nnp.random.default_rng(2)
    a = _rand_c(rng, (5, 7, 3, 4))
    b = _rand_c(rng, (5, 7, 4, 2))
    ad, bd = np.asarray(a), np.asarray(b)
    # max_k=3 < K=4 -> exact matmul
    assert nnp.array_equal(_host(fm._small_matmul(ad, bd, max_k=3)), _host(np.matmul(ad, bd)))


@skip_on_cpu
def test_small_matmul_2d_inputs():
    rng = nnp.random.default_rng(3)
    a, b = _rand_c(rng, (4, 3)), _rand_c(rng, (3, 5))
    out = _host(fm._small_matmul(np.asarray(a), np.asarray(b)))
    assert nnp.allclose(out, a @ b, rtol=1e-13, atol=1e-13)


# ---------------------------------------------------------------------------
# 2. _dmPhasors
# ---------------------------------------------------------------------------

def _fake_phasor_model(heights, kc, n=24, dk=0.5):
    """Bare instance carrying what _dmPhasors reads, on a synthetic AO grid."""
    k1 = (nnp.arange(n) - n//2) * dk
    kx, ky = nnp.meshgrid(k1, k1, indexing='ij')
    fao = object.__new__(fourierModel)
    fao.ao = SimpleNamespace(dms=SimpleNamespace(heights=nnp.asarray(heights, float)))
    fao.freq = SimpleNamespace(kxAO_=np.asarray(kx), kyAO_=np.asarray(ky),
                               k2AO_=np.asarray(kx**2 + ky**2),
                               kc_=nnp.asarray(kc, float))
    return fao, kx, ky


@pytest.mark.parametrize('dtype', [nnp.complex128, nnp.complex64])
def test_dm_phasors_modulus_mask_and_phase(dtype):
    heights, kc = [0.0, 3000.0, 12000.0], [3.0, 4.0, 2.0]
    ax, ay = 1.2e-6, -0.7e-6
    fao, kx, ky = _fake_phasor_model(heights, kc)
    out = _host(fao._dmPhasors(ax, ay, dtype))
    n = kx.shape[0]
    assert out.shape == (n, n, 3) and out.dtype == dtype
    k = nnp.hypot(kx, ky)
    tol = 1e-6 if dtype == nnp.complex64 else 1e-12
    for j in range(3):
        inside = k <= kc[j]
        assert inside.any() and (~inside).any()
        assert nnp.all(out[~inside, j] == 0)
        assert nnp.allclose(nnp.abs(out[inside, j]), 1.0, atol=tol)
    assert nnp.all(out[k <= kc[0], 0] == 1)        # DM at height 0: exactly 1 inside
    # phase at a few sample points, plain scalars
    for (ix, iy, j) in [(n//2 + 2, n//2 + 1, 1), (n//2 - 3, n//2 + 2, 2), (n//2 + 1, n//2 - 2, 1)]:
        if k[ix, iy] > kc[j]:
            continue
        ref = nnp.exp(2j*nnp.pi*heights[j]*(ax*kx[ix, iy] + ay*ky[ix, iy]))
        # compare through the phasor product to avoid wrapping ambiguities
        assert abs(out[ix, iy, j] - ref) < (1e-3 if dtype == nnp.complex64 else 1e-12)


# ---------------------------------------------------------------------------
# 4. controller: _controller_layer_term / _controller_layer / controller()
# ---------------------------------------------------------------------------

def _ref_controller_term(kx, ky, ax, ay, Ts, delay, gain, eps):
    """Explicit float64 numpy ATF = H z^-d / (1 + H z^-d), H = g / (1 - z^-1)."""
    z = nnp.exp(2j*nnp.pi*Ts*(ax*kx + ay*ky))
    den = 1 - 1/z
    den = nnp.where(nnp.abs(den) < eps, eps, den)
    H = gain/den
    zd = z**(-delay)
    den2 = 1 + H*zd
    den2 = nnp.where(nnp.abs(den2) < eps, eps, den2)
    atf = H*zd/den2
    return atf, z


def _synthetic_grid(n=33, dk=0.4):
    k1 = (nnp.arange(n) - n//2) * dk
    return nnp.meshgrid(k1, k1, indexing='ij')


# 1.7 and 2.3 are not exactly representable in float16 (see the comment in controller())
@pytest.mark.parametrize('delay', [2, 1.5, 1.7, 2.3])
def test_controller_layer_term_vs_explicit(delay):
    kx, ky = _synthetic_grid()
    Ts, gain, eps, w = 1e-3, 0.4, nnp.finfo(float).eps, 0.37
    ax, ay = 11.0, -5.0
    c = nnp.complex128(2j*nnp.pi*Ts)
    rng = nnp.random.default_rng(0)
    h1_0 = _rand_c(rng, kx.shape)
    h2_0 = rng.random(kx.shape)
    hn_0 = rng.random(kx.shape)
    h1, h2, hn = (_host(x) for x in fm._controller_layer_term(
        np.asarray(h1_0), np.asarray(h2_0), np.asarray(hn_0),
        np.asarray(kx), np.asarray(ky), ax, ay, c, delay, gain, eps, w))
    atf, z = _ref_controller_term(kx, ky, ax, ay, Ts, delay, gain, eps)
    assert nnp.allclose(h1, h1_0 + w*atf, rtol=1e-12, atol=1e-12)
    assert nnp.allclose(h2, h2_0 + w*nnp.abs(atf)**2, rtol=1e-12, atol=1e-12)
    assert nnp.allclose(hn, hn_0 + w*nnp.abs(atf/z)**2, rtol=1e-12, atol=1e-12)


@skip_on_cpu
@pytest.mark.parametrize('delay', [2, 1.5, 1.7, 2.3])
def test_controller_layer_fused_equals_term(delay):
    """Fused kernel called with numpy scalars, as controller() does: inside cp.fuse a
    unary op on a Python scalar alone (-delay) would be computed in float16."""
    kx, ky = _synthetic_grid()
    Ts, gain, eps, w = 1e-3, 0.5, nnp.finfo(float).eps, 0.25
    c = nnp.complex128(2j*nnp.pi*Ts)
    args = (np.zeros(kx.shape, dtype=complex), np.zeros(kx.shape), np.zeros(kx.shape),
            np.asarray(kx), np.asarray(ky), nnp.float64(7.0), nnp.float64(3.0), c,
            nnp.float64(delay), nnp.float64(gain), nnp.float64(eps), nnp.float64(w))
    for a, b in zip(fm._controller_layer(*args), fm._controller_layer_term(*args)):
        a, b = _host(a), _host(b)
        assert nnp.max(nnp.abs(a - b)) <= 1e-12*max(nnp.max(nnp.abs(b)), 1e-300)


def test_controller_layer_term_low_frequency_passes():
    """An integrator has |ATF| -> 1 as the temporal frequency -> 0."""
    kx, ky = _synthetic_grid()
    Ts, gain, eps = 1e-3, 0.5, nnp.finfo(float).eps
    c = complex(2j*nnp.pi*Ts)
    _, h2, _ = fm._controller_layer_term(np.zeros(kx.shape, dtype=complex), np.zeros(kx.shape),
                                         np.zeros(kx.shape), np.asarray(kx), np.asarray(ky),
                                         1.0, 0.0, c, 2, gain, eps, 1.0)
    h2 = _host(h2)
    f = nnp.abs(kx)                     # temporal frequency up to v.k, in Hz (v = 1 m/s)
    low = (f > 0) & (f < 0.5)
    assert nnp.all(h2[low] > 0.9)
    assert nnp.allclose(h2[f == 0], 1.0, atol=1e-6)


@pytest.fixture(scope='module')
def one_layer_model(tmp_path_factory):
    fao = _build_lgs(tmp_path_factory.mktemp('onelayer'), 90e3, heights=[5000.0], weights=[1.0])
    assert fao.ao.atm.nL == 1
    return fao


@pytest.mark.parametrize('wdir', [0.0, 35.0])
def test_controller_one_layer(one_layer_model, wdir):
    fao = one_layer_model
    fao.ao.atm.wDir = nnp.array([wdir])
    fao.controller(nTh=1)
    h1, h2, hn = _host(fao.h1), _host(fao.h2), _host(fao.hn)
    kx, ky = _host(fao.freq.kxAO_), _host(fao.freq.kyAO_)
    Ts = 1/fao.ao.rtc.holoop['rate']
    v = float(_host(fao.ao.atm.wSpeed)[0])
    vx, vy = v*nnp.cos(nnp.deg2rad(wdir)), v*nnp.sin(nnp.deg2rad(wdir))
    z = nnp.exp(2j*nnp.pi*Ts*(vx*kx + vy*ky))
    atf, _ = _ref_controller_term(kx, ky, vx, vy, Ts, fao.ao.rtc.holoop['delay'],
                                  fao.ao.rtc.holoop['gain'], nnp.finfo(float).eps)
    tol = 1e-12 if h2.dtype == nnp.float64 else 1e-5
    assert nnp.allclose(h2, nnp.abs(h1)**2, rtol=tol, atol=tol)
    assert nnp.allclose(hn, nnp.abs(h1/z)**2, rtol=tol, atol=tol)
    assert nnp.allclose(h1, atf, rtol=tol, atol=tol)
    # integrator passes the lowest non-zero temporal frequencies
    vk = nnp.abs(vx*kx + vy*ky)
    low = vk[vk > 0].min()
    assert nnp.all(h2[nnp.isclose(vk, low)] > 0.9)
    assert h2.max() <= (nnp.abs(atf)**2).max()*(1 + tol)


@pytest.mark.parametrize('delay', [1.7, 2.3])
def test_controller_one_layer_fractional_delay(one_layer_model, delay):
    """End to end: a delay that is not float16-exact must reach the kernel unrounded."""
    fao = one_layer_model
    holoop = fao.ao.rtc.holoop
    old = holoop['delay']
    fao.ao.atm.wDir = nnp.array([0.0])
    holoop['delay'] = delay
    try:
        fao.controller(nTh=1)
        h1, h2 = _host(fao.h1), _host(fao.h2)
    finally:
        holoop['delay'] = old
    kx, ky = _host(fao.freq.kxAO_), _host(fao.freq.kyAO_)
    Ts = 1/holoop['rate']
    v = float(_host(fao.ao.atm.wSpeed)[0])
    atf, _ = _ref_controller_term(kx, ky, v, 0.0, Ts, delay, holoop['gain'], nnp.finfo(float).eps)
    tol = 1e-12 if h2.dtype == nnp.float64 else 1e-5
    assert nnp.max(nnp.abs(h1 - atf)) <= tol*nnp.max(nnp.abs(atf))
    assert nnp.max(nnp.abs(h2 - nnp.abs(atf)**2)) <= tol*nnp.max(nnp.abs(atf)**2)


def test_controller_wind_angle_average(one_layer_model):
    """nTh angles: h2 is the mean of |ATF|^2 over the angles, bounded by its maximum.

    Note: the source (before and after the fusion) scales the whole wind vector (vx, vy)
    by cos(theta) instead of rotating it; the reference reproduces that behaviour."""
    fao = one_layer_model
    fao.ao.atm.wDir = nnp.array([0.0])
    nTh = 4
    fao.controller(nTh=nTh)
    h2 = _host(fao.h2)
    kx, ky = _host(fao.freq.kxAO_), _host(fao.freq.kyAO_)
    Ts = 1/fao.ao.rtc.holoop['rate']
    v = float(_host(fao.ao.atm.wSpeed)[0])
    acc, mx = 0, 0
    for th in nnp.linspace(0, 2*nnp.pi - 2*nnp.pi/nTh, nTh):
        atf, _ = _ref_controller_term(kx, ky, v*nnp.cos(th), 0.0, Ts,
                                      fao.ao.rtc.holoop['delay'], fao.ao.rtc.holoop['gain'],
                                      nnp.finfo(float).eps)
        acc = acc + nnp.abs(atf)**2/nTh
        mx = max(mx, (nnp.abs(atf)**2).max())
    tol = 1e-12 if h2.dtype == nnp.float64 else 1e-5
    assert nnp.allclose(h2, acc, rtol=tol, atol=tol)
    assert h2.max() <= mx*(1 + tol)


# ---------------------------------------------------------------------------
# 5. aliasing
# ---------------------------------------------------------------------------

def test_aliasing_layer_term_factorization():
    rng = nnp.random.default_rng(5)
    d, T, td = 0.5, 2e-3, 4e-3
    nS, nK = 6, 20
    kx, ky = rng.uniform(-1, 1, nK)/d, rng.uniform(-1, 1, nK)/d
    m = rng.integers(-3, 4, nS).astype(float)
    n = rng.integers(-3, 4, nS).astype(float)
    km, kn = kx[None, :] - m[:, None]/d, ky[None, :] - n[:, None]/d
    tf = _rand_c(rng, nK)
    avr = _rand_c(rng, (nS, nK))
    vx, vy, w = 9.0, -4.0, 0.3
    two_pi_i = 2j*nnp.pi
    row = w*tf*nnp.exp(two_pi_i*(kx*vx + ky*vy)*td)
    shift = nnp.exp(-two_pi_i*(m*vx + n*vy)*td/d)
    out = _host(fm._aliasing_layer_term(np.asarray(avr), np.asarray(km), np.asarray(kn), vx, vy, T,
                                        np.asarray(shift[:, None]), np.asarray(row[None, :])))
    direct = avr + w*nnp.sinc(km*vx*T)*nnp.sinc(kn*vy*T)*nnp.exp(two_pi_i*(km*vx + kn*vy)*td)*tf[None, :]
    assert nnp.max(nnp.abs(out - direct)) <= 1e-12*nnp.max(nnp.abs(direct))


def test_aliasing_layer_term_zero_wind_zero_delay():
    """vx = vy = 0, td = 0: the layer average is sum(w) tf."""
    rng = nnp.random.default_rng(6)
    nS, nK = 5, 12
    km, kn = rng.standard_normal((nS, nK)), rng.standard_normal((nS, nK))
    tf = _rand_c(rng, nK)
    ws = [0.5, 0.3, 0.2]
    avr = np.zeros((nS, nK), dtype=complex)
    for w in ws:
        avr = fm._aliasing_layer_term(avr, np.asarray(km), np.asarray(kn), 0.0, 0.0, 1e-3,
                                      np.asarray(nnp.ones((nS, 1), complex)),
                                      np.asarray((w*tf)[None, :]))
    assert nnp.allclose(_host(avr), sum(ws)*tf[None, :].repeat(nS, 0), rtol=1e-13, atol=1e-13)


def test_aliasing_chunked_gpu_branch_equals_cpu_branch(monkeypatch):
    if fm._aliasing_layer_fused is None:
        pytest.skip('fused aliasing kernel only exists on the cupy backend')
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        fao = _build_lgs(pathlib.Path(td), 90e3, multi=False)
    fao.controller()
    psd_fused = _host(fao._aliasing_psd_chunked())
    monkeypatch.setattr(fm, '_aliasing_layer_fused', None)
    psd_cpu = _host(fao._aliasing_psd_chunked())
    assert nnp.all(psd_fused >= 0) and psd_fused.max() > 0
    assert nnp.max(nnp.abs(psd_fused - psd_cpu)) <= 1e-12*nnp.max(nnp.abs(psd_cpu))


@pytest.fixture(scope='module')
def aliasing_single_double(tmp_path_factory):
    """Aliasing PSD of the same model in double and single precision; for the single one
    the dtypes returned by the per-layer fused kernel (the avr_sum accumulator) are recorded."""
    if fm._aliasing_layer_fused is None:
        pytest.skip('fused aliasing kernel only exists on the cupy backend')
    out = {}
    orig = fm._aliasing_layer_fused
    for prec in ('double', 'single'):
        td = tmp_path_factory.mktemp('alias_' + prec)
        ini = _write_ini(td, 90e3, multi=False)
        with open(ini, 'a') as f:
            f.write('\n[COMPUTATION]\nprecision = %s\n' % prec)
        fao = fourierModel(ini, calcPSF=False, verbose=False, display=False,
                           reduce_memory=False, computeFocalAnisoCov=False,
                           getErrorBreakDown=False)
        assert fao.complex_dtype == (np.complex64 if prec == 'single' else np.complex128)
        fao.controller()
        acc_dtypes = []

        def spy(*a, **k):
            r = orig(*a, **k)
            acc_dtypes.append(r.dtype)
            return r

        fm._aliasing_layer_fused = spy
        try:
            out[prec] = (fao._aliasing_psd_chunked(), acc_dtypes)
        finally:
            fm._aliasing_layer_fused = orig
    return out


def test_aliasing_chunked_gpu_single_precision_accumulator(aliasing_single_double):
    """row, phase_shift, T and d (via km, kn, Q) keep the model precision, so avr_sum stays complex64
    over all layers (before the fix it was promoted to complex128 by the first layer)."""
    _, acc32 = aliasing_single_double['single']
    _, acc64 = aliasing_single_double['double']
    assert len(acc32) > 0 and all(dt == nnp.complex64 for dt in acc32)
    assert all(dt == nnp.complex128 for dt in acc64)


def test_aliasing_chunked_gpu_single_precision_matches_double(aliasing_single_double):
    """Single vs double PSD: float32 eps = 1.2e-7 per operation, accumulated over the layers
    and the shifts and amplified by the squared modulus; tolerance 1e-4 of the maximum."""
    p32 = _host(aliasing_single_double['single'][0]).astype(float)
    p64 = _host(aliasing_single_double['double'][0])
    assert p64.max() > 0
    assert nnp.max(nnp.abs(p32 - p64)) <= 1e-4*nnp.max(p64)


def test_aliasing_chunked_gpu_single_precision_result_dtype(aliasing_single_double):
    assert aliasing_single_double['single'][0].dtype == nnp.float32


# ---------------------------------------------------------------------------
# 6. MCAO cone filter (batched over layers)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def cone_geom(tmp_path_factory):
    fao = _build_cone(tmp_path_factory.mktemp('conefilt'))
    geom = fao._mcaoConeGeometry()
    assert geom is not None
    return geom


def test_mcao_cone_filter_batched_vs_per_layer(cone_geom):
    geom = cone_geom
    nLayer, nSrc = nnp.asarray(cpuArray(geom['f_cut'])).shape
    n_checked = 0
    for s in range(nSrc):
        idx = nnp.where(_host(geom['mask'])[:, s])[0]
        if idx.size == 0:
            continue
        out = _host(fourierModel._mcaoConeFilter(geom, idx, s))
        assert out.shape == geom['z_ao'].shape + (idx.size,)
        z_ao = _host(geom['z_ao'])
        for j, l in enumerate(idx):
            f_cut = float(_host(geom['f_cut'])[l, s])
            G2 = float(_host(geom['G2'])[l, s])
            zPole = nnp.exp(2*nnp.pi*f_cut/float(geom['fs']))
            H = z_ao*(1 - zPole)/(z_ao - zPole)
            ref = nnp.maximum(G2*(1 - nnp.abs(H)**2), 0)
            assert nnp.allclose(out[:, :, j], ref, rtol=1e-12, atol=1e-14)
            assert out[:, :, j].min() >= 0
            assert out[:, :, j].max() <= G2*(1 + 1e-12)
            n_checked += 1
    assert n_checked > 0


# ---------------------------------------------------------------------------
# 7. footprint inner gap, batched over layers (GPU) vs per layer
# ---------------------------------------------------------------------------

def _gap_geometry(D, gs_zen, gs_az, src_zen, src_az, z=90e3):
    fao = object.__new__(fourierModel)
    fao.ao = SimpleNamespace(tel=SimpleNamespace(D=D),
                             src=SimpleNamespace(zenith=nnp.asarray(src_zen, float),
                                                 azimuth=nnp.asarray(src_az, float)))
    fao.gs = SimpleNamespace(height=nnp.full(len(gs_zen), z), zenith=nnp.asarray(gs_zen, float),
                             azimuth=nnp.asarray(gs_az, float))
    return fao


@pytest.mark.skipif(_ON_CPU, reason='batched version is only used (and meaningful) on the cupy backend')
@pytest.mark.parametrize('gs_zen, gs_az, src_zen, src_az, h', [
    # 3 LGS at 30" (footprints separate at high altitude), layer at h = 0 and a high layer
    ([30, 30, 30], [0, 120, 240], [0, 20, 40], [0, 90, 200], [0.0, 500.0, 4000.0, 12000.0, 20000.0]),
    # 4 LGS on a wider asterism, off-axis sources, cones shrinking towards the LGS height
    ([60, 60, 60, 60], [0, 90, 180, 270], [0, 35, 50], [0, 45, 300], [0.0, 2000.0, 9000.0, 30000.0, 60000.0]),
    # wide 3-LGS asterism: central hole and fully separated footprints at high altitude
    ([120, 120, 120], [10, 130, 250], [0, 40, 100], [0, 0, 120], [0.0, 1000.0, 15000.0, 40000.0]),
])
def test_footprint_inner_gap_batched_equals_per_layer(gs_zen, gs_az, src_zen, src_az, h):
    fao = _gap_geometry(8.0, gs_zen, gs_az, src_zen, src_az)
    batched = nnp.asarray(fao._footprint_inner_gap_layers_gpu(h))
    assert batched.shape == (len(h), len(src_zen))
    per_layer = nnp.stack([fao._footprint_inner_gap(hi) for hi in h])
    assert nnp.max(nnp.abs(batched - per_layer)) <= 1e-12*8.0
    assert batched.max() > 0     # the geometry exercises non-trivial gaps
