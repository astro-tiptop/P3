#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the reduced sensing volume (cone effect) term of multi-LGS systems
([sensor_HO] addMcaoWFsensConeError): footprint geometry and PSD/error breakdown."""

from types import SimpleNamespace

import numpy as nnp
import pytest
from scipy import ndimage

from p3.aoSystem import FourierUtils
from p3.aoSystem.FourierUtils import cpuArray
from p3.aoSystem.fourierModel import fourierModel
from test_low_altitude_lgs import _INI_TEMPLATE

RAD2ARC = 180 / nnp.pi * 3600
Z_LGS = 90e3


# ---------------------------------------------------------------------------
# Geometry: _footprint_inner_gap on a bare instance
# ---------------------------------------------------------------------------

def _geom(D, gs_zen, gs_az, src_zen=(0.0,), src_az=(0.0,), z=Z_LGS):
    """Minimal fourierModel instance carrying only what _footprint_inner_gap reads."""
    fao = object.__new__(fourierModel)
    n = len(gs_zen)
    fao.ao = SimpleNamespace(tel=SimpleNamespace(D=D),
                             src=SimpleNamespace(zenith=nnp.asarray(src_zen, float),
                                                 azimuth=nnp.asarray(src_az, float)))
    fao.gs = SimpleNamespace(height=nnp.full(n, z), zenith=nnp.asarray(gs_zen, float),
                             azimuth=nnp.asarray(gs_az, float))
    return fao


def _arcsec(rho_m, h):
    """Angle [arcsec] subtending rho_m at altitude h."""
    return rho_m / h * RAD2ARC


def _brute_gap(fao, h, n=1024):
    """Reference gap [m] per source: Euclidean distance transform on a fine grid."""
    D = float(fao.ao.tel.D)
    z = float(fao.gs.height[0])
    R = D/2 * (1 - h/z)

    def centres(zen, az):
        zen, az = nnp.asarray(zen, float)/RAD2ARC, nnp.deg2rad(nnp.asarray(az, float))
        return nnp.stack([zen*nnp.cos(az), zen*nnp.sin(az)], -1) * h

    cs, cg = centres(fao.ao.src.zenith, fao.ao.src.azimuth), centres(fao.gs.zenith, fao.gs.azimuth)
    r_env = nnp.hypot(*cg.T).max() + R
    pix = D / n
    u = (nnp.arange(n) - n/2 + 0.5) * pix
    X, Y = nnp.meshgrid(u, u, indexing='ij')
    out = []
    for c in cs:
        ok = nnp.hypot(X, Y) < D/2
        ok &= nnp.hypot(X + c[0], Y + c[1]) < r_env
        for g in cg:
            ok &= nnp.hypot(X + c[0] - g[0], Y + c[1] - g[1]) >= R
        out.append(2 * max(ndimage.distance_transform_edt(ok).max() - 0.5, 0) * pix if ok.any() else 0.0)
    return nnp.array(out), pix


def test_gap_full_coverage_is_zero():
    """Concentric LGS footprints larger than the science footprint cover everything."""
    D, h = 8.0, 10e3
    for zen, az in [([0.0], [0.0]), ([0.0, 0.0], [0.0, 90.0])]:
        assert nnp.all(_geom(D, zen, az)._footprint_inner_gap(h) == 0)


def test_gap_small_ring_nearly_covered():
    """Four LGS very close to the axis leave at most thin crescents at the envelope edge."""
    D, h = 8.0, 10e3
    fao = _geom(D, [_arcsec(0.3, h)]*4, [0, 90, 180, 270], src_zen=[0.0, 5.0], src_az=[0.0, 30.0])
    assert nnp.all(fao._footprint_inner_gap(h) < 0.05*D)


def test_gap_central_hole_four_lgs():
    """Four LGS with R < rho: central hole, gap matches the brute-force inscribed circle."""
    D, h, rho = 8.0, 10e3, 4.0
    R = D/2 * (1 - h/Z_LGS)
    fao = _geom(D, [_arcsec(rho, h)]*4, [0, 90, 180, 270])
    gap = fao._footprint_inner_gap(h)
    ref, pix = _brute_gap(fao, h)
    assert ref[0] >= 2*(rho - R) - 2*pix   # at least the on-axis hole
    assert gap[0] == pytest.approx(ref[0], rel=0.02, abs=2*pix)
    assert gap[0] <= ref[0] + 2*pix        # distance is exact: never above the true maximum


def test_gap_separated_footprints_is_full_footprint():
    """Science footprint disjoint from all LGS footprints and inside the envelope: gap = D."""
    D, h = 8.0, 10e3
    fao = _geom(D, [_arcsec(10.0, h)]*3, [0, 120, 240], src_zen=[0.0, _arcsec(1.0, h)],
                src_az=[0.0, 45.0])
    assert fao._footprint_inner_gap(h) == pytest.approx([D, D], rel=1e-2)


def test_gap_shape_and_nonnegative():
    fao = _geom(2.2, [420.0]*3, [0, 120, 270], src_zen=[0, 7, 14], src_az=[0, 10, 20])
    gap = fao._footprint_inner_gap(5e3)
    assert gap.shape == (3,) and nnp.all(gap >= 0) and nnp.all(nnp.isfinite(gap))


@pytest.mark.parametrize('seed', range(6))
def test_gap_random_configurations_vs_brute_force(seed):
    rng = nnp.random.default_rng(seed)
    D, h = 8.0, 10e3
    nG, nS = rng.integers(3, 7), 3
    rho = rng.uniform(2.0, 8.0, nG)
    fao = _geom(D, _arcsec(rho, h), rng.uniform(0, 360, nG),
                src_zen=_arcsec(rng.uniform(0, 6.0, nS), h), src_az=rng.uniform(0, 360, nS))
    gap = fao._footprint_inner_gap(h)
    ref, pix = _brute_gap(fao, h, n=768)
    # coarse-grid maximisation can only underestimate, by about a coarse pixel
    assert nnp.all(gap <= ref + 2*pix)
    assert nnp.all(gap >= ref - 0.05*D)


def test_gap_ring_of_lgs_below_analytic_protrusion():
    """MAVIS-like ring: inside the asterism the inner gap stays negligible, so that the
    analytic (edge) term dominates as in the old model."""
    D, th = 8.0, 17.5
    fao = _geom(D, [th]*16, nnp.arange(16)*22.5, src_zen=[0, 5, 10, 14, 17.5], src_az=[0]*5)
    d_ang = nnp.minimum(fao.ao.src.zenith, th) - 0.5*(2*th - D/Z_LGS*RAD2ARC)
    for h in (1e3, 4e3, 10e3, 16e3):
        analytic = nnp.maximum(d_ang*h/RAD2ARC, 0)
        assert nnp.all(fao._footprint_inner_gap(h) <= analytic + 0.02*D)


# ---------------------------------------------------------------------------
# fourierModel: PSD and error breakdown
# ---------------------------------------------------------------------------

def _ini(tmp_path, cone=True, ring=False, nG=3, z=Z_LGS, name='cone.ini'):
    """Reduced multi-LGS ini built from the low-altitude-LGS template.

    ring=False: GLAO-like, D = 2.2 m, `nG` LGS at 420 arcsec (central hole).
    ring=True: MAVIS-like, D = 8 m, 8 LGS ring at 17.5 arcsec, high-altitude layers.
    """
    if ring:
        nG, zen, az = 8, [17.5]*8, [22.5 + 45*i for i in range(8)]
        heights, weights = [0.0, 4000.0, 10000.0, 16000.0], [0.5, 0.2, 0.2, 0.1]
    else:
        zen, az = [420.0]*nG, [360.0*i/nG for i in range(nG)]
        heights, weights = [0.0, 500.0, 2000.0, 8000.0], [0.5, 0.2, 0.2, 0.1]
    nL = len(heights)
    cfg = _INI_TEMPLATE.format(
        zenith=0.0, weights=weights, heights=heights, wspeed=[8.0]*nL, wdir=[0.0]*nL,
        ho_zen=zen, ho_az=az, lgs_height=z, phot=[1e5]*nG, ron=0.0, nlens=[16]*nG,
        noisevar=[None], opt_zen='[0, 30, 30]', opt_az='[0, 0, 120]', opt_w='[1, 1, 1]',
        nrec=nL)
    if ring:
        cfg = cfg.replace('TelescopeDiameter = 2.2', 'TelescopeDiameter = 8.0')
        cfg = cfg.replace('DmPitchs = [0.105]', 'DmPitchs = [0.25]')
        cfg = cfg.replace('NumberActuators = [17]', 'NumberActuators = [33]')
        # off-axis sources: the analytic protrusion must fall below kcMax
        cfg = cfg.replace('Zenith = [0.0, 14.0]\nAzimuth = [0.0, 0.0]',
                          'Zenith = [0.0, 14.0, 17.5]\nAzimuth = [0.0, 0.0, 0.0]')
    cfg = cfg.replace('NoiseVariance = [None]',
                      'NoiseVariance = [None]\naddMcaoWFsensConeError = %s' % cone)
    path = tmp_path / name
    path.write_text(cfg)
    return str(path)


def _build(tmp_path, breakdown=False, verbose=False, **kw):
    return fourierModel(_ini(tmp_path, **kw), calcPSF=False, verbose=verbose, display=False,
                        reduce_memory=False, computeFocalAnisoCov=False,
                        getErrorBreakDown=breakdown)


def _cone_wfe(psd_cone):
    return nnp.sqrt(nnp.asarray(cpuArray(psd_cone), float).sum(axis=(0, 1)))


def _zero_gap(monkeypatch):
    monkeypatch.setattr(fourierModel, '_footprint_inner_gap',
                        lambda self, h, **kw: nnp.zeros(len(self.ao.src.zenith)))


def _old_cone_psd(fao, psd_res):
    """Analytic-only reference of mcaoWFsensConePSD (model without the inner gap)."""
    D = float(fao.ao.tel.D)
    k = nnp.asarray(cpuArray(nnp.sqrt(cpuArray(fao.freq.k2_))), float)
    fs = k.max() * 2
    z = nnp.exp(1j * k / (fs/2) * nnp.pi)
    n, res = fao.freq.nOtf, fao.freq.resAO
    id1, id2 = int(nnp.ceil(n/2 - res/2)), int(nnp.ceil(n/2 + res/2))
    atmo = nnp.asarray(cpuArray(fao.ao.atm.spectrum(FourierUtils.np.sqrt(fao.freq.k2_))), float)
    pf = nnp.asarray(cpuArray(FourierUtils.pistonFilter(D, FourierUtils.np.sqrt(fao.freq.k2_))), float)
    src = nnp.asarray(cpuArray(fao.ao.src.zenith), float)
    gmax = float(nnp.max(cpuArray(fao.gs.zenith)))
    lfov = 2*gmax
    efov = lfov - D/float(cpuArray(fao.gs.height[0]))*RAD2ARC
    dE = nnp.minimum(src, gmax) - (efov/2 if efov > 0 else efov)
    dL = nnp.maximum(src - lfov/2, 0)
    heights = nnp.asarray(cpuArray(fao.ao.atm.heights), float)
    weights = nnp.asarray(cpuArray(fao.ao.atm.weights), float)
    kc = float(cpuArray(fao.freq.kcMax_))
    out = nnp.zeros((n, n, len(src)))
    dpsd = nnp.maximum(atmo[id1:id2, id1:id2, None] - nnp.asarray(cpuArray(psd_res), float)[id1:id2, id1:id2], 0)
    dpsd = dpsd * pf[id1:id2, id1:id2, None]
    for h, w, sens in zip(heights, weights, cpuArray(fao.sensedLayers)):
        if h <= 0 or not sens:
            continue
        for s in range(len(src)):
            if dE[s] <= 0:
                continue
            fcut = RAD2ARC / (dE[s]*h)
            eqD = min(D - dL[s]*h/RAD2ARC, D)
            if not (fcut < kc and eqD > 0):
                continue
            zp = nnp.exp(2*nnp.pi*fcut/fs)
            lp = z[id1:id2, id1:id2]*(1 - zp)/(z[id1:id2, id1:id2] - zp)
            out[id1:id2, id1:id2, s] += w*nnp.maximum((1 - abs(lp)**2)*(eqD/D)**2, 0)*dpsd[:, :, s]
    return out


@pytest.fixture(scope='module')
def mcao(tmp_path_factory):
    """MAVIS-like model with the cone term enabled."""
    return _build(tmp_path_factory.mktemp('ring'), ring=True)


@pytest.fixture(scope='module')
def glao(tmp_path_factory):
    """GLAO-like model (3 LGS at 420 arcsec) with the cone term enabled."""
    return _build(tmp_path_factory.mktemp('glao'))


def _res_psd(fao):
    return 0.3 * nnp.asarray(cpuArray(fao.ao.atm.spectrum(FourierUtils.np.sqrt(fao.freq.k2_))))[:, :, None] \
        * nnp.ones(fao.ao.src.nSrc)


def test_analytic_part_matches_old_model(mcao, monkeypatch):
    """With the inner gap disabled the PSD is the old analytic-only one."""
    psd_res = _res_psd(mcao)
    ref = _old_cone_psd(mcao, psd_res)
    assert ref.sum() > 0                      # off-axis source (17.5") is affected
    _zero_gap(monkeypatch)
    new = nnp.asarray(cpuArray(mcao.mcaoWFsensConePSD(FourierUtils.np.asarray(psd_res))), float)
    nnp.testing.assert_allclose(new, ref, rtol=1e-6, atol=1e-12*ref.max())


def test_mavis_like_cone_wfe_essentially_unchanged(mcao, monkeypatch):
    """Real gap vs analytic only on a classic MCAO: cone wfe changes by < 3 %."""
    psd_res = FourierUtils.np.asarray(_res_psd(mcao))
    new = _cone_wfe(mcao.mcaoWFsensConePSD(psd_res))
    _zero_gap(monkeypatch)
    old = _cone_wfe(mcao.mcaoWFsensConePSD(psd_res))
    assert old.max() > 0
    nnp.testing.assert_allclose(new, old, rtol=0.03)


def test_glao_central_hole_adds_cone_term(glao, monkeypatch):
    """3 LGS at 420 arcsec leave a hole: cone term on axis > 0, zero in the old model."""
    on_axis = int(nnp.argmin(cpuArray(glao.ao.src.zenith)))
    assert _cone_wfe(glao.psdMcaoWFsensCone)[on_axis] > 0
    _zero_gap(monkeypatch)
    old = glao.mcaoWFsensConePSD(FourierUtils.np.asarray(_res_psd(glao)))
    assert nnp.all(nnp.asarray(cpuArray(old)) == 0)
    assert glao._mcaoConeGeometry() is None


def test_cone_geometry_structure(glao):
    geom = glao._mcaoConeGeometry()
    nlay, nsrc = geom['f_cut'].shape
    assert nsrc == glao.ao.src.nSrc and nlay == len(geom['layer_idx'])
    assert geom['mask'].shape == geom['G2'].shape == (nlay, nsrc)
    assert nnp.all((geom['G2'] >= 0) & (geom['G2'] <= 1 + 1e-12))
    assert geom['mask'].any()


def test_error_breakdown_cone_per_source(tmp_path):
    fao = _build(tmp_path, breakdown=True)
    nS = fao.ao.src.nSrc
    cone = nnp.asarray(cpuArray(fao.wfeMcaoCone))
    tot = nnp.asarray(cpuArray(fao.wfeTot))
    assert cone.shape == (nS,) and nnp.all(nnp.isfinite(cone)) and cone.max() > 0
    assert tot.shape == (nS,) and nnp.all(tot**2 >= cone**2)
    # PSD (rad^2 per pixel) -> nm
    rad2nm = 2*float(cpuArray(fao.freq.kcMax_))/fao.freq.resAO * float(cpuArray(fao.freq.wvlRef))*1e9/2/nnp.pi
    nnp.testing.assert_allclose(cone, _cone_wfe(fao.psdMcaoWFsensCone) * rad2nm, rtol=1e-6)


def test_error_breakdown_verbose_prints_on_axis_cone(tmp_path, capsys):
    _build(tmp_path, breakdown=True, verbose=True)
    assert '.Mcao Cone:' in capsys.readouterr().out


def test_flag_off_no_cone_term(tmp_path, glao):
    off = _build(tmp_path, cone=False, breakdown=True, name='off.ini')
    assert getattr(off, 'psdMcaoWFsensCone', None) is None
    assert off.wfeMcaoCone == 0
    on = nnp.asarray(cpuArray(glao.PSD), float)
    diff = on.sum(axis=(0, 1)) - nnp.asarray(cpuArray(off.PSD), float).sum(axis=(0, 1))
    assert nnp.all(diff > 0)


def test_single_lgs_cone_term_not_applied(tmp_path):
    """SLAO: the flag has no effect on the PSD."""
    def single(cone, name):
        cfg = open(_ini(tmp_path, cone=cone, nG=1, name=name)).read()
        cfg = cfg.replace('Zenith = [420.0]', 'Zenith = [0.0]')
        (tmp_path / name).write_text(cfg)
        return fourierModel(str(tmp_path / name), calcPSF=False, verbose=False, display=False,
                            reduce_memory=False, computeFocalAnisoCov=False)
    on, off = single(True, 'on.ini'), single(False, 'off.ini')
    assert on.nGs == 1
    assert getattr(on, 'psdMcaoWFsensCone', None) is None
    nnp.testing.assert_array_equal(nnp.asarray(cpuArray(on.PSD)), nnp.asarray(cpuArray(off.PSD)))


def test_single_lgs_flag_on_error_breakdown(tmp_path):
    cfg = open(_ini(tmp_path, nG=1)).read().replace('Zenith = [420.0]', 'Zenith = [0.0]')
    (tmp_path / 's.ini').write_text(cfg)
    fao = fourierModel(str(tmp_path / 's.ini'), calcPSF=False, verbose=False, display=False,
                       reduce_memory=False, computeFocalAnisoCov=False, getErrorBreakDown=True)
    assert fao.wfeMcaoCone == 0
