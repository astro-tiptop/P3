#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests of the layer/GS-batched tomographic matrices built in
fourierModel.spatialReconstructor (PbetaDM, MPalphaL) against explicit scalar references.

Convention: in p3.aoSystem `np` is the dispatched backend, `nnp` is real numpy."""

import numpy as nnp
import pytest

from p3.aoSystem import np, cpuArray
from test_low_altitude_lgs import _build as _build_lgs


def _host(a):
    return nnp.asarray(cpuArray(a))


def _sinc(x):
    return 1.0 if x == 0 else nnp.sin(nnp.pi*x)/(nnp.pi*x)


def _ref_mpalpha(fao, ix, iy, g, h):
    """Scalar MPalphaL[ix, iy, g, h] from the documented formula."""
    kx = float(_host(fao.freq.kxAO_)[ix, iy])
    ky = float(_host(fao.freq.kyAO_)[ix, iy])
    k = nnp.hypot(kx, ky)
    wdir = nnp.deg2rad(float(_host(fao.ao.atm.wDir)[h]))
    wspeed = float(_host(fao.ao.atm.wSpeed)[h])
    Hs = float(_host(fao.ao.atm.heights)[h]) * float(_host(fao.strechFactor)[h])
    gdir = _host(fao.gs.direction)
    dsub = fao.ao.wfs.optics[g].dsub
    sampTime = 1/fao.ao.rtc.holoop['rate']
    freq_t = nnp.cos(wdir)*kx + nnp.sin(wdir)*ky
    www = 2j*nnp.pi*k*_sinc(sampTime*fao.ao.wfs.detector[g].clock_rate*wspeed*freq_t)
    return (www*_sinc(dsub*kx)*_sinc(dsub*ky)
            * nnp.exp(2j*nnp.pi*Hs*(gdir[0, g]*kx + gdir[1, g]*ky)))


def _build_tomo(tmp_path, lgs_height):
    fao = _build_lgs(tmp_path, lgs_height, multi=True)
    # distinct wind per layer so that the freq_t and sinc arguments are exercised
    nL = fao.ao.atm.nL
    fao.ao.atm.wDir = nnp.linspace(0, 120, nL)
    fao.ao.atm.wSpeed = nnp.linspace(5, 20, nL)
    fao.reduce_memory = False
    fao.spatialReconstructor()
    return fao


@pytest.fixture(scope='module')
def tomo_all_sensed(tmp_path_factory):
    return _build_tomo(tmp_path_factory.mktemp('tomo_a'), 90e3)


@pytest.fixture(scope='module')
def tomo_unsensed(tmp_path_factory):
    # LGS at 2 km: the two highest layers are not sensed
    return _build_tomo(tmp_path_factory.mktemp('tomo_b'), 2000.0)


def _sample_points(fao):
    n = fao.freq.resAO
    return [(n//2, n//2), (n//2 + 1, n//2), (n//2, n//2 - 3), (3, n - 5), (n - 2, 4), (n//2 + 5, n//2 + 2)]


def test_mpalpha_matches_scalar_reference(tomo_all_sensed):
    fao = tomo_all_sensed
    assert fao.nGs >= 2 and fao.sensedLayers.all()
    M = _host(fao.MPalphaL)
    assert M.shape == (fao.freq.resAO, fao.freq.resAO, fao.nGs, fao.ao.atm.nL)
    tol = 1e-12 if M.dtype == nnp.complex128 else 1e-4
    scale = nnp.abs(M).max()
    for (ix, iy) in _sample_points(fao):
        for g in range(fao.nGs):
            for h in range(fao.ao.atm.nL):
                ref = _ref_mpalpha(fao, ix, iy, g, h)
                assert abs(M[ix, iy, g, h] - ref) <= tol*scale


def test_mpalpha_unsensed_layers_exactly_zero(tomo_unsensed):
    fao = tomo_unsensed
    sensed = nnp.asarray(fao.sensedLayers, dtype=bool)
    assert sensed.any() and (~sensed).any()
    M = _host(fao.MPalphaL)
    for h in range(fao.ao.atm.nL):
        if sensed[h]:
            assert nnp.abs(M[:, :, :, h]).max() > 0
        else:
            assert nnp.all(M[:, :, :, h] == 0)
    tol = 1e-12 if M.dtype == nnp.complex128 else 1e-4
    scale = nnp.abs(M).max()
    for (ix, iy) in _sample_points(fao):
        for h in nnp.where(sensed)[0]:
            assert abs(M[ix, iy, 0, h] - _ref_mpalpha(fao, ix, iy, 0, h)) <= tol*scale


def test_pbetadm_ground_dm_is_cutoff_mask(tomo_all_sensed):
    """DM at height 0: PbetaDM = 1 inside the DM cut-off, exactly 0 outside, for every source."""
    fao = tomo_all_sensed
    assert float(_host(fao.ao.dms.heights)[0]) == 0.0
    k = nnp.sqrt(_host(fao.freq.k2AO_))
    kc = float(_host(fao.freq.kc_)[0])
    inside = k <= kc
    assert inside.any() and (~inside).any()
    assert len(fao.PbetaDM) == fao.ao.src.nSrc
    for P in fao.PbetaDM:
        P = _host(P)
        assert P.shape == (fao.freq.resAO, fao.freq.resAO, 1, 1)
        assert nnp.all(P[inside, 0, 0] == 1)
        assert nnp.all(P[~inside, 0, 0] == 0)


def test_walpha_is_w_times_mpalpha(tomo_all_sensed):
    fao = tomo_all_sensed
    W, M, Wa = _host(fao.W), _host(fao.MPalphaL), _host(fao.Walpha)
    ref = nnp.einsum('xyij,xyjk->xyik', W, M)
    assert Wa.dtype in (nnp.complex64, nnp.complex128)
    tol = 1e-5 if Wa.dtype == nnp.complex64 else 1e-10
    assert nnp.max(nnp.abs(Wa - ref)) <= tol*nnp.abs(ref).max()
