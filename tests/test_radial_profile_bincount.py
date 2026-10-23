#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the bincount-based radial_profile against the old sort-based one."""

import numpy as np
import pytest
from scipy.special import j0, j1

from p3.aoSystem.FourierUtils import radial_profile, getEncircledEnergy


def _radial_profile_reference(image, pixelscale=1, ee=False, center=None, binsize=None,
                              maxradius=None, normalize='None', pa_range=None, nargout=2):
    """Old argsort/cumsum implementation (default path only), kept as reference.

    Parameters
    ----------
    image : numpy.ndarray
        2D image (normalization is applied in place, as in the original).

    Returns
    -------
    profile, (rr, profile) or (rr, profile, ee) depending on nargout and ee.
    """
    if normalize.lower() == 'peak':
        image /= image.max()
    elif normalize.lower() == 'total':
        image /= image.sum()

    if binsize is None:
        binsize = pixelscale
    if center is None:
        center = tuple((a - 1) / 2.0 for a in image.shape[::-1])

    y, x = np.indices(image.shape, dtype=float)
    x -= center[0]
    y -= center[1]
    r = np.sqrt(x ** 2 + y ** 2) * pixelscale / binsize

    if pa_range is None:
        ind = np.argsort(r.flat)
        sr = r.flat[ind]
        sim = image.flat[ind]
    else:
        pa = np.rad2deg(np.arctan2(-x, y))
        mask = (pa >= pa_range[0]) & (pa <= pa_range[1])
        ind = np.argsort(r[mask].flat)
        sr = r[mask].flat[ind]
        sim = image[mask].flat[ind]

    ri = np.round(sr).astype(int)
    deltar = ri[1:] - ri[:-1]
    rind = np.where(deltar)[0]
    nr = rind[1:] - rind[:-1]
    csim = np.nan_to_num(sim).cumsum(dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]
    radialprofile = tbin / nr

    radialprofile2 = np.empty(len(radialprofile) + 1)
    if rind[0] != 0:
        radialprofile2[0] = csim[rind[0]] / (rind[0] + 1)
    else:
        radialprofile2[0] = csim[0]
    radialprofile2[1:] = radialprofile

    rr = np.arange(r.min(), r.min() + len(radialprofile2)) * binsize
    if maxradius is not None:
        crop = rr < maxradius
        rr = rr[crop]
        radialprofile2 = radialprofile2[crop]

    if nargout == 1:
        return radialprofile2
    if ee:
        ee = csim[rind]
        ee /= np.max(ee)
        return rr, radialprofile2, ee
    return rr, radialprofile2


def _random_image(n, seed=0):
    return np.random.default_rng(seed).random((n, n)) + 0.1


def _psf_like(n=129, seed=1):
    rng = np.random.default_rng(seed)
    y, x = np.indices((n, n), dtype=float)
    r = np.hypot(x - (n - 1) / 2, y - (n - 1) / 2)
    return np.exp(-r ** 2 / (2 * 2.0 ** 2)) + 1e-3 / (1 + r) ** 2 + 1e-6 * rng.random((n, n))


def _assert_same(img, **kw):
    """Run old and new implementations on copies and compare all outputs."""
    scale = np.nanmax(np.abs(img))
    atol = 1e-14 * scale
    ee = kw.get('ee', False)
    new = radial_profile(img.copy(), **kw)
    old = _radial_profile_reference(img.copy(), **kw)
    assert len(new) == len(old)
    for a, b in zip(new, old):
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=atol)
    if ee:
        assert len(new) == 3


@pytest.mark.parametrize("n", [64, 65, 256])
@pytest.mark.parametrize("ee", [False, True])
def test_default_center(n, ee):
    _assert_same(_random_image(n, seed=n), ee=ee)


@pytest.mark.parametrize("center", [(32, 32), (30.0, 33.0), (10.5, 40.25)])
@pytest.mark.parametrize("ee", [False, True])
def test_explicit_centers(center, ee):
    _assert_same(_random_image(65, seed=3), ee=ee, center=center)


@pytest.mark.parametrize("binsize", [None, 7.0, 1.0])
def test_pixelscale_with_and_without_binsize(binsize):
    _assert_same(_random_image(65, seed=4), ee=True, pixelscale=3.5, binsize=binsize)


def test_maxradius_crop():
    img = _random_image(65, seed=5)
    _assert_same(img, ee=True, maxradius=20.0)
    _assert_same(img, ee=False, pixelscale=3.5, maxradius=50.0)


@pytest.mark.parametrize("n", [64, 65])
def test_pa_range(n):
    _assert_same(_random_image(n, seed=6), ee=True, pa_range=[-30, 60])
    _assert_same(_random_image(n, seed=6), ee=False, pa_range=[-30, 60], center=(30.0, 33.0))


@pytest.mark.parametrize("normalize", ['total', 'peak', 'None'])
def test_normalize(normalize):
    _assert_same(_random_image(65, seed=7), ee=True, normalize=normalize)


def test_nargout_1():
    img = _random_image(65, seed=8)
    new = radial_profile(img.copy(), nargout=1)
    old = _radial_profile_reference(img.copy(), nargout=1)
    assert new.shape == old.shape
    np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-14)


def test_nan_treated_as_zero():
    img = _random_image(65, seed=9)
    img[5, 7] = np.nan
    img[32, 32] = np.nan
    img[60, 3] = np.nan
    new = radial_profile(img.copy(), ee=True)
    old = _radial_profile_reference(img.copy(), ee=True)
    for a, b in zip(new, old):
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("kw", [dict(), dict(center=(60.0, 66.5)), dict(pixelscale=5.0),
                                dict(pa_range=[-30, 60], normalize='total')])
def test_psf_like(kw):
    _assert_same(_psf_like(), ee=True, **kw)


@pytest.mark.parametrize("center", [None, (30.0, 33.0)])
def test_get_encircled_energy(center):
    psf = _psf_like(65)
    ee_new, rr_new = getEncircledEnergy(psf.copy(), pixelscale=2.5, center=center, nargout=2)
    rr_old, _, ee_old = _radial_profile_reference(psf.copy(), pixelscale=2.5, center=center, ee=True)
    assert ee_new.shape == ee_old.shape and rr_new.shape == rr_old.shape
    np.testing.assert_allclose(ee_new, ee_old, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(rr_new, rr_old, rtol=1e-10, atol=1e-14)
    ee_only = getEncircledEnergy(psf.copy(), pixelscale=2.5, center=center)
    np.testing.assert_allclose(ee_only, ee_old, rtol=1e-10, atol=1e-14)


def test_airy_profile_and_encircled_energy():
    """Airy pattern: I = (2 J1(x)/x)^2, EE(x) = 1 - J0(x)^2 - J1(x)^2."""
    n, c, step = 801, 400, 0.1  # step in lambda/D per pixel
    y, x = np.indices((n, n), dtype=float)
    r = np.hypot(x - c, y - c)
    xx = np.pi * r * step
    safe = np.where(xx == 0, 1.0, xx)
    img = np.where(xx == 0, 1.0, (2 * j1(safe) / safe) ** 2)
    # Zero the corners beyond the inscribed disk: otherwise the outer annuli are
    # only partially sampled and the last-bin normalization of EE is not E(x_last).
    r_max = float(c)
    img[r > r_max] = 0.0

    rr, prof, ee = radial_profile(img, ee=True, center=(c, c), pixelscale=1)

    assert rr[0] == 0.0
    assert abs(prof[0] - 1.0) < 1e-12

    # Bin 12 (r in [11.5, 12.5) px) straddles the first zero at 12.2 px (3.8317/(pi*0.1));
    # averaging I ~ (0.21 dx)^2 over the bin gives ~5e-4, so 1e-3 would be too tight.
    assert prof[12] < 2e-3

    # EE index j includes all pixels with round(r) <= j, i.e. outer radius rr[j]+0.5 px.
    def E(x_):
        return 1 - j0(x_) ** 2 - j1(x_) ** 2

    e_last = E(np.pi * step * (r_max + 0.5))
    r_out = rr + 0.5
    for r_ld, tol in [(1.22, 0.01), (2.23, 0.01), (5.0, 0.01)]:
        ee_num = np.interp(r_ld / step, r_out, ee)
        ee_ana = E(np.pi * r_ld) / e_last
        assert ee_num == pytest.approx(ee_ana, rel=tol)
    assert E(np.pi * 1.22) == pytest.approx(0.838, abs=2e-3)
    assert ee[-1] == pytest.approx(1.0)
