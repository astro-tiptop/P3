#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the bounding-box-cropped find_contour_points against the pre-optimization version (main, 1ea47cd)."""

import subprocess
import types

import numpy as np
import pytest

from p3.aoSystem.FourierUtils import find_contour_points, getFWHM

P3_DIR = '/raid1/guido/pythonLib/P3'


def _load_old_module() -> types.ModuleType:
    """Load FourierUtils from commit 1ea47cd (main before the optimization) as a standalone module."""
    try:
        res = subprocess.run(['git', '-C', P3_DIR, 'show', '1ea47cd:p3/aoSystem/FourierUtils.py'],
                             capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        pytest.skip('git not available')
    if res.returncode != 0 or not res.stdout:
        pytest.skip('cannot read FourierUtils.py from commit 1ea47cd')
    old = types.ModuleType('fu_old')
    old.__package__ = 'p3.aoSystem'
    exec(compile(res.stdout, 'fu_old', 'exec'), old.__dict__)
    return old


@pytest.fixture(scope='module')
def old():
    return _load_old_module()


def _gauss(ny, nx, sx, sy, cx=None, cy=None, theta=0.0):
    """Peak-1 elliptical Gaussian (sigmas in px, theta in rad), centered by default."""
    cx = (nx - 1) / 2 if cx is None else cx
    cy = (ny - 1) / 2 if cy is None else cy
    y, x = np.indices((ny, nx), dtype=float)
    x, y = x - cx, y - cy
    u = x * np.cos(theta) + y * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta)
    return np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))


def _two_blobs():
    n = 101
    return _gauss(n, n, 4, 4, 25, 30) + 0.9 * _gauss(n, n, 5, 3, 70, 65)


def _noisy():
    rng = np.random.default_rng(42)
    return _gauss(101, 101, 7, 5, theta=0.4) + 0.05 * rng.standard_normal((101, 101))


CASES = {
    'round_odd': lambda: _gauss(101, 101, 5, 5),
    'round_even': lambda: _gauss(100, 100, 5, 5),
    'elliptical_odd': lambda: _gauss(101, 101, 8, 4),
    'elliptical_even': lambda: _gauss(100, 100, 8, 4),
    'rotated': lambda: _gauss(101, 101, 9, 4, theta=np.deg2rad(35)),
    'rotated_nonsquare': lambda: _gauss(90, 120, 9, 4, theta=np.deg2rad(-20)),
    'sharp': lambda: _gauss(65, 65, 0.8, 0.8),
    'sharp_even': lambda: _gauss(64, 64, 0.7, 0.9),
    'broad': lambda: _gauss(121, 121, 121 / 3 / 2.355, 121 / 3 / 2.355),
    'near_border': lambda: _gauss(101, 101, 3, 3, cx=50, cy=3),
    'near_border_corner': lambda: _gauss(101, 101, 3, 3, cx=3, cy=3),
    'two_blobs': _two_blobs,
    'noisy': _noisy,
}


def _run(fn, *args, **kw):
    """Return ('ok', result) or ('err', exception type)."""
    try:
        return 'ok', fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        return 'err', type(e)


def _assert_identical(a, b):
    assert a[0] == b[0]
    if a[0] == 'err':
        assert a[1] is b[1]
        return
    ra, rb = a[1], b[1]
    if isinstance(ra, (tuple, list)) and not isinstance(rb, np.ndarray):
        assert len(ra) == len(rb)
        for x, y in zip(ra, rb):
            assert np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
    else:
        assert np.array_equal(np.asarray(ra), np.asarray(rb), equal_nan=True)


@pytest.mark.parametrize('name', list(CASES))
@pytest.mark.parametrize('frac', [0.5, 0.3, 0.8])
def test_find_contour_points_identical(old, name, frac):
    img = CASES[name]()
    level = frac * img.max()
    new = find_contour_points(img, level)
    ref = old.find_contour_points(img, level)
    assert np.array_equal(np.asarray(new), np.asarray(ref))


@pytest.mark.parametrize('name', list(CASES))
def test_getfwhm_contour_identical(old, name):
    img = CASES[name]()
    new = _run(getFWHM, img, 1.7, method='contour', nargout=4)
    ref = _run(old.getFWHM, img, 1.7, method='contour', nargout=4)
    if ref[0] == 'err' and issubclass(ref[1], ValueError) and name.startswith('near_border'):
        # half-max region touching the edge: the old fwhm_1d raised, the fixed one returns
        assert new[0] == 'ok'
        return
    _assert_identical(new, ref)


@pytest.mark.parametrize('name', ['round_odd', 'rotated', 'noisy', 'two_blobs'])
def test_getfwhm_contour_rebin2_identical(old, name):
    img = CASES[name]()
    new = _run(getFWHM, img, 1.7, rebin=2, method='contour', nargout=4)
    ref = _run(old.getFWHM, img, 1.7, rebin=2, method='contour', nargout=4)
    _assert_identical(new, ref)


def test_peak_on_edge_no_longer_raises(old):
    # A peak exactly on the edge row made fwhm_1d raise ValueError
    # (profile[left-1:left+2] too short) in the old code; fwhm_1d now uses the
    # edge as the crossing on that side (FWHM truncated, a lower bound).
    img = _gauss(101, 101, 3, 3, cx=50, cy=0)
    ref = _run(old.getFWHM, img, 1.0, method='contour', nargout=4)
    assert ref[0] == 'err' and issubclass(ref[1], ValueError)
    new = _run(getFWHM, img, 1.0, method='contour', nargout=4)
    assert new[0] != 'err'
    assert all(np.isfinite(v) and v > 0 for v in new[1][:2])


@pytest.mark.parametrize('name', ['round_odd', 'two_blobs', 'near_border'])
def test_level_above_max_returns_zero(old, name):
    img = CASES[name]()
    level = 2 * img.max()
    new = find_contour_points(img, level)
    ref = old.find_contour_points(img, level)
    assert new == [0]
    assert ref == [0]


def test_level_equal_to_max_returns_zero(old):
    # Strict '>' comparison: no pixel is above a level equal to the maximum.
    img = _gauss(101, 101, 5, 5)
    assert find_contour_points(img, img.max()) == old.find_contour_points(img, img.max()) == [0]


def test_elliptical_gaussian_fwhm():
    """FWHM of a Gaussian is 2*sqrt(2 ln2)*sigma.

    getFWHM returns fwhmX = 2*max radius (major axis) and fwhmY = 2*min radius
    (minor axis), in mas. Tolerance 3%: the contour is a set of pixel-resolved
    boundary points on a discrete grid (quantization ~0.5-1 px out of ~14-28 px
    half widths, i.e. a few %), not a sub-pixel iso-line.
    """
    ps = 2.0
    k = 2 * np.sqrt(2 * np.log(2))
    img = _gauss(201, 201, 6, 4)
    fwhmX, fwhmY = getFWHM(img, pixelScale=ps, method='contour', nargout=2)
    assert fwhmX == pytest.approx(k * 6 * ps, rel=0.03)
    assert fwhmY == pytest.approx(k * 4 * ps, rel=0.03)
