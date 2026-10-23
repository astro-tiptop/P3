"""Regression tests: enlargeSupport rounding, DL PSF peak, getEnsquaredEnergy wrap-around."""

import numpy as np
import pytest
from scipy.special import erf

from p3.aoSystem import np as xp  # cupy when the GPU is enabled
from p3.aoSystem.FourierUtils import (
    cpuArray,
    enlargeSupport,
    getEnsquaredEnergy,
    otf2psf,
    telescopeOtf,
)


def make_pupil(n: int, obs: float = 0.0) -> np.ndarray:
    """Binary circular pupil of diameter n pixels centred at n/2."""
    x = np.arange(n) - n / 2
    r = np.hypot(*np.meshgrid(x, x))
    return ((r <= n / 2) & (r >= obs * n / 2)).astype(np.float64)


def gaussian(shape: tuple, y0: int, x0: int, sigma: float) -> np.ndarray:
    """Separable 2D Gaussian centred on pixel (y0, x0)."""
    y, x = np.indices(shape)
    return np.exp(-((y - y0) ** 2 + (x - x0) ** 2) / (2 * sigma**2))


@pytest.mark.parametrize("nx, n, expected", [(100, 2.3, 230), (128, 3.0, 384),
                                             (150, 2.0, 300), (100, 1.0, 100),
                                             (60, 2.5, 150)])
def test_enlarge_support_size_and_centering(nx, n, expected):
    rng = np.random.default_rng(0)
    a = rng.random((nx, nx)) + 0.1
    out = cpuArray(enlargeSupport(xp.asarray(a), n))
    assert out.shape == (expected, expected)
    assert out.sum() == pytest.approx(a.sum(), rel=1e-12)
    p = (expected - nx) // 2
    assert np.array_equal(out[p:p + nx, p:p + nx], a)
    out[p:p + nx, p:p + nx] = 0
    assert not out.any()


def test_dl_peak_matches_pixelated_pupil_non_integer_samp():
    nx, samp = 100, 2.3
    pupil = make_pupil(nx)
    otf = telescopeOtf(xp.asarray(pupil), samp)
    psf = cpuArray(otf2psf(otf, psfInOnePix=True))
    assert psf.shape[0] == 230
    peak = psf.max() / psf.sum()
    expected = pupil.sum() / (samp * nx) ** 2
    print(f"peak={peak:.6e} expected={expected:.6e} rel={peak / expected - 1:.2e}")
    assert peak == pytest.approx(expected, rel=5e-3)


@pytest.mark.parametrize("shape, peak", [((64, 64), (32, 32)),
                                         ((65, 65), (32, 32)),
                                         ((64, 64), (10, 40))])
def test_ensquared_energy_no_wraparound(shape, peak):
    sigma = 3.0
    psf = gaussian(shape, peak[0], peak[1], sigma)
    ee = getEnsquaredEnergy(xp.asarray(psf))
    # tolerance: sums over different slices differ by float round-off (~1e-16)
    assert np.all(np.diff(ee) >= -1e-12)
    assert ee[-1] == pytest.approx(1.0, abs=1e-12)
    for k in range(2, 7):
        expected = erf((k + 0.5) / (np.sqrt(2) * sigma)) ** 2
        assert ee[k] == pytest.approx(expected, rel=1e-2), k


def _old_ensquared_energy(psf: np.ndarray) -> np.ndarray:
    """Previous implementation (no clamping), valid for a centred even-size PSF."""
    S = psf.sum()
    nY, nX = psf.shape
    y0, x0 = np.unravel_index(psf.argmax(), psf.shape)
    nEE = int(min(nY - y0, nX - x0))
    EE = np.zeros(nEE + 1, dtype=psf.dtype)
    for n in range(nEE + 1):
        EE[n] = psf[y0 - n:y0 + n + 1, x0 - n:x0 + n + 1].sum()
    return EE / S


def test_ensquared_energy_even_centred_identical_to_old():
    psf = gaussian((64, 64), 32, 32, 3.0)
    new = getEnsquaredEnergy(xp.asarray(psf))
    old = _old_ensquared_energy(psf)
    assert np.array_equal(cpuArray(new), old)
