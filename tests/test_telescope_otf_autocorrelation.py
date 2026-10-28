#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests: telescopeOtf is the pupil autocorrelation; getStrehl clip."""

import unittest

import numpy as np

from p3.aoSystem import np as xp  # cupy when the GPU is enabled
from p3.aoSystem.FourierUtils import (
    cpuArray,
    enlargeSupport,
    getStrehl,
    otf2psf,
    telescopeOtf,
)

SIZES = (64, 128, 150, 65)
SAMPS = (2.0, 3.04, 3.62)


def make_pupil(n: int, obs: float = 0.28, spiders: bool = False) -> np.ndarray:
    """Binary circular pupil with central obstruction and optional spiders.

    Parameters
    ----------
    n : int
        Pupil size in pixels (centred on the array centre, n/2).
    obs : float
        Central obstruction ratio.
    spiders : bool
        If True, two thin zero bands (horizontal and vertical) are added.
    """
    x = np.arange(n) - n / 2
    xx, yy = np.meshgrid(x, x)
    r = np.hypot(xx, yy)
    pupil = ((r <= n / 2) & (r >= obs * n / 2)).astype(np.float64)
    if spiders:
        pupil[np.abs(xx) < 1] = 0
        pupil[np.abs(yy) < 1] = 0
    return pupil


def dl_psf_numpy(pupil: np.ndarray, samp: float) -> np.ndarray:
    """Unit-sum |FFT(P_pad)|^2 with the peak on pixel M//2, padded as P3 does."""
    pad = enlargeSupport(xp.asarray(pupil), samp)
    psf = np.abs(np.fft.fft2(np.fft.fftshift(cpuArray(pad)))) ** 2
    psf = np.fft.fftshift(psf)
    return psf / psf.sum()


class TestTelescopeOtfAutocorrelation(unittest.TestCase):
    def _cases(self):
        for n in SIZES:
            for samp in SAMPS:
                yield n, samp, make_pupil(n)
        for samp in SAMPS:
            yield 128, samp, make_pupil(128, spiders=True)

    def test_dl_psf_non_negative(self):
        for n, samp, pupil in self._cases():
            with self.subTest(n=n, samp=samp):
                otf = telescopeOtf(xp.asarray(pupil), samp)
                psf = cpuArray(otf2psf(otf, psfInOnePix=True))
                neg = psf[psf < 0].sum() / psf.sum()
                print(f"n={n} samp={samp} negFrac={neg:.3e}")
                self.assertGreater(neg, -1e-10)
                self.assertGreaterEqual(psf.min(), -1e-12 * psf.max())

    def test_otf_hermitian_and_unit_centre(self):
        for n, samp, pupil in self._cases():
            with self.subTest(n=n, samp=samp):
                otf = cpuArray(telescopeOtf(xp.asarray(pupil), samp))
                m = otf.shape[0]
                c = m // 2
                self.assertAlmostEqual(abs(otf[c, c]), 1.0, places=10)
                self.assertAlmostEqual(abs(otf).max(), 1.0, places=10)
                # OTF(-f) = conj(OTF(f)) about the centre pixel c
                k = min(c, m - c - 1)
                sub = otf[c - k:c + k + 1, c - k:c + k + 1]
                self.assertTrue(np.allclose(sub, np.conj(sub[::-1, ::-1]),
                                            atol=1e-10))

    def test_dl_peak_matches_pixelated_pupil(self):
        for n, samp, pupil in self._cases():
            with self.subTest(n=n, samp=samp):
                otf = telescopeOtf(xp.asarray(pupil), samp)
                m = otf.shape[0]
                self.assertEqual(m, enlargeSupport(xp.asarray(pupil), samp).shape[0])
                peak = cpuArray(otf2psf(otf, psfInOnePix=True)).max()
                expected = dl_psf_numpy(pupil, samp).max()
                # binary pupil: sum(P)^2 / (M^2 sum P^2) = sum(P) / M^2
                self.assertAlmostEqual(expected, pupil.sum() / m**2,
                                       delta=1e-12)
                print(f"n={n} samp={samp} peak={peak:.8e} expected={expected:.8e}")
                self.assertTrue(np.isclose(peak, expected, rtol=1e-6))


class TestGetStrehl(unittest.TestCase):
    def test_max_strehl_of_ideal_psf_is_one(self):
        for n, samp, pupil in [(64, 2.0, make_pupil(64)),
                               (128, 3.04, make_pupil(128)),
                               (150, 3.62, make_pupil(150)),
                               (65, 2.0, make_pupil(65))]:
            with self.subTest(n=n, samp=samp):
                psf = dl_psf_numpy(pupil, samp)
                sr = float(cpuArray(getStrehl(xp.asarray(psf), xp.asarray(pupil), samp, method='max',
                                     psfInOnePix=True)))
                self.assertAlmostEqual(sr, 1.0, delta=1e-4)

    def test_max_strehl_does_not_modify_input(self):
        pupil = make_pupil(64)
        psf = dl_psf_numpy(pupil, 2.0) * 7.3  # not unit-sum on purpose
        psf[3, 3] = -0.5  # negative value that must be clipped on a copy
        before = psf.copy()
        psf_dev = xp.asarray(psf)  # the very array handed to getStrehl
        getStrehl(psf_dev, xp.asarray(pupil), 2.0, method='max',
                  psfInOnePix=True)
        self.assertTrue(np.array_equal(cpuArray(psf_dev), before))

    def test_max_strehl_clips_negative_values(self):
        pupil = make_pupil(64)
        psf = dl_psf_numpy(pupil, 2.0)
        noisy = psf.copy()
        noisy[3, 3] = -0.2  # would shift the normalization if not clipped
        sr = float(cpuArray(getStrehl(xp.asarray(noisy), xp.asarray(pupil), 2.0, method='max', psfInOnePix=True)))
        # after clipping the negative pixel is 0, so the PSF is the ideal one
        self.assertAlmostEqual(sr, 1.0, delta=1e-4)

    def test_max_half_pixel_reference_requires_samp_ge_2(self):
        pupil = make_pupil(64)
        psf = dl_psf_numpy(pupil, 2.0)
        for samp in (0.5, 1.0, 1.5):
            with self.subTest(samp=samp):
                with self.assertRaises(ValueError):
                    getStrehl(xp.asarray(psf), xp.asarray(pupil), samp, method='max', psfInOnePix=False)

    def test_max_strehl_of_half_pixel_centred_dl_psf_is_one(self):
        n = 64
        pupil = make_pupil(n)
        xs = np.arange(n) - n / 2  # pupil coordinates in pixels, as in make_pupil
        for samp in (2.0, 3.0):
            with self.subTest(samp=samp):
                m = telescopeOtf(xp.asarray(pupil), samp).shape[0]
                # exact DFT with the PSF peak midway between pixels m//2-1 and m//2 on both axes
                e = np.exp(-2j * np.pi * np.outer(np.arange(m) - m // 2 - 0.5, xs) / m)
                psf = np.abs(e @ pupil @ e.T) ** 2
                psf /= psf.sum()
                sr = float(cpuArray(getStrehl(xp.asarray(psf), xp.asarray(pupil), samp, method='max',
                                              psfInOnePix=False)))
                self.assertAlmostEqual(sr, 1.0, delta=1e-3)

    def test_max_strehl_in_one_pix_works_below_nyquist(self):
        pupil = make_pupil(64)
        # DL PSF at samp 1.5 point-sampled every 2 pixels = samp 0.75, peak on the centre pixel
        fine = dl_psf_numpy(pupil, 1.5)
        c = fine.shape[0] // 2
        idx = c + 2 * np.arange(-(c // 2), c // 2)
        psf = fine[idx][:, idx]
        psf /= psf.sum()
        sr = float(cpuArray(getStrehl(xp.asarray(psf), xp.asarray(pupil), 0.75, method='max',
                                      psfInOnePix=True)))
        self.assertAlmostEqual(sr, 1.0, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
