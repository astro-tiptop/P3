#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests: psf2otf is the inverse of otf2psf (odd sizes); telescopeOtf
for pixels coarser than lambda/D (samp < 1) is the point-sampled DL PSF."""

import unittest

import numpy as np

from p3.aoSystem import np as xp  # cupy when the GPU is enabled
from p3.aoSystem.FourierUtils import (
    cpuArray,
    getStrehl,
    otf2psf,
    psf2otf,
    telescopeOtf,
)

NX = 40


def disk_pupil(n: int) -> np.ndarray:
    """Binary disk pupil of diameter n pixels centred at (n-1)/2."""
    x = np.arange(n) - (n - 1) / 2
    r = np.hypot(*np.meshgrid(x, x))
    return (r <= n / 2).astype(np.float64)


def point_sampled_peak(pupil: np.ndarray, samp: float) -> float:
    """Unit-sum peak of the DL PSF point-sampled at pixel scale lambda/D / samp.

    A very fine PSF (K points per coarse pixel) is computed by zero padding and
    decimated every K pixels keeping the peak (pixel M//2) in the sample set.
    """
    nx = pupil.shape[0]
    k = int(np.ceil(8 / samp))
    m = int(round(k * samp * nx))
    assert abs(k * samp * nx - m) < 1e-9
    pp = np.zeros((m, m))
    pp[:nx, :nx] = pupil
    f = np.fft.fftshift(np.abs(np.fft.fft2(pp)) ** 2)
    c = m // 2
    coarse = f[c % k::k, c % k::k]
    return coarse.max() / coarse.sum()


def dl_peak(samp: float) -> float:
    otf = telescopeOtf(xp.asarray(disk_pupil(NX)), samp)
    return float(cpuArray(otf2psf(otf, psfInOnePix=True)).max())


class TestPsf2Otf(unittest.TestCase):
    def test_round_trip_odd_and_even(self):
        rng = np.random.default_rng(0)
        for n in (64, 65, 101):
            with self.subTest(n=n):
                psf = rng.random((n, n)) * 1e-3
                psf[n // 2, n // 2] = 1.0
                psf /= psf.sum()
                otf = psf2otf(xp.asarray(psf))
                back = cpuArray(otf2psf(xp.asarray(np.fft.fftshift(cpuArray(otf))),
                                        psfInOnePix=True))
                err = np.abs(back - psf).max()
                print(f"n={n} round-trip max err={err:.2e}")
                self.assertLess(err, 1e-12)

    def test_otf_strehl_of_dl_psf_is_one(self):
        for n in (64, 65, 101):
            with self.subTest(n=n):
                pupil = disk_pupil(n)
                samp = 2.0
                otf = telescopeOtf(xp.asarray(pupil), samp)
                psf = otf2psf(otf, psfInOnePix=True)
                sr = float(cpuArray(getStrehl(psf, xp.asarray(pupil), samp,
                                              method='otf')))
                print(f"n={n} SR(otf)={sr:.6f}")
                self.assertAlmostEqual(sr, 1.0, delta=1e-3)


class TestCoarseSampling(unittest.TestCase):
    def test_peak_matches_point_sampled_reference(self):
        pupil = disk_pupil(NX)
        for samp in (0.5, 0.8, 0.9, 1.0, 1.6, 2.0):
            with self.subTest(samp=samp):
                peak = dl_peak(samp)
                ref = point_sampled_peak(pupil, samp)
                print(f"samp={samp} peak={peak:.6e} ref={ref:.6e} "
                      f"rel={peak / ref - 1:.2e}")
                self.assertTrue(np.isclose(peak, ref, rtol=1e-3))

    def test_peak_monotonic_with_pixel_size(self):
        p05, p08, p10 = dl_peak(0.5), dl_peak(0.8), dl_peak(1.0)
        print(f"peaks 0.5/0.8/1.0: {p05:.6e} {p08:.6e} {p10:.6e}")
        self.assertGreater(p05, p08)
        self.assertGreater(p08, p10)


if __name__ == '__main__':
    unittest.main()
