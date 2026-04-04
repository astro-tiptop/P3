#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace

import numpy as np

from p3.aoSystem.FourierUtils import (
    create_wavelength_vector,
    cropSupport,
    getStaticOTF,
    interpolateSupport,
    normalizeImage,
    pistonFilter,
    psd2cov,
)


class TestFourierUtils(unittest.TestCase):
    def test_crop_support_returns_centered_view(self):
        image = np.arange(25, dtype=np.float32).reshape(5, 5)

        out = cropSupport(image, 5 / 3)

        self.assertEqual(out.shape, (3, 3))
        self.assertEqual(out.dtype, image.dtype)
        self.assertTrue(np.array_equal(out, image[1:4, 1:4]))

    def test_create_wavelength_vector_multiple_sources(self):
        ao = SimpleNamespace(
            cam=SimpleNamespace(nWvl=3, bandwidth=0.2),
            src=SimpleNamespace(wvl=np.array([1.0, 2.0])),
        )

        wvl, nwvl = create_wavelength_vector(ao, dtype=np.float64)

        expected = np.array([0.9, 1.0, 1.1, 1.9, 2.0, 2.1])
        self.assertEqual(nwvl, 6)
        self.assertTrue(np.allclose(wvl, expected))

    def test_interpolate_support_nearest_respects_target_shape(self):
        image = np.arange(12, dtype=np.float32).reshape(3, 4)

        out = interpolateSupport(image, (6, 2), kind='nearest')

        self.assertEqual(out.shape, (6, 2))
        self.assertEqual(out.dtype, image.dtype)

    def test_interpolate_support_complex_nearest_preserves_dtype(self):
        image = (
            np.arange(9, dtype=np.float32).reshape(3, 3)
            + 1j * np.eye(3, dtype=np.float32)
        ).astype(np.complex64)

        out = interpolateSupport(image, (5, 4), kind='nearest')

        self.assertEqual(out.shape, (5, 4))
        self.assertEqual(out.dtype, image.dtype)
        self.assertTrue(np.iscomplexobj(out))

    def test_get_static_otf_does_not_mutate_tel_opd_map(self):
        opd_map = np.arange(9, dtype=np.float64).reshape(3, 3)
        tel = SimpleNamespace(
            pupil=np.ones((3, 3), dtype=np.float64),
            apodizer=np.ones((3, 3), dtype=np.float64),
            opdMap_on=opd_map.copy(),
            statModes=None,
        )

        getStaticOTF(tel, nOtf=3, samp=1, wvl=1.0, theta_ext=15, dtype=np.float64)

        self.assertTrue(np.array_equal(tel.opdMap_on, opd_map))

    def test_normalize_image_minmax_roundtrip(self):
        image = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64)

        normalized, param = normalizeImage(image, normType=2)
        restored = normalizeImage(normalized, normType=2, param=param)

        self.assertTrue(np.allclose(normalized.min(), 0.0))
        self.assertTrue(np.allclose(normalized.max(), 1.0))
        self.assertTrue(np.allclose(restored, image))

    def test_normalize_image_constant_input_gives_zeroed_minmax(self):
        image = np.full((3, 3), 5.0, dtype=np.float64)

        normalized, param = normalizeImage(image, normType=2)
        restored = normalizeImage(normalized, normType=2, param=param)

        self.assertTrue(np.array_equal(normalized, np.zeros_like(image)))
        self.assertTrue(np.allclose(restored, image))

    def test_piston_filter_does_not_modify_input(self):
        freq = np.array([0.0, 0.1, 0.2], dtype=np.float32)
        original = freq.copy()

        filt = pistonFilter(8.0, freq, dtype=np.float32)

        self.assertTrue(np.array_equal(freq, original))
        self.assertEqual(filt.shape, (3, 3))
        self.assertTrue(np.all(filt >= 0))

    def test_psd2cov_rejects_unsupported_dimensions(self):
        psd = np.ones((2, 2, 2), dtype=np.float32)

        with self.assertRaises(ValueError):
            psd2cov(psd, pixelScale=1.0)


if __name__ == '__main__':
    unittest.main()
