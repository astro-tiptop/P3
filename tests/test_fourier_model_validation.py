#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from p3.aoSystem.fourierModel import fourierModel


class TestFourierModelValidation(unittest.TestCase):
    @staticmethod
    def _base_ao(fov_in_pix=8):
        return SimpleNamespace(
            error=False,
            dtype=np.float64,
            complex_dtype=np.complex128,
            my_data_map={},
            cam=SimpleNamespace(fovInPix=fov_in_pix),
            lgs=None,
            ngs=SimpleNamespace(nSrc=1, wvl=np.array([1.0e-6])),
            dms=SimpleNamespace(nRecLayers=None),
            atm=SimpleNamespace(
                wvl=500e-9,
                r0=0.15,
                weights=np.array([1.0]),
                heights=np.array([0.0]),
                wSpeed=np.array([10.0]),
                wDir=np.array([0.0]),
                L0=25.0,
            ),
        )

    def test_init_computations_raises_when_notf_smaller_than_resao(self):
        ao = self._base_ao(fov_in_pix=8)
        fake_freq = SimpleNamespace(nOtf=16, resAO=32, kRef_=2)

        with patch(
            "p3.aoSystem.fourierModel.FourierUtils.create_wavelength_vector",
            return_value=(np.array([1.65e-6]), 1),
        ), patch("p3.aoSystem.fourierModel.frequencyDomain", return_value=fake_freq):
            with self.assertRaisesRegex(
                ValueError,
                "Invalid PSF size configuration after frequencyDomain init",
            ) as ctx:
                fourierModel(path_ini="unused.ini", ao=ao, doComputations=True, calcPSF=False)

        msg = str(ctx.exception)
        self.assertIn("nOtf=16", msg)
        self.assertIn("resAO=32", msg)
        self.assertIn("fovInPix=8", msg)

    def test_init_computations_raises_when_fov_too_small_for_ao_area(self):
        ao = self._base_ao(fov_in_pix=8)
        fake_freq = SimpleNamespace(
            nOtf=32,
            resAO=32,
            kRef_=1,
            psInMas=np.array([1.0]),
            kcInMas=np.array([20.0]),
            wvlRef=1.65e-6,
        )

        fake_refraction = SimpleNamespace(get_refractive_index=lambda _wvl: 1.0)

        with patch(
            "p3.aoSystem.fourierModel.FourierUtils.create_wavelength_vector",
            return_value=(np.array([1.65e-6]), 1),
        ), patch("p3.aoSystem.fourierModel.frequencyDomain", return_value=fake_freq), patch(
            "p3.aoSystem.fourierModel.MatharAirRefraction", return_value=fake_refraction
        ), patch(
            "p3.aoSystem.fourierModel.atmosphere", return_value=SimpleNamespace(wvl=None)
        ):
            with self.assertRaisesRegex(
                ValueError,
                "PSF field of view is too small to simulate the AO correction area",
            ) as ctx:
                fourierModel(path_ini="unused.ini", ao=ao, doComputations=True, calcPSF=False)

        msg = str(ctx.exception)
        self.assertIn("nOtf=32", msg)
        self.assertIn("resAO=32", msg)
        self.assertIn("fovInPix=8", msg)
