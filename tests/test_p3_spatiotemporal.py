#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import time
import unittest

import numpy as nnp

import p3.aoSystem as aoSystemMain
from p3.aoSystem import asnumpy
from p3.aoSystem.fourierModel import fourierModel


def _p3_path():
    return str(pathlib.Path(aoSystemMain.__file__).parent.parent.parent.absolute())


def _spatiotemporal_reference(fao):
    """Reference implementation matching the pre-refactor loop structure."""
    nK = fao.freq.resAO
    real_dtype = nnp.dtype(fao.dtype)
    complex_dtype = nnp.dtype(fao.complex_dtype)
    psd = nnp.zeros((nK, nK, fao.ao.src.nSrc), dtype=real_dtype)
    i = fao.complex_dtype(1j)
    nH = fao.ao.atm.nL
    Hs = asnumpy(fao.ao.atm.heights) * asnumpy(fao.strechFactor)
    Ws = asnumpy(fao.ao.atm.weights)
    deltaT = fao.ao.rtc.holoop['delay'] / fao.ao.rtc.holoop['rate']
    wDir = asnumpy(fao.ao.atm.wDir)
    wSpeed = asnumpy(fao.ao.atm.wSpeed)
    wDir_x = nnp.cos(wDir * nnp.pi / 180)
    wDir_y = nnp.sin(wDir * nnp.pi / 180)
    Watm = asnumpy(fao.Wphi * fao.freq.pistonFilterAO_)
    F = asnumpy(fao.Rx * fao.SxAv + fao.Ry * fao.SyAv)
    kx = asnumpy(fao.freq.kxAO_)
    ky = asnumpy(fao.freq.kyAO_)
    msk = asnumpy(fao.freq.mskInAO_)
    piston = asnumpy(fao.freq.pistonFilterAO_)
    h1 = asnumpy(fao.h1)
    h2 = asnumpy(fao.h2)
    for s in range(fao.ao.src.nSrc):
        if fao.nGs < 2:
            th = asnumpy(fao.ao.src.direction[:, s] - fao.gs.direction[:, 0])
            if nnp.any(nnp.asarray(th)):
                A = nnp.zeros((nK, nK), dtype=complex_dtype)
                for l in range(fao.ao.atm.nL):
                    A = A + Ws[l] * nnp.exp(
                        2 * i * nnp.pi * Hs[l] * (kx * th[1] + ky * th[0])
                    )
            else:
                A = nnp.ones((fao.freq.resAO, fao.freq.resAO), dtype=complex_dtype)

            if fao.ao.rtc.holoop['gain'] == 0:
                psd[:, :, s] = abs(1 - F) ** 2 * Watm
            else:
                psd[:, :, s] = (
                    msk
                    * (1 + abs(F) ** 2 * h2 - 2 * nnp.real(F * h1 * A))
                    * Watm
                )
        else:
            pbeta_dm = asnumpy(fao.PbetaDM[s])
            walpha = asnumpy(fao.Walpha)
            cphi = asnumpy(fao.Cphi)
            beta = asnumpy(fao.ao.src.direction[:, s])
            PbetaL = nnp.zeros([nK, nK, 1, nH], dtype=complex_dtype)
            fx = beta[0] * kx
            fy = beta[1] * ky
            for j in range(nH):
                freq_t = wDir_x[j] * kx + wDir_y[j] * ky
                delta_h = Hs[j] * (fx + fy) - deltaT * wSpeed[j] * freq_t
                PbetaL[:, :, 0, j] = nnp.exp(i * 2 * nnp.pi * delta_h)

            proj = PbetaL - nnp.matmul(pbeta_dm, walpha)
            proj_t = nnp.conj(proj.transpose(0, 1, 3, 2))
            tmp = nnp.matmul(proj, nnp.matmul(cphi, proj_t)).real
            psd[:, :, s] = msk * tmp[:, :, 0, 0] * piston

    return psd


class TestSpatioTemporalPSD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_p3 = _p3_path()
        cls.scao_ini = cls.path_p3 + '/tests/scao_test_wvl1100nm.ini'
        cls.ltao_ini = cls.path_p3 + '/tests/MAVIStest.ini'

    def _build_model(self, ini_path):
        return fourierModel(
            ini_path,
            path_root=self.path_p3,
            calcPSF=False,
            verbose=False,
            display=False,
            reduce_memory=False,
        )

    def _assert_close(self, a, b, rtol=1e-7, atol=1e-9):
        aa = nnp.asarray(asnumpy(a), dtype=nnp.float64)
        bb = nnp.asarray(asnumpy(b), dtype=nnp.float64)
        self.assertEqual(aa.shape, bb.shape)
        self.assertTrue(nnp.all(nnp.isfinite(aa)))
        self.assertTrue(nnp.all(nnp.isfinite(bb)))
        self.assertTrue(nnp.allclose(aa, bb, rtol=rtol, atol=atol))

    def test_spatiotemporal_scao_matches_reference(self):
        fao = self._build_model(self.scao_ini)
        ref = _spatiotemporal_reference(fao)
        new = fao.spatioTemporalPSD()
        self._assert_close(ref, new)

    def test_spatiotemporal_tomographic_matches_reference(self):
        fao = self._build_model(self.ltao_ini)
        ref = _spatiotemporal_reference(fao)
        new = fao.spatioTemporalPSD()
        self._assert_close(ref, new)

    def test_spatiotemporal_speed_not_regressed(self):
        fao = self._build_model(self.ltao_ini)

        _ = _spatiotemporal_reference(fao)
        _ = fao.spatioTemporalPSD()

        ref_times = []
        new_times = []
        for _ in range(2):
            t0 = time.perf_counter()
            _ = _spatiotemporal_reference(fao)
            ref_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            _ = fao.spatioTemporalPSD()
            new_times.append(time.perf_counter() - t0)

        ref_avg = sum(ref_times) / len(ref_times)
        new_avg = sum(new_times) / len(new_times)

        # Allow some platform variance while guarding against major regression.
        self.assertLessEqual(new_avg, ref_avg * 1.5)


if __name__ == '__main__':
    unittest.main()
