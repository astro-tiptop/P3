#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import time
import unittest

import numpy as nnp

import p3.aoSystem as aoSystemMain
from p3.aoSystem import cpuArray
from p3.aoSystem.fourierModel import fourierModel


def _p3_path():
    return str(pathlib.Path(aoSystemMain.__file__).parent.parent.parent.absolute())


def _spatiotemporal_reference(fao):
    """Reference implementation matching the pre-refactor loop structure."""
    nK = fao.freq.resAO
    psd = nnp.zeros((nK, nK, fao.ao.src.nSrc), dtype=fao.dtype)
    i = fao.complex_dtype(1j)
    nH = fao.ao.atm.nL
    Hs = fao.ao.atm.heights * fao.strechFactor
    Ws = fao.ao.atm.weights
    deltaT = fao.ao.rtc.holoop['delay'] / fao.ao.rtc.holoop['rate']
    wDir_x = nnp.cos(fao.ao.atm.wDir * nnp.pi / 180)
    wDir_y = nnp.sin(fao.ao.atm.wDir * nnp.pi / 180)
    Watm = fao.Wphi * fao.freq.pistonFilterAO_
    F = fao.Rx * fao.SxAv + fao.Ry * fao.SyAv

    for s in range(fao.ao.src.nSrc):
        if fao.nGs < 2:
            th = fao.ao.src.direction[:, s] - fao.gs.direction[:, 0]
            if nnp.any(nnp.asarray(th)):
                A = nnp.zeros((nK, nK), dtype=fao.complex_dtype)
                for l in range(fao.ao.atm.nL):
                    A = A + Ws[l] * nnp.exp(
                        2 * i * nnp.pi * Hs[l] * (fao.freq.kxAO_ * th[1] + fao.freq.kyAO_ * th[0])
                    )
            else:
                A = nnp.ones((fao.freq.resAO, fao.freq.resAO), dtype=fao.complex_dtype)

            if fao.ao.rtc.holoop['gain'] == 0:
                psd[:, :, s] = abs(1 - F) ** 2 * Watm
            else:
                psd[:, :, s] = (
                    fao.freq.mskInAO_
                    * (1 + abs(F) ** 2 * fao.h2 - 2 * nnp.real(F * fao.h1 * A))
                    * Watm
                )
        else:
            beta = [fao.ao.src.direction[0, s], fao.ao.src.direction[1, s]]
            PbetaL = nnp.zeros([nK, nK, 1, nH], dtype=fao.complex_dtype)
            fx = beta[0] * fao.freq.kxAO_
            fy = beta[1] * fao.freq.kyAO_
            for j in range(nH):
                freq_t = wDir_x[j] * fao.freq.kxAO_ + wDir_y[j] * fao.freq.kyAO_
                delta_h = Hs[j] * (fx + fy) - deltaT * fao.ao.atm.wSpeed[j] * freq_t
                PbetaL[:, :, 0, j] = nnp.exp(i * 2 * nnp.pi * delta_h)

            proj = PbetaL - nnp.matmul(fao.PbetaDM[s], fao.Walpha)
            proj_t = nnp.conj(proj.transpose(0, 1, 3, 2))
            tmp = nnp.matmul(proj, nnp.matmul(fao.Cphi, proj_t)).real
            psd[:, :, s] = fao.freq.mskInAO_ * tmp[:, :, 0, 0] * fao.freq.pistonFilterAO_

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
        aa = nnp.asarray(cpuArray(a), dtype=nnp.float64)
        bb = nnp.asarray(cpuArray(b), dtype=nnp.float64)
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
