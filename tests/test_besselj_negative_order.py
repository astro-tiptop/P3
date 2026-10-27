#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as nnp
from scipy.special import jv

from p3.aoSystem import np, cpuArray
from p3.aoSystem.fourierModel import besselj__n


class TestBesselJNegativeOrder(unittest.TestCase):

    def test_orders_match_scipy_jv(self):
        """besselj__n(n, z) for n = -4..4 must match scipy.special.jv, i.e. J_{-n} = (-1)^n J_n."""
        z_host = nnp.linspace(0.5, 20.0, 400)  # z=0 excluded: the recurrence divides by z
        z = np.asarray(z_host)

        # Measured max abs error is 2.4e-14 (GPU) / 1.2e-14 (CPU), worst for |n|=4 at z=0.5
        # where the upward recurrence is least accurate (z < n); atol=1e-12 leaves ~40x margin.
        # Even orders (n = -2, -4) are the ones the old -1**(-n) bug got wrong; odd orders were right by accident.
        for n in [0, 1, 2, 3, 4, -1, -2, -3, -4]:
            with self.subTest(n=n):
                result = cpuArray(besselj__n(n, z))
                nnp.testing.assert_allclose(result, jv(n, z_host), rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
