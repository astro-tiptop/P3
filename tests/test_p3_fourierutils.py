#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import warnings

import numpy as nnp

from p3.aoSystem.FourierUtils import pistonFilter


class TestPistonFilterApiHardening(unittest.TestCase):
    def test_warns_when_shift_args_are_ignored_for_multidimensional_input(self):
        f2d = nnp.ones((4, 4), dtype=nnp.float64)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = pistonFilter(8.0, f2d, fm=1.0, fn=-0.5)

        messages = [str(w.message) for w in caught]
        self.assertTrue(any("ignored" in m for m in messages))

    def test_no_warning_for_1d_input_with_shifts(self):
        f1d = nnp.linspace(0.0, 1.0, 8, dtype=nnp.float64)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = pistonFilter(8.0, f1d, fm=1.0, fn=-0.5)

        self.assertEqual(len(caught), 0)
        self.assertEqual(out.shape, (8, 8))


if __name__ == '__main__':
    unittest.main()
