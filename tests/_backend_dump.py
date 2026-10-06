"""Helper for test_p3_bugfixes_noise_tomo.py: dump tomographic outputs to .npz.

Run as ``python _backend_dump.py <ini> <out.npz>``; the backend is selected
by the P3_DISABLE_GPU env var at import time.
"""
import pathlib
import sys

import numpy as nnp

import p3.aoSystem as aoSystemMain
from p3.aoSystem import cpuArray
from p3.aoSystem.fourierModel import fourierModel

ini, out = sys.argv[1], sys.argv[2]
root = str(pathlib.Path(aoSystemMain.__file__).parent.parent.absolute())
fao = fourierModel(ini, path_root=root, calcPSF=False, verbose=False,
                   display=False, reduce_memory=False,
                   computeFocalAnisoCov=False)
nnp.savez(out, gpu=bool(aoSystemMain.gpuEnabled),
          Popt=cpuArray(fao.Popt), Wtomo=cpuArray(fao.Wtomo),
          W=cpuArray(fao.W), PSD=cpuArray(fao.PSD))
