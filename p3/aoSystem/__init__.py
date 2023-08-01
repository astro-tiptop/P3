import numpy as np
import os

gpuEnabled = False
cp = None

systemDisable = os.environ.get('P3_DISABLE_GPU', 'FALSE')
if systemDisable=='FALSE':
    try:
        import cupy as cp
        print("Cupy import successfull. Installed version is:", cp.__version__)
        gpuEnabled = True
    except:
        print("Cupy import failed. P3 will fall back to CPU use.")
        cp = np
else:
    print("env variable P3_DISABLE_GPU prevents using the GPU.")
    cp = np
