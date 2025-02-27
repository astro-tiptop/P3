import numpy as np
import os
import warnings

gpuEnabled = False
cp = None
fft = None
spc = None
interp = None
scnd = None
rotate = None
nnp = np
RectBivariateSpline = None

systemDisable = os.environ.get('P3_DISABLE_GPU', 'FALSE')
if systemDisable=='FALSE':
    try:
        import cupy as cp
        print("Cupy import successfull. Installed version is:", cp.__version__)
        gpuEnabled = True
        import cupy.fft as fftI
        import cupyx.scipy.special as spcI
        from  scipy.interpolate import RectBivariateSpline as RectBivariateSplineI
        import cupyx.scipy.ndimage as scndI
        from cupyx.scipy.ndimage import rotate as rotateI
        np = cp
    except:
        print("Cupy import failed. P3 will fall back to CPU use.")
        cp = np
        import numpy.fft as fftI
        import scipy.special as spcI
        from scipy.interpolate import RectBivariateSpline as RectBivariateSplineI
        import scipy.ndimage as scndI
        from scipy.ndimage import rotate as rotateI

else:
    print("env variable P3_DISABLE_GPU prevents using the GPU.")
    cp = np
    import numpy.fft as fftI
    import scipy.special as spcI
    from scipy.interpolate import RectBivariateSpline as RectBivariateSplineI
    import scipy.ndimage as scndI
    from scipy.ndimage import rotate as rotateI

fft = fftI
spc = spcI
RectBivariateSpline = RectBivariateSplineI
scnd = scndI
rotate = rotateI

def cpuArray(v):
    if isinstance(v,nnp.ndarray) or isinstance(v, list):
        return v
    else:
        return v.get()
