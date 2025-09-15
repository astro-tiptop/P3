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
        trapz = cp.trapz
    except:
        print("Cupy import failed. P3 will fall back to CPU use.")
        cp = np
        import numpy.fft as fftI
        import scipy.special as spcI
        from scipy.interpolate import RectBivariateSpline as RectBivariateSplineI
        import scipy.ndimage as scndI
        from scipy.ndimage import rotate as rotateI
        try:
            trapz = np.trapezoid
        except AttributeError:
            trapz = np.trapz
else:
    print("env variable P3_DISABLE_GPU prevents using the GPU.")
    cp = np
    import numpy.fft as fftI
    import scipy.special as spcI
    from scipy.interpolate import RectBivariateSpline as RectBivariateSplineI
    import scipy.ndimage as scndI
    from scipy.ndimage import rotate as rotateI
    try:
        trapz = np.trapezoid
    except AttributeError:
        trapz = np.trapz

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

def resolve_config_path(path_value, path_root, path_p3, path_tiptop=None):
    """
    Resolve configuration file paths for both P3 and TIPTOP
    - path_root has priority if it is not empty.
    - aoSystem/... => resolved under path_p3
    - tiptop/...   => resolved under path_tiptop (if available)
    - otherwise: returns as is (absolute or current relative)
    """
    if not path_value or path_value == '':
        return ''
   
    # Explicit path_root has priority
    if path_root:
        return os.path.join(path_root, path_value)
   
    # Clean path for consistent checking (remove leading slash)
    clean_path = path_value.lstrip('/')
   
    # P3 relative paths
    if clean_path.startswith('aoSystem'):
        return os.path.join(path_p3, clean_path)
   
    # TIPTOP relative paths
    if path_tiptop and clean_path.startswith('tiptop'):
        return os.path.join(path_tiptop, clean_path)
   
    # Default: use as-is (could be absolute or relative to current dir)
    return path_value

def detect_tiptop_path():
    """Auto-detect TIPTOP path from the call stack"""
    import inspect
    from pathlib import Path
    try:
        # context=0 avoids collecting source lines; faster and lighter
        for frame_info in inspect.stack(context=0):
            p = Path(frame_info.filename).resolve()
            # Walk the file's directory and its parents, looking for a folder named "tiptop"
            for parent in (p, *p.parents):
                if parent.name == 'tiptop':
                    return str(parent.parent)  # Repository root = parent of the "tiptop" directory
    except Exception:
        pass
    return None

# Try to auto-detect TIPTOP path
PATH_TIPTOP = detect_tiptop_path()