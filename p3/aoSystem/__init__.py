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
    """
    Convert GPU arrays to CPU arrays, or return as-is for CPU arrays and scalars.
    
    Parameters:
    -----------
    v : array-like, scalar, or list
        Input value to convert
        
    Returns:
    --------
    CPU-compatible array or scalar
    """
    if nnp.isscalar(v) or isinstance(v,nnp.ndarray) or isinstance(v, list):
        return v
    else:
        return v.get()


def asnumpy(v):
    """Return a NumPy array/scalar from either NumPy or CuPy-backed inputs."""
    if isinstance(v, list):
        return [asnumpy(item) for item in v]
    if isinstance(v, tuple):
        return tuple(asnumpy(item) for item in v)
    if nnp.isscalar(v) or isinstance(v, nnp.ndarray) or isinstance(v, nnp.generic):
        return nnp.asarray(v)
    if gpuEnabled and cp is not None and hasattr(cp, 'ndarray') and isinstance(v, cp.ndarray):
        return cp.asnumpy(v)
    if hasattr(v, 'get'):
        return nnp.asarray(v.get())
    return nnp.asarray(v)

def resolve_config_path(path_value, path_root, path_p3):
    """
    Resolve configuration file paths.
    - path_root has priority if it is not empty (used e.g. for tiptop/... paths).
    - aoSystem/... => resolved under path_p3
    - otherwise: returns as is (absolute or current relative)
    - if all else fails and path starts with '/', try without the leading slash
    """
    if not path_value or path_value == '':
        return ''

    # Explicit path_root has priority
    if path_root:
        candidate = os.path.join(path_root, path_value)
        if os.path.isfile(candidate):
            return candidate

    # Clean path for consistent checking (remove leading slash)
    clean_path = path_value.lstrip('/')

    # P3 relative paths
    if clean_path.startswith('aoSystem'):
        return os.path.join(path_p3, clean_path)

    # Try as-is first
    if os.path.isfile(path_value):
        return path_value

    # Last resort: if nothing worked and path starts with '/', try with path_p3
    if path_value.startswith('/') and clean_path != path_value:
        candidate = os.path.join(path_p3, clean_path)
        if os.path.isfile(candidate):
            return candidate

    # Default: use as-is (could be absolute or relative to current dir)
    return path_value