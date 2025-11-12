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
    """
    Auto-detect the TIPTOP project root path.

    It first tries to find the path via the standard installed 'tiptop' package.
    If the package is not found (e.g., in a development environment), it falls
    back to inspecting the call stack to locate the project repository.

    Returns:
        The absolute path to the project root as a string, or None if not found.
    """
    from pathlib import Path
    # --- Method 1: Standard package inspection (preferred, fast, and reliable) ---
    try:
        import tiptop
        # The project root is assumed to be the parent of the 'tiptop' package directory.
        # e.g., from /path/to/project/tiptop/__init__.py -> get /path/to/project
        project_root = Path(tiptop.__file__).resolve().parent.parent
        return str(project_root)
   
    except ImportError:
        import inspect
        # --- Method 2: Fallback via call stack inspection ---
        # This is useful when running from a source checkout without installation.
        try:
            # context=0 avoids collecting source lines; it's faster and lighter.
            for frame_info in inspect.stack(context=0):
                p = Path(frame_info.filename).resolve()
               
                # Walk the file's directory and its parents, looking for a folder named "tiptop"
                for parent in (p, *p.parents):
                    if parent.name == 'tiptop':
                        # The repository root is the parent of the "tiptop" directory
                        return str(parent.parent)
                       
        except Exception:
            # If stack inspection fails for any reason, pass silently and return None later.
            pass

    # Return None if neither method succeeded
    return None

# Try to auto-detect TIPTOP path
PATH_TIPTOP = detect_tiptop_path()