"""GPU utilities: detect CuPy and provide an array-module alias `xp`.

Usage:
  from gpu_utils import xp, has_gpu, to_cpu, array

`xp` will be `cupy` only when CuPy can actually reach a CUDA device.
Otherwise it falls back to `numpy`.
"""
import numpy as _np


def _init_backend():
    try:
        import cupy as _cupy
    except Exception:
        return _np, False

    try:
        # This catches driver/runtime problems such as cudaErrorInsufficientDriver.
        if _cupy.cuda.runtime.getDeviceCount() <= 0:
            return _np, False
    except Exception:
        return _np, False

    return _cupy, True


xp, has_gpu = _init_backend()


def to_cpu(arr):
    """Return a NumPy array on host regardless of backend.
    If `arr` is a CuPy array, copy to host.
    """
    if has_gpu:
        try:
            return xp.asnumpy(arr)
        except Exception:
            return _np.array(arr)
    else:
        return _np.array(arr)


def array(x, dtype=None):
    """Create array using selected backend."""
    if has_gpu:
        return xp.array(x, dtype=dtype)
    return _np.array(x, dtype=dtype)
