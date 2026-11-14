"""Runtime device checks shared by the Python backend tests."""
import ctypes
from functools import lru_cache
import sys


@lru_cache(maxsize=1)
def mps_available():
    if sys.platform != 'darwin':
        return False
    metal = ctypes.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    return bool(metal.MTLCreateSystemDefaultDevice())
