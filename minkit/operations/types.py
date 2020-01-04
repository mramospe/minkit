'''
Define the types for the variables to use in the package.
'''
import ctypes
import numpy as np

__all__ = []

# Types for numpy.ndarray objects
cpu_type = np.float64
cpu_complex = np.complex128
cpu_int = np.int32
cpu_bool = np.bool

# Types to handle with ctypes
c_int = ctypes.c_int
c_double = ctypes.c_double
c_double_p = ctypes.POINTER(c_double)
