'''
Define the types for the variables to use in the package.
'''
import ctypes
import numpy as np

__all__ = []

# Types for numpy.ndarray objects
cpu_type = np.float64  # double
cpu_complex = np.complex128  # complex double
cpu_int = np.int32  # int
# unsigned integer (char does not seem to be allowed in CUDA), and int16 is too small (must not be equal to cpu_int)
cpu_bool = np.uint32
cpu_real_bool = np.bool  # bool (not allowed in PyOpenCL)

# Types to handle with ctypes
c_int = ctypes.c_int  # int
c_double = ctypes.c_double  # double
c_double_p = ctypes.POINTER(c_double)  # double*
