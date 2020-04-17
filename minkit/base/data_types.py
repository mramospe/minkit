'''
Define the data_types.for the variables to use in the package.
'''
import ctypes
import numpy as np

__all__ = []

# Types for numpy.ndarray objects
cpu_float = np.float64  # double
cpu_complex = np.complex128  # complex double
cpu_int = np.int32  # int
# unsigned integer (char does not seem to be allowed in CUDA), and int16 is too small (must not be equal to cpu_int)
cpu_bool = np.uint32
cpu_real_bool = np.bool  # bool (not allowed in PyOpenCL)


def array_float(*args, **kwargs):
    return np.array(*args, dtype=cpu_float, **kwargs)


def array_int(*args, **kwargs):
    return np.array(*args, dtype=cpu_int, **kwargs)


def empty_float(*args, **kwargs):
    return np.empty(*args, dtype=cpu_float, **kwargs)


def empty_int(*args, **kwargs):
    return np.empty(*args, dtype=cpu_int, **kwargs)


def fromiter_float(i):
    return np.fromiter(i, dtype=cpu_float)


def fromiter_int(i):
    return np.fromiter(i, dtype=cpu_int)


# Types to handle with ctypes
c_double = ctypes.c_double  # double
c_double_p = ctypes.POINTER(c_double)  # double*
c_int = ctypes.c_int  # int
c_int_p = ctypes.POINTER(c_int)  # int*
