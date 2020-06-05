########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
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


def full_int(n, i):
    return np.full(n, i, dtype=cpu_int)


# Expose ctypes objects
c_double = ctypes.c_double
c_double_p = ctypes.POINTER(c_double)
c_int = ctypes.c_int
c_int_p = ctypes.POINTER(c_int)
py_object = ctypes.py_object

# Functions to deal with ctypes and numpy value types


def as_c_double(*args):
    '''
    Transform arguments to a :mod:`ctypes` "double".
    '''
    if len(args) == 1:
        return c_double(*args)
    else:
        return tuple(c_double(a) for a in args)


def as_double(*args):
    '''
    Transform arguments to a :mod:`numpy` "double".
    '''
    if len(args) == 1:
        return cpu_float(*args)
    else:
        return tuple(cpu_float(a) for a in args)


def data_as_c_double(*args):
    '''
    Transform arguments to a :mod:`ctypes` "double*".
    '''
    if len(args) == 1:
        return args[0].ctypes.data_as(c_double_p)
    else:
        return tuple(a.ctypes.data_as(c_double_p) for a in args)


def as_c_integer(*args):
    '''
    Transform arguments to a :mode:`ctypes` "int".
    '''
    if len(args) == 1:
        return c_int(*args)
    else:
        return tuple(c_int(a) for a in args)


def as_integer(*args):
    '''
    Transform arguments to a :mod:`numpy` "integral" type.
    '''
    if len(args) == 1:
        return cpu_int(*args)
    else:
        return tuple(cpu_int(a) for a in args)


def data_as_c_int(*args):
    '''
    Transform arguments to a :mod:`ctypes` "int*".
    '''
    if len(args) == 1:
        return args[0].ctypes.data_as(c_int_p)
    else:
        return tuple(a.ctypes.data_as(c_int_p) for a in args)


def as_py_object(*args):
    '''
    Transform arguments to a :mod:`ctypes` "PyObject".
    '''
    if len(args) == 1:
        return py_object(*args)
    else:
        return tuple(py_object(a) for a in args)
