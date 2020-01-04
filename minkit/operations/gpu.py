'''
Operations with GPU objects
'''
from . import PACKAGE_PATH
from . import docstrings
from . import types
from .gpu_core import get_sizes, THREAD

from reikna.cluda import functions
from reikna.fft import FFT
import functools
import numpy as np
import os

# Save the FFT compiled objects
FFT_CACHE = {}


def return_dtype(dtype):
    '''
    Wrap a function automatically creating an output array with the
    same shape as that of the input but with possible different data type.
    '''
    def __wrapper(function):
        '''
        Internal wrapper.
        '''
        @functools.wraps(function)
        def __wrapper(arr, *args, **kwargs):
            '''
            Internal wrapper.
            '''
            out = THREAD.array(arr.shape, dtype=dtype)
            gs, ls = get_sizes(len(arr))
            function(out, arr, *args, global_size=gs, local_size=ls, **kwargs)
            return out
        return __wrapper
    return __wrapper


RETURN_COMPLEX = return_dtype(types.cpu_complex)
RETURN_DOUBLE = return_dtype(types.cpu_type)
RETURN_BOOL = return_dtype(types.cpu_bool)


# Compile general GPU functions.
with open(os.path.join(PACKAGE_PATH, 'src/functions_by_element.c')) as fi:
    FUNCS_BY_ELEMENT = THREAD.compile(fi.read())

# These functions take an array of doubles and return another array of doubles
for function in ('exponential_complex',):
    setattr(FUNCS_BY_ELEMENT, function, RETURN_COMPLEX(
        getattr(FUNCS_BY_ELEMENT, function)))

# These functions take an array of doubles and return another array of doubles
for function in ('exponential_double', 'logarithm', 'real'):
    setattr(FUNCS_BY_ELEMENT, function, RETURN_DOUBLE(
        getattr(FUNCS_BY_ELEMENT, function)))

# These functions take an array of doubles as an input, and return an array of bool
for function in ('geq', 'leq', 'logical_and', 'logical_or'):
    setattr(FUNCS_BY_ELEMENT, function, RETURN_BOOL(
        getattr(FUNCS_BY_ELEMENT, function)))


def creating_array_dtype(dtype):
    '''
    Wrap a function automatically creating an output array with the
    same shape as that of the input but with possible different data type.
    '''
    def __wrapper(function):
        '''
        Internal wrapper.
        '''
        @functools.wraps(function)
        def __wrapper(size, *args, **kwargs):
            '''
            Internal wrapper.
            '''
            out = THREAD.array((size,), dtype=dtype)
            gs, ls = get_sizes(size)
            function(out, *args, global_size=gs, local_size=ls, **kwargs)
            return out
        return __wrapper
    return __wrapper


CREATE_COMPLEX = creating_array_dtype(types.cpu_complex)
CREATE_DOUBLE = creating_array_dtype(types.cpu_type)
CREATE_INT = creating_array_dtype(types.cpu_int)
CREATE_BOOL = creating_array_dtype(types.cpu_bool)

# These functions create e new array of complex numbers
for function in ('arange_complex',):
    setattr(FUNCS_BY_ELEMENT, function, CREATE_COMPLEX(
        getattr(FUNCS_BY_ELEMENT, function)))

# These functions create e new array of doubles
for function in ('linspace', 'ones_double', 'zeros_double'):
    setattr(FUNCS_BY_ELEMENT, function, CREATE_DOUBLE(
        getattr(FUNCS_BY_ELEMENT, function)))

# These functions create e new array of integers
for function in ('arange_int',):
    setattr(FUNCS_BY_ELEMENT, function, CREATE_INT(
        getattr(FUNCS_BY_ELEMENT, function)))

# These functions create e new array of bool
for function in ('true_till', 'ones_bool', 'zeros_bool'):
    setattr(FUNCS_BY_ELEMENT, function, CREATE_BOOL(
        getattr(FUNCS_BY_ELEMENT, function)))


def reikna_fft(a, inverse=False):
    '''
    Get the FFT to calculate the FFT of an array, keeping the compiled
    source in a cache.
    '''
    global FFT_CACHE

    # Compile the FFT
    cf = FFT_CACHE.get(a.shape, None)
    if cf is None:
        f = FFT(a)
        cf = f.compile(THREAD)
        FFT_CACHE[a.shape] = cf

    # Calculate the value
    output = THREAD.array(a.shape, dtype=types.cpu_complex)

    cf(output, a, inverse=inverse)

    return output


@docstrings.set_docstring
def arange(n, dtype=types.cpu_int):
    if dtype == types.cpu_int:
        return FUNCS_BY_ELEMENT.arange_int(n, 0)
    elif dtype == types.cpu_complex:
        return FUNCS_BY_ELEMENT.arange_complex(n, types.cpu_complex(0. + 0.j))
    else:
        raise NotImplementedError(
            f'Function not implemented for data type "{dtype}"')


@docstrings.set_docstring
def array(a, copy=True, convert=True):
    if convert:
        return THREAD.to_device(a)
    # Is assumed to be a GPU-array
    if copy:
        return a.copy()
    else:
        return a


@docstrings.set_docstring
def concatenate(*arrays):
    return THREAD.to_device(np.concatenate(tuple(a.get() for a in arrays)))


@docstrings.set_docstring
def exp(a):
    if a.dtype == types.cpu_complex:
        return FUNCS_BY_ELEMENT.exponential_complex(a)
    elif a.dtype == types.cpu_type:
        return FUNCS_BY_ELEMENT.exponential_double(a)
    else:
        raise NotImplementedError(
            f'Function not implemented for data type "{a.dtype}"')


@docstrings.set_docstring
def extract_ndarray(a):
    return a.get()


@docstrings.set_docstring
def fft(a):
    return reikna_fft(a.astype(types.cpu_complex))


@docstrings.set_docstring
def fftconvolve(a, b, data):

    fa = fft(a)
    fb = fft(b)

    shift = fftshift(data)

    output = ifft(fa * shift * fb)

    return output * (data[1].get() - data[0].get())


@docstrings.set_docstring
def fftshift(a):
    n0 = sum(a < 0)
    nt = len(a)
    com = types.cpu_complex(+2.j * np.pi * n0 / nt)
    rng = arange(nt, dtype=types.cpu_complex)
    return exp(com * rng)


@docstrings.set_docstring
def geq(a, v):
    return FUNCS_BY_ELEMENT.geq(a, types.cpu_type(v))


@docstrings.set_docstring
def ifft(a):
    return reikna_fft(a, inverse=True)


@docstrings.set_docstring
def interpolate_linear(x, xp, yp):
    return THREAD.to_device(np.interp(x.get(), xp.get(), yp.get()))


@docstrings.set_docstring
def leq(a, v):
    return FUNCS_BY_ELEMENT.leq(a, types.cpu_type(v))


@docstrings.set_docstring
def linspace(vmin, vmax, size):
    return FUNCS_BY_ELEMENT.linspace(size,
                                     types.cpu_type(vmin),
                                     types.cpu_type(vmax),
                                     types.cpu_int(size))


@docstrings.set_docstring
def log(a):
    return FUNCS_BY_ELEMENT.logarithm(a)


@docstrings.set_docstring
def logical_and(a, b):
    return FUNCS_BY_ELEMENT.logical_and(a, b)


@docstrings.set_docstring
def logical_or(a, b):
    return FUNCS_BY_ELEMENT.logical_or(a, b)


@docstrings.set_docstring
def max(a):
    return np.max(a.get())


@docstrings.set_docstring
def meshgrid(*arrays):
    a = map(np.ndarray.flatten, np.meshgrid(*tuple(a.get() for a in arrays)))
    return tuple(map(THREAD.to_device, a))


@docstrings.set_docstring
def min(a):
    return np.min(a.get())


@docstrings.set_docstring
def ones(n, dtype=types.cpu_type):
    if dtype == types.cpu_type:
        return FUNCS_BY_ELEMENT.ones_double(n)
    elif dtype == types.cpu_bool:
        return FUNCS_BY_ELEMENT.ones_bool(n)
    else:
        raise NotImplementedError(
            f'Function not implemented for data type "{dtype}"')


@docstrings.set_docstring
def random_uniform(vmin, vmax, size):
    return THREAD.to_device(np.random.uniform(vmin, vmax, size))


@docstrings.set_docstring
def real(a):
    return FUNCS_BY_ELEMENT.real(a)


@docstrings.set_docstring
def shuffling_index(n):
    indices = np.arange(n)
    np.random.shuffle(indices)
    return THREAD.to_device(indices)


@docstrings.set_docstring
def sum(a):
    return np.sum(a.get())


@docstrings.set_docstring
def slice_from_boolean(a, valid):
    return THREAD.to_device(a.get()[valid.astype(types.cpu_bool).get()])


@docstrings.set_docstring
def slice_from_integer(a, indices):
    return THREAD.to_device(a.get()[indices.astype(types.cpu_int).get()])


@docstrings.set_docstring
def true_till(N, n):
    return FUNCS_BY_ELEMENT.true_till(N, types.cpu_int(n))


@docstrings.set_docstring
def zeros(n, dtype=types.cpu_type):
    if dtype == types.cpu_type:
        return FUNCS_BY_ELEMENT.zeros_double(n)
    elif dtype == types.cpu_bool:
        return FUNCS_BY_ELEMENT.zeros_bool(n)
    else:
        raise NotImplementedError(
            f'Function not implemented for data type "{dtype}"')
