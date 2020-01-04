'''
Operations with numpy objects
'''
from . import docstrings
from . import types

import numpy as np


@docstrings.set_docstring
def arange(*args, **kwargs):
    return np.arange(*args, **kwargs)


@docstrings.set_docstring
def array(a, copy=True, convert=True):
    if copy:
        return np.array(a, dtype=types.cpu_type)
    else:
        return a


@docstrings.set_docstring
def concatenate(*arrays):
    return np.concatenate(arrays)


@docstrings.set_docstring
def exp(a):
    return np.exp(a)


@docstrings.set_docstring
def extract_ndarray(a):
    return a


@docstrings.set_docstring
def fft(a):
    return np.fft.fft(a)


@docstrings.set_docstring
def fftconvolve(a, b, data):

    fa = fft(a)
    fb = fft(b)

    shift = fftshift(data)

    output = ifft(fa * shift * fb)

    return output * (data[1] - data[0])


@docstrings.set_docstring
def fftshift(a):
    n0 = sum(a < 0)
    nt = len(a)
    com = types.cpu_complex(+2.j * np.pi * n0 / nt)
    rng = arange(nt, dtype=types.cpu_int).astype(types.cpu_complex)
    return exp(com * rng)


@docstrings.set_docstring
def geq(a, v):
    return a >= v


@docstrings.set_docstring
def ifft(a):
    return np.fft.ifft(a)


@docstrings.set_docstring
def interpolate_linear(x, xp, yp):
    return np.interp(x, xp, yp)


@docstrings.set_docstring
def leq(a, v):
    return a <= v


@docstrings.set_docstring
def linspace(vmin, vmax, size):
    return np.linspace(vmin, vmax, size, dtype=types.cpu_type)


@docstrings.set_docstring
def log(a):
    return np.log(a)


@docstrings.set_docstring
def logical_and(a, b):
    return np.logical_and(a, b)


@docstrings.set_docstring
def logical_or(a, b):
    return np.logical_or(a, b)


@docstrings.set_docstring
def max(a):
    return np.max(a)


@docstrings.set_docstring
def meshgrid(*arrays):
    return tuple(map(np.ndarray.flatten, np.meshgrid(*arrays)))


@docstrings.set_docstring
def min(a):
    return np.min(a)


@docstrings.set_docstring
def ones(n, dtype=types.cpu_type):
    return np.ones(n, dtype=dtype)


@docstrings.set_docstring
def random_uniform(vmin, vmax, size):
    return np.random.uniform(vmin, vmax, size)


@docstrings.set_docstring
def real(a):
    return a.real


@docstrings.set_docstring
def shuffling_index(n):
    indices = np.arange(n)
    np.random.shuffle(indices)
    return indices


@docstrings.set_docstring
def sum(a):
    return np.sum(a)


@docstrings.set_docstring
def slice_from_boolean(a, valid):
    return a[valid]


@docstrings.set_docstring
def slice_from_integer(a, indices):
    return a[indices]


@docstrings.set_docstring
def true_till(N, n):
    a = np.ones(N, dtype=types.cpu_bool)
    a[n:] = False
    return a


@docstrings.set_docstring
def zeros(n, dtype=types.cpu_type):
    return np.zeros(n, dtype=dtype)
