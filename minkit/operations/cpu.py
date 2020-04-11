'''
Operations with numpy objects
'''
from . import docstrings
from . import types

import numpy as np


@docstrings.set_docstring
def arange(n, dtype=types.cpu_int):
    if dtype == types.cpu_int:
        return np.arange(n, dtype=dtype)
    elif dtype == types.cpu_complex:
        return np.arange(n, dtype=dtype).astype(types.cpu_complex)
    else:
        raise NotImplementedError(
            f'Function not implemented for data type "{dtype}"')


@docstrings.set_docstring
def ale(a1, a2):
    return a1 < a2


@docstrings.set_docstring
def concatenate(arrays, maximum=None):
    if maximum is not None:
        return np.concatenate(arrays)[:maximum]
    else:
        return np.concatenate(arrays)


@docstrings.set_docstring
def count_nonzero(a):
    return np.count_nonzero(a)


@docstrings.set_docstring
def data_array(a, copy=True, convert=True, dtype=types.cpu_type):
    if copy:
        return np.array(a, dtype=dtype)
    else:
        if a.dtype != dtype:
            return a.astype(dtype)
        return a


@docstrings.set_docstring
def empty(size, dtype=types.cpu_type):
    return np.empty(size, dtype=dtype)


@docstrings.set_docstring
def exp(a):
    return np.exp(a)


@docstrings.set_docstring
def extract_ndarray(a):
    return a


@docstrings.set_docstring
def false_till(N, n):
    a = np.zeros(N, dtype=types.cpu_real_bool)
    a[n:] = True
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
def indices_to_mask(n, indices):
    a = np.zeros(n, dtype=types.cpu_real_bool)
    a[indices] = True
    return a


@docstrings.set_docstring
def interpolate_linear(idx, x, xp, yp):
    nd = len(idx)
    ln = len(x) // nd
    st = idx[0]
    return np.interp(x[st:st + ln], xp, yp)  # 1D case


@docstrings.set_docstring
def is_inside(data, lb, ub):
    if len(lb) > 1:
        ln = len(data) // len(lb)
        c = np.ones(ln, dtype=types.cpu_real_bool)
        for i, (l, u) in enumerate(zip(lb, ub)):
            d = data[i * ln:(i + 1) * ln]
            c = np.logical_and(c, np.logical_and(d >= l, d < u))
    else:
        c = np.logical_and(data >= lb, data < ub)
    return c


@docstrings.set_docstring
def keep_to_limit(maximum, ndim, len, data):
    return np.concatenate(tuple(data[i * len:i * len + maximum] for i in range(ndim)))


@docstrings.set_docstring
def le(a, v):
    return a < v


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
    if dtype == types.cpu_bool:
        # Hack due to lack of "bool" in PyOpenCL
        return np.ones(n, dtype=types.cpu_real_bool)
    else:
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
def sum(a, *args):
    if len(args) == 0:
        return np.sum(a)
    else:
        return np.sum((a, *args), axis=0)


@docstrings.set_docstring
def sum_inside(indices, gaps, centers, edges, values=None):

    nd = len(gaps)
    lc = len(centers) // nd

    c = [centers[i * lc:(i + 1) * lc] for i in range(nd)]
    e = [edges[p:n] for p, n in zip(indices[:-1], indices[1:])]

    out, _ = np.histogramdd(c, bins=e, weights=values)

    return out.T.flatten()


@docstrings.set_docstring
def slice_from_boolean(a, valid, dim=1):
    return a[np.tile(valid, dim)]


@docstrings.set_docstring
def slice_from_integer(a, indices, dim=1):
    return a[np.tile(indices, dim)]


@docstrings.set_docstring
def true_till(N, n):
    a = np.ones(N, dtype=types.cpu_real_bool)
    a[n:] = False
    return a


@docstrings.set_docstring
def zeros(n, dtype=types.cpu_type):
    if dtype == types.cpu_bool:
        # Hack due to lack of "bool" in PyOpenCL
        return np.zeros(n, dtype=types.cpu_real_bool)
    else:
        return np.zeros(n, dtype=dtype)
