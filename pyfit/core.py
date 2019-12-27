'''
Definition of the backend where to store the data and run the jobs.
'''
import functools
import logging
import numpy as np
import os

from . import types

__all__ = ['array', 'extract_ndarray', 'initialize', 'random_uniform', 'zeros', 'max', 'min', 'concatenate', 'linspace', 'shuffling_index']

# Current backend
BACKEND = None

# CPU backend
CPU = 'cpu'
# OpenCL backend
OPENCL = 'opencl'
# CUDA backend
CUDA = 'cuda'

NOT_IMPLEMENTED = NotImplementedError(f'Function not implemented for backend "{BACKEND}"')

logger = logging.getLogger(__name__)


def calling_base_class_method( method ):
    '''
    Decorator to always call a base class method.
    The execution will take place before the call to the derived class method.

    :param method: method to decorate.
    :type method: function
    :returns: wrapper
    :rtype: function
    '''
    def __wrapper( self, *args, **kwargs ):
        '''
        Internal wrapper.
        '''
        getattr(super(self.__class__, self), method.__name__)(*args, **kwargs)
        return method(self)
    return __wrapper


def with_backend( func ):
    '''
    Check whether a backend has been defined and raise an error if not.

    :raises RuntimeError: if a backend is not set.
    '''
    @functools.wraps(func)
    def __wrapper( *args, **kwargs ):
        '''
        Internal wrapper around the decorated function.
        '''
        if BACKEND is None:
            raise RuntimeError(f'A backend must be specified before a call to "{func.__name__}"')
        return func(*args, **kwargs)
    return __wrapper


@with_backend
def array( *args, **kwargs ):
    '''
    Create an array using the current backend.
    Arguments are directly forwarded to the constructor.

    :returns: input data converted to the correct type for the current backend.
    :rtype: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
    '''
    if BACKEND is CPU:
        return np.array(*args, dtype=types.cpu_type, **kwargs)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def extract_ndarray( obj ):
    '''
    Get an :class:`numpy.ndarray` class from the input object.

    :param obj: input data.
    :type obj: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
    :returns: data as :class:`numpy.ndarray`
    :rtype: numpy.ndarray
    '''
    if BACKEND is CPU:
        return np.array(obj)
    else:
        raise NOT_IMPLEMENTED


def initialize( backend = CPU, interactive = False ):
    '''
    Initialize the package, setting the backend and determining what packages to
    use accordingly.
    '''
    global BACKEND

    if BACKEND is not None:
        logger.error(f'Unable to set backend to "{backend}"; already set ({BACKEND})')
        return

    BACKEND = backend


def linspace( vmin, vmax, size ):
    '''
    '''
    if BACKEND == CPU:
        return np.linspace(vmin, vmax, size)
    else:
        raise NOT_IMPLEMENTED


def meshgrid( *arrays ):
    '''
    '''
    if BACKEND == CPU:
        return tuple(map(np.ndarray.flatten, np.meshgrid(*arrays)))
    else:
        raise NOT_IMPLEMENTED


@with_backend
def concatenate( *arrs ):
    '''
    '''
    if BACKEND == CPU:
        return np.concatenate(arrs)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def random_uniform( vmin, vmax, size ):
    '''
    Create data following an uniform distribution between 0 and 1, with size "size".

    :param size: size of the data to create
    :type size: int
    :returns: data following an uniform distribution between 0 and 1.
    :rtype: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
    '''
    if BACKEND == CPU:
        return np.random.uniform(vmin, vmax, size)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def log( array ):
    '''
    '''
    if BACKEND is CPU:
        return np.log(array)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def sum( array ):
    '''
    '''
    if BACKEND is CPU:
        return np.sum(array)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def max( arr ):
    '''
    '''
    if BACKEND == CPU:
        return np.max(arr)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def min( arr ):
    '''
    '''
    if BACKEND == CPU:
        return np.min(arr)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def arange( *args, **kwargs ):
    '''
    '''
    if BACKEND == CPU:
        return np.arange(*args, **kwargs)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def ones( *args, dtype=types.cpu_type, **kwargs ):
    '''
    Create an array filled with ones using the current backend.
    Arguments are directly forwarded to the constructor.

    :returns: zeros following the given shape.
    :rtype: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
    '''
    if BACKEND is CPU:
        return np.ones(*args, dtype=dtype, **kwargs)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def zeros( *args, dtype=types.cpu_type, **kwargs ):
    '''
    Create an array filled with zeros using the current backend.
    Arguments are directly forwarded to the constructor.

    :returns: zeros following the given shape.
    :rtype: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
    '''
    if BACKEND is CPU:
        return np.zeros(*args, dtype=dtype, **kwargs)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def logical_and( a, b ):
    '''
    Do the logical "and" operation between two arrays.
    '''
    if BACKEND == CPU:
        return np.logical_and(a, b)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def logical_or( a, b ):
    '''
    Do the logical "or" operation between two arrays.
    '''
    if BACKEND == CPU:
        return np.logical_or(a, b)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def fft( a ):
    '''
    Calculate the fast-Fourier transform of an array.
    '''
    if BACKEND == CPU:
        return np.fft.fft(a)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def ifft( a ):
    '''
    Calculate the fast-Fourier transform of an array.
    '''
    if BACKEND == CPU:
        return np.fft.ifft(a)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def interpolate_linear( x, xp, yp):
    '''
    '''
    if BACKEND == CPU:
        return np.interp(x, xp, yp)
    else:
        raise NOT_IMPLEMENTED


def fftconvolve( a, b, data = None, normalized = True ):
    '''
    Calculate the convolution using a FFT of two arrays.

    :param a: first array.
    :type a: numpy.ndarray
    :param b: first array.
    :type b: numpy.ndarray
    :param data: possible values where "a" and "b" where taken from.
    :type data: numpy.ndarray
    :param normalized: whether to return a normalized output or not.
    :type normalized: bool
    :returns: convolution of the two array.
    :rtype: numpy.ndarray
    '''
    fa = fft(a)
    fb = fft(b)

    if data is not None:
        shift = fftshift(data)
    else:
        shift = 1.

    prod = fa * shift * fb

    output = fft(prod)

    if normalized:
        return output * (data[1] - data[0]) / len(output)
    else:
        return output


def fftshift( a ):
    '''
    Compute the shift in frequencies needed for "fftconvolve" to return values in the same data range.

    :param a: input array.
    :type a: numpy.ndarray
    :returns: shift in the frequency domain.
    :rtype: numpy.ndarray
    '''
    n0  = sum(a < 0)
    nt  = len(a)
    com = types.cpu_complex(+2.j * np.pi * n0 / nt)
    rng = arange(nt, dtype=types.cpu_int).astype(types.cpu_complex)
    return exp(com * rng)


@with_backend
def exp( a ):
    if BACKEND == CPU:
        return np.exp(a)
    else:
        raise NOT_IMPLEMENTED


@with_backend
def shuffling_index( n ):
    '''
    Return an array to shuffle data, with length "n".

    :param n: size of the data.
    :type n: int
    :returns: shuffling array.
    :rtype: numpy.ndarray
    '''
    if BACKEND == CPU:
        indices = np.arange(n)
        np.random.shuffle(indices)
        return indices
    else:
        raise NOT_IMPLEMENTED
