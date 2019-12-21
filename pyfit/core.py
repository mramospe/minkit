'''
Definition of the backend where to store the data and run the jobs.
'''
import functools
import logging
import os

from . import types

__all__ = ['array', 'extract_ndarray', 'initialize', 'random_uniform', 'zeros']

# Current backend
BACKEND = None

# CPU backend
CPU = 'cpu'
# OpenCL backend
OPENCL = 'opencl'
# CUDA backend
CUDA = 'cuda'

logger = logging.getLogger(__name__)


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
        import numpy as np
        return np.array(*args, dtype=types.cpu_type, **kwargs)
    else:
        raise NotImplementedError(f'Function not implemented for backend "{BACKEND}"')


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
        import numpy as np
        return np.array(obj)
    else:
        raise NotImplementedError(f'Function not implemented for backend "{BACKEND}"')


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


@with_backend
def random_uniform( size ):
    '''
    Create data following an uniform distribution between 0 and 1, with size "size".

    :param size: size of the data to create
    :type size: int
    :returns: data following an uniform distribution between 0 and 1.
    :rtype: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
    '''
    if BACKEND == CPU:
        import numpy as np
        return np.random.uniform(0, 1, size)
    else:
        raise NotImplementedError(f'Function not implemented for backend "{BACKEND}"')


@with_backend
def log( array ):
    '''
    '''
    if BACKEND is CPU:
        import numpy as np
        return np.log(array)
    else:
        raise NotImplementedError(f'Function not implemented for backend "{BACKEND}"')


@with_backend
def sum( array ):
    '''
    '''
    if BACKEND is CPU:
        import numpy as np
        return np.sum(array)
    else:
        raise NotImplementedError(f'Function not implemented for backend "{BACKEND}"')


@with_backend
def zeros( *args, **kwargs ):
    '''
    Create an array filled with zeros using the current backend.
    Arguments are directly forwarded to the constructor.

    :returns: zeros following the given shape.
    :rtype: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
    '''
    if BACKEND is CPU:
        import numpy as np
        return np.zeros(*args, dtype=types.cpu_type, **kwargs)
    else:
        raise NotImplementedError(f'Function not implemented for backend "{BACKEND}"')
