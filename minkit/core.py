'''
Definition of the backend where to store the data and run the jobs.
'''
import atexit
import contextlib
import functools
import logging
import numpy as np
import os
import time

from .operations import types
from .operations import gpu_core

__all__ = ['aop', 'free_cache', 'initialize', 'timer']


# CPU backend
CPU = 'cpu'
# OpenCL backend
OPENCL = gpu_core.OPENCL
# CUDA backend
CUDA = gpu_core.CUDA

# Keep the module to do operation on arrays
ARRAY_OPERATION = None

# Current backend (by default use CPU)
BACKEND = None

NOT_IMPLEMENTED = NotImplementedError(
    f'Function not implemented for backend "{BACKEND}"')

logger = logging.getLogger(__name__)


def calling_base_class_method(method):
    '''
    Decorator to always call a base class method.
    The execution will take place before the call to the derived class method.

    :param method: method to decorate.
    :type method: function
    :returns: wrapper
    :rtype: function
    '''

    def __wrapper(self, *args, **kwargs):
        '''
        Internal wrapper.
        '''
        getattr(super(self.__class__, self), method.__name__)(*args, **kwargs)
        return method(self)
    return __wrapper


def with_backend(function):
    '''
    Check whether a backend has been defined and raise an error if not.

    :param function: function to decorate.
    :type function: function
    :raises RuntimeError: if a backend is not set.
    '''
    @functools.wraps(function)
    def __wrapper(*args, **kwargs):
        '''
        Internal wrapper around the decorated function.
        '''
        if BACKEND is None:
            raise RuntimeError(
                f'A backend must be specified before a call to "{function.__name__}"')
        return function(*args, **kwargs)
    return __wrapper


class meta_operation(type):
    '''
    Metaclass to hold the operations.
    '''
    @with_backend
    def __getattr__(cls, name):
        '''
        Get an operation with name "name".

        :param name: name of the operation.
        :type name: str
        :returns: operation
        :rtype: function
        '''
        return getattr(ARRAY_OPERATION, name)


class aop(metaclass=meta_operation):
    '''
    Access the operations on arrays for the current backend.
    '''
    pass


def free_cache(self):
    '''
    Free the cache of arrays. Only works in GPU mode, with
    backend "cuda" or "opencl".
    In :py:mod:`minkit`, when the backend is set to GPU, arrays will
    tend to be reused and kept in the device till a call to :func:`free_cache`
    is done.
    This call will eliminate all arrays that are not being used, what will
    free space in memory.
    '''
    if BACKEND != CPU:
        aop.free_cache()


def initialize(backend=CPU, **kwargs):
    '''
    Initialize the package, setting the backend and determining what packages to
    use accordingly.

    :param backend: backend to use. It must be any of "cpu", "cuda" or "opencl".
    :type backend: str
    :param kwargs: meant to be used in the CUDA or OpenCL backend, it may contain \
    any of the following values: \
    - interactive: (bool) whether to select the device manually (defaults to False) \
    - device: (int) number of the device to use (defaults to None).
    :type kwargs: dict

    .. note:: The backend can be also specified as an environmental variable \
    MINKIT_BACKEND, overriding that specified in any :func:`initialize` call.
    '''
    global BACKEND
    global ARRAY_OPERATION

    # Override the backend from the environmental variable "MINKIT_BACKEND"
    backend = os.environ.get('MINKIT_BACKEND', backend).lower()

    if BACKEND is not None and backend != BACKEND:
        logger.error(
            f'Unable to set backend to "{backend}"; already set ({BACKEND})')
        return
    elif backend == BACKEND:
        # It is asking to initialize the same backend again; skip
        return

    BACKEND = backend

    logger.info(f'Using backend "{BACKEND}"')

    if BACKEND == CPU:
        from .operations import cpu
        ARRAY_OPERATION = cpu
    elif BACKEND == CUDA or BACKEND == OPENCL:

        # Must be called only once
        gpu_core.initialize_gpu(BACKEND, **kwargs)

        from .operations import gpu
        ARRAY_OPERATION = gpu

    else:
        raise ValueError(f'Unknown backend "{BACKEND}"')


@contextlib.contextmanager
def timer():
    '''
    Create an object that, on exit, displays the time elapsed.
    '''
    start = time.time()
    yield
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info(f'Time elapsed: {hours}h {minutes}m {seconds}s')
