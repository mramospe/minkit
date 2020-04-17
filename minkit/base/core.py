'''
Definition of the backend where to store the data and run the jobs.
'''
import atexit
import contextlib
import functools
import importlib
import inspect
import logging
import numpy as np
import os
import pkgutil
import time

from . import data_types

__all__ = ['Backend', 'free_cache', 'timer']


logger = logging.getLogger(__name__)


def get_exposed_package_objects(path):
    '''
    Process a given path, taking all the exposed objects in it and returning
    a dictionary with their names and respective pointers.

    :param path: path to the package.
    :type path: str
    :returns: names and objects that are exposed.
    :rtype: dict(str, object)
    '''
    pkg = os.path.normpath(path[path.rfind('minkit'):]).replace(os.sep, '.')

    dct = {}

    for loader, module_name, ispkg in pkgutil.walk_packages([path]):

        if module_name.endswith('setup') or module_name.endswith('__'):
            continue

        # Import all classes and functions
        mod = importlib.import_module('.' + module_name, package=pkg)

        for n, c in inspect.getmembers(mod):
            if n in mod.__all__:
                dct[n] = c

    return dct


def free_cache(self):
    '''
    Free the cache of arrays. Only works in GPU mode, with
    backend *cuda* or *opencl*.
    In :py:mod:`minkit`, when the backend is set to GPU, arrays will
    tend to be reused and kept in the device till a call to :func:`free_cache`
    is done.
    This call will eliminate all arrays that are not being used, what will
    free space in memory.
    '''
    if BACKEND != CPU:
        aop.free_cache()


def initialize(backend='cpu', **kwargs):
    '''
    Initialize the package, setting the backend and determining what packages to
    use accordingly.

    :param backend: backend to use. It must be any of *cpu*, *cuda* or *opencl*.
    :type backend: str
    :param kwargs: meant to be used in the CUDA or OpenCL backend, it may contain \
    any of the following values: \
    - interactive: (bool) whether to select the device manually (defaults to False) \
    - device: (int) number of the device to use (defaults to None).
    :type kwargs: dict

    .. note:: The backend can be also specified as an environment variable \
    MINKIT_BACKEND, overriding that specified in any :func:`initialize` call. \
    The device can be selected using the MINKIT_DEVICE variable as well, with \
    an identical behaviour.
    '''
    global BACKEND
    global ARRAY_OPERATION

    # Override the backend from the environment variable "MINKIT_BACKEND"
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
        from ..backends import cpu
        ARRAY_OPERATION = cpu
    elif BACKEND == CUDA or BACKEND == OPENCL:

        # Must be called only once
        gpu_core.initialize_gpu(BACKEND, **kwargs)

        from ..backends import gpu
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
