########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Core to do operations in GPUs.

IMPORTANT: do not expose any pycuda or reikna function or class, since this module
is imported by the "core.py" module.
'''
from . import PACKAGE_PATH
from .core import CUDA, OPENCL

import functools
import logging
import math
import numpy as np
import os
import warnings

GPU_SRC = os.path.join(PACKAGE_PATH, 'src', 'gpu')

logger = logging.getLogger(__name__)


def active_context(method):
    '''
    Decorator to set the context of a class as the current
    context before calling the method.
    '''
    @functools.wraps(method)
    def __wrapper(self, *args, **kwargs):
        self.make_active()
        return method(self, *args, **kwargs)
    return __wrapper


def closest_number_divisible(n, m):
    '''
    Find the closest number to *n* divisible by *m*, but being always
    equal or greater than *n*.

    :param n: reference number.
    :type n: int
    :param m: number to be divisible by.
    :type m: int
    :returns: Output number.
    :rtype: int
    '''
    if n % m == 0:
        return n
    elif n > m:
        return m * ((n // m) + 1)
    else:
        return m


class Context(object):

    __CUDA_CONTEXT = None  # only necessary for CUDA

    def __init__(self, api, device, backend):
        '''
        Proxy for a GPU device.
        '''
        super().__init__()

        # Create the context and thread
        if backend == CUDA:
            self.__context = device.make_context()
            xsize, ysize = device.MAX_BLOCK_DIM_X, device.MAX_BLOCK_DIM_Y
        else:
            # OPENCL
            import pyopencl
            self.__context = pyopencl.Context([device])

            xsize, ysize, _ = device.max_work_item_sizes

        if xsize != ysize:
            warnings.warn(
                'Maximum number of threads allowed in X and Y are different; choosing the smallest value', RuntimeWarning)

        self.__in_cuda = (backend == CUDA)  # to properly handle "__del__"
        self.__device = device
        self.__thread = api.Thread(self.__context)

        self.__max_local_size = int(min(xsize, ysize))
        self.__min_local_size_2d = 2  # the only value that is hardcoded
        self.__sqrt_max_local_size = int(math.sqrt(self.__max_local_size))

    def __del__(self):
        '''
        Detach the underlying context if in CUDA backend.
        '''
        if self.__in_cuda:
            self.__context.pop()
            self.__context.detach()

    def __getattr__(self, name):
        '''
        Get an attribute of the underlying :meth:`reikna.cluda.api.Thread`
        object.

        :param name: name of the attribute.
        :type name: str
        '''
        return getattr(self.__thread, name)

    @property
    def context(self):
        '''
        Underlying context.
        '''
        return self.__context

    @property
    def device(self):
        '''
        Underlying GPU device.
        '''
        return self.__device

    def _get_sizes_2d_gl(self, size_f, size_s):
        '''
        Return the standard sizes in two dimensions, where the first size is
        assumed to be greater than or equal to the second.

        :param sizes: size of the arrays to work.
        :type: int
        :returns: global and local sizes, that depend on the number of dimensions.
        :rtype: int, int or tuple(int, ...), tuple(int, ...)
        '''
        assert size_f >= size_s

        d = np.power(self.__min_local_size_2d, size_f // size_s - 1)

        if d > 1:

            if d < self.__sqrt_max_local_size:
                max_size = self.__sqrt_max_local_size // d
            else:
                max_size = self.__min_local_size_2d  # minimum number of threads per block

            ls_f = self.__max_local_size // max_size
            ls_s = max_size

        else:  # Both are set to have similar sizes
            ls_f = ls_s = self.__sqrt_max_local_size

        size_f = closest_number_divisible(size_f, ls_f)
        size_s = closest_number_divisible(size_s, ls_s)

        return size_f, ls_f, size_s, ls_s

    @active_context
    def array(self, *args, **kwargs):
        '''
        Create a new array in the device. Arguments are forwarded to
        :meth:`reikna.cluda.api.Thread.array`.
        '''
        return self.__thread.array(*args, **kwargs)

    @active_context
    def compile(self, *args, **kwargs):
        '''
        Compile code in the device. Arguments are forwarded to
        :meth:`reikna.cluda.api.Thread.compile`.
        '''
        return Module(self.__thread.compile(*args, **kwargs), self)

    def get_sizes(self, size_x, size_y=None):
        '''
        Return the standard sizes for a given array. In case a size in
        "y" is provided, it is considered to be smaller than the size in "x".

        :param sizes: size of the arrays to work.
        :type: int
        :returns: global and local sizes, that depend on the number of dimensions.
        :rtype: int, int or tuple(int, ...), tuple(int, ...)
        '''
        if size_y is None:

            if size_x > self.__max_local_size:

                ls_x = self.__max_local_size
                size_x = closest_number_divisible(size_x, ls_x)

                return int(size_x), int(ls_x)
            else:
                # we can run everything in parallel
                return self.__max_local_size, self.__max_local_size
        else:
            if size_x * size_y > self.__max_local_size:

                if size_x > size_y:
                    size_x, ls_x, size_y, ls_y = self._get_sizes_2d_gl(
                        size_x, size_y)
                else:
                    size_y, ls_y, size_x, ls_x = self._get_sizes_2d_gl(
                        size_y, size_x)

            else:
                ls_x = ls_y = self.__sqrt_max_local_size
                size_x = closest_number_divisible(size_x, ls_x)
                size_y = closest_number_divisible(size_y, ls_y)

            return int(size_x), int(ls_x), int(size_y), int(ls_y)

    def make_active(self):
        '''
        Make this context the current context.
        '''
        if self.__in_cuda and not self.__context is Context.__CUDA_CONTEXT:

            if Context.__CUDA_CONTEXT is not None:
                Context.__CUDA_CONTEXT.pop()

            self.__context.push()

            Context.__CUDA_CONTEXT = self.__context

    @active_context
    def to_device(self, a):
        '''
        Copy an array to the device.

        :param a: input array.
        :type a: numpy.ndarray
        '''
        return self.__thread.to_device(a)


class Module(object):

    def __init__(self, module, context):
        '''
        Wrapper around a :class:`reikna.cluda.api.Program` instance.

        :param module: compiled module.
        :type module: reikna.cluda.api.Program
        :param context: context where the module lives.
        '''
        super().__init__()

        self.__module = module
        self.__context = context

    def __getattr__(self, name):
        '''
        Get an attribute of the underlying module.

        :param name: name of the attribute.
        :type name: str
        '''
        self.__context.make_active()
        return getattr(self.__module, name)


def device_lookup(devices, device=None, interactive=False):
    '''
    Function to look for GPU devices.

    :param devices: list of available devices
    :type devices: list(Device)
    :param device: index of the possible device to use.
    :type device: int
    :param interactive: whether to ask the user for input.
    :type interactive: bool
    :returns: the selected device.
    :rtype: Device

    .. note:: The device can be selected using the MINKIT_DEVICE environment variable.
    '''
    if len(devices) == 0:
        raise LookupError('No devices have been found')

    default = 0  # This is the default device to use

    # Override the device from the environment variable "MINKIT_DEVICE"
    device = os.environ.get('MINKIT_DEVICE', device)

    if device is not None:

        device = int(device)

        # Use the specified device (if available)
        if device > len(devices) - 1:
            logger.warning(f'Specified a device number ({device}) '
                           'greater than the maximum number '
                           'of devices (maximum allowed: {n - 1})')
        else:
            return device
    elif len(devices) == 1:
        # Use the default value
        if interactive:
            logger.warning('There is only one device available; will use that')
        return default

    if not interactive:
        # Use the default value
        logger.info('Using the first encountered device')
        return default
    else:
        # Ask the user to select a device
        print(f'Found {len(devices)} available devices:')
        for i, (p, d) in enumerate(devices):
            print(f'- ({p.name}) {d.name} [{i}]')

        device = -1
        while int(device) not in range(len(devices)):
            device = input('Select a device (default {}): '.format(default))
            if device.strip() == '':
                # Set to default value
                return default

        return device


def initialize_gpu(backend, **kwargs):
    '''
    Initialize a new GPU context.

    :param backend: backend to use. It must be any of "cuda" or "opencl".
    :type backend: str
    :param kwargs: it may contain any of the following values: \
    - interactive: (bool) whether to select the device manually (defaults to False) \
    - device: (int) number of the device to use (defaults to None).
    :type kwargs: dict

    .. note:: The device can be selected using the MINKIT_DEVICE environment variable.
    '''
    from reikna import cluda

    if backend == CUDA:
        api = cluda.cuda_api()
    elif backend == OPENCL:
        api = cluda.ocl_api()
    else:
        raise ValueError(f'Unknown backend type "{backend}"')

    # Get all available devices
    platforms = api.get_platforms()

    all_devices = [(p, d) for p in platforms for d in p.get_devices()]

    # Determine the device to use
    idev = device_lookup(all_devices, **kwargs)

    platform, device = all_devices[idev]

    logger.info(
        f'Selected device "{device.name}" ({idev}) (platform: {platform.name})')

    return Context(api, device, backend)
