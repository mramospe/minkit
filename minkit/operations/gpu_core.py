'''
Core to do operations in GPUs.

IMPORTANT: do not expose any pycuda or reikna function or class, since this module
is imported by the "core.py" module.
'''
import atexit
import logging
import math
import numpy as np
import os

# GPU variables
DEVICE = None
CONTEXT = None
THREAD = None

# GPU backend
API = None
BACKEND = None

# OpenCL backend
OPENCL = 'opencl'
# CUDA backend
CUDA = 'cuda'

# Maximum value for the local size
MAX_LOCAL_SIZE = 100


logger = logging.getLogger(__name__)


def get_sizes(size):
    '''
    Return the standard sizes for a given array.

    :param size: size of the arrays to work.
    :type: int
    :returns: global and local sizes.
    :rtype: int, int or tuple(int, ...), tuple(int, ...)
    '''
    a = size % MAX_LOCAL_SIZE
    if a == 0:
        gs, ls = size, MAX_LOCAL_SIZE
    elif size < MAX_LOCAL_SIZE:
        gs, ls = size, 1
    else:
        a = np.arange(1, min(MAX_LOCAL_SIZE, math.ceil(math.sqrt(size))))
        a = a[size % a == 0]
        ls = int(a[np.argmin(np.abs(a - MAX_LOCAL_SIZE))])
        gs = size
    return int(gs), int(ls)


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
        print(f'Found {n} available devices:')
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
    global BACKEND
    global DEVICE
    global CONTEXT
    global THREAD

    from reikna import cluda

    # Establish the backend
    if BACKEND is not None and backend != BACKEND:
        raise RuntimeError(
            f'Attempt to change backend from "{BACKEND}" to "{backend}"; not supported')
    elif backend == CUDA:
        API = cluda.cuda_api()
    elif backend == OPENCL:
        API = cluda.ocl_api()
    elif backend == BACKEND:
        # Using same backend
        return
    else:
        raise ValueError(f'Unknown backend type "{backend}"')

    BACKEND = backend

    # Get all available devices
    platforms = API.get_platforms()

    all_devices = [(p, d) for p in platforms for d in p.get_devices()]

    # Determine the device to use
    idev = device_lookup(all_devices, **kwargs)

    platform, device = all_devices[idev]

    logger.info(
        f'Selected device "{device.name}" ({idev}) (platform: {platform.name})')

    DEVICE = device

    # Create the context and thread
    if BACKEND == CUDA:
        CONTEXT = DEVICE.make_context()

        def clear_cuda_context():
            from pycuda.tools import clear_context_caches
            CONTEXT.pop()
            clear_context_caches()
        atexit.register(clear_cuda_context)
    else:
        # OPENCL
        import pyopencl
        CONTEXT = pyopencl.Context([DEVICE])

    THREAD = API.Thread(CONTEXT)
