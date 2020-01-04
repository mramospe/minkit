'''
Core to do operations in GPUs.

IMPORTANT: do not expose any pycuda or reikna function or class, since this module
is imported by the "core.py" module.
'''
import atexit
import logging

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


logger = logging.getLogger(__name__)


def get_sizes(size):
    '''
    Return the standard sizes for a given array.

    :param size: size of the arrays to work.
    :type: int
    :returns: global and local sizes.
    :rtype: int, int or tuple(int, ...), tuple(int, ...)
    '''
    frac = (size // 1000) or 1
    while size % frac != 0:
        frac -= 1

    return size, frac  # TODO: OPTIMIZE


def device_lookup(devices, device=None, interactive=False):
    '''
    Function to look for GPU devices.
    '''
    if len(devices) == 0:
        raise LookupError('No devices have been found')

    default = 0  # This is the default device to use

    if device is not None:
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
    Initialize a new pycuda context.
    The argument "kwargs" may contain any of the following values:
    - interactive: (bool) whether to select the device manually (defaults to False)
    - device: (int) number of the device to use (defaults to None).
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
