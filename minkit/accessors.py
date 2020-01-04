'''
Accessors to PDF in code.
'''
from . import PACKAGE_PATH
from . import core
from .operations import gpu_core
from .operations import types
from distutils import ccompiler

from ctypes import cdll
import functools
import numpy as np
import os
import tempfile

__all__ = ['add_pdf_src']

TMPDIR = tempfile.TemporaryDirectory()

# Caches for the C++ PDFs
CPU_PDF_MODULE_CACHE = {}
CPU_PDF_CACHE = {}

# Caches for the OpenCL and CUDA PDFs
GPU_PDF_MODULE_CACHE = {}
GPU_PDF_CACHE = {}

# Path to directories where to search for PDFs
PDF_PATHS = []

# Save the C-related flags to compile
CFLAGS = os.environ.get('CFLAGS', '').split()


def access_cpu_module(name):
    '''
    Access a C++ module, compiling it if it has not been done yet.

    :param name: name of the module.
    :type name: str
    :returns: compiled module.
    :rtype: module
    '''
    global CPU_PDF_MODULE_CACHE

    if name in CPU_PDF_MODULE_CACHE:
        # Do not compile again the PDF source if it has already been done
        module = CPU_PDF_MODULE_CACHE[name]
    else:
        # By default take it from this package
        source = os.path.join(PACKAGE_PATH, f'cpu/{name}.cpp')

        # Check if it exists in any of the provided paths
        for p in PDF_PATHS:
            fp = os.path.join(p, f'{name}.cpp')
            if os.path.isfile(fp):
                source = fp
                break

        # Compile the C++ code and load the library
        if not os.path.isfile(source):
            raise RuntimeError(
                f'Function {name} does not exist in "{source}" or any of the provided paths')

        compiler = ccompiler.new_compiler(plat='x86_64-centos7-gcc8-opt')
        objects = compiler.compile(
            [source], output_dir=TMPDIR.name, extra_preargs=CFLAGS)
        libname = os.path.join(TMPDIR.name, f'lib{name}.so')
        compiler.link(f'{name} library', objects, libname,
                      extra_preargs=CFLAGS, libraries=['stdc++'])

        module = cdll.LoadLibrary(libname)

        CPU_PDF_MODULE_CACHE[name] = module

    return module


def add_pdf_src(path):
    '''
    This function adds a new path where to look for user-defined PDFs.
    PDFs are searched for in paths opposite to the order they are appended
    (PDFs are taken from the last appended paths first).

    :param path: new path to be considered.
    :type path: str
    '''
    PDF_PATHS.insert(0, path)


def create_cpu_function_proxies(module, name, ndata_pars, narg_pars=0, nvar_arg_pars=None):
    '''
    Create proxies for C++ that handle correctly the input and output data
    types of a fucntion.
    '''
    # Get the functions
    function = module.function
    evaluate = module.evaluate

    if hasattr(module, 'normalization'):
        normalization = module.normalization
    else:
        normalization = None

    # Define the types of the input arguments
    partypes = [types.c_double for _ in range(narg_pars)]

    if nvar_arg_pars is not None:
        partypes += [types.c_int, types.c_double_p]

    # Define the types passed to the function
    functiontypes = [types.c_double for _ in range(ndata_pars)]
    functiontypes += partypes
    function.argtypes = functiontypes
    function.restype = types.c_double

    @functools.wraps(function)
    def __function(*args):
        '''
        Internal wrapper.
        '''
        if nvar_arg_pars is not None:
            var_arg_pars = args[-1]
            vals = tuple(map(types.c_double, args[:-1])) + (
                types.c_int(nvar_arg_pars), var_arg_pars.ctypes.data_as(types.c_double_p))
        else:
            vals = tuple(map(types.c_double, args))

        return function(*vals)

    # Define the types for the arguments passed to the evaluate function
    argtypes = [types.c_int, types.c_double_p]
    argtypes += [types.c_double_p for _ in range(ndata_pars)]
    argtypes += partypes
    evaluate.argtypes = argtypes

    @functools.wraps(evaluate)
    def __evaluate(output_array, *args):
        '''
        Internal wrapper.
        '''
        op = output_array.ctypes.data_as(types.c_double_p)
        ips = tuple(d.ctypes.data_as(types.c_double_p)
                    for d in args[:ndata_pars])

        if nvar_arg_pars is not None:

            # Variable arguments must be the last in the list
            var_arg_pars = args[-1]

            vals = tuple(map(types.c_double, args[ndata_pars:-1])) + (
                nvar_arg_pars, var_arg_pars.ctypes.data_as(types.c_double_p))

        else:
            vals = tuple(map(types.c_double, args[ndata_pars:]))

        return evaluate(types.c_int(len(output_array)), op, *ips, *vals)

    if normalization is not None:

        # Define the types passed to the normalization function
        normtypes = partypes
        normtypes += [types.c_double for _ in range(2 * ndata_pars)]
        normalization.argtypes = normtypes
        normalization.restype = types.c_double

        @functools.wraps(normalization)
        def __normalization(*args):
            '''
            Internal wrapper.
            '''
            if nvar_arg_pars is not None:
                if nvar_arg_pars == 0:
                    # This is special case, must handle index carefully
                    var_arg_pars = np.array([], dtype=types.c_double)
                else:
                    var_arg_pars = args[-1 - 2 * ndata_pars]
                # Normal arguments are parse first
                vals = tuple(map(types.c_double, args[:-1 - 2 * ndata_pars]))
                # Variable number of arguments follow
                vals += (types.c_int(nvar_arg_pars),
                         var_arg_pars.ctypes.data_as(types.c_double_p))
                # Finally, the integration limits must be specified
                vals += tuple(map(types.c_double, args[-2 * ndata_pars:]))
            else:
                vals = tuple(map(types.c_double, args))

            return normalization(*vals)
    else:
        __normalization = None

    return __function, __evaluate, __normalization


def create_gpu_function_proxy(name, ndata_pars, narg_pars, nvar_arg_pars):
    '''
    Creates a proxy for a function writen in GPU.
    '''
    global GPU_PDF_MODULE_CACHE

    if name in GPU_PDF_MODULE_CACHE:
        # Return the existing module if it is in the cache
        return GPU_PDF_MODULE_CACHE[name]
    else:
        # By default take it from this package
        source = os.path.join(PACKAGE_PATH, f'gpu/{name}.c')

        # Check if it exists in any of the provided paths
        for p in PDF_PATHS:
            fp = os.path.join(p, f'{name}.c')
            if os.path.isfile(fp):
                source = fp
                break

        if not os.path.isfile(source):
            raise RuntimeError(
                f'Function {name} does not exist in "{source}" or any of the provided paths')

        # Compile the code
        with open(source) as fi:
            module = gpu_core.THREAD.compile(fi.read())

        GPU_PDF_MODULE_CACHE[name] = module

        # Access the function in the module
        evaluate = module.evaluate

        @functools.wraps(evaluate)
        def __evaluate(output_array, *args):
            '''
            Internal wrapper.
            '''
            ips = args[:ndata_pars]

            if nvar_arg_pars is not None:

                # Variable arguments must be the last in the list
                var_arg_pars = args[-1]

                vals = tuple(map(types.cpu_type, args[ndata_pars:-1])) + (
                    types.cpu_int(nvar_arg_pars), var_arg_pars)
            else:
                vals = tuple(map(types.cpu_type, args[ndata_pars:]))

            global_size, local_size = gpu_core.get_sizes(len(output_array))

            return evaluate(output_array, *ips, *vals, global_size=global_size, local_size=local_size)

        return __evaluate


@core.with_backend
def access_pdf(name, ndata_pars, narg_pars=0, nvar_arg_pars=None):
    '''
    Access a PDF with the given name and number of data and parameter arguments.

    :returns: PDF and normalization function.
    :rtype: tuple(function, function)
    '''
    global CPU_PDF_MODULE_CACHE
    global CPU_PDF_CACHE

    # Access the C++ module
    cpu_module = access_cpu_module(name)

    # Get the functions
    try:
        modname = name if nvar_arg_pars is None else f'{name}{nvar_arg_pars}'

        if modname in CPU_PDF_CACHE:
            cpu_output = CPU_PDF_CACHE[modname]
        else:
            cpu_output = create_cpu_function_proxies(
                cpu_module, name, ndata_pars, narg_pars, nvar_arg_pars)
            CPU_PDF_CACHE[modname] = cpu_output

    except AttributeError as ex:
        raise RuntimeError(
            f'Error loading function "{name}"; make sure that both "{name}" and "normalization" are defined inside "{source}"')

    if core.BACKEND == core.CPU:
        # Return the CPU version directly
        return cpu_output

    elif core.BACKEND == core.CUDA or core.BACKEND == core.OPENCL:

        # Function and normalization are taken from the C++ version
        function, _, normalization = cpu_output

        # Get the "evaluate" function from source
        if modname in GPU_PDF_CACHE:
            evaluate = GPU_PDF_CACHE[modname]
        else:
            evaluate = create_gpu_function_proxy(
                name, ndata_pars, narg_pars, nvar_arg_pars)
            GPU_PDF_CACHE[modname] = evaluate

        return function, evaluate, normalization
    else:
        raise NotImplementedError(
            f'Function not implemented for backend "{core.BACKEND}"')
