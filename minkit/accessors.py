'''
Accessors to PDF in code.
'''
from . import PACKAGE_PATH
from . import autocode
from . import core
from .operations import gpu_core
from .operations import types
from distutils import ccompiler

from ctypes import cdll
import functools
import logging
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

logger = logging.getLogger(__name__)


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
        xml_source = os.path.join(PACKAGE_PATH, f'src/{name}.xml')

        # Check if it exists in any of the provided paths
        for p in PDF_PATHS:
            fp = os.path.join(p, f'{name}.xml')
            if os.path.isfile(fp):
                xml_source = fp
                break

        if not os.path.isfile(xml_source):
            raise RuntimeError(
                f'XML file for function {name} not found in "{PACKAGE_PATH}/src" or any of the provided paths')

        # Write the code
        source = os.path.join(TMPDIR.name, f'{name}.cpp')
        code = autocode.generate_code(xml_source, core.CPU)
        with open(source, 'wt') as f:
            f.write(code)

        # Compile the C++ code and load the library
        compiler = ccompiler.new_compiler()

        try:
            objects = compiler.compile(
                [source], output_dir=TMPDIR.name, extra_preargs=CFLAGS)
            libname = os.path.join(TMPDIR.name, f'lib{name}.so')
            compiler.link(f'{name} library', objects, libname,
                          extra_preargs=CFLAGS, libraries=['stdc++'])
        except Exception as ex:
            nl = len(str(code.count('\n')))
            code = '\n'.join(f'{i + 1:>{nl}}: {l}' for i,
                             l in enumerate(code.split('\n')))
            logger.error(f'Error found compiling:\n{code}')
            raise ex

        module = cdll.LoadLibrary(libname)

        CPU_PDF_MODULE_CACHE[name] = module

    return module


def access_gpu_module(name):
    '''
    Access a GPU module, compiling it if it has not been done yet.

    :param name: name of the module.
    :type name: str
    :returns: compiled module.
    :rtype: module
    '''
    global GPU_PDF_MODULE_CACHE

    if name in GPU_PDF_MODULE_CACHE:
        # Do not compile again the PDF source if it has already been done
        module = GPU_PDF_MODULE_CACHE[name]
    else:
        # By default take it from this package
        xml_source = os.path.join(PACKAGE_PATH, f'src/{name}.xml')

        # Check if it exists in any of the provided paths
        for p in PDF_PATHS:
            fp = os.path.join(p, f'{name}.xml')
            if os.path.isfile(fp):
                xml_source = fp
                break

        if not os.path.isfile(xml_source):
            raise RuntimeError(
                f'XML file for function {name} not found in "{PACKAGE_PATH}/src" or any of the provided paths')

        # Write the code
        source = os.path.join(TMPDIR.name, f'{name}.c')
        code = autocode.generate_code(xml_source, core.GPU)
        with open(source, 'wt') as f:
            f.write(code)

        # Compile the code
        with open(source) as fi:
            try:
                module = gpu_core.THREAD.compile(fi.read())
            except Exception as ex:
                nl = len(str(code.count('\n')))
                code = '\n'.join(f'{i + 1:>{nl}}: {l}' for i,
                                 l in enumerate(code.split('\n')))
                logger.error(f'Error found compiling:\n{code}')
                raise ex

        GPU_PDF_MODULE_CACHE[name] = module

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


def create_cpu_function_proxies(module, ndata_pars, narg_pars=0, nvar_arg_pars=None):
    '''
    Create proxies for C++ that handle correctly the input and output data
    types of a function.

    :param module: module where to load the functions from.
    :type module: module
    :param ndata_pars: number of data parameters.
    :type ndata_pars: int
    :param narg_pars: number of fixed argument parameters.
    :type narg_pars: int
    :param nvar_arg_pars: number of variable argument parameters.
    :type nvar_arg_pars: int
    :returns: proxies for the function, the array-like function and normalization.
    :rtype: function, function, function
    '''
    # Get the functions
    function = module.function
    evaluate = module.evaluate

    if hasattr(module, 'evaluate_binned'):
        evaluate_binned = module.evaluate_binned
    else:
        evaluate_binned = None

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
            if nvar_arg_pars != 0:
                var_arg_pars = types.cpu_type(args[-nvar_arg_pars:])
                pos_args = tuple(
                    map(types.c_double, args[ndata_pars:-nvar_arg_pars]))
            else:
                var_arg_pars = types.cpu_type([])
                pos_args = tuple(map(types.c_double, args[ndata_pars:]))

            vals = pos_args + (types.c_int(nvar_arg_pars),
                               var_arg_pars.ctypes.data_as(types.c_double_p))
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

            if nvar_arg_pars != 0:
                var_arg_pars = types.cpu_type(args[-nvar_arg_pars:])
                pos_args = tuple(
                    map(types.c_double, args[ndata_pars:-nvar_arg_pars]))
            else:
                var_arg_pars = types.cpu_type([])
                pos_args = tuple(map(types.c_double, args[ndata_pars:]))

            vals = pos_args + (types.c_int(nvar_arg_pars),
                               var_arg_pars.ctypes.data_as(types.c_double_p))

        else:
            vals = tuple(map(types.c_double, args[ndata_pars:]))

        return evaluate(types.c_int(len(output_array)), op, *ips, *vals)

    if normalization is not None and evaluate_binned is None:
        logger.warning(
            'If you are able to define a normalization function you can also provide an evaluation for binned data sets.')

    if evaluate_binned is not None:

        argtypes = [types.c_int, types.c_double_p]
        argtypes += partypes
        argtypes += [types.c_int for _ in range(2 * ndata_pars)]
        argtypes += [types.c_double_p for _ in range(ndata_pars)]
        evaluate_binned.argtypes = argtypes

        @functools.wraps(evaluate_binned)
        def __evaluate_binned(output_array, *args):
            '''
            Internal wrapper.
            '''
            op = output_array.ctypes.data_as(types.c_double_p)

            bs = 3 * ndata_pars

            if nvar_arg_pars is not None:

                var_arg_pars = types.cpu_type(
                    args[-bs - nvar_arg_pars: -bs])

                # Normal arguments are parsed first
                vals = tuple(
                    map(types.c_double, args[:-nvar_arg_pars - bs]))
                # Variable number of arguments follow
                vals += (types.c_int(nvar_arg_pars),
                         var_arg_pars.ctypes.data_as(types.c_double_p))
                # Add the number of edges and gap values
                vals += tuple(map(types.c_int, args[-bs:-ndata_pars]))
                # Finally, the integration limits must be specified
                vals += tuple(a.ctypes.data_as(types.c_double_p)
                              for a in args[-ndata_pars:])

            else:
                vals = tuple(map(types.c_double, args[:-bs])) + tuple(
                    map(types.c_int, args[-bs:-ndata_pars])) + tuple(
                    a.ctypes.data_as(types.c_double_p) for a in args[-ndata_pars:])

            return evaluate_binned(types.c_int(len(output_array)), op, *vals)
    else:
        __evaluate_binned = None

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

                bs = 2 * ndata_pars

                var_arg_pars = types.cpu_type(args[-bs - nvar_arg_pars: -bs])

                # Normal arguments are parse first
                vals = tuple(map(types.c_double, args[:-nvar_arg_pars - bs]))
                # Variable number of arguments follow
                vals += (types.c_int(nvar_arg_pars),
                         var_arg_pars.ctypes.data_as(types.c_double_p))
                # Finally, the integration limits must be specified
                vals += tuple(map(types.c_double, args[-bs:]))
            else:
                vals = tuple(map(types.c_double, args))

            return normalization(*vals)
    else:
        __normalization = None

    return __function, __evaluate, __evaluate_binned, __normalization


def create_gpu_function_proxy(module, ndata_pars, narg_pars, nvar_arg_pars):
    '''
    Creates a proxy for a function writen in GPU.

    :param module: module containing the function to wrap.
    :type module: module
    :param ndata_pars: number of data parameters.
    :type ndata_pars: int
    :param narg_pars: number of fixed argument parameters.
    :type narg_pars: int
    :param nvar_arg_pars: number of variable argument parameters.
    :type nvar_arg_pars: int
    :returns: proxy for the array-like function.
    :rtype: function
    '''
    # Access the function in the module
    evaluate = module.evaluate

    try:
        # Can not use "hasattr"
        evaluate_binned = module.evaluate_binned
    except:
        evaluate_binned = None

    @functools.wraps(evaluate)
    def __evaluate(output_array, *args):
        '''
        Internal wrapper.
        '''
        ips = args[:ndata_pars]

        if nvar_arg_pars is not None:

            # Variable arguments must be the last in the list
            if nvar_arg_pars != 0:
                var_arg_pars = types.cpu_type(args[-nvar_arg_pars:])
                pos_args = tuple(
                    map(types.cpu_type, args[ndata_pars:-nvar_arg_pars]))
            else:
                # This is necessary because arrays with zero size are not allowed
                # in reikna
                var_arg_pars = types.cpu_type([0])
                pos_args = tuple(map(types.cpu_type, args[ndata_pars:]))

            vals = pos_args + (types.cpu_int(nvar_arg_pars),
                               core.aop.data_array(var_arg_pars))
        else:
            vals = tuple(map(types.cpu_type, args[ndata_pars:]))

        global_size, local_size = gpu_core.get_sizes(len(output_array))

        return evaluate(output_array, *ips, *vals, global_size=global_size, local_size=local_size)

    if evaluate_binned is not None:

        @functools.wraps(evaluate_binned)
        def __evaluate_binned(output_array, *args):
            '''
            Internal wrapper.
            '''
            bs = 3 * ndata_pars

            if nvar_arg_pars is not None:

                var_arg_pars = types.cpu_type(
                    args[-bs - nvar_arg_pars: -bs])

                # Normal arguments are parse first
                vals = tuple(
                    map(types.cpu_type, args[:-nvar_arg_pars - bs]))
                # Variable number of arguments follow
                vals += (types.cpu_int(nvar_arg_pars),
                         core.aop.data_array(var_arg_pars))
                # Add the number of edges and gap values
                vals += tuple(map(types.c_int, args[-bs:-ndata_pars]))
                # Finally, the integration limits must be specified
                vals += args[-ndata_pars:]

            else:
                vals = tuple(map(
                    types.cpu_type, args[:-bs])) + tuple(map(
                        types.cpu_int, args[-bs:-ndata_pars])) + args[-ndata_pars:]

            global_size, local_size = gpu_core.get_sizes(len(output_array))

            return evaluate_binned(output_array, *vals, global_size=global_size, local_size=local_size)
    else:
        __evaluate_binned = None

    return __evaluate, __evaluate_binned


@core.with_backend
def access_pdf(name, ndata_pars, narg_pars=0, nvar_arg_pars=None):
    '''
    Access a PDF with the given name and number of data and parameter arguments.

    :param name: name of the PDF.
    :type name: str
    :param ndata_pars: number of data parameters.
    :type ndata_pars: int
    :param narg_pars: number of fixed argument parameters.
    :type narg_pars: int
    :param nvar_arg_pars: number of variable argument parameters.
    :type nvar_arg_pars: int
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
                cpu_module, ndata_pars, narg_pars, nvar_arg_pars)
            CPU_PDF_CACHE[modname] = cpu_output

    except AttributeError as ex:
        raise RuntimeError(
            f'Error loading function "{name}"; make sure that at least "evaluate" and "function" are defined inside the source file')

    if core.BACKEND == core.CPU:
        # Return the CPU version directly
        return cpu_output

    elif core.BACKEND == core.CUDA or core.BACKEND == core.OPENCL:

        # Function and normalization are taken from the C++ version
        function, _, _, normalization = cpu_output

        # Get the "evaluate" function from source
        if modname in GPU_PDF_CACHE:
            evaluate, evaluate_binned = GPU_PDF_CACHE[modname]
        else:
            # Access the GPU module
            gpu_module = access_gpu_module(name)

            evaluate, evaluate_binned = create_gpu_function_proxy(
                gpu_module, ndata_pars, narg_pars, nvar_arg_pars)
            GPU_PDF_CACHE[modname] = evaluate, evaluate_binned

        return function, evaluate, evaluate_binned, normalization
    else:
        raise NotImplementedError(
            f'Function not implemented for backend "{core.BACKEND}"')
