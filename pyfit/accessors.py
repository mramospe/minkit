'''
Accessors to PDF in code.
'''
from . import PACKAGE_PATH
from . import core
from . import parameters
from . import types
from distutils import ccompiler

from ctypes import cdll
import functools
import numpy as np
import os
import tempfile

__all__ = []

TMPDIR = tempfile.TemporaryDirectory()

PDF_MODULE_CACHE = {}

PDF_CACHE = {}


def create_cpp_function_proxy( module, name, ndata_pars, narg_pars = 0, nvar_arg_pars = None ):
    '''
    Create a proxy for C++ that handles correctly the input and output data
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
    def __function( *args ):
        '''
        Internal wrapper.
        '''
        if nvar_arg_pars is not None:
            var_arg_pars = args[-1]
            vals = tuple(map(types.c_double, args[:-1])) + (nvar_arg_pars, var_arg_pars.ctypes.data_as(types.c_double_p))
        else:
            vals = tuple(map(types.c_double, args))

        return function(*vals)

    # Define the types for the arguments passed to the evaluate function
    argtypes = [types.c_int, types.c_double_p]
    argtypes += [types.c_double_p for _ in range(ndata_pars)]
    argtypes += partypes
    evaluate.argtypes = argtypes
    
    @functools.wraps(evaluate)
    def __evaluate( output_array, *args ):
        '''
        Internal wrapper.
        '''
        op   = output_array.ctypes.data_as(types.c_double_p)
        ips  = tuple(d.ctypes.data_as(types.c_double_p) for d in args[:ndata_pars])

        if nvar_arg_pars is not None:

            # Variable arguments must be the last in the list
            var_arg_pars = args[-1]

            vals = tuple(map(types.c_double, args[ndata_pars:-1])) + (nvar_arg_pars, var_arg_pars.ctypes.data_as(types.c_double_p))

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
        def __normalization( *args ):
            '''
            Internal wrapper.
            '''
            if nvar_arg_pars is not None:
                if nvar_arg_pars == 0:
                    # This is special case, must handle index carefully
                    var_arg_pars = np.array([], dtype=types.c_double)
                else:
                    var_arg_pars = args[-1 -2 * ndata_pars]
                # Normal arguments are parse first
                vals = tuple(map(types.c_double, args[:-1 -2 * ndata_pars]))
                # Variable number of arguments follow
                vals += (types.c_int(nvar_arg_pars), var_arg_pars.ctypes.data_as(types.c_double_p))
                # Finally, the integration limits must be specified
                vals += tuple(map(types.c_double, args[-2 * ndata_pars:]))
            else:
                vals = tuple(map(types.c_double, args))

            return normalization(*vals)
    else:
        __normalization = None

    return __function, __evaluate, __normalization


def access_pdf( name, ndata_pars, narg_pars = 0, nvar_arg_pars = None ):
    '''
    Access a PDF with the given name and number of data and parameter arguments.

    :returns: PDF and normalization function.
    :rtype: tuple(function, function)
    '''
    global PDF_MODULE_CACHE
    global PDF_CACHE

    if core.BACKEND == core.CPU:

        source = os.path.join(PACKAGE_PATH, f'cpp/{name}.cpp')

        if name in PDF_MODULE_CACHE:
            # Do not compile again the PDF source if it has already been done
            module = PDF_MODULE_CACHE[name]
        else:
            # Compile the C++ code and load the library
            if not os.path.isfile(source):
                raise RuntimeError(f'Function {name} does not exist in "{source}"')

            compiler = ccompiler.new_compiler()
            objects  = compiler.compile([source], output_dir=TMPDIR.name)
            libname  = os.path.join(TMPDIR.name, f'lib{name}.so')
            compiler.link(f'{name} library', objects, libname)

            module = cdll.LoadLibrary(libname)

            PDF_MODULE_CACHE[name] = module

        # Get the functions
        try:
            modname = name if nvar_arg_pars is not None else f'{name}{nvar_arg_pars}'

            if modname in PDF_CACHE:
                output = PDF_CACHE[modname]
            else:
                output = create_cpp_function_proxy(module, name, ndata_pars, narg_pars, nvar_arg_pars)
                PDF_CACHE[modname] = output

        except AttributeError as ex:
            raise RuntimeError(f'Error loading function "{name}"; make sure that both "{name}" and "normalization" are defined inside "{source}"')
    else:
        raise NotImplementedError(f'Function not implemented for backend "{core.BACKEND}"')

    return output
