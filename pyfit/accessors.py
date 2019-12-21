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
import os
import tempfile

__all__ = []

TMPDIR = tempfile.TemporaryDirectory()


def create_cpp_function_proxy( module, name, ndata_pars, narg_pars ):
    '''
    Create a proxy for C++ that handles correctly the input and output data
    types of a fucntion.
    '''
    # Get the functions
    function      = getattr(module, name)
    normalization = getattr(module, 'normalization')

    # Define the types of the input arguments
    partypes = [types.c_double for _ in range(narg_pars)]

    # Define the types for the arguments passed to the function
    argtypes = [types.c_int, types.c_double_p]
    argtypes += [types.c_double_p for _ in range(ndata_pars)]
    argtypes += partypes
    function.argtypes = argtypes

    # Define the types passed to the normalization function
    normtypes = partypes
    normtypes += [types.c_double for _ in range(2 * ndata_pars)]
    normalization.argtypes = normtypes
    normalization.restype = types.c_double

    @functools.wraps(function)
    def __function( output_array, *args ):
        '''
        Internal wrapper.
        '''
        op   = output_array.ctypes.data_as(types.c_double_p)
        ips  = tuple(d.ctypes.data_as(types.c_double_p) for d in args[:ndata_pars])
        vals = tuple(map(types.c_double, args[ndata_pars:]))
        return function(types.c_int(len(output_array)), op, *ips, *vals)

    @functools.wraps(normalization)
    def __normalization( *args ):
        '''
        Internal wrapper.
        '''
        return normalization(*tuple(map(types.c_double, args)))

    return __function, __normalization


def access_pdf( name, ndata_pars, narg_pars ):
    '''
    Access a PDF with the given name and number of data and parameter arguments.

    :returns: PDF and normalization function.
    :rtype: tuple(function, function)
    '''
    if core.BACKEND == core.CPU:

        # Compile the C++ code and load the library
        source = os.path.join(PACKAGE_PATH, f'cpp/{name}.cpp')
        if not os.path.isfile(source):
            raise RuntimeError(f'Function {name} does not exist in "{source}"')

        compiler = ccompiler.new_compiler()
        objects  = compiler.compile([source], output_dir=TMPDIR.name)
        libname  = os.path.join(TMPDIR.name, f'lib{name}.so')
        compiler.link(f'{name} library', objects, libname)

        module = cdll.LoadLibrary(libname)

        # Get the functions
        try:
            function, normalization = create_cpp_function_proxy(module, name, ndata_pars, narg_pars)
        except AttributeError as ex:
            raise RuntimeError(f'Error loading function "{name}"; make sure that both "{name}" and "normalization" are defined inside "{source}"')
    else:
        raise NotImplementedError(f'Function not implemented for backend "{core.BACKEND}"')

    return function, normalization
