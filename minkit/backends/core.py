########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Accessors to PDF in code.
'''
from . import PACKAGE_PATH

import collections
import os

__all__ = ['add_pdf_src']

# CPU backend
CPU = 'cpu'
# OpenCL backend
OPENCL = 'opencl'
# CUDA backend
CUDA = 'cuda'
# Internl use only
GPU = 'gpu'

# Define the default object to do array operations
DEFAULT_AOP = None

# Path to directories where to search for PDFs
PDF_PATHS = [os.path.join(PACKAGE_PATH, 'src', 'xml')]

# This class is meant to store run-time compiled functions
FunctionsProxy = collections.namedtuple('FunctionsProxy', [
                                        'function', 'integral', 'evaluate', 'evaluate_binned', 'evaluate_binned_numerical', 'numerical_integral'])


def add_pdf_src(path):
    '''
    This function adds a new path where to look for user-defined PDFs.
    PDFs are searched for in paths opposite to the order they are appended
    (PDFs are taken from the last appended paths first).

    :param path: new path to be considered.
    :type path: str
    '''
    PDF_PATHS.insert(0, path)


def document_operations_method(method):
    '''
    Build the documentation of the given method.
    '''
    method.__doc__ = f'''
This function resolves the :method:`ArrayOperations.{method.__name__}` method
in this backend.
'''
    return method


def common_backend(objects):
    '''
    Check whether the input objets have the same backend and return it.

    :param objects: objects with the "backend" attribute defined.
    :type objects: list
    :returns: common backend.
    :rtype: Backend
    :raises RuntimeError: If the objects do not have a common backend.
    '''
    b = objects[0].backend
    if any(not o.backend is b for o in objects):
        raise RuntimeError('The provided objects do not have the same backend')
    return b


def get_pdf_src():
    '''
    Get the list of paths to search for PDFs.

    :returns: list of directories.
    :rtype: list(str)
    '''
    return PDF_PATHS


def is_gpu_backend(backend):
    '''
    Return whether the backend is of GPU type.

    :param backend: backend type.
    :type backend: str
    :returns: whether the backend is of GPU type or not.
    :rtype: bool
    '''
    return backend is not None and (backend.lower() != CPU)


def parse_backend(backend=None):
    '''
    Function to parse a backend an return the object to do array operations.

    :param backend: input backend.
    :type backend: Backend or None
    :returns: object to do array operations.
    :rtype: CPUOperations or GPUOperations
    '''
    if backend is None:
        if DEFAULT_AOP is None:
            raise RuntimeError(
                'Default object to do operations on arrays is not defined, and it should be automatically done; please report the bug')
        return DEFAULT_AOP
    else:
        return backend.aop


def parse_module_name(name, nvar_arg_pars=None):
    '''
    Parse the module name for a given number of variable arguments.

    :param name: name of the module.
    :type name: str
    :param nvar_arg_pars: number of variable arguments.
    :type nvar_arg_pars: int
    :returns: name of the module.
    :rtype: str
    '''
    return name if nvar_arg_pars is None else f'{name}{nvar_arg_pars}'


def set_default_aop(aop):
    '''
    Set the default object to do operations on arrays.

    :param aop: object to do operations on arrays.
    :type aop: CPUOperations or GPUOperations
    '''
    global DEFAULT_AOP
    DEFAULT_AOP = aop
