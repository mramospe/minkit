'''
Definition of functions for GPUs.

NOTE: All functions in this module accept a single type of value.
'''
from . import PACKAGE_PATH
from .gpu_core import get_sizes
from ..base import data_types

import collections
import functools
import numpy as np
import os
import reikna

GPU_SRC = os.path.join(PACKAGE_PATH, 'src', 'gpu')

ReduceFunctionsProxy = collections.namedtuple(
    'ReduceFunctionsProxy', ['amax', 'amin', 'rsum', 'count_nonzero'])


def creating_array_dtype(ops_mgr, dtype):
    '''
    Wrap a function automatically creating an output array with the
    same shape as that of the input but with possible different data type.
    '''
    cache_mgr = ops_mgr.get_array_cache(dtype)

    def __wrapper(function):
        @functools.wraps(function)
        def __wrapper(size, *args, **kwargs):
            gs, ls = get_sizes(size)
            out = cache_mgr.get_array(size)
            function(out, *args, global_size=gs, local_size=ls, **kwargs)
            return out
        return __wrapper
    return __wrapper


def return_dtype(ops_mgr, dtype):
    '''
    Wrap a function automatically creating an output array with the
    same shape as that of the input but with possible different data type.
    '''
    cache_mgr = ops_mgr.get_array_cache(dtype)

    def __wrapper(function):
        @functools.wraps(function)
        def __wrapper(arr, *args, **kwargs):
            gs, ls = get_sizes(len(arr))
            out = cache_mgr.get_array(len(arr))
            function(out, arr, *args, global_size=gs, local_size=ls, **kwargs)
            return out
        return __wrapper
    return __wrapper


def call_simple(function):
    '''
    Call a function setting the global size and the local size. The output
    array must be provided to the function as the first argument.
    '''
    @functools.wraps(function)
    def __wrapper(out, *args, **kwargs):
        gs, ls = get_sizes(len(out))
        function(out, *args, global_size=gs, local_size=ls, **kwargs)
        return out
    return __wrapper


def create_reduce_function(ops_mgr, function, arr, default):
    '''
    Create a :class:`reikna.algorithms.Reduce` object with the given function
    implemented.

    :param function: function to parse.
    :type function: function
    :param arr: array to process.
    :type arr: reikna.cluda.Array
    :returns: object that applies the function on a given array.
    :rtype: numpy.float64
    '''
    snippet = reikna.cluda.Snippet.create(function)
    predicate = reikna.algorithms.Predicate(snippet, default)
    return reikna.algorithms.Reduce(arr, predicate).compile(ops_mgr.thread)


def declare_reduce_function(ops_mgr, function_proxy, default):
    '''
    Return a decorator to create a :class:`reikna.algorithms.Reduce` object
    to apply a reduction of an array to a single value.

    :param ops_mgr: operations manager.
    :type ops_mgr: GPUOperations
    :param function_proxy: function to pass to :class:`reikna.cluda.Snippet`.
    :type function_proxy: function
    '''
    cache = {}

    def __wrapper(arr):

        callobj = cache.get(arr.shape, None)

        if callobj is None:
            callobj = create_reduce_function(
                ops_mgr, function_proxy, arr, default)
            cache[arr.shape] = callobj

        result = ops_mgr.thread.array((1,), dtype=arr.dtype)

        callobj(result, arr)

        return result.get().item()

    return __wrapper


def declare_template_functions(ops_mgr):
    '''
    '''
    cache = {}

    def __wrapper(ls):

        functions = cache.get(ls, None)

        if functions is None:
            with open(os.path.join(GPU_SRC, 'templates.c')) as f:
                code = f.read().format(threads_per_block=ls)
                functions = ops_mgr.thread.compile(code)
            cache[ls] = functions

        return functions

    return __wrapper


def make_functions(ops_mgr):
    '''
    Compile the GPU functions for the given operation manager.

    :param ops_mgr: operations manager.
    :type ops_mgr: GPUOperations
    :returns: functions
    :rtype: list, list
    '''
    return_complex = return_dtype(ops_mgr, data_types.cpu_complex)
    return_double = return_dtype(ops_mgr, data_types.cpu_float)
    return_bool = return_dtype(ops_mgr, data_types.cpu_bool)

    # Compile general GPU functions by element.
    with open(os.path.join(GPU_SRC, 'functions_by_element.c')) as fi:
        funcs_by_element = ops_mgr.thread.compile(fi.read())

    with open(os.path.join(GPU_SRC, 'multiparameter.c')) as fi:
        funcs_multipar = ops_mgr.thread.compile(fi.read())

    # These functions take an array of doubles and return another array of doubles
    for function in ('exponential_complex',):
        setattr(funcs_by_element, function, return_complex(
                getattr(funcs_by_element, function)))

    # These functions take an array of doubles and return another array of doubles
    for function in ('exponential_double', 'logarithm', 'real'):
        setattr(funcs_by_element, function, return_double(
                getattr(funcs_by_element, function)))

    # These functions take an array of doubles as an input, and return an array of bool,
    # but the output array is provided by the user.
    for function in ('logical_and', 'logical_or'):
        setattr(funcs_by_element, f'{function}_to_output',
                call_simple(getattr(funcs_by_element, function)))

    # These functions take an array of doubles as an input, and return an array of bool
    for function in ('alt', 'ge', 'lt', 'le', 'logical_and', 'logical_or'):
        setattr(funcs_by_element, function, return_bool(
                getattr(funcs_by_element, function)))

    create_complex = creating_array_dtype(ops_mgr, data_types.cpu_complex)
    create_double = creating_array_dtype(ops_mgr, data_types.cpu_float)
    create_int = creating_array_dtype(ops_mgr, data_types.cpu_int)
    create_bool = creating_array_dtype(ops_mgr, data_types.cpu_bool)

    # These functions create e new array of complex numbers
    for function in ('arange_complex',):
        setattr(funcs_by_element, function, create_complex(
                getattr(funcs_by_element, function)))

    # These functions create e new array of doubles
    for function in ('interpolate', 'linspace', 'ones_double', 'zeros_double'):
        setattr(funcs_by_element, function, create_double(
                getattr(funcs_by_element, function)))

    # These functions create e new array of integers
    for function in ('arange_int', 'invalid_indices'):
        setattr(funcs_by_element, function, create_int(
                getattr(funcs_by_element, function)))

    # These functions create e new array of bool
    for function in ('ones_bool', 'zeros_bool'):
        setattr(funcs_by_element, function, create_bool(
                getattr(funcs_by_element, function)))

    # Reduce functions
    amax = declare_reduce_function(ops_mgr,
                                   lambda f, s: 'return ${f} > ${s} ? ${f} : ${s};', default=np.finfo(data_types.cpu_float).min)
    amin = declare_reduce_function(ops_mgr,
                                   lambda f, s: 'return ${f} < ${s} ? ${f} : ${s};', default=np.finfo(data_types.cpu_float).max)
    rsum = declare_reduce_function(
        ops_mgr, lambda f, s: 'return ${f} + ${s};', default=0)
    count_nonzero = declare_reduce_function(ops_mgr,
                                            lambda f, s: 'return ${f} + ${s};', default=0)

    reduce_functions = ReduceFunctionsProxy(amax, amin, rsum, count_nonzero)

    # Templated functions
    templated_functions = declare_template_functions(ops_mgr)

    return funcs_by_element, funcs_multipar, reduce_functions, templated_functions
