########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Definition of functions for GPUs.

NOTE: All functions in this module accept a single type of value.
'''
from . import gpu_cache
from .gpu_core import GPU_SRC
from ..base import data_types

import collections
import functools
import numpy as np
import os

ReduceFunctionsProxy = collections.namedtuple(
    'ReduceFunctionsProxy', ['amax', 'amin', 'rsum', 'count_nonzero'])


def return_dtype(ops_mgr, dtype):
    '''
    Wrap a function automatically creating an output array with the
    same shape as that of the input but with possible different data type.

    :param ops_mgr: object to do operations on arrays.
    :type ops_mgr: ArrayOperations
    :param dtype: data type.
    :type dtype: numpy.dtype
    '''
    cache_mgr = ops_mgr.get_array_cache(dtype)

    def __wrapper(function):
        @functools.wraps(function)
        def __wrapper(arr, *args, **kwargs):
            # for floating-point types only 1D is supported
            out = cache_mgr.get_array(len(arr))
            gs, ls = ops_mgr.context.get_sizes(out.length)
            function(out.length, out.ua, arr.ua, *args,
                     global_size=gs, local_size=ls, **kwargs)
            return out
        return __wrapper
    return __wrapper


def creating_array_dtype(ops_mgr, dtype):
    '''
    Wrap a function automatically creating an output array with the
    same shape as that of the input but with possible different data type.

    :param ops_mgr: object to do operations on arrays.
    :type ops_mgr: ArrayOperations
    :param dtype: data type.
    :type dtype: numpy.dtype
    '''
    cache_mgr = ops_mgr.get_array_cache(dtype)

    def __wrapper(function):
        @functools.wraps(function)
        def __wrapper(size, *args, **kwargs):
            out = cache_mgr.get_array(size)
            gs, ls = ops_mgr.context.get_sizes(out.length)
            function(out.length, out.ua, *args,
                     global_size=gs, local_size=ls, **kwargs)
            return out
        return __wrapper
    return __wrapper


def call_simple(ops_mgr):
    '''
    Call a function setting the global size and the local size. The output
    array must be provided to the function as the first argument.
    '''
    def __wrapper(function):
        @functools.wraps(function)
        def __wrapper(out, *args, **kwargs):
            gs, ls = ops_mgr.context.get_sizes(out.length)
            function(out.length, out.ua, *args,
                     global_size=gs, local_size=ls, **kwargs)
            return out
        return __wrapper
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
        funcs_by_element = ops_mgr.context.compile(fi.read())

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
                call_simple(ops_mgr)(getattr(funcs_by_element, function)))

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
    for function in ('linspace', 'ones_double'):
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
    amax = gpu_cache.ReduceFunctionCache(ops_mgr.context,
                                         lambda f, s: 'return ${f} > ${s} ? ${f} : ${s};', default=np.finfo(data_types.cpu_float).min)
    amin = gpu_cache.ReduceFunctionCache(ops_mgr.context,
                                         lambda f, s: 'return ${f} < ${s} ? ${f} : ${s};', default=np.finfo(data_types.cpu_float).max)
    rsum = gpu_cache.ReduceFunctionCache(
        ops_mgr.context, lambda f, s: 'return ${f} + ${s};', default=0)
    count_nonzero = gpu_cache.ReduceFunctionCache(ops_mgr.context,
                                                  lambda f, s: 'return ${f} + ${s};', default=0)

    reduce_functions = ReduceFunctionsProxy(amax, amin, rsum, count_nonzero)

    # Templated functions
    templated_functions_1d = gpu_cache.TemplateFunctionCache1D(ops_mgr.context)
    templated_functions_2d = gpu_cache.TemplateFunctionCache2D(ops_mgr.context)

    return funcs_by_element, reduce_functions, templated_functions_1d, templated_functions_2d
