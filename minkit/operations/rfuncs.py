'''
Definition of functions to reduce 1D arrays to a single element.

NOTE: All functions in this module accept a single type of value.
'''
import numpy as np
import reikna
from .gpu_core import THREAD
from . import types


def create_reduce_function(function, arr, default):
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
    return reikna.algorithms.Reduce(arr, predicate).compile(THREAD)


def declare_reduce_function(function_proxy, default):
    '''
    Return a decorator to create a :class:`reikna.algorithms.Reduce` object
    to apply a reduction of an array to a single value.

    :param function_proxy: function to pass to :class:`reikna.cluda.Snippet`.
    :type function_proxy: function
    '''
    cache = {}

    def __wrapper(arr):

        callobj = cache.get(arr.shape, None)

        if callobj is None:
            callobj = create_reduce_function(function_proxy, arr, default)
            cache[arr.shape] = callobj

        result = THREAD.array((1,), dtype=arr.dtype)

        callobj(result, arr)

        return result.get().item()

    return __wrapper


# The declaration of functions starts here
amax = declare_reduce_function(
    lambda f, s: 'return ${f} > ${s} ? ${f} : ${s};', default=np.finfo(types.cpu_type).min)
amin = declare_reduce_function(
    lambda f, s: 'return ${f} < ${s} ? ${f} : ${s};', default=np.finfo(types.cpu_type).max)
rsum = declare_reduce_function(lambda f, s: 'return ${f} + ${s};', default=0)
count_nonzero = declare_reduce_function(
    lambda f, s: 'return ${f} + ${s};', default=0)
