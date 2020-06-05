########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Objects to manage the cache for GPU implementations.
'''
from . import arrays
from . import gpu_core
from ..base import data_types
from ..base import exceptions
from ..base.core import DocMeta

from reikna.fft import FFT

from string import Template

import contextlib
import os
import reikna
import sys


class FunctionCache(metaclass=DocMeta):

    def __init__(self, thread):
        '''
        Base class for function cache objects.

        :param thread: thread to compile the function.
        '''
        super().__init__()

        self.__cache = None
        self._thread = thread

    @contextlib.contextmanager
    def activate(self):
        '''
        Activate this cache.
        '''
        self.__cache = {}
        yield self
        self.__cache = None  # free the cache

    def build_object(self, arg):
        '''
        Build an object with the argument used as a key for the cache.

        :param size: size associated to the object.
        :type size: int or tuple
        :returns: object related to this cache.
        '''
        raise exceptions.MethodNotDefinedError(self.__class__, 'build_object')

    def get_object(self, identifier, build_arg=None):
        '''
        Get an object in the cache. The argument *identifier* is used to access the
        cache, so it must be *hashable*.

        :param identifier: argument defining the operation to do.
        :param build_arg: if provided, this argument will be passed to \
        :meth:`FunctionCache.build_object` instead of *identifier*.
        :returns: object related to this cache.
        '''
        if self.__cache is not None:

            obj = self.__cache.get(identifier, None)

            if obj is None:
                obj = self.build_object(
                    build_arg if build_arg is not None else identifier)

            self.__cache[identifier] = obj
        else:
            obj = self.build_object(
                build_arg if build_arg is not None else identifier)

        return obj


class FFTCache(FunctionCache):

    def __init__(self, thread):
        '''
        Cache for FFT functions.

        :param thread: thread to compile the FFT.
        '''
        super().__init__(thread)

    def build_object(self, arr):
        f = FFT(arr)
        return f.compile(self._thread)

    def __call__(self, oa, ia, inverse=False):
        '''
        Build the FFT in the output array, for a given input array.

        :param oa: output array.
        :type opa: carray
        :param ia: input array.
        :type ia: carray
        :param inverse: whether to calculate the inverse FFT.
        :type inverse: bool
        :returns: FFT or inverse FFT.
        :rtype: carray
        '''
        self.get_object(ia.length, ia)(oa.ua, ia.ua, inverse=inverse)
        return oa


class TemplateFunctionCache1D(FunctionCache):

    with open(os.path.join(gpu_core.GPU_SRC, 'templates_1d.c')) as f:
        template = Template(f.read())

    def __init__(self, thread):
        '''
        Functions in one dimension with a fixed number of threads per block.

        :param thread: thread to compile the FFT.
        '''
        super().__init__(thread)

    def build_object(self, size):
        code = TemplateFunctionCache1D.template.substitute(
            threads_per_block=size)
        return self._thread.compile(code)


class TemplateFunctionCache2D(FunctionCache):

    with open(os.path.join(gpu_core.GPU_SRC, 'templates_2d.c')) as f:
        template = Template(f.read())

    def __init__(self, thread):
        '''
        Functions in two dimensions with a fixed number of threads per block.

        :param thread: thread to compile the FFT.
        '''
        super().__init__(thread)

    def build_object(self, sizes):
        lsx, lsy = sizes
        code = TemplateFunctionCache2D.template.substitute(threads_per_block_x=lsx,
                                                           threads_per_block_y=lsy)
        return self._thread.compile(code)


class ReduceFunctionCache(FunctionCache):

    def __init__(self, thread, function, default):
        '''
        Cache for function doing reduction operations.

        :param thread: thread to compile the function.
        :param function: string with the code to execute.
        :type function: str
        :param default: initial value of the reduction.
        :type default: int or float
        '''
        super().__init__(thread)

        self.__snippet = reikna.cluda.Snippet.create(function)
        self.__default = default

    def build_object(self, arr):
        predicate = reikna.algorithms.Predicate(self.__snippet, self.__default)
        return reikna.algorithms.Reduce(arr, predicate).compile(self._thread)

    def __call__(self, arr):
        callobj = self.get_object(len(arr), arr)
        result = self._thread.array((1,), dtype=arr.dtype)
        callobj(result, arr.ua)
        return result.get().item()


class ArrayCacheManager(object):

    def __init__(self, backend, thread, dtype):
        '''
        Object that keeps array in the GPU device in order to avoid creating
        and destroying them many times, and calls functions with them.

        :param dtype: data type of the output arrays.
        :type dtype: numpy.dtype
        '''
        super().__init__()

        self._backend = backend
        self._cache = None
        self._dtype = dtype
        self._thread = thread

        if dtype == data_types.cpu_float:
            self._array_builder = arrays.darray
        elif dtype == data_types.cpu_int:
            self._array_builder = arrays.iarray
        elif dtype == data_types.cpu_bool:
            self._array_builder = arrays.barray
        elif dtype == data_types.cpu_complex:
            self._array_builder = arrays.carray
        else:
            raise ValueError(
                f'Unknown data type "{dtype}"; please report the bug')

    @contextlib.contextmanager
    def activate(self):
        '''
        Activate this cache.
        '''
        self._cache = {}
        yield self
        self._cache = None  # free the cache

    def get_array(self, size):
        '''
        Get the array with size "size" from the cache, if it exists.

        :param size: size of the output array.
        :type size: int
        '''
        if size == 0:
            return self._array_builder(None, 0, self._backend)  # empty array

        if self._cache is not None:

            elements = self._cache.get(size, None)

            if elements is not None:
                for el in elements:
                    # This class is the only one that owns it, together with "elements" and "el"
                    if sys.getrefcount(el) == 3:
                        return self._array_builder(el, size, self._backend)
            else:
                self._cache[size] = []

        out = self._thread.array((size,), dtype=self._dtype)

        if self._cache is not None:
            self._cache[size].append(out)

        return self._array_builder(out, size, self._backend)


class FloatArrayCacheManager(ArrayCacheManager):

    def __init__(self, backend, thread, dtype):
        '''
        Similarly to :class:`ArrayCacheManager` allows to manage GPU arrays
        in a cache, but it correctly treats the number of dimensions for
        floating point arrays.

        :param dtype: data type of the output arrays.
        :type dtype: numpy.dtype
        '''
        super().__init__(backend, thread, dtype)

    def get_array(self, size, ndim=1):
        '''
        Get the array with size "size" from the cache, if it exists.

        :param size: size of the output array.
        :type size: int
        '''
        if size == 0:
            # empty array
            return self._array_builder(None, 0, 0, self._backend)

        true_size = size * ndim

        if self._cache is not None:

            elements = self._cache.get(true_size, None)

            if elements is not None:
                for el in elements:
                    # This class is the only one that owns it, together with "elements" and "el"
                    if sys.getrefcount(el) == 3:
                        return self._array_builder(el, ndim, size, self._backend)
            else:
                self._cache[true_size] = []

        out = self._thread.array((true_size,), dtype=self._dtype)

        if self._cache is not None:
            self._cache[true_size].append(out)

        return self._array_builder(out, ndim, size, self._backend)
