'''
Application interface to deal with arrays. The different array classes
can not define the __getitem__ since numpy objects would use it to do
arithmetic operations.
'''
from . import core
from ..base import data_types

import functools
import numpy as np


__all__ = ['array_operations', 'marray', 'barray', 'farray', 'iarray']


def arithmetic_operation(method):
    @functools.wraps(method)
    def wrapper(self, other):
        if np.array(other).dtype == np.dtype(object):
            return method(self, other.ua)
        else:
            return method(self, other)
    return wrapper


def comparison_operation(method):
    @functools.wraps(method)
    def wrapper(self, a, b):

        if np.array(a).dtype == np.dtype(object):
            a = a.ua

        if np.array(b).dtype == np.dtype(object):
            b = b.ua

        return method(self, a, b)

    return wrapper


def return_barray(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return barray(method(self, *args, **kwargs), self.backend)
    return wrapper


def return_farray(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return farray(method(self, *args, **kwargs), self.backend)
    return wrapper


def return_iarray(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return iarray(method(self, *args, **kwargs), self.backend)
    return wrapper


class array_operations(object):

    def __init__(self, backend, **kwargs):
        '''
        Build the object to do operations within a backend. Only the necessary
        operators have been defined.

        :param btype: backend type ('cpu', 'cuda', 'opencl').
        :type btype: str
        :param kwargs: arguments forwarded to the backend constructor \
        (cuda and opencl backends only). It can contain any of the following \
        keys: \
        * device: \
        * interactive: \
        :type kwargs: dict
        '''
        super(array_operations, self).__init__()

        self.__backend = backend

        if self.__backend.btype == core.CPU:
            from .cpu import CPUOperations
            self.__oper = CPUOperations(**kwargs)
        else:
            from .gpu import GPUOperations
            self.__oper = GPUOperations(backend.btype, **kwargs)

    @property
    def backend(self):
        '''
        Backend interface.
        '''
        return self.__backend

    def access_pdf(self, name, ndata_pars, nvar_arg_pars=None):

        fun, ev, evb, inte = self.__oper.access_pdf(
            name, ndata_pars, nvar_arg_pars)

        @functools.wraps(ev)
        def evaluate(output_array, data_idx, input_array, args):
            return ev(output_array.ua, data_idx, input_array.ua, args)

        if evb is not None:
            @functools.wraps(evb)
            def evaluate_binned(output_array, gaps_idx, gaps, edges, args):
                return evb(output_array.ua, gaps_idx, gaps, edges.ua, args)
        else:
            evaluate_binned = None

        return fun, evaluate, evaluate_binned, inte

    @return_iarray
    def arange(self, n):
        return self.__oper.arange(n, dtype=data_types.cpu_int)

    def ndarray_to_barray(self, a):
        return barray.from_ndarray(a, self.__backend)

    def ndarray_to_farray(self, a):
        return farray.from_ndarray(a, self.__backend)

    def ndarray_to_iarray(self, a):
        return iarray.from_ndarray(a, self.__backend)

    def ndarray_to_backend(self, a):
        return self.__oper.ndarray_to_backend(a)

    @return_farray
    def concatenate(self, arrays, maximum=None):
        return self.__oper.concatenate(tuple(a.ua for a in arrays), maximum)

    def count_nonzero(self, a):
        return self.__oper.count_nonzero(a.ua)

    @return_farray
    def fempty(self, l):
        return self.__oper.empty(l, dtype=data_types.cpu_float)

    @return_iarray
    def iempty(self, l):
        return self.__oper.empty(l, dtype=data_types.cpu_int)

    @return_farray
    def exp(self, a):
        return self.__oper.exp(a.ua)

    @return_farray
    def fftconvolve(self, a, b, data):
        return self.__oper.fftconvolve(a.ua, b.ua, data.ua)

    @return_barray
    @comparison_operation
    def ge(self, a, b):
        return self.__oper.ge(a, b)

    @return_farray
    def interpolate_linear(self, idx, x, xp, yp):
        return self.__oper.interpolate_linear(idx, x.ua, xp.ua, yp.ua)

    @property
    def bool_type(self):
        if core.is_gpu_backend(self.__backend.btype):
            return data_types.cpu_bool
        else:
            return data_types.cpu_real_bool

    def is_bool(self, a):
        '''
        '''
        return a.dtype == self.bool_type

    def is_float(self, a):
        '''
        '''
        return a.dtype == data_types.cpu_float

    def is_int(self, a):
        return a.dtype == data_types.cpu_int

    @return_barray
    def is_inside(self, data, lb, ub):
        return self.__oper.is_inside(data.ua, lb, ub)

    @return_barray
    @comparison_operation
    def lt(self, a, b):
        return self.__oper.lt(a, b)

    @return_barray
    @comparison_operation
    def le(self, a, b):
        return self.__oper.le(a, b)

    @return_farray
    def linspace(self, vmin, vmax, size):
        return self.__oper.linspace(vmin, vmax, size)

    @return_farray
    def log(self, a):
        return self.__oper.log(a.ua)

    @return_barray
    def logical_and(self, a, b, out=None):
        if out is not None:
            out = out.ua
        return self.__oper.logical_and(a.ua, b.ua, out)

    @return_barray
    def logical_or(self, a, b, out=None):
        if out is not None:
            out = out.ua
        return self.__oper.logical_or(a.ua, b.ua, out)

    def max(self, a):
        return self.__oper.max(a.ua)

    def meshgrid(self, *arrays):
        return tuple(map(lambda t: farray(t, self.__backend), self.__oper.meshgrid(*tuple(a.ua for a in arrays))))

    def min(self, a):
        return self.__oper.min(a.ua)

    @return_barray
    def bones(self, size):
        return self.__oper.ones(size, dtype=self.bool_type)

    @return_farray
    def fones(self, size):
        return self.__oper.ones(size, dtype=data_types.cpu_float)

    @return_farray
    def random_uniform(self, vmin, vmax, size):
        return self.__oper.random_uniform(vmin, vmax, size)

    @return_farray
    def restrict_data_size(self, maximum, ndim, len, data):
        return self.__oper.restrict_data_size(maximum, ndim, len, data.ua)

    @return_iarray
    def shuffling_index(self, n):
        return self.__oper.shuffling_index(n)

    def sum(self, a):
        return self.__oper.sum(a.ua)

    def sum_arrays(self, arrays):
        out = self.fzeros(len(arrays[0]))
        for a in arrays:
            out += a
        return out

    @return_farray
    def sum_inside(self, indices, gaps, centers, edges, values=None):
        if values is not None:
            values = values.ua
        return self.__oper.sum_inside(indices, gaps, centers.ua, edges.ua, values)

    @return_farray
    def slice_from_boolean(self, a, v, dim=1):
        return self.__oper.slice_from_boolean(a.ua, v.ua, dim)

    @return_farray
    def slice_from_integer(self, a, i, dim=1):
        return self.__oper.slice_from_integer(a.ua, i.ua, dim)

    @return_barray
    def bzeros(self, l):
        return self.__oper.zeros(l, dtype=self.bool_type)

    @return_farray
    def fzeros(self, l):
        return self.__oper.zeros(l, dtype=data_types.cpu_float)


class marray(object):

    def __init__(self, array, backend=None):
        '''
        Wrapper over the arrays to do operations in CPU or GPU devices.

        :param array: original array.
        :type array: numpy.ndarray or reikna.cluda.Array
        :param backend: backend where to put the array.
        :tye backend: Backend
        '''
        super(marray, self).__init__()
        self.__aop = core.parse_backend(backend)
        self.__array = array

    def __len__(self):
        '''
        Get the length of the array.

        :returns: Length of the array.
        :rtype: int
        '''
        return len(self.__array)

    @property
    def aop(self):
        '''
        Associated object to do array operations.
        '''
        return self.__aop

    @property
    def backend(self):
        '''
        Backend interface.
        '''
        return self.__aop.backend

    @property
    def dtype(self):
        '''
        Data type.
        '''
        return self.__array.dtype

    @property
    def ua(self):
        '''
        Underlying array.
        '''
        return self.__array

    @ua.setter
    def ua(self, a):
        self.__array = a

    def as_ndarray(self):

        if core.is_gpu_backend(self.__aop.backend.btype):
            return self.__array.get()
        else:
            return self.__array

    def get(self, i):
        if core.is_gpu_backend(self.__aop.backend.btype):
            return self.__array[i].get()
        else:
            return self.__array[i]

    def take_each(self, n, start=0):
        return self.__class__(self.__array[start::n], self.__aop.backend)

    def take_slice(self, start=0, end=None):
        end = end if end is not None else len(self)
        return self.__class__(self.__array[start:end])

    def to_backend(self, backend):
        '''
        Send the array to the given backend.

        :param backend: backend where to transfer the array.
        :type backend: Backend
        '''
        if self.__aop.backend is backend:
            return self
        else:
            if core.is_gpu_backend(self.__aop.backend.btype):
                a = backend.aop.ndarray_to_backend(self.__array.get())
            else:
                a = backend.aop.ndarray_to_backend(self.__array)
        return self.__class__(a, backend)


class barray(marray):

    def __init__(self, array, backend=None):
        '''
        Array of booleans.

        :param array: original array.
        :type array: numpy.ndarray or reikna.cluda.Array
        :param backend: backend where to put the array.
        :tye backend: Backend
        '''
        super(barray, self).__init__(array, backend)
        assert self.aop.is_bool(self)

    def __and__(self, other):
        return self.aop.logical_and(self, other)

    def __iand__(self, other):
        return self.aop.logical_and(self, other, out=self)

    def __or__(self, other):
        return self.aop.logical_or(self, other)

    def __ior__(self, other):
        return self.aop.logical_or(self, other, out=self)

    @classmethod
    def from_ndarray(cls, a, backend):
        if not backend.aop.is_bool(a):
            a = a.astype(backend.aop.bool_type)
        return cls(backend.aop.ndarray_to_backend(a), backend)

    def count_nonzero(self):
        '''
        Count the number of elements that are different from zero.

        :returns: number of elements different from zero.
        :rtype: int
        '''
        return self.aop.count_nonzero(self)


class farray(marray):

    def __init__(self, array, backend=None):
        '''
        Array of floats.

        :param array: original array.
        :type array: numpy.ndarray or reikna.cluda.Array
        :param backend: backend where to put the array.
        :tye backend: Backend
        '''
        super(farray, self).__init__(array, backend)
        assert self.aop.is_float(self)

    @classmethod
    def from_ndarray(cls, a, backend):
        if not backend.aop.is_float(a):
            a = a.astype(data_types.cpu_float)
        return cls(backend.aop.ndarray_to_backend(a), backend)

    @arithmetic_operation
    def __add__(self, other):
        return self.__class__(self.ua + other, self.backend)

    @arithmetic_operation
    def __radd__(self, other):
        return self.__class__(other + self.ua, self.backend)

    @arithmetic_operation
    def __iadd__(self, other):
        self.ua += other
        return self

    @arithmetic_operation
    def __truediv__(self, other):
        return self.__class__(self.ua / other, self.backend)

    @arithmetic_operation
    def __itruediv__(self, other):
        self.ua /= other
        return self

    def __lt__(self, other):
        return self.aop.lt(self, other)

    def __le__(self, other):
        return self.aop.le(self, other)

    def __ge__(self, other):
        return self.aop.ge(self, other)

    def __pow__(self, n):
        return self.__class__(self.ua**n, self.backend)

    @arithmetic_operation
    def __mul__(self, other):
        return self.__class__(self.ua * other, self.backend)

    @arithmetic_operation
    def __rmul__(self, other):
        return self.__class__(other * self.ua, self.backend)

    @arithmetic_operation
    def __imul__(self, other):
        self.ua *= other
        return self

    @arithmetic_operation
    def __sub__(self, other):
        return self.__class__(self.ua - other, self.backend)

    @arithmetic_operation
    def __rsub__(self, other):
        return self.__class__(other - self.ua, self.backend)

    @arithmetic_operation
    def __isub__(self, other):
        self.ua -= other.ua
        return self

    def min(self):
        return self.aop.min(self)

    def max(self):
        return self.aop.max(self)

    def slice(self, a, dim=1):
        if self.aop.is_bool(a):
            return self.aop.slice_from_boolean(self, a, dim)
        elif self.aop.is_int(a):
            return self.aop.slice_from_integer(self, a, dim)
        else:
            raise NotImplementedError(
                f'Method not implemented for data type {a.dtype}')

    def sum(self):
        return self.aop.sum(self)


class iarray(marray):

    def __init__(self, array, backend=None):
        '''
        Array of integers.

        :param array: original array.
        :type array: numpy.ndarray or reikna.cluda.Array
        :param backend: backend where to put the array.
        :tye backend: Backend
        '''
        super(iarray, self).__init__(array, backend)
        assert self.aop.is_int(self)

    @classmethod
    def from_ndarray(cls, a, backend):
        if not backend.aop.is_int(a):
            a = a.astype(data_types.cpu_int)
        return cls(backend.aop.ndarray_to_backend(a), backend)
