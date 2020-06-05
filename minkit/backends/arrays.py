########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Define array types to be used within the package.
'''
from . import core
from ..base import data_types
from ..base import exceptions
from ..base.core import DocMeta

import functools
import numpy as np


__all__ = ['marray', 'barray', 'farray', 'carray', 'darray', 'iarray']


def arithmetic_operation(method):
    '''
    Wrap a method to do arithmetic operations so the underlying array is
    accessed in case the input arguments are :class:`marray` instances.

    :param method: method to wrap.
    :type method: class method
    :returns: decorated function
    :rtype: function
    '''
    @functools.wraps(method)
    def wrapper(self, other):
        if np.asarray(other).dtype == object:
            return method(self, other.ua)
        else:
            return method(self, other)
    return wrapper


def comparison_operation(method):
    '''
    Wrap a method to do comparison operations so the underlying array is
    accessed in case the input arguments are :class:`marray` instances.

    :param method: method to wrap.
    :type method: class method
    :returns: decorated function
    :rtype: function
    '''
    @functools.wraps(method)
    def wrapper(self, a, b):

        if np.asarray(a).dtype == object:
            a = a.ua

        if np.asarray(b).dtype == object:
            b = b.ua

        return method(self, a, b)

    return wrapper


class marray(object, metaclass=DocMeta):

    def __init__(self, array, dtype, length=None, backend=None):
        '''
        Wrapper over the arrays to do operations in CPU or GPU devices.

        :param array: original array.
        :type array: numpy.ndarray or reikna.cluda.api.Array
        :param int: (GPU only) actual size of the array.
        :type length: int
        :param backend: backend where to put the array.
        :type backend: Backend
        '''
        super().__init__()
        self.__aop = core.parse_backend(backend)
        self.__array = array
        self.__dtype = dtype
        self.__length = data_types.cpu_int(
            length if length is not None else len(array))

    def __len__(self):
        '''
        Get the length of the array.

        :returns: Length of the array.
        :rtype: int
        '''
        return self.__length

    def __repr__(self):
        '''
        Attributes of the class as a string.
        '''
        return f'''dtype  = {self.dtype}
length = {self.length}
array  = {self.as_ndarray()}'''

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
        return self.__dtype

    @property
    def length(self):
        '''
        Length as a :class:`numpy.int32` instance.
        '''
        return self.__length

    @property
    def shape(self):
        '''
        Shape of the array.
        '''
        return (self.__length,)

    @property
    def ua(self):
        '''
        Underlying array.
        '''
        return self.__array

    @ua.setter
    def ua(self, a):
        self.__array = a

    @classmethod
    def from_ndarray(cls, a, backend):
        '''
        Create this class from a :class:`numpy.ndarray` instance.

        :param a: array.
        :type a: numpy.ndarray
        :param backend: backend where to create this array.
        :type backend: Backend
        :returns: Newly created array.
        '''
        raise exceptions.MethodNotDefinedError(cls, 'from_ndarray')

    def as_ndarray(self):
        '''
        Return the underlying array as a :class:`numpy.ndarray` instance. If
        the underlying array is already of this type, no copy is done.

        :returns: underlying array as a :class:`numpy.ndarray`.
        :rtype: numpy.ndarray
        '''
        if core.is_gpu_backend(self.__aop.backend.btype):
            return self.__array.get()[:self.__length]
        else:
            return self.__array

    def copy(self):
        '''
        Copy this array.

        :returns: Copy of this array.
        '''
        return self.__class__(self.__array.copy(), self.__length, self.__backend)

    def get(self, i):
        '''
        Get an element of the array. The output type depends on the type of
        array.

        :param i: index.
        :type i: int
        :returns: value at the given index.
        :rtype: float, int or bool
        '''
        if i >= self.__length:
            raise IndexError(
                f'Index is out of range for array of length {self.__length}')

        if core.is_gpu_backend(self.__aop.backend.btype):
            return self.__array[i].get()
        else:
            return self.__array[i]

    def to_backend(self, backend):
        '''
        Send the array to the given backend.

        :param backend: backend where to transfer the array.
        :type backend: Backend
        :returns: this array on a new backend.
        :rtype: marray
        '''
        if self.__aop.backend is backend:
            return self
        else:
            if core.is_gpu_backend(self.__aop.backend.btype):
                a, s = backend.aop.ndarray_to_backend(
                    self.__array.get()[:self.__length])
            else:
                a, s = backend.aop.ndarray_to_backend(self.__array)
        return self.__class__(a, s, backend)


class barray(marray):

    def __init__(self, array, length=None, backend=None):
        '''
        Array of booleans.

        :param array: original array.
        :type array: numpy.ndarray or reikna.cluda.api.Array
        :param backend: backend where to put the array.
        :tye backend: Backend
        '''
        super().__init__(
            array, backend.aop.bool_type, length, backend)

    @property
    def dtype(self):
        return self.aop.bool_type

    def __and__(self, other):
        '''
        Logical "and" operator (element by element).

        :param other: array to do the comparison.
        :type other: barray
        :returns: result of the comparison.
        :rtype: barray
        '''
        return self.aop.logical_and(self, other)

    def __iand__(self, other):
        '''
        Logical in-place "and" operator (element by element).

        :param other: array to do the comparison.
        :type other: barray
        :returns: this array.
        :rtype: barray
        '''
        return self.aop.logical_and(self, other, out=self)

    def __or__(self, other):
        '''
        Logical "or" operator (element by element).

        :param other: array to do the comparison.
        :type other: barray
        :returns: result of the comparison.
        :rtype: barray
        '''
        return self.aop.logical_or(self, other)

    def __ior__(self, other):
        '''
        Logical in-place "or" operator (element by element).

        :param other: array to do the comparison.
        :type other: barray
        :returns: this array.
        :rtype: barray
        '''
        return self.aop.logical_or(self, other, out=self)

    @classmethod
    def from_ndarray(cls, a, backend):

        if not backend.aop.is_bool(a):
            a = a.astype(backend.aop.bool_type)

        return cls(*backend.aop.ndarray_to_backend(a), backend)

    def as_ndarray(self):

        if core.is_gpu_backend(self.aop.backend.btype):
            return self.ua.get().astype(data_types.cpu_real_bool)[:self.length]
        else:
            return self.ua

    def count_nonzero(self):
        '''
        Count the number of elements that are different from zero.

        :returns: number of elements different from zero.
        :rtype: int
        '''
        return self.aop.count_nonzero(self)


class farray(marray):

    def __init__(self, array, dtype, ndim=1, length=None, backend=None):
        '''
        Array of floats.

        :param array: original array.
        :type array: numpy.ndarray or reikna.cluda.api.Array
        :param backend: backend where to put the array.
        :tye backend: Backend
        '''
        if length is None:
            length = len(array) // ndim

        super().__init__(array, dtype, length, backend)

        self.__ndim = data_types.cpu_int(ndim)

    def __repr__(self):
        '''
        Attributes of the class as a string.
        '''
        return f'''dtype  = {self.dtype}
length = {self.length}
ndim   = {self.ndim}
array  = {self.as_ndarray()}'''

    @arithmetic_operation
    def __add__(self, other):
        '''
        Add the elements from another array to this one.

        :param other: array to sum element by element.
        :type other: farray
        :returns: newly created array.
        :rtype: farray
        '''
        return self.__class__(self.ua + other, self.__ndim, self.length, self.backend)

    @arithmetic_operation
    def __radd__(self, other):
        '''
        Add the elements from another array to this one.

        :param other: array to sum element by element.
        :type other: farray
        :returns: newly created array.
        :rtype: farray
        '''
        return self.__class__(other + self.ua, self.__ndim, self.length, self.backend)

    @arithmetic_operation
    def __iadd__(self, other):
        '''
        Add the elements from another array in-place.

        :param other: array to add element by element.
        :type other: farray
        :returns: this array.
        :rtype: farray
        '''
        self.ua += other
        return self

    @arithmetic_operation
    def __truediv__(self, other):
        '''
        Divide the elements of this array by those from another array.

        :param other: array to divide element by element.
        :type other: farray
        :returns: newly created array.
        :rtype: farray
        '''
        return self.__class__(self.ua / other, self.__ndim, self.length, self.backend)

    @arithmetic_operation
    def __itruediv__(self, other):
        '''
        Divide the elements of this array by those from another array in-place.

        :param other: array to divide element by element.
        :type other: farray
        :returns: this array.
        :rtype: farray
        '''
        self.ua /= other
        return self

    @arithmetic_operation
    def __mul__(self, other):
        '''
        Multiply the elements from another array.

        :param other: array to multiply element by element.
        :type other: farray
        :returns: newly created array.
        :rtype: farray
        '''
        return self.__class__(self.ua * other, self.__ndim, self.length, self.backend)

    @arithmetic_operation
    def __rmul__(self, other):
        '''
        Multiply the elements from another array.

        :param other: array to multiply element by element.
        :type other: farray
        :returns: newly created array.
        :rtype: farray
        '''
        return self.__class__(other * self.ua, self.__ndim, self.length, self.backend)

    @arithmetic_operation
    def __imul__(self, other):
        '''
        Multiply the elements from another array in-place.

        :param other: array to multiply element by element.
        :type other: farray
        :returns: this array.
        :rtype: farray
        '''
        self.ua *= other
        return self

    @arithmetic_operation
    def __sub__(self, other):
        '''
        Subtract the elements from another array to this one.

        :param other: array to subtract element by element.
        :type other: farray
        :returns: newly created array.
        :rtype: farray
        '''
        return self.__class__(self.ua - other, self.__ndim, self.length, self.backend)

    @arithmetic_operation
    def __rsub__(self, other):
        '''
        Subtract the elements from another array to this one.

        :param other: array to subtract element by element.
        :type other: farray
        :returns: newly created array.
        :rtype: farray
        '''
        return self.__class__(other - self.ua, self.__ndim, self.length, self.backend)

    @arithmetic_operation
    def __isub__(self, other):
        '''
        Subtract the elements from another array in-place.

        :param other: array to subtract element by element.
        :type other: farray
        :returns: this array.
        :rtype: farray
        '''
        self.ua -= other.ua
        return self

    @property
    def ndim(self):
        '''
        Number of dimensions of the array.
        '''
        return self.__ndim

    @property
    def shape(self):
        '''
        Shape of the array.
        '''
        if self.__ndim == 1:
            return (self.length,)
        else:
            return (self.length, self.__ndim)  # same length in all dimensions

    @property
    def size(self):
        '''
        Size of the array.
        '''
        return self.length * self.__ndim

    def as_ndarray(self):

        if core.is_gpu_backend(self.aop.backend.btype):
            return self.ua.get()[:self.length * self.__ndim]
        else:
            return self.ua

    def astype(self, dtype):
        '''
        Convert the array into the given data type. Only conversions from
        :class:`farray` and :class:`carray` objects are allowed.

        :param dtype: data type.
        :type dtype: numpy.dtype
        :returns: Converted array.
        :rtype: farray or carray
        :raises ValueError: If the conversion is not allowed.
        '''
        if dtype == self.dtype:
            return self

        if dtype == data_types.cpu_float:
            builder = farray
        elif dtype == data_types.cpu_complex:
            builder = carray
        else:
            raise ValueError(
                f'Conversion not allowed from data type "{self.dtype}" to "{dtype}"')

        return builder(self.ua.astype(dtype), self.__ndim, self.length, self.backend)

    def copy(self):
        return self.__class__(self.ua.copy(), self.__ndim, self.length, self.backend)

    @classmethod
    def from_ndarray(cls, a, backend):
        '''
        Create this class from a :class:`numpy.ndarray` instance. If the number
        of dimensions is greater than one, the array is flattened, and each
        column is assumed to belong to a different parameter.

        :param a: array.
        :type a: numpy.ndarray
        :param backend: backend where to create this array.
        :type backend: Backend
        :returns: Newly created array.
        '''
        raise exceptions.MethodNotDefinedError(cls, 'from_ndarray')

    def to_backend(self, backend):

        if self.aop.backend is backend:
            return self
        else:
            if core.is_gpu_backend(self.aop.backend.btype):
                a, s = backend.aop.ndarray_to_backend(
                    self.ua.get()[:self.ndim * self.length])
            else:
                a, s = backend.aop.ndarray_to_backend(self.ua)

        return self.__class__(a, self.ndim, s // self.ndim, backend)


class carray(farray):

    def __init__(self, array, ndim=1, length=None, backend=None):
        '''
        They can be of complex type.

        :param array: original array.
        :type array: numpy.ndarray or reikna.cluda.api.Array
        :param backend: backend where to put the array.
        :tye backend: Backend
        '''
        super().__init__(
            array, data_types.cpu_complex, ndim, length, backend)

    @classmethod
    def from_ndarray(cls, a, backend):

        if not backend.aop.is_complex(a):
            a = a.astype(data_types.cpu_complex)

        ndim = a.ndim
        if ndim != 1:
            a = a.flatten()

        a, l = backend.aop.ndarray_to_backend(a)

        return cls(a, ndim, l // ndim, backend)


class darray(farray):

    def __init__(self, array, ndim=1, length=None, backend=None):
        '''
        Array of floats.

        :param array: original array.
        :type array: numpy.ndarray or reikna.cluda.api.Array
        :param backend: backend where to put the array.
        :tye backend: Backend
        '''
        super().__init__(
            array, data_types.cpu_float, ndim, length, backend)

    @classmethod
    def from_ndarray(cls, a, backend):

        if not backend.aop.is_float(a):
            a = a.astype(data_types.cpu_float)

        ndim = a.ndim
        if ndim != 1:
            a = a.flatten()

        a, l = backend.aop.ndarray_to_backend(a)
        return cls(a, ndim, l // ndim, backend)

    def __lt__(self, other):
        r'''
        Comparison operator :math:`a < b`.

        :param other: array to compare element by element.
        :type other: darray
        :returns: newly created array.
        :rtype: barray
        '''
        return self.aop.lt(self, other)

    def __le__(self, other):
        r'''
        Comparison operator :math:`a \leq b`.

        :param other: array to compare element by element.
        :type other: darray
        :returns: newly created array.
        :rtype: barray
        '''
        return self.aop.le(self, other)

    def __ge__(self, other):
        r'''
        Comparison operator :math:`a \geq b`.

        :param other: array to compare element by element.
        :type other: darray
        :returns: newly created array.
        :rtype: barray
        '''
        return self.aop.ge(self, other)

    def __pow__(self, n):
        '''
        Calculate the n-th power of the array.

        :param n: power.
        :type n: int
        :returns: newly created array.
        :rtype: darray
        '''
        return self.__class__(self.ua**n, self.ndim, self.length, self.backend)

    def min(self):
        '''
        Minimum value of the elements in the array.

        :returns: minimum of the elements.
        :rtype: float
        '''
        return self.aop.min(self)

    def max(self):
        '''
        Minimum value of the elements in the array.

        :returns: maximum of the elements.
        :rtype: float
        '''
        return self.aop.max(self)

    def slice(self, a):
        '''
        Get a slice of this array, that can be done using a boolean mask
        or an array of integers.

        :param a: mask or indices array.
        :type a: barray or iarray
        :returns: slice of this array.
        :rtype: darray
        '''
        if self.aop.is_bool(a):
            return self.aop.slice_from_boolean(self, a)
        elif self.aop.is_int(a):
            return self.aop.slice_from_integer(self, a)
        else:
            raise ValueError(f'Method not implemented for data type {a.dtype}')

    def sum(self):
        '''
        Sum the elements in the array.

        :returns: sum of elements.
        :rtype: float
        '''
        return self.aop.sum(self)

    def take_column(self, i=0):
        '''
        Take elements of the array using a period.

        :param i: column to take the elements.
        :type i: int
        :returns: reduced array.
        :rtype: marray
        '''
        return self.aop.take_column(self, i)

    def take_slice(self, start=0, end=None):
        '''
        Take a slice of entries from the array.

        :param start: where to start taking entries.
        :type start: int
        :param end: where to end taking entries.
        :type end: int
        :returns: slice of the array.
        :rtype: marray
        '''
        return self.aop.take_slice(self, start, end)


class iarray(marray):

    def __init__(self, array, length=None, backend=None):
        '''
        Array of integers.

        :param array: original array.
        :type array: numpy.ndarray or reikna.cluda.api.Array
        :param backend: backend where to put the array.
        :tye backend: Backend
        '''
        super().__init__(
            array, data_types.cpu_int, length, backend)

    @classmethod
    def from_ndarray(cls, a, backend):

        if not backend.aop.is_int(a):
            a = a.astype(data_types.cpu_int)

        return cls(*backend.aop.ndarray_to_backend(a), backend)
