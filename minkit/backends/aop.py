########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Application interface to deal with arrays. The different array classes
can not define the __getitem__ since numpy objects would use it to do
arithmetic operations.
'''
from . import arrays
from . import core
from ..base import data_types

import contextlib


__all__ = ['ArrayOperations']


class ArrayOperations(object):

    def __init__(self, backend, **kwargs):
        '''
        Build the object to do operations within a backend. Only the necessary
        operators have been defined.

        :param btype: backend type ('cpu', 'cuda', 'opencl').
        :type btype: str
        :param kwargs: arguments forwarded to the backend constructor \
        (cuda and opencl backends only).
        :type kwargs: dict

        The possible keyword arguments in GPU backends are:

        * *device*: device to be used.
        * *interactive*: whether to ask the user a device if not specified or
          if the proposed device is not available.
        '''
        super().__init__()

        self.__backend = backend

        if self.__backend.btype == core.CPU:
            from .cpu import CPUOperations
            self.__oper = CPUOperations(backend, **kwargs)
        else:
            from .gpu import GPUOperations
            self.__oper = GPUOperations(backend, **kwargs)

    @property
    def backend(self):
        '''
        Backend interface.
        '''
        return self.__backend

    def access_pdf(self, name, ndata_pars, nvar_arg_pars=None):
        '''
        Access the PDF with the given name.

        :param name: name of the PDF.
        :type name: str
        :param ndata_pars: number of data parameters.
        :type ndata_pars: int
        :param nvar_arg_pars: number of variable arguments.
        :type nvar_arg_pars: int
        :returns: proxy for the different evaluation functions.
        :rtype: FunctionsProxy
        '''
        return self.__oper.access_pdf(name, ndata_pars, nvar_arg_pars)

    def carange(self, n):
        '''
        Create an array of elements from 0 to n.

        :param n: the next to last item in the array.
        :type n: int
        :returns: output array.
        :rtype: carray
        '''
        return self.__oper.carange(n)

    def iarange(self, n):
        '''
        Create an array of elements from 0 to n.

        :param n: the next to last item in the array.
        :type n: int
        :returns: output array.
        :rtype: iarray
        '''
        return self.__oper.iarange(n)

    def ndarray_to_backend(self, a):
        '''
        Create an array in the backend of this object. This will directly
        create an object of the underlying array type of the :class:`minkit.barray`,
        :class:`minkit.farray` or :class:`minkit.iarray` instance.

        :param a: input array.
        :type a: numpy.ndarray
        :returns: array representation in the given backend.
        :rtype: numpy.ndarray or reikna.cluda.api.Array

        .. warning:: This function must be used carefully since the output array \
           does not track to which backend it belongs.
        '''
        return self.__oper.ndarray_to_backend(a)

    def concatenate(self, arrays, maximum=None):
        '''
        Concatenate several arrays. If "maximum" is specified, the last elements
        will be dropped, so the length of the output array is "maximum".

        :param arrays: collection of arrays.
        :type arrays: tuple(farray)
        :param maximum: possible maximum length of the output array.
        :type maximum: int
        :returns: concatenated array.
        :rtype: farray
        '''
        return self.__oper.concatenate(tuple(a for a in arrays), maximum)

    def count_nonzero(self, a):
        '''
        Count the number of elements that are different from zero.

        :param a: input array.
        :type a: barray
        :returns: number of elements that are different from zero.
        :rtype: int
        '''
        return self.__oper.count_nonzero(a)

    def bempty(self, size):
        '''
        Create an empty :class:`barray` instance with the given length.

        :param size: length of the output array.
        :type size: int
        :returns: empty array.
        :rtype: barray
        '''
        return self.__oper.bempty(size)

    def fempty(self, size, ndim=1):
        '''
        Create an empty :class:`darray` instance with the given length.

        :param size: length of the output array.
        :type size: int
        :returns: empty array.
        :rtype: darray
        '''
        return self.__oper.fempty(size, ndim)

    def iempty(self, size):
        '''
        Create an empty :class:`iarray` instance with the given length.

        :param size: length of the output array.
        :type size: int
        :returns: empty array.
        :rtype: iarray
        '''
        return self.__oper.iempty(size)

    def cexp(self, a):
        '''
        Calculate the exponential of an array of numbers.

        :param a: input array.
        :type a: carray
        :returns: exponential values.
        :rtype: carray
        '''
        return self.__oper.cexp(a)

    def fexp(self, a):
        '''
        Calculate the exponential of an array of numbers. The input array must
        be unidimensional.

        :param a: input array.
        :type a: darray
        :returns: exponential values.
        :rtype: darray
        '''
        return self.__oper.fexp(a)

    def fftconvolve(self, a, b, data):
        '''
        Calculate the convolution of two arrays that correspond to the output
        from the evaluation of two PDFs in the given data.

        :param a: first array.
        :type a: farray
        :param b: second array.
        :type b: farray
        :param data: array of data (only 1D is supported).
        :type data: farray
        :returns: convolution of the two arrays.
        :rtype: farray
        '''
        return self.__oper.fftconvolve(a, b, data)

    def ge(self, a, b):
        r'''
        Compare two arrays :math:`a \geq b`. The input arrays must be unidimensional.

        :param a: first array.
        :type a: farray
        :param b: second array.
        :type b: farray
        :returns: result of the comparison.
        :rtype: barray
        '''
        return self.__oper.ge(a, b)

    def make_linear_interpolator(self, xp, yp):
        '''
        Create a function that, given an array of points in *x*, provides the
        interpolated values using a linear interpolator.

        :param xp: reference points in the x axis.
        :type xp: darray
        :param yp: reference points in the y axis.
        :type yp: darray
        :returns: interpolator.
        '''
        return self.__oper.make_linear_interpolator(xp, yp)

    def make_spline_interpolator(self, xp, yp):
        '''
        Create a function that, given an array of points in *x*, provides the
        interpolated values using an spline of order 3.

        :param xp: reference points in the x axis.
        :type xp: darray
        :param yp: reference points in the y axis.
        :type yp: darray
        :returns: interpolator.
        '''
        return self.__oper.make_spline_interpolator(xp, yp)

    def product_by_zero_is_zero(self, f, s):
        '''
        Do a product of two arrays in such a way that if any of the elements
        is zero the output will be zero.

        :param f: first array.
        :type f: darray
        :param s: second array.
        :type f: darray
        :returns: Product of the two arrays.
        :rtype: darray
        '''
        return self.__oper.product_by_zero_is_zero(f, s)

    @property
    def bool_type(self):
        '''
        Data type used for boolean comparisons.
        '''
        if core.is_gpu_backend(self.__backend.btype):
            return data_types.cpu_bool
        else:
            return data_types.cpu_real_bool

    def is_bool(self, a):
        '''
        Check whether the data type of the given array is of boolean type.

        :param a: input array.
        :type a: marray, numpy.ndarray or reikna.cluda.api.Array
        :returns: decision.
        :rtype: bool
        '''
        return a.dtype == self.bool_type

    def is_complex(self, a):
        '''
        Check whether the data type of the given array is of complex type.

        :param a: input array.
        :type a: marray, numpy.ndarray or reikna.cluda.api.Array
        :returns: decision.
        :rtype: bool
        '''
        return a.dtype == data_types.cpu_complex

    def is_float(self, a):
        '''
        Check whether the data type of the given array is of floating point type.

        :param a: input array.
        :type a: marray, numpy.ndarray or reikna.cluda.api.Array
        :returns: decision.
        :rtype: bool
        '''
        return a.dtype == data_types.cpu_float

    def is_int(self, a):
        '''
        Check whether the data type of the given array is of integral type.

        :param a: input array.
        :type a: marray, numpy.ndarray or reikna.cluda.api.Array
        :returns: decision.
        :rtype: bool
        '''
        return a.dtype == data_types.cpu_int

    def is_inside(self, data, lb, ub):
        '''
        Return the decision whether the input data is within the given bounds
        or not.

        :param data: input data array.
        :type data: farray
        :param lb: lower bounds.
        :type lb: numpy.ndarray
        :param ub: upper bounds.
        :type ub: numpy.ndarray
        :returns: array with the decisions.
        :rtype: barray
        '''
        return self.__oper.is_inside(data, lb, ub)

    def le(self, a, b):
        r'''
        Compare two arrays :math:`a \leq b`. The input arrays must be unidimensional.

        :param a: first array.
        :type a: farray
        :param b: second array.
        :type b: farray
        :returns: result of the comparison.
        :rtype: barray
        '''
        return self.__oper.le(a, b)

    def lt(self, a, b):
        r'''
        Compare two arrays :math:`a < b`. The input arrays must be unidimensional.

        :param a: first array.
        :type a: farray
        :param b: second array.
        :type b: farray
        :returns: result of the comparison.
        :rtype: barray
        '''
        return self.__oper.lt(a, b)

    def linspace(self, vmin, vmax, size):
        '''
        Create an array where the values are in increasing order by a constant
        step. Points in "vmin" and "vmax" are included.

        :param vmin: where to start generating values.
        :type vmin: float
        :param vmax: where to end generating values.
        :type vmax: float
        :param size: length of the output array.
        :type size: int
        :returns: array.
        :rtype: farray
        '''
        return self.__oper.linspace(vmin, vmax, size)

    def log(self, a):
        '''
        Calculate the logarithm of an array of numbers. The input array must
        be unidimensional.

        :param a: input array.
        :type a: farray
        :returns: logarithm values.
        :rtype: farray
        '''
        return self.__oper.log(a)

    def logical_and(self, a, b, out=None):
        '''
        Do the logical "and" operation element by element.

        :param a: first array.
        :type a: farray
        :param b: second array.
        :type b: farray
        :param out: possible output array to write the data.
        :type out: barray
        :returns: array of decisions.
        :rtype: barray
        '''
        if out is not None:
            self.__oper.logical_and(a, b, out)
            return out
        else:
            return arrays.barray(*self.__oper.logical_and(a, b), backend=self.backend)

    def logical_or(self, a, b, out=None):
        '''
        Do the logical "or" operation element by element.

        :param a: first array.
        :type a: farray
        :param b: second array.
        :type b: farray
        :param out: possible output array to write the data.
        :type out: barray
        :returns: array of decisions.
        :rtype: barray
        '''
        if out is not None:
            self.__oper.logical_or(a, b, out)
            return out
        else:
            return arrays.barray(*self.__oper.logical_or(a, b), backend=self.backend)

    def max(self, a):
        '''
        Calculate the maximum value in an array.

        :param a: input array.
        :type a: farray
        :returns: maximum value.
        :rtype: float
        '''
        return self.__oper.max(a)

    def meshgrid(self, lb, ub, size):
        '''
        Create a grid of values using the given set of bounds. The size can
        be specified as a single value or as a set of sizes for each dimension.
        If it is a single value, then the output array will be of length
        :math:`size \times size \times ...`, depending on the number of dimensions.

        :param lb: lower bounds.
        :type lb: numpy.ndarray
        :param ub: upper bounds.
        :type ub: numpy.ndarray
        :param size: size of the grid to generate.
        :type size: numpy.ndarray
        '''
        return self.__oper.meshgrid(lb, ub, size)

    def min(self, a):
        '''
        Calculate the minimum value in an array.

        :param a: input array.
        :type a: farray
        :returns: minimum value.
        :rtype: float
        '''
        return self.__oper.min(a)

    def bones(self, size):
        '''
        Create an array of "true" values of the given size.

        :param size: length of the output array.
        :type size: int
        :returns: array of "true".
        :rtype: barray
        '''
        return self.__oper.bones(size)

    def bzeros(self, size):
        '''
        Create an array of "false" values of the given size.

        :param size: length of the output array.
        :type size: int
        :returns: array of "false".
        :rtype: barray
        '''
        return self.__oper.bzeros(size)

    def fones(self, size):
        '''
        Create an array of the given size filled with ones.

        :param size: length of the output array.
        :type size: int
        :returns: array of ones.
        :rtype: farray
        '''
        return self.__oper.fones(size)

    def fzeros(self, size, ndim=1):
        '''
        Create an array of the given size filled with zeros.

        :param size: length of the output array.
        :type size: int
        :returns: array of zeros.
        :rtype: farray
        '''
        return self.__oper.fzeros(size, ndim)

    def random_grid(self, lb, ub, size):
        '''
        Create a random grid using the given bounds. The size can be specified
        as a single value or as a set of sizes for each dimension. If it is a
        single value, then the output array will be of length
        :math:`size \times size \times ...`, depending on the number of dimensions.

        :param lb: lower bounds.
        :type lb: numpy.ndarray
        :param ub: upper bounds.
        :type ub: numpy.ndarray
        :param size: size of the grid.
        :type size: int or numpy.ndarray
        :returns: random grid.
        :rtype: farray
        '''
        return self.__oper.random_grid(lb, ub, size)

    def random_uniform(self, vmin, vmax, size):
        return self.__oper.random_uniform(vmin, vmax, size)

    def restrict_data_size(self, maximum, data):
        '''
        Drop the last elements of an array of data.

        :param maximum: maximum length of the data sample.
        :type maximum: int
        :param data: data sample.
        :type data: farray
        :returns: reduced data array.
        :rtype: farray
        '''
        return self.__oper.restrict_data_size(maximum, data)

    def set_rndm_seed(self, seed):
        '''
        Set the seed of the random number generator.

        :param seed: new seed to use.
        :type seed: int
        '''
        self.__oper.set_rndm_seed(seed)

    def shuffling_index(self, n):
        '''
        Calculate a set of indices to shuffle an array.

        :param n: length of the array.
        :type n: int
        :returns: array of indices.
        :rtype: iarray
        '''
        return self.__oper.shuffling_index(n)

    def sum(self, a):
        '''
        Sum the elements of the given array.

        :param a: input array.
        :type a: farray
        :returns: sum of elements.
        :rtype: float
        '''
        return self.__oper.sum(a)

    def sum_arrays(self, arrays):
        '''
        Sum arrays element by element.

        :param arrays: set of arrays.
        :type arrays: tuple(farray)
        :returns: array with the sum.
        :rtype: farray
        '''
        out = self.fzeros(len(arrays[0]))
        for a in arrays:
            out += a
        return out

    def sum_inside(self, indices, gaps, centers, edges, values=None):
        '''
        Sum the occurrences inside the given bounds. If "values" is specified,
        then the values in it are used instead.

        :param indices: array of indices.
        :type indices: numpy.ndarray
        :param gaps: gaps to access the data.
        :type gaps: numpy.ndarray
        :param centers: centers corresponding to a data sample.
        :type centers: farray
        :param edges: edges defining the bins.
        :type edges: farray
        :param values: possible values (or weights) associated to the data sample.
        :type values: farray or None
        :returns: sum inside each bin.
        :rtype: farray
        '''
        return self.__oper.sum_inside(indices, gaps, centers, edges, values)

    def slice_from_boolean(self, a, v):
        '''
        Get a slice of the given array using a mask array.

        :param a: input data array.
        :type a: farray
        :param v: mask array.
        :type v: barray
        :returns: slice of the array.
        :rtype: farray
        '''
        return self.__oper.slice_from_boolean(a, v)

    def slice_from_integer(self, a, i):
        '''
        Get a slice of the given array using an array of indices.

        :param a: input data array.
        :type a: farray
        :param i: array of indices.
        :type i: iarray
        :returns: slice of the array.
        :rtype: farray
        '''
        return self.__oper.slice_from_integer(a, i)

    def take_column(self, a, i=0):
        '''
        Take elements of the array using a period.

        :param i: column to take the elements.
        :type i: int
        :returns: reduced array.
        :rtype: marray
        '''
        return self.__oper.take_column(a, i)

    def take_slice(self, a, start=0, end=None):
        '''
        Take a slice of entries from the array.

        :param start: where to start taking entries.
        :type start: int
        :param end: where to end taking entries.
        :type end: int
        :returns: slice of the array.
        :rtype: marray
        '''
        end = end if end is not None else len(a)
        return self.__oper.take_slice(a, start, end)

    @contextlib.contextmanager
    def using_caches(self):
        '''
        Use the caches (GPU only). Only those caches related to arrays and functions
        (not the PDFs) will be removed.
        '''
        if core.is_gpu_backend(self.__backend.btype):
            with self.__oper.using_caches():
                yield self
        else:
            yield self
