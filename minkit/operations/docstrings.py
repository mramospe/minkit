'''
Contains the docstrings for the different available functions
'''


def set_docstring(function):
    '''
    Decorator to define the docstring of a function.
    '''
    function.__doc__ = globals()[function.__name__]
    return function


ale = '''
Evaluate the condition "a1[i] < a2[i]" for all elements in the two arrays.

:param a1: first array.
:type a1: numpy.ndarray or reikna.cluda.Array
:param a2: second array.
:type a2: numpy.ndarray or reikna.cluda.Array
:returns: boolean array with the results of the evaluation.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
arange = '''
Create an array with integers or complex numbers (depending on "dtype") from 0 to "n".

:param n: length of the array and next-to-maximum number in it.
:type n: int
:param dtype: data type for the array. It can be :class:`numpy.int32` or :class:`numpy.complex128`.
:type dtype: numpy.dtype
:returns: array with values starting from zero and increasing by one.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
concatenate = '''
Concatenate many arrays into one, trimming the output so its length is "maximum",
if specified.

:param array: arrays to concatenate.
:type array: tuple(numpy.ndarray, ...) or tuple(reikna.cluda.Array, ...)
:param maximum: possible maximum size for the output array.
:type maximum: int or None
:returns: concatenated array.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
data_array = '''
Create an array using the current backend.
Arguments are directly forwarded to the constructor.

:returns: input data converted to the correct type for the current backend.
:rtype: numpy.ndarray, reikna.cluda.Array
'''
count_nonzero = '''
Count the number of True occurrences on a boolean array.

:param a: input array.
:type a: numpy.ndarray or reikna.cluda.Array
:returns: number of occurrences different from zero.
:rtype: int

.. note:: Input array must be of type :class:`minkit.operations.types.cpu_bool`.
'''
empty = '''
Create an empty array.

:param size: size of the output array.
:type size: int
:param dtype: data type.
:type dtype: numpy.dtype
:returns: empty array.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
exp = '''
Calculate the exponential for each element in an array.

:param a: input array.
:type a: numpy.ndarray, reikna.cluda.Array
:returns: array with the function evaluated.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
extract_ndarray = '''
Get an :class:`numpy.ndarray` class from the input object.

:param obj: input data.
:type obj: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
:returns: data as :class:`numpy.ndarray`
:rtype: numpy.ndarray
'''
false_till = '''
Create an array of False values till a given index.

:param N: size of the array.
:type N: int
:param n: value where to stop putting False.
:type n: int
:returns: output array.
:rtype: numpy.ndarray(bool) or reikna.cluda.Array(bool)
'''
fft = '''
Calculate the fast-Fourier transform of an array.

:param a: input array.
:type a: numpy.ndarray or reikna.cluda.Array
:returns: array with the fast-fourier transform.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
fftconvolve = '''
Calculate the convolution using a FFT of two arrays.

:param a: first array.
:type a: numpy.ndarray or reikna.cluda.Array
:param b: first array.
:type b: numpy.ndarray or reikna.cluda.Array
:param data: possible values where "a" and "b" where taken from.
:type data: numpy.ndarray or reikna.cluda.Array
:returns: convolution of the two array.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
fftshift = '''
Compute the shift in frequencies needed for "fftconvolve" to return values in the same data range.

:param a: input array.
:type a: numpy.ndarray or reikna.cluda.Array
:returns: shift in the frequency domain.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
geq = '''
Return a mask array with the places where the input array is greater or equal than the provided value evaluated to True.

:param a: input array.
:type a: numpy.ndarray or reikna.cluda.Array
:returns: mask array.
:rtype: numpy.ndarray(bool) or reikna.cluda.Array(bool)
'''
ifft = '''
Calculate the inverse of the fast-Fourier transform of an array.

:param a: input array.
:type a: numpy.ndarray or reikna.cluda.Array
:returns: array with the inverse fast-fourier transform.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
interpolate_linear = '''
Do a linear interpolation.

:param x: points that need to be interpolated.
:type x: numpy.ndarray or reikna.cluda.Array
:param xp: true data points.
:type xp: numpy.ndarray or reikna.cluda.Array
:param fp: true evaluation points.
:type fp: numpy.ndarray or reikna.cluda.Array
:returns: values associated to the "x" points.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
le = '''
Return a mask array with the places where the input array is less than the provided value evaluated to True.

:param a: input array.
:type a: numpy.ndarray or reikna.cluda.Array
:returns: mask array.
:rtype: numpy.ndarray(bool) or reikna.cluda.Array(bool)
'''
leq = '''
Return a mask array with the places where the input array is less or equal than the provided value evaluated to True.

:param a: input array.
:type a: numpy.ndarray or reikna.cluda.Array
:returns: mask array.
:rtype: numpy.ndarray(bool) or reikna.cluda.Array(bool)
'''
linspace = '''
Create an array with "size" values in increasing order from "vmin" to "vmax".

:param vmin: minimum value of the array.
:type vmin: float
:param vmax: maximum value of the array.
:type vmax: float
:param size: size of the array.
:type size: int
:returns: output array.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
log = '''
Calculate the logarithm for each element in an array.

:param a: input array.
:type a: numpy.ndarray, reikna.cluda.Array
:returns: array with the function evaluated.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
logical_and = '''
Do the logical "and" operation between two arrays.

:param a: first array.
:type a: numpy.ndarray(bool) or reikna.cluda.Array(bool)
:param b: second array.
:type b: numpy.ndarray(bool) or reikna.cluda.Array(bool)
:returns: evaluation of the "and" operation.
:rtype: numpy.ndarray(bool) or reikna.cluda.Array(bool)
'''
logical_or = '''
Do the logical "or" operation between two arrays.

:param a: first array.
:type a: numpy.ndarray(bool) or reikna.cluda.Array(bool)
:param b: second array.
:type b: numpy.ndarray(bool) or reikna.cluda.Array(bool)
:returns: evaluation of the "or" operation.
:rtype: numpy.ndarray(bool) or reikna.cluda.Array(bool)
'''
max = '''
Get the maximum value of an array.

:param a: input array.
:type a: numpy.ndarray or reikna.cluda.Array
:returns: maximum value.
:rtype: float

.. note:: Input array must be of type :class:`minkit.operations.types.cpu_type`.
'''
meshgrid = '''
Create a meshgrid from the input arrays.
This is similar to :func:`numpy.meshgrid`, but the output arrays are flattened.

:param arrays: arrays to create the grid.
:type arrays: tuple(numpy.ndarray, ...) or tuple(reikna.cluda.Array, ...)
:returns: output grid.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
min = '''
Get the minimum value of an array.

:param a: input array.
:type a: numpy.ndarray or reikna.cluda.Array
:returns: minimum value.
:rtype: float

.. note:: Input array must be of type :class:`minkit.operations.types.cpu_type`.
'''
ones = '''
Create an array filled with ones using the current backend.
Arguments are directly forwarded to the constructor.

:returns: zeros following the given shape.
:rtype: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
'''
random_uniform = '''
Create data following an uniform distribution between 0 and 1, with size "size".

:param size: size of the data to create
:type size: int
:returns: data following an uniform distribution between 0 and 1.
:rtype: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
'''
real = '''
Get the real part of an array of complex numbers.

:param a: input array.
:type a: numpy.ndarray(numpy.complex128) or reikna.cluda.Array(numpy.complex128)
:returns: real part of "a".
:rtype: numpy.ndarray(numpy.float64) or reikna.cluda.Array(numpy.float64)
'''
shuffling_index = '''
Return an array to shuffle data, with length "n".

:param n: size of the data.
:type n: int
:returns: shuffling array.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
sum = '''
Sum the elements of an array, or sum many arrays into one.

:param a: first array.
:type a: numpy.ndarray or reikna.cluda.Array
:param args: possible additional arrays.
:type args: tuple(numpy.ndarray, ...) or tuple(reikna.cluda.Array, ...)
:returns: sum of the elements of "a" if no more arrays are provided, or sum of "a" with "args".
:rtype: numpy.ndarray or reikna.cluda.Array
'''
sum_inside = '''
Sum the occurrences of values inside the provided edges.

:param centers: centers of the points.
:type centers: numpy.ndarray or reikna.cluda.Array
:param values: values to add.
:type values: numpy.ndarray or reikna.cluda.Array
:param edges: edges defining the bins.
:type edges: numpy.ndarray or reikna.cluda.Array
:returns: sum of "values" inside each bin.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
slice_from_boolean = '''
Get elements from an array given a mask array.

:param a: input array.
:type a: numpy.ndarray or reikna.cluda.Array
:param valid: mask array.
:type valid: numpy.ndarray(bool) or reikna.cluda.Array(bool)
:returns: output array.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
slice_from_integer = '''
Get elements from an array given an array of indices.

:param a: input array.
:type a: numpy.ndarray or reikna.cluda.Array
:param valid: mask array.
:type valid: numpy.ndarray(int) or reikna.cluda.Array(int)
:returns: output array.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
true_till = '''
Create an array of True values till a given index.

:param N: size of the array.
:type N: int
:param n: value where to stop putting ones.
:type n: int
:returns: output array.
:rtype: numpy.ndarray(bool) or reikna.cluda.Array(bool)
'''
zeros = '''
Create an array filled with zeros using the current backend.
Arguments are directly forwarded to the constructor.

:param n: size of the array.
:type n: int
:returns: zeros following the given shape.
:rtype: numpy.ndarray or reikna.cluda.Array
'''
