'''
Contains the docstrings for the different available functions
'''


def set_docstring(function):
    '''
    Decorator to define the docstring of a function.
    '''
    function.__doc__ = globals()[function.__name__]
    return function


arange = '''

'''
array = '''
Create an array using the current backend.
Arguments are directly forwarded to the constructor.

:returns: input data converted to the correct type for the current backend.
:rtype: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
'''
concatenate = '''

'''
exp = '''

'''
extract_ndarray = '''
Get an :class:`numpy.ndarray` class from the input object.

:param obj: input data.
:type obj: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
:returns: data as :class:`numpy.ndarray`
:rtype: numpy.ndarray
'''
fft = '''
Calculate the fast-Fourier transform of an array.
'''
fftconvolve = '''
Calculate the convolution using a FFT of two arrays.

:param a: first array.
:type a: numpy.ndarray
:param b: first array.
:type b: numpy.ndarray
:param data: possible values where "a" and "b" where taken from.
:type data: numpy.ndarray
:returns: convolution of the two array.
:rtype: numpy.ndarray
'''
fftshift = '''
Compute the shift in frequencies needed for "fftconvolve" to return values in the same data range.

:param a: input array.
:type a: numpy.ndarray
:returns: shift in the frequency domain.
:rtype: numpy.ndarray
'''
geq = '''

'''
ifft = '''
Calculate the fast-Fourier transform of an array.
'''
interpolate_linear = '''

'''
leq = '''

'''
linspace = '''

'''
log = '''

'''
logical_and = '''
Do the logical "and" operation between two arrays.
'''
logical_or = '''
Do the logical "or" operation between two arrays.
'''
max = '''

'''
meshgrid = '''

'''
min = '''

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

'''
shuffling_index = '''
Return an array to shuffle data, with length "n".

:param n: size of the data.
:type n: int
:returns: shuffling array.
:rtype: numpy.ndarray
'''
sum = '''

'''
slice_from_boolean = '''

'''
slice_from_integer = '''

'''
true_till = '''

'''
zeros = '''
Create an array filled with zeros using the current backend.
Arguments are directly forwarded to the constructor.

:returns: zeros following the given shape.
:rtype: numpy.ndarray, pyopencl.Buffer or pycuda.gpuarray
'''
