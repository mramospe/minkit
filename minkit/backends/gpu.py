'''
Operations with GPU objects. All the functions in this module expect objects
of type :class:`reikna.cluda.Array` or :class:`numpy.ndarray`.
'''
from . import autocode
from . import core
from . import cpu
from . import docstrings
from .gpu_core import get_sizes, initialize_gpu
from .gpu_functions import make_functions
from ..base import data_types

from reikna.fft import FFT
import functools
import logging
import numpy as np
import os
import sys
import tempfile
import threading

logger = logging.getLogger(__name__)


class ArrayCacheManager(object):

    def __init__(self, thread, dtype):
        '''
        Object that keeps array in the GPU device in order to avoid creating
        and destroying them many times, and calls functions with them.

        :param dtype: data type of the output arrays.
        :type dtype: numpy.dtype
        '''
        self.__cache = {}
        self.__dtype = dtype
        self.__lock = threading.Lock()
        self.__thread = thread
        super(ArrayCacheManager, self).__init__()

    @property
    def cpu_module_cache(self):
        return self.__cpu_module_cache

    @property
    def cpu_pdf_cache(self):
        return self.__cpu_pdf_cache

    def free_cache(self):
        '''
        Free the cache of this object, removing the arrays not being used.
        '''
        with self.__lock:

            # Process the arrays to look for those not being used
            remove = []
            for s, elements in self.__cache.items():
                for i, el in reversed(enumerate(elements)):
                    if sys.getrefcount(el) == 1:
                        remove.append((s, i))
            for s, i in remove:
                self.__cache[s].pop(i)

            # Clean empty lists
            remove = []
            for s, lst in self.__cache.items():
                if len(lst) == 0:
                    remove.append(s)

            for s in remove:
                self.__cache.pop(s)

    def get_array(self, size):
        '''
        Get the array with size "size" from the cache, if it exists.

        :param size: size of the output array.
        :type size: int
        '''
        with self.__lock:

            elements = self.__cache.get(size, None)

            if elements is not None:
                for el in elements:
                    # This class is the only one that owns it, together with "elements"
                    if sys.getrefcount(el) == 3:
                        return el
            else:
                self.__cache[size] = []

            out = self.__thread.array((size,), dtype=self.__dtype)
            self.__cache[size].append(out)

            return out


class GPUOperations(object):

    def __init__(self, btype, **kwargs):
        '''
        Initialize the class with the interface to the user backend.

        :param kwargs: it may contain any of the following values: \
        - interactive: (bool) whether to select the device manually (defaults to False) \
        - device: (int) number of the device to use (defaults to None).
        :type kwargs: dict

        .. note:: The device can be selected using the MINKIT_DEVICE environment variable.
        '''
        self.__api, self.__device, self.__context, self.__thread = initialize_gpu(
            btype, **kwargs)

        self.__tmpdir = tempfile.TemporaryDirectory()
        self.__cpu_aop = cpu.CPUOperations(self.__tmpdir)

        # Cache for GPU objects
        self.__array_cache = {}
        self.__fft_cache = {}

        # Cache for the PDFs
        self.__gpu_module_cache = {}
        self.__gpu_pdf_cache = {}

        # Compile the functions
        self.__fbe, self.__fmp, self.__rfu, self.__tplf = make_functions(self)

    @property
    def gpu_module_cache(self):
        '''
        Cache for GPU modules.
        '''
        return self.__gpu_module_cache

    @property
    def gpu_pdf_cache(self):
        '''
        Cache for GPU PDFs.
        '''
        return self.__gpu_pdf_cache

    @property
    def thread(self):
        return self.__thread

    def _access_gpu_module(self, name):
        '''
        Access a GPU module, compiling it if it has not been done yet.

        :param name: name of the module.
        :type name: str
        :returns: compiled module.
        :rtype: module
        '''
        pdf_paths = core.get_pdf_src()

        if name in self.__gpu_module_cache:
            # Do not compile again the PDF source if it has already been done
            module = self.__gpu_module_cache[name]
        else:
            # Check if it exists in any of the provided paths
            for p in pdf_paths:
                fp = os.path.join(p, f'{name}.xml')
                if os.path.isfile(fp):
                    xml_source = fp
                    break

            if not os.path.isfile(xml_source):
                raise RuntimeError(
                    f'XML file for function {name} not found in any of the provided paths: {pdf_paths}')

            # Write the code
            source = os.path.join(self.__tmpdir.name, f'{name}.c')
            code = autocode.generate_code(xml_source, core.GPU)
            with open(source, 'wt') as f:
                f.write(code)

            # Compile the code
            with open(source) as fi:
                try:
                    module = self.__thread.compile(fi.read())
                except Exception as ex:
                    nl = len(str(code.count('\n')))
                    code = '\n'.join(f'{i + 1:>{nl}}: {l}' for i,
                                     l in enumerate(code.split('\n')))
                    logger.error(f'Error found compiling:\n{code}')
                    raise ex

            self.__gpu_module_cache[name] = module

        return module

    def _create_gpu_function_proxy(self, module, ndata_pars, nvar_arg_pars):
        '''
        Creates a proxy for a function writen in GPU.

        :param module: module containing the function to wrap.
        :type module: module
        :param ndata_pars: number of data parameters.
        :type ndata_pars: int
        :param nvar_arg_pars: number of variable argument parameters.
        :type nvar_arg_pars: int
        :returns: proxy for the array-like function.
        :rtype: function
        '''
        # Access the function in the module
        evaluate = module.evaluate

        try:
            # Can not use "hasattr"
            evaluate_binned = module.evaluate_binned
        except:
            evaluate_binned = None

        @functools.wraps(evaluate)
        def __evaluate(output_array, data_idx, input_array, args):
            '''
            Internal wrapper.
            '''
            ic = self.args_to_array(data_idx, dtype=data_types.cpu_int)

            if len(args) == 0:
                # It seems we can not pass a null pointer in OpenCL
                ac = self.zeros(1)
            else:
                ac = self.args_to_array(args, dtype=data_types.cpu_float)

            if nvar_arg_pars is not None:
                vals = (data_types.cpu_int(nvar_arg_pars), ac)
            else:
                vals = (ac,)

            global_size, local_size = get_sizes(len(output_array))

            return evaluate(output_array, ic, input_array, *vals, global_size=global_size, local_size=local_size)

        if evaluate_binned is not None:

            @functools.wraps(evaluate_binned)
            def __evaluate_binned(output_array, gaps_idx, gaps, edges, args):
                '''
                Internal wrapper.
                '''
                gi = self.args_to_array(
                    gaps_idx, dtype=data_types.cpu_int)
                gp = self.args_to_array(gaps, dtype=data_types.cpu_int)
                nd = data_types.cpu_int(len(gaps))

                if len(args) == 0:
                    # It seems we can not pass a null pointer in OpenCL
                    ac = self.zeros(1)
                else:
                    ac = self.args_to_array(
                        args, dtype=data_types.cpu_float)

                if nvar_arg_pars is not None:
                    vals = (data_types.cpu_int(nvar_arg_pars), ac)
                else:
                    vals = (ac,)

                global_size, local_size = get_sizes(len(output_array))

                return evaluate_binned(output_array, nd, gi, gp, edges, *vals, global_size=global_size, local_size=local_size)
        else:
            __evaluate_binned = None

        return __evaluate, __evaluate_binned

    def access_pdf(self, name, ndata_pars, nvar_arg_pars=None):
        '''
        Access a PDF with the given name and number of data and parameter arguments.

        :param name: name of the PDF.
        :type name: str
        :param ndata_pars: number of data parameters.
        :type ndata_pars: int
        :param nvar_arg_pars: number of variable argument parameters.
        :type nvar_arg_pars: int
        :returns: PDF and integral function.
        :rtype: tuple(function, function)
        '''
        # Function and integral are taken from the C++ version
        function, _, _, integral = self.__cpu_aop.access_pdf(
            name, ndata_pars, nvar_arg_pars)

        modname = core.parse_module_name(name, nvar_arg_pars)

        # Get the "evaluate" function from source
        if modname in self.__gpu_pdf_cache:
            evaluate, evaluate_binned = self.__gpu_pdf_cache[modname]
        else:
            # Access the GPU module
            gpu_module = self._access_gpu_module(name)

            evaluate, evaluate_binned = self._create_gpu_function_proxy(
                gpu_module, ndata_pars, nvar_arg_pars)

            self.__gpu_pdf_cache[modname] = evaluate, evaluate_binned

        return function, evaluate, evaluate_binned, integral

    def free_gpu_cache(self):
        '''
        Free the arrays saved in the GPU cache.
        '''
        self.__fft_cache.clear()
        for c in self.__array_cache.values():
            c.free_cache()

    def get_array_cache(self, dtype):
        '''
        Given a data type, return the associated array cache.

        :param dtype: data type.
        :type dtype: numpy.dtype
        :returns: array cache.
        :rtype: ArrayCacheManager
        '''
        c = self.__array_cache.get(dtype, None)
        if c is None:
            c = ArrayCacheManager(self.__thread, dtype)
            self.__array_cache[dtype] = c
        return c

    def reikna_fft(self, a, inverse=False):
        '''
        Get the FFT to calculate the FFT of an array, keeping the compiled
        source in a cache.
        '''
        # Compile the FFT
        cf = self.__fft_cache.get(a.shape, None)
        if cf is None:
            f = FFT(a)
            cf = f.compile(self.__thread)
            self.__fft_cache[a.shape] = cf

        # Calculate the value
        output = self.get_array_cache(data_types.cpu_complex).get_array(len(a))

        cf(output, a, inverse=inverse)

        return output

    @docstrings.set_docstring
    def arange(self, n, dtype=data_types.cpu_int):
        if dtype == data_types.cpu_int:
            return self.__fbe.arange_int(n, data_types.cpu_int(0))
        elif dtype == data_types.cpu_complex:
            return self.__fbe.arange_complex(n, data_types.cpu_float(0))
        else:
            raise NotImplementedError(
                f'Function not implemented for data type "{dtype}"')

    @docstrings.set_docstring
    def args_to_array(self, a, dtype=data_types.cpu_float):
        if a.dtype != dtype:
            a = a.astype(dtype)
        return self.__thread.to_device(a)

    def ndarray_to_backend(self, a):
        return self.__thread.to_device(a)

    @docstrings.set_docstring
    def concatenate(self, arrays, maximum=None):

        maximum = maximum if maximum is not None else np.sum(
            np.fromiter(map(len, arrays), dtype=data_types.cpu_int))

        dtype = arrays[0].dtype

        if dtype == data_types.cpu_float:
            function = self.__fbe.assign_double
        elif dtype == data_types.cpu_bool:
            function = self.__fbe.assign_bool
        else:
            raise NotImplementedError(
                f'Function not implemented for data type "{dtype}"')

        out = self.get_array_cache(dtype).get_array(maximum)

        offset = data_types.cpu_int(0)
        for a in arrays:
            l = data_types.cpu_int(len(a))
            gs, ls = get_sizes(data_types.cpu_int(
                l if l + offset <= maximum else maximum - offset))
            function(out, a, data_types.cpu_int(
                offset), global_size=gs, local_size=ls)
            offset += l

        return out

    @docstrings.set_docstring
    def count_nonzero(self, a):
        return self.__rfu.count_nonzero(a)

    @docstrings.set_docstring
    def data_array(self, a, dtype=data_types.cpu_float):
        if a.dtype != dtype:
            a = a.astype(dtype)
        return self.__thread.to_device(a)

    @docstrings.set_docstring
    def empty(self, size, dtype=data_types.cpu_float):
        return self.get_array_cache(dtype).get_array(size)

    @docstrings.set_docstring
    def exp(self, a):
        if a.dtype == data_types.cpu_complex:
            return self.__fbe.exponential_complex(a)
        elif a.dtype == data_types.cpu_float:
            return self.__fbe.exponential_double(a)
        else:
            raise NotImplementedError(
                f'Function not implemented for data type "{a.dtype}"')

    @docstrings.set_docstring
    def extract_ndarray(self, a):
        return a.get()

    @docstrings.set_docstring
    def fft(self, a):
        return self.reikna_fft(a.astype(data_types.cpu_complex))

    @docstrings.set_docstring
    def fftconvolve(self, a, b, data):

        fa = self.fft(a)
        fb = self.fft(b)

        shift = self.fftshift(data)

        output = self.ifft(fa * shift * fb)

        return self.__fbe.real(output * (data[1].get() - data[0].get()))

    @docstrings.set_docstring
    def fftshift(self, a):
        n0 = self.count_nonzero(self.le(a, 0))
        nt = len(a)
        com = data_types.cpu_complex(+2.j * np.pi * n0 / nt)
        rng = self.arange(nt, dtype=data_types.cpu_complex)
        return self.exp(com * rng)

    @docstrings.set_docstring
    def ge(self, a, v):
        return self.__fbe.ge(a, data_types.cpu_float(v))

    @docstrings.set_docstring
    def ifft(self, a):
        return self.reikna_fft(a, inverse=True)

    @docstrings.set_docstring
    def interpolate_linear(self, idx, x, xp, yp):
        ix = self.args_to_array(idx, dtype=data_types.cpu_int)
        nd = data_types.cpu_int(len(idx))
        lp = data_types.cpu_int(len(xp))
        ln = data_types.cpu_int(len(x))
        return self.__fbe.interpolate(ln, nd, ln, ix, x, lp, xp, yp)

    @docstrings.set_docstring
    def is_inside(self, data, lb, ub):

        if lb.ndim == 0:
            lb, ub = self.data_array(data_types.array_float(
                [lb])), self.data_array(data_types.array_float([ub]))
        else:
            lb, ub = self.data_array(lb), self.data_array(ub)

        ndim = data_types.cpu_int(len(lb))
        lgth = data_types.cpu_int(len(data) // ndim)

        out = self.get_array_cache(data_types.cpu_bool).get_array(lgth)

        gs, ls = get_sizes(lgth)

        self.__fbe.is_inside(out, lgth, data, ndim, lb, ub,
                             global_size=gs, local_size=ls)

        return out

    @docstrings.set_docstring
    def restrict_data_size(self, maximum, ndim, lgth, data):

        maximum, ndim, lgth = (data_types.cpu_int(i)
                               for i in (maximum, ndim, lgth))

        out = self.get_array_cache(data_types.cpu_float).get_array(maximum)

        gs, ls = get_sizes(maximum // ndim)

        self.__fbe.keep_to_limit(out, maximum, ndim, lgth, data,
                                 global_size=gs, local_size=ls)

        return out

    @docstrings.set_docstring
    def lt(self, a, v):
        if np.array(v).dtype == np.dtype(object):
            return self.__fbe.alt(a, v)
        else:
            return self.__fbe.lt(a, data_types.cpu_float(v))

    @docstrings.set_docstring
    def le(self, a, v):
        return self.__fbe.le(a, data_types.cpu_float(v))

    @docstrings.set_docstring
    def linspace(self, vmin, vmax, size):
        return self.__fbe.linspace(size,
                                   data_types.cpu_float(vmin),
                                   data_types.cpu_float(vmax),
                                   data_types.cpu_int(size))

    @docstrings.set_docstring
    def log(self, a):
        return self.__fbe.logarithm(a)

    @docstrings.set_docstring
    def logical_and(self, a, b, out=None):
        if out is None:
            return self.__fbe.logical_and(a, b)
        else:
            return self.__fbe.logical_and_to_output(out, a, b)

    @docstrings.set_docstring
    def logical_or(self, a, b, out=None):
        if out is None:
            return self.__fbe.logical_or(out, a, b)
        else:
            return self.__fbe.logical_or_to_output(out, a, b)

    @docstrings.set_docstring
    def max(self, a):
        return self.__rfu.amax(a)

    @docstrings.set_docstring
    def meshgrid(self, *arrays):
        a = map(np.ndarray.flatten, np.meshgrid(
            *tuple(a.get() for a in arrays)))
        return tuple(map(self.__thread.to_device, a))

    @docstrings.set_docstring
    def min(self, a):
        return self.__rfu.amin(a)

    @docstrings.set_docstring
    def ones(self, n, dtype=data_types.cpu_float):
        if dtype == data_types.cpu_float:
            return self.__fbe.ones_double(n)
        elif dtype == data_types.cpu_bool:
            return self.__fbe.ones_bool(n)
        else:
            raise NotImplementedError(
                f'Function not implemented for data type "{dtype}"')

    @docstrings.set_docstring
    def random_uniform(self, vmin, vmax, size):
        return self.__thread.to_device(np.random.uniform(vmin, vmax, size))

    @docstrings.set_docstring
    def shuffling_index(self, n):
        indices = np.arange(n)
        np.random.shuffle(indices)
        return self.__thread.to_device(indices)

    @docstrings.set_docstring
    def sum(self, a, *args):
        if len(args) == 0:
            if a.dtype == data_types.cpu_float:
                return self.__rfu.rsum(a)
            else:
                raise NotImplementedError(
                    f'Function not implemented for data type {a.dtype}')
        else:
            r = a
            for a in args:
                r += a
            return r

    @docstrings.set_docstring
    def sum_inside(self, indices, gaps, centers, edges, values=None):

        ndim = data_types.cpu_int(len(gaps))
        lgth = data_types.cpu_int(len(centers) // ndim)

        n = np.prod(indices[1:] - indices[:-1] - 1)

        out = self.zeros(n, dtype=data_types.cpu_float)

        gs, ls = get_sizes(n)

        gs = (int(lgth), gs)
        ls = (1, ls)

        gidx = self.__thread.to_device(gaps)

        if values is None:
            self.__fmp.sum_inside_bins(out, lgth, centers, ndim, gidx, edges,
                                       global_size=gs, local_size=ls)
        else:
            self.__fmp.sum_inside_bins_with_values(out, lgth, centers, ndim, gidx, edges, values,
                                                   global_size=gs, local_size=ls)

        return out

    @docstrings.set_docstring
    def slice_from_boolean(self, a, valid, dim=1):

        nz = self.__rfu.count_nonzero(valid)

        if nz == 0:
            return None  # Empty array

        # Calculate the compact indices
        indices = self.__fbe.invalid_indices(len(valid))

        gs, ls = get_sizes(len(indices))

        sizes = self.get_array_cache(data_types.cpu_int).get_array(gs // ls)

        self.__tplf(ls).compact_indices(indices, sizes,
                                        valid, global_size=gs, local_size=ls)

        # Build the output array
        out = self.get_array_cache(data_types.cpu_float).get_array(nz * dim)

        self.__fbe.take(out, data_types.cpu_int(dim), data_types.cpu_int(
            nz), sizes, indices, a, global_size=gs, local_size=ls)

        return out

    @docstrings.set_docstring
    def slice_from_integer(self, a, indices, dim=1):
        l = len(indices)
        out = self.get_array_cache(data_types.cpu_float).get_array(l * dim)
        gs, ls = get_sizes(l)
        self.__fbe.slice_from_integer(out, dim, len(a), a, l, indices,
                                      global_size=gs, local_size=ls)
        return out

    @docstrings.set_docstring
    def zeros(self, n, dtype=data_types.cpu_float):
        if dtype == data_types.cpu_float:
            return self.__fbe.zeros_double(n)
        elif dtype == data_types.cpu_bool:
            return self.__fbe.zeros_bool(n)
        else:
            raise NotImplementedError(
                f'Function not implemented for data type "{dtype}"')
