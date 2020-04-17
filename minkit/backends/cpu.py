'''
Operations with numpy objects. All the functions in this module expect objects
of type :class:`numpy.ndarray`.
'''
from . import autocode
from . import core
from ..base import data_types

from distutils import ccompiler
import ctypes
import functools
import logging
import numpy as np
import os
import tempfile

# Save the C-related flags to compile
CFLAGS = os.environ.get('CFLAGS', '').split()

logger = logging.getLogger(__name__)


class CPUOperations(object):

    def __init__(self, tmpdir=None):
        '''
        Initialize the class with the interface to the user backend.

        :param backend: user interface to the backend.
        :type backend: Backend
        '''
        # Cache for the PDFs
        self.__cpu_module_cache = {}
        self.__cpu_pdf_cache = {}

        if tmpdir is None:
            self.__tmpdir = tempfile.TemporaryDirectory()
        else:
            self.__tmpdir = tmpdir

    @property
    def cpu_module_cache(self):
        return self.__cpu_module_cache

    @property
    def cpu_pdf_cache(self):
        return self.__cpu_pdf_cache

    def _access_cpu_module(self, name):
        '''
        Access a C++ module, compiling it if it has not been done yet.

        :param name: name of the module.
        :type name: str
        :returns: compiled module.
        :rtype: module
        '''
        pdf_paths = core.get_pdf_src()

        if name in self.__cpu_module_cache:
            # Do not compile again the PDF source if it has already been done
            module = self.__cpu_module_cache[name]
        else:
            xml_source = None

            # Check if it exists in any of the provided paths
            for p in pdf_paths:
                fp = os.path.join(p, f'{name}.xml')
                if os.path.isfile(fp):
                    xml_source = fp
                    break

            if xml_source is None:
                raise RuntimeError(
                    f'XML file for function {name} not found in any of the provided paths: {pdf_paths}')

            # Write the code
            source = os.path.join(self.__tmpdir.name, f'{name}.cpp')
            code = autocode.generate_code(xml_source, core.CPU)
            with open(source, 'wt') as f:
                f.write(code)

            # Compile the C++ code and load the library
            compiler = ccompiler.new_compiler()

            try:
                objects = compiler.compile(
                    [source], output_dir=self.__tmpdir.name, extra_preargs=CFLAGS)
                libname = os.path.join(self.__tmpdir.name, f'lib{name}.so')
                compiler.link(f'{name} library', objects, libname,
                              extra_preargs=CFLAGS, libraries=['stdc++'])
            except Exception as ex:
                nl = len(str(code.count('\n')))
                code = '\n'.join(f'{i + 1:>{nl}}: {l}' for i,
                                 l in enumerate(code.split('\n')))
                logger.error(f'Error found compiling:\n{code}')
                raise ex

            module = ctypes.cdll.LoadLibrary(libname)

            self.__cpu_module_cache[name] = module

        return module

    def _create_cpu_function_proxies(self, module, ndata_pars, nvar_arg_pars=None):
        '''
        Create proxies for C++ that handle correctly the input and output data
        types of a function.

        :param module: module where to load the functions from.
        :type module: module
        :param ndata_pars: number of data parameters.
        :type ndata_pars: int
        :param nvar_arg_pars: number of variable argument parameters.
        :type nvar_arg_pars: int
        :returns: proxies for the function, the array-like function and integral.
        :rtype: function, function, function
        '''
        # Get the functions
        function = module.function
        evaluate = module.evaluate

        if hasattr(module, 'evaluate_binned'):
            evaluate_binned = module.evaluate_binned
        else:
            evaluate_binned = None

        if hasattr(module, 'integral'):
            integral = module.integral
        else:
            integral = None

        # Define the types of the input arguments
        if nvar_arg_pars is not None:
            partypes = [data_types.c_int, data_types.c_double_p]
        else:
            partypes = [data_types.c_double_p]

        # Define the types passed to the function
        function.argtypes = [data_types.c_double_p] + partypes
        function.restype = data_types.c_double

        @functools.wraps(function)
        def __function(data, args):
            '''
            Internal wrapper.
            '''
            dv = data.ctypes.data_as(data_types.c_double_p)
            ac = args.ctypes.data_as(data_types.c_double_p)

            if nvar_arg_pars is not None:
                return function(dv, data_types.c_int(nvar_arg_pars), ac)
            else:
                return function(dv, ac)

        # Define the types for the arguments passed to the evaluate function
        evaluate.argtypes = [data_types.c_int, data_types.c_double_p,
                             data_types.c_int_p, data_types.c_double_p] + partypes

        @functools.wraps(evaluate)
        def __evaluate(output_array, data_idx, input_array, args):
            '''
            Internal wrapper.
            '''
            op = output_array.ctypes.data_as(data_types.c_double_p)
            ip = input_array.ctypes.data_as(data_types.c_double_p)
            di = data_idx.ctypes.data_as(data_types.c_int_p)
            ac = args.ctypes.data_as(data_types.c_double_p)

            l = data_types.c_int(len(output_array))

            if nvar_arg_pars is not None:
                return evaluate(l, op, di, ip, data_types.c_int(nvar_arg_pars), ac)
            else:
                return evaluate(l, op, di, ip, ac)

        # Check the functions that need the integral to be defined
        if integral is not None and evaluate_binned is None:
            logger.warning(
                'If you are able to define a integral function you can also provide an evaluation for binned data sets.')

        if evaluate_binned is not None:

            # Define the types for the arguments passed to the evaluate_binned function
            evaluate_binned.argtypes = [data_types.c_int, data_types.c_double_p, data_types.c_int,
                                        data_types.c_int_p, data_types.c_int_p, data_types.c_double_p] + partypes

            @functools.wraps(evaluate_binned)
            def __evaluate_binned(output_array, gaps_idx, gaps, edges, args):
                '''
                Internal wrapper.
                '''
                op = output_array.ctypes.data_as(data_types.c_double_p)
                ed = edges.ctypes.data_as(data_types.c_double_p)
                gi = gaps_idx.ctypes.data_as(data_types.c_int_p)
                gp = gaps.ctypes.data_as(data_types.c_int_p)
                ac = args.ctypes.data_as(data_types.c_double_p)

                l = data_types.c_int(len(output_array))

                vals = (l, op, data_types.c_int(len(gaps)), gi, gp, ed)

                if nvar_arg_pars is not None:
                    return evaluate_binned(*vals, data_types.c_int(nvar_arg_pars), ac)
                else:
                    return evaluate_binned(*vals, ac)
        else:
            __evaluate_binned = None

        if integral is not None:

            # Define the types passed to the integral function
            integral.argtypes = [data_types.c_double_p,
                                 data_types.c_double_p] + partypes
            integral.restype = data_types.c_double

            @functools.wraps(integral)
            def __integral(lb, ub, args):
                '''
                Internal wrapper.
                '''
                if lb.ndim == 0:
                    lb, ub = data_types.array_float(
                        lb), data_types.array_float(ub)

                lb = lb.ctypes.data_as(data_types.c_double_p)
                ub = ub.ctypes.data_as(data_types.c_double_p)
                ac = args.ctypes.data_as(data_types.c_double_p)

                if nvar_arg_pars is not None:
                    return integral(lb, ub, data_types.c_int(nvar_arg_pars), ac)
                else:
                    return integral(lb, ub, ac)
        else:
            __integral = None

        return __function, __evaluate, __evaluate_binned, __integral

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
        :rtype: tuple(function, function) or tuple(function, function, function, function)
        '''
        module = self._access_cpu_module(name)

        try:
            modname = core.parse_module_name(name, nvar_arg_pars)

            if modname in self.__cpu_pdf_cache:
                output = self.__cpu_pdf_cache[modname]
            else:
                output = self._create_cpu_function_proxies(
                    module, ndata_pars, nvar_arg_pars)
                self.__cpu_pdf_cache[modname] = output

        except AttributeError as ex:
            raise RuntimeError(
                f'Error loading function "{name}"; make sure that at least "evaluate" and "function" are defined inside the source file')

        return output

    def _fft(self, a):
        return np.fft.fft(a)

    def _fftshift(self, a):
        n0 = sum(a < 0)
        nt = len(a)
        com = data_types.cpu_complex(+2.j * np.pi * n0 / nt)
        rng = self.arange(nt, dtype=data_types.cpu_int).astype(
            data_types.cpu_complex)
        return np.exp(com * rng)

    def _ifft(self, a):
        return np.fft.ifft(a)

    def arange(self, n, dtype=data_types.cpu_int):
        if dtype == data_types.cpu_int:
            return np.arange(n, dtype=dtype)
        elif dtype == data_types.cpu_complex:
            return np.arange(n, dtype=dtype).astype(data_types.cpu_complex)
        else:
            raise NotImplementedError(
                f'Function not implemented for data type "{dtype}"')

    def args_to_array(self, a, dtype=data_types.cpu_float):
        return a

    def ndarray_to_backend(self, a):
        return a

    def concatenate(self, arrays, maximum=None):
        if maximum is not None:
            return np.concatenate(arrays)[:maximum]
        else:
            return np.concatenate(arrays)

    def count_nonzero(self, a):
        return np.count_nonzero(a)

    def empty(self, size, dtype=data_types.cpu_float):
        return np.empty(size, dtype=dtype)

    def exp(self, a):
        return np.exp(a)

    def extract_ndarray(self, a):
        return a

    def fftconvolve(self, a, b, data):

        fa = self._fft(a)
        fb = self._fft(b)

        shift = self._fftshift(data)

        output = self._ifft(fa * shift * fb)

        return np.real(output * (data[1] - data[0]))

    def ge(self, a, v):
        return a >= v

    def interpolate_linear(self, idx, x, xp, yp):
        nd = len(idx)
        ln = len(x) // nd
        st = idx[0]
        return np.interp(x[st:st + ln], xp, yp)  # 1D case

    def is_inside(self, data, lb, ub):
        if len(lb) > 1:
            ln = len(data) // len(lb)
            c = np.ones(ln, dtype=data_types.cpu_real_bool)
            for i, (l, u) in enumerate(zip(lb, ub)):
                d = data[i * ln:(i + 1) * ln]
                np.logical_and(c, np.logical_and(d >= l, d < u), out=c)
        else:
            c = np.logical_and(data >= lb, data < ub)
        return c

    def lt(self, a, v):
        return a < v

    def le(self, a, v):
        return a <= v

    def linspace(self, vmin, vmax, size):
        return np.linspace(vmin, vmax, size, dtype=data_types.cpu_float)

    def log(self, a):
        return np.log(a)

    def logical_and(self, a, b,  out=None):
        return np.logical_and(a, b, out=out)

    def logical_or(self, a, b, out=None):
        return np.logical_or(a, b, out=out)

    def max(self, a):
        return np.max(a)

    def meshgrid(self, *arrays):
        return tuple(map(np.ndarray.flatten, np.meshgrid(*arrays)))

    def min(self, a):
        return np.min(a)

    def ones(self, n, dtype=data_types.cpu_float):
        if dtype == data_types.cpu_bool:
            # Hack due to lack of "bool" in PyOpenCL
            return np.ones(n, dtype=data_types.cpu_real_bool)
        else:
            return np.ones(n, dtype=dtype)

    def random_uniform(self, vmin, vmax, size):
        return np.random.uniform(vmin, vmax, size)

    def restrict_data_size(self, maximum, ndim, len, data):
        return np.concatenate(tuple(data[i * len:i * len + maximum] for i in range(ndim)))

    def shuffling_index(self, n):
        indices = np.arange(n)
        np.random.shuffle(indices)
        return indices

    def sum(self, a, *args):
        if len(args) == 0:
            return np.sum(a)
        else:
            return np.sum((a, *args), axis=0)

    def sum_inside(self, indices, gaps, centers, edges, values=None):

        nd = len(gaps)
        lc = len(centers) // nd

        c = [centers[i * lc:(i + 1) * lc] for i in range(nd)]
        e = [edges[p:n] for p, n in zip(indices[:-1], indices[1:])]

        out, _ = np.histogramdd(c, bins=e, weights=values)

        return out.T.flatten()

    def slice_from_boolean(self, a, valid, dim=1):
        return a[np.tile(valid, dim)]

    def slice_from_integer(self, a, indices, dim=1):
        return a[np.tile(indices, dim)]

    def zeros(self, n, dtype=data_types.cpu_float):
        if dtype == data_types.cpu_bool:
            # Hack due to lack of "bool" in PyOpenCL
            return np.zeros(n, dtype=data_types.cpu_real_bool)
        else:
            return np.zeros(n, dtype=dtype)
