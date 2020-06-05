########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Operations with numpy objects. All the functions in this module expect objects
of type :class:`numpy.ndarray`.
'''
from . import autocode
from . import arrays
from . import core
from . import gsl_api
from ..base import data_types
from ..base.data_types import c_int, c_int_p, c_double, c_double_p

from distutils import ccompiler, sysconfig
from scipy.interpolate import make_interp_spline
import ctypes
import functools
import logging
import numpy as np
import os
import tempfile

# Default seed for the random number generators
DEFAULT_SEED = 49763

# Save the C-related flags to compile
CFLAGS = os.environ.get('CFLAGS', '').split()

logger = logging.getLogger(__name__)


def return_barray(method):
    '''
    Wrapper to automatically create a :class:`minkit.barray` object with
    the output from a method call.

    :param method: method of the class.
    :type method: class method
    :returns: decorated function
    :rtype: function
    '''
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return arrays.barray(method(self, *args, **kwargs), backend=self.backend)
    return wrapper


def return_carray(method):
    '''
    Wrapper to automatically create a :class:`minkit.carray` object with
    the output from a method call.

    :param method: method of the class.
    :type method: class method
    :returns: decorated function
    :rtype: function
    '''
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return arrays.carray(method(self, *args, **kwargs), backend=self.backend)
    return wrapper


def return_darray(method):
    '''
    Wrapper to automatically create a :class:`minkit.darray` object with
    the output from a method call.

    :param method: method of the class.
    :rtype: function
    :returns: decorated function
    :rtype: function
    '''
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return arrays.darray(*method(self, *args, **kwargs), backend=self.backend)
    return wrapper


def return_darray_one_dim(method):
    '''
    Wrapper to automatically create a :class:`minkit.darray` object with
    the output from a method call.

    :param method: method of the class.
    :rtype: function
    :returns: decorated function
    :rtype: function
    '''
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return arrays.darray(method(self, *args, **kwargs), backend=self.backend)
    return wrapper


def return_iarray(method):
    '''
    Wrapper to automatically create a :class:`minkit.iarray` object with
    the output from a method call.

    :param method: method of the class.
    :rtype: function
    :returns: decorated function
    :rtype: function
    '''
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return arrays.iarray(method(self, *args, **kwargs), backend=self.backend)
    return wrapper


class CPUOperations(object):

    def __init__(self, backend, tmpdir=None):
        '''
        Initialize the class with the interface to the user backend.

        :param backend: user interface to the backend.
        :type backend: Backend
        '''
        self.__backend = backend

        # Cache for the PDFs
        self.__cpu_module_cache = {}
        self.__cpu_pdf_cache = {}

        if tmpdir is None:
            self.__tmpdir = tempfile.TemporaryDirectory()
        else:
            self.__tmpdir = tmpdir

        # Random state
        self.__rndm_gen = np.random.RandomState(seed=DEFAULT_SEED)

    @property
    def backend(self):
        '''
        Backend interface.
        '''
        return self.__backend

    @property
    def cpu_module_cache(self):
        '''
        Cache for CPU modules.
        '''
        return self.__cpu_module_cache

    @property
    def cpu_pdf_cache(self):
        '''
        Cache for CPU PDFs.
        '''
        return self.__cpu_pdf_cache

    def _access_cpu_module(self, name, nvar_arg_pars):
        '''
        Access a C++ module, compiling it if it has not been done yet.

        :param name: name of the module.
        :type name: str
        :returns: compiled module.
        :rtype: module
        '''
        modname = core.parse_module_name(name, nvar_arg_pars)

        pdf_paths = core.get_pdf_src()

        if modname in self.__cpu_module_cache:
            # Do not compile again the PDF source if it has already been done
            module = self.__cpu_module_cache[modname]
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
            source = os.path.join(self.__tmpdir.name, f'{modname}.cpp')
            code = autocode.generate_code(xml_source, core.CPU, nvar_arg_pars)
            with open(source, 'wt') as f:
                f.write(code)

            # Compile the C++ code and load the library
            compiler = ccompiler.new_compiler()

            try:
                objects = compiler.compile(
                    [source], output_dir=self.__tmpdir.name,
                    include_dirs=[sysconfig.get_python_inc()],
                    extra_preargs=CFLAGS)
                libname = os.path.join(self.__tmpdir.name, f'lib{modname}.so')
                compiler.link(f'{modname} library', objects, libname,
                              extra_preargs=CFLAGS, libraries=['stdc++', 'gsl', 'gslcblas'])
            except Exception as ex:
                nl = len(str(code.count('\n')))
                code = '\n'.join(f'{i + 1:>{nl}}: {l}' for i,
                                 l in enumerate(code.split('\n')))
                logger.error(f'Error found compiling:\n{code}')
                raise ex

            module = ctypes.cdll.LoadLibrary(libname)

            self.__cpu_module_cache[modname] = module

        return modname, module

    def _create_cpu_function_proxies(self, module, ndata_pars):
        '''
        Create proxies for C++ that handle correctly the input and output data
        types of a function.

        :param module: module where to load the functions from.
        :type module: module
        :param ndata_pars: number of data parameters.
        :type ndata_pars: int
        :returns: proxies for the function, the array-like function and integral.
        :rtype: function, function, function
        '''
        # Get the functions
        function = module.function
        evaluate = module.evaluate
        evaluate_binned_numerical = module.evaluate_binned_numerical

        if hasattr(module, 'evaluate_binned'):
            evaluate_binned = module.evaluate_binned
        else:
            evaluate_binned = None

        if hasattr(module, 'integral'):
            integral = module.integral
        else:
            integral = None

        # Define the types of the input arguments
        partypes = [c_double_p]

        # Define the types passed to the function
        function.argtypes = [c_double_p] + partypes
        function.restype = c_double

        @functools.wraps(function)
        def __function(data, args):
            dv, ac = data_types.data_as_c_double(data, args)
            return function(dv, ac)

        # Define the types for the arguments passed to the evaluate function
        evaluate.argtypes = [c_int, c_double_p,
                             c_int, c_int_p, c_double_p] + partypes

        @functools.wraps(evaluate)
        def __evaluate(output_array, data_idx, input_array, args):

            op, ip, ac = data_types.data_as_c_double(
                output_array.ua, input_array.ua, args)
            di = data_types.data_as_c_int(data_idx)

            l = c_int(len(output_array))
            n = c_int(input_array.ndim)

            return evaluate(l, op, n, di, ip, ac)

        # Define the types for the arguments passed to the evaluate_binned_numerical function
        evaluate_binned_numerical.argtypes = [
            c_int, c_double_p, c_int, c_int_p, c_int_p, c_double_p, c_int] + partypes

        @functools.wraps(evaluate_binned_numerical)
        def __evaluate_binned_numerical(output_array, gaps_idx, gaps, edges, nsteps, args):

            op, ed, ac = data_types.data_as_c_double(
                output_array.ua, edges.ua, args)
            gi, gp = data_types.data_as_c_int(gaps_idx, gaps)

            lg = c_int(len(output_array))
            ng = c_int(len(gaps))
            ns = c_int(nsteps)

            vals = (lg, op, ng, gi, gp, ed, ns)

            return evaluate_binned_numerical(*vals, ac)

        # Check the functions that need the integral to be defined
        if integral is not None and evaluate_binned is None:
            logger.warning(
                'If you are able to define a integral function you can also provide an evaluation for binned data sets.')

        if evaluate_binned is not None:

            # Define the types for the arguments passed to the evaluate_binned function
            evaluate_binned.argtypes = [c_int, c_double_p, c_int,
                                        c_int_p, c_int_p, c_double_p] + partypes

            @functools.wraps(evaluate_binned)
            def __evaluate_binned(output_array, gaps_idx, gaps, edges, args):

                op, ed, ac = data_types.data_as_c_double(
                    output_array.ua, edges.ua, args)
                gi, gp = data_types.data_as_c_int(gaps_idx, gaps)

                l = c_int(len(output_array))

                vals = (l, op, c_int(len(gaps)), gi, gp, ed)

                return evaluate_binned(*vals, ac)
        else:
            __evaluate_binned = None

        if integral is not None:

            # Define the types passed to the integral function
            integral.argtypes = [c_double_p, c_double_p] + partypes
            integral.restype = c_double

            @functools.wraps(integral)
            def __integral(lb, ub, args):

                if lb.ndim == 0:
                    lb, ub = data_types.array_float(
                        lb), data_types.array_float(ub)

                lb, ub, ac = data_types.data_as_c_double(lb, ub, args)

                return integral(lb, ub, ac)
        else:
            __integral = None

        # Parse the numerical integration function
        numerical_integration = gsl_api.parse_functions(module)

        proxy = core.FunctionsProxy(
            __function, __integral, __evaluate, __evaluate_binned, __evaluate_binned_numerical, numerical_integration)

        return proxy

    @core.document_operations_method
    def access_pdf(self, name, ndata_pars, nvar_arg_pars=None):

        modname, module = self._access_cpu_module(name, nvar_arg_pars)

        try:
            if modname in self.__cpu_pdf_cache:
                output = self.__cpu_pdf_cache[modname]
            else:
                output = self._create_cpu_function_proxies(
                    module, ndata_pars)
                self.__cpu_pdf_cache[modname] = output

        except AttributeError:
            raise RuntimeError(
                f'Error loading function "{name}"; make sure that at least "evaluate" and "function" are defined inside the source file')

        return output

    @return_carray
    @core.document_operations_method
    def carange(self, n):
        return np.arange(n, dtype=data_types.cpu_int).astype(data_types.cpu_complex)

    @return_iarray
    @core.document_operations_method
    def iarange(self, n):
        return np.arange(n, dtype=data_types.cpu_int)

    @return_barray
    @core.document_operations_method
    def bones(self, n):
        return np.ones(n, dtype=data_types.cpu_real_bool)

    @return_barray
    @core.document_operations_method
    def bzeros(self, n):
        return np.zeros(n, dtype=data_types.cpu_real_bool)

    @return_darray
    @core.document_operations_method
    def fempty(self, size, ndim=1):
        return np.empty(ndim * size, dtype=data_types.cpu_float), ndim

    @return_iarray
    @core.document_operations_method
    def iempty(self, size):
        return np.empty(size, dtype=data_types.cpu_int)

    @return_darray_one_dim
    @core.document_operations_method
    def fones(self, n):
        return np.ones(n, dtype=data_types.cpu_float)

    @return_darray
    @core.document_operations_method
    def fzeros(self, n, ndim=1):
        return np.zeros(ndim * n, dtype=data_types.cpu_float), ndim

    @return_darray
    @core.document_operations_method
    def concatenate(self, arrays, maximum=None):
        a = np.concatenate(tuple(a.ua for a in arrays))
        if maximum is not None:
            return a[:maximum * arrays[0].ndim], arrays[0].ndim
        else:
            return a, arrays[0].ndim

    @core.document_operations_method
    def count_nonzero(self, a):
        return np.count_nonzero(a.ua)

    @return_carray
    @core.document_operations_method
    def cexp(self, a):
        return np.exp(a.ua)

    @return_darray
    @core.document_operations_method
    def fexp(self, a):
        return np.exp(a.ua), a.ndim

    @return_darray_one_dim
    @core.document_operations_method
    def fftconvolve(self, a, b, data):

        # Calculate the FFT of the input signals
        fa = np.fft.fft(a.ua)
        fb = np.fft.fft(b.ua)

        # Calculate the shift in time
        n0 = np.count_nonzero(data.ua < 0)
        nt = len(data)
        com = data_types.cpu_complex(+2.j * np.pi * n0 / nt)
        rng = self.carange(nt)
        shift = np.exp(com * rng.ua)

        # Calculate the inverse of the FFT
        output = np.fft.ifft(fa * shift * fb)

        return np.real(np.real(output * (data.ua[1] - data.ua[0])))

    @return_barray
    @core.document_operations_method
    def ge(self, a, v):
        return a.ua >= v

    @core.document_operations_method
    def make_linear_interpolator(self, xp, yp):

        def wrapper(idx, x):
            a = np.interp(x.ua[idx::x.ndim], xp.ua, yp.ua)
            return arrays.darray(a, backend=self.backend)

        return wrapper

    @core.document_operations_method
    def make_spline_interpolator(self, xp, yp):

        spline = make_interp_spline(xp.ua, yp.ua, k=3)  # cubic spline

        def wrapper(idx, x):
            a = spline(x.ua[idx::x.ndim])
            return arrays.darray(a, backend=self.backend)

        return wrapper

    @return_barray
    @core.document_operations_method
    def is_inside(self, data, lb, ub):
        if len(lb) > 1:
            ln = len(data)
            c = np.ones(ln, dtype=data_types.cpu_real_bool)
            for i, (l, u) in enumerate(zip(lb, ub)):
                d = data.ua[i::data.ndim]
                np.logical_and(c, np.logical_and(d >= l, d < u), out=c)
        else:
            c = np.logical_and(data.ua >= lb, data.ua < ub)
        return c

    @return_barray
    @core.document_operations_method
    def lt(self, a, v):
        if np.asarray(v).dtype == object:
            return a.ua < v.ua
        else:
            return a.ua < v

    @return_barray
    @core.document_operations_method
    def le(self, a, v):
        return a.ua <= v

    @return_darray_one_dim
    @core.document_operations_method
    def linspace(self, vmin, vmax, size):
        return np.linspace(vmin, vmax, size, dtype=data_types.cpu_float)

    @return_darray
    @core.document_operations_method
    def log(self, a):
        return np.log(a.ua), a.ndim

    @return_barray
    @core.document_operations_method
    def logical_and(self, a, b,  out=None):
        if out is None:
            np.logical_and(a.ua, b.ua)
        else:
            return np.logical_and(a.ua, b.ua, out=out.ua)

    @return_barray
    @core.document_operations_method
    def logical_or(self, a, b, out=None):
        if out is None:
            return np.logical_or(a.ua, b.ua)
        else:
            return np.logical_or(a.ua, b.ua, out=out.ua)

    @core.document_operations_method
    def max(self, a):
        return np.max(a.ua)

    @return_darray
    @core.document_operations_method
    def meshgrid(self, lb, ub, sizes):
        arrays = tuple(np.linspace(l, u, s) for l, u, s in zip(lb, ub, sizes))
        mesh = data_types.array_float([a.flatten()
                                       for a in np.meshgrid(*arrays)])
        return mesh.T.flatten(), len(lb)

    @core.document_operations_method
    def min(self, a):
        return np.min(a.ua)

    @core.document_operations_method
    def ndarray_to_backend(self, a):
        return a, len(a)

    @core.document_operations_method
    def product_by_zero_is_zero(self, f, s):
        c = np.logical_or(f.ua == 0., s.ua == 0.)
        out = f * s
        out.ua[c] = 0.
        return out

    @return_darray
    @core.document_operations_method
    def random_grid(self, lb, ub, n):
        mesh = data_types.array_float(
            [self.__rndm_gen.uniform(l, u, n) for l, u in zip(lb, ub)])
        return mesh.T.flatten(), len(lb)

    @return_darray_one_dim
    @core.document_operations_method
    def random_uniform(self, vmin, vmax, size):
        return self.__rndm_gen.uniform(vmin, vmax, size)

    @return_darray
    @core.document_operations_method
    def restrict_data_size(self, maximum, data):
        return data.ua[:data.ndim * maximum], data.ndim

    @core.document_operations_method
    def set_rndm_seed(self, seed):
        self.__rndm_gen.seed(seed)

    @core.document_operations_method
    def sum(self, a):
        return np.sum(a.ua)

    @return_darray_one_dim
    @core.document_operations_method
    def sum_inside(self, indices, gaps, centers, edges, values=None):

        values = values if values is None else values.ua

        nd = len(gaps)

        c = [centers.ua[i::nd] for i in range(nd)]
        e = [edges.ua[p:n] for p, n in zip(indices[:-1], indices[1:])]

        out, _ = np.histogramdd(c, bins=e, weights=values)

        return out.T.flatten()

    @return_darray
    @core.document_operations_method
    def slice_from_boolean(self, a, valid):
        return a.ua[np.repeat(valid.ua, a.ndim)], a.ndim

    @return_darray
    @core.document_operations_method
    def slice_from_integer(self, a, indices):
        idx = np.repeat(a.ndim * indices.ua, a.ndim) + \
            np.tile(np.arange(a.ndim), len(indices))
        return a.ua[idx], a.ndim

    @return_darray_one_dim
    @core.document_operations_method
    def take_column(self, a, i):
        return a.ua[i::a.ndim]

    @return_darray_one_dim
    @core.document_operations_method
    def take_slice(self, a, start, end):
        return a.ua[start:end]
