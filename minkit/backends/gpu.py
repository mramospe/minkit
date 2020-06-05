########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Operations with GPU objects. All the functions in this module expect objects
of type :class:`reikna.cluda.api.Array` or :class:`numpy.ndarray`.
'''
from . import autocode
from . import arrays
from . import core
from . import cpu
from . import gpu_cache
from . import gpu_core
from .gpu_functions import make_functions
from ..base import data_types

from scipy.interpolate import splrep

import contextlib
import functools
import logging
import numpy as np
import os
import tempfile

# Default seed for the random number generators
DEFAULT_SEED = 49763


logger = logging.getLogger(__name__)


class GPUOperations(object):

    def __init__(self, backend, **kwargs):
        '''
        Initialize the class with the interface to the user backend.

        :param kwargs: it may contain any of the following values: \
        - interactive: (bool) whether to select the device manually (defaults to False) \
        - device: (int) number of the device to use (defaults to None).
        :type kwargs: dict

        .. note:: The device can be selected using the MINKIT_DEVICE environment variable.
        '''
        self.__context = gpu_core.initialize_gpu(
            backend.btype, **kwargs)

        self.__backend = backend

        self.__tmpdir = tempfile.TemporaryDirectory()
        self.__cpu_aop = cpu.CPUOperations(self.__tmpdir)

        # Cache for GPU objects
        self.__array_cache = {}
        self.__fft_cache = gpu_cache.FFTCache(self.__context)

        # Cache for the PDFs
        self.__gpu_module_cache = {}
        self.__gpu_pdf_cache = {}

        # To generate random numbers
        self.__rndm_gen = RandomUniformGenerator(self)

        # Compile the functions
        self.__fbe, self.__rfu, self.__tplf_1d, self.__tplf_2d = make_functions(
            self)

    @property
    def backend(self):
        '''
        Backend interface.
        '''
        return self.__backend

    @property
    def context(self):
        '''
        Proxy for the device.
        '''
        return self.__context

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

    def _access_gpu_module(self, name, nvar_arg_pars):
        '''
        Access a GPU module, compiling it if it has not been done yet.

        :param name: name of the module.
        :type name: str
        :returns: compiled module.
        :rtype: module
        '''
        pdf_paths = core.get_pdf_src()

        modname = core.parse_module_name(name, nvar_arg_pars)

        if modname in self.__gpu_module_cache:
            # Do not compile again the PDF source if it has already been done
            module = self.__gpu_module_cache[modname]
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
            code = autocode.generate_code(xml_source, core.GPU, nvar_arg_pars)
            with open(source, 'wt') as f:
                f.write(code)

            # Compile the code
            with open(source) as fi:
                try:
                    module = self.__context.compile(fi.read())
                except Exception as ex:
                    nl = len(str(code.count('\n')))
                    code = '\n'.join(f'{i + 1:>{nl}}: {l}' for i,
                                     l in enumerate(code.split('\n')))
                    logger.error(f'Error found compiling:\n{code}')
                    raise ex

            self.__gpu_module_cache[modname] = module

        return module

    def _create_gpu_function_proxies(self, module, ndata_pars):
        '''
        Creates a proxy for a function writen in GPU.

        :param module: module containing the function to wrap.
        :type module: module
        :param ndata_pars: number of data parameters.
        :type ndata_pars: int
        :returns: proxy for the array-like function.
        :rtype: function
        '''
        # Access the function in the module
        evaluate = module.evaluate
        evaluate_binned_numerical = module.evaluate_binned_numerical

        try:
            # Can not use "hasattr"
            evaluate_binned = module.evaluate_binned
        except:
            evaluate_binned = None

        @functools.wraps(evaluate)
        def __evaluate(output_array, data_idx, input_array, args):

            ic = self.args_to_array(data_idx, dtype=data_types.cpu_int)

            if len(args) == 0:
                # It seems we can not pass a null pointer in OpenCL
                ac = self.fzeros(1).ua
            else:
                ac = self.args_to_array(args, dtype=data_types.cpu_float)

            global_size, local_size = self.__context.get_sizes(
                output_array.length)

            return evaluate(output_array.length, output_array.ua, input_array.ndim, ic, input_array.ua, ac, global_size=global_size, local_size=local_size)

        @functools.wraps(evaluate_binned_numerical)
        def __evaluate_binned_numerical(output_array, gaps_idx, gaps, edges, nsteps, args):

            gi = self.args_to_array(
                gaps_idx, dtype=data_types.cpu_int)
            gp = self.args_to_array(gaps, dtype=data_types.cpu_int)
            nd = data_types.cpu_int(len(gaps))
            ns = data_types.cpu_int(nsteps)

            if len(args) == 0:
                # It seems we can not pass a null pointer in OpenCL
                ac = self.fzeros(1).ua
            else:
                ac = self.args_to_array(
                    args, dtype=data_types.cpu_float)

            global_size, local_size = self.__context.get_sizes(
                output_array.length)

            return evaluate_binned_numerical(output_array.length, output_array.ua, nd, gi, gp, edges.ua, ns, ac, global_size=global_size, local_size=local_size)

        if evaluate_binned is not None:

            @functools.wraps(evaluate_binned)
            def __evaluate_binned(output_array, gaps_idx, gaps, edges, args):

                gi = self.args_to_array(
                    gaps_idx, dtype=data_types.cpu_int)
                gp = self.args_to_array(gaps, dtype=data_types.cpu_int)
                nd = data_types.cpu_int(len(gaps))

                if len(args) == 0:
                    # It seems we can not pass a null pointer in OpenCL
                    ac = self.fzeros(1).ua
                else:
                    ac = self.args_to_array(
                        args, dtype=data_types.cpu_float)

                global_size, local_size = self.__context.get_sizes(
                    output_array.length)

                return evaluate_binned(output_array.length, output_array.ua, nd, gi, gp, edges.ua, ac, global_size=global_size, local_size=local_size)
        else:
            __evaluate_binned = None

        return __evaluate, __evaluate_binned, __evaluate_binned_numerical

    @core.document_operations_method
    def access_pdf(self, name, ndata_pars, nvar_arg_pars=None):

        # Function and integral are taken from the C++ version
        cpu_proxy = self.__cpu_aop.access_pdf(
            name, ndata_pars, nvar_arg_pars)

        modname = core.parse_module_name(name, nvar_arg_pars)

        # Get the "evaluate" function from source
        if modname in self.__gpu_pdf_cache:
            gpu_functions = self.__gpu_pdf_cache[modname]
        else:
            # Access the GPU module
            gpu_module = self._access_gpu_module(name, nvar_arg_pars)

            gpu_functions = self._create_gpu_function_proxies(
                gpu_module, ndata_pars)

            self.__gpu_pdf_cache[modname] = gpu_functions

        proxy = core.FunctionsProxy(
            cpu_proxy.function, cpu_proxy.integral, *gpu_functions, cpu_proxy.numerical_integral)

        return proxy

    @contextlib.contextmanager
    @core.document_operations_method
    def using_caches(self):
        '''
        Use the caches. Only those caches related to arrays and functions
        (not the PDFs) will be removed.
        '''
        with contextlib.ExitStack() as stack:

            stack.enter_context(self.__fft_cache.activate())
            stack.enter_context(self.__tplf_1d.activate())
            stack.enter_context(self.__tplf_2d.activate())

            for c in self.__array_cache.values():
                stack.enter_context(c.activate())

            for c in self.__rfu:
                stack.enter_context(c.activate())

            yield self

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
            if dtype in (data_types.cpu_float, data_types.cpu_complex):
                c = gpu_cache.FloatArrayCacheManager(
                    self.__backend, self.__context, dtype)
            else:
                c = gpu_cache.ArrayCacheManager(
                    self.__backend, self.__context, dtype)
            self.__array_cache[dtype] = c
        return c

    def reikna_fft(self, a, inverse=False):
        '''
        Get the FFT to calculate the FFT of an array, keeping the compiled
        source in a cache.
        '''
        # Calculate the value
        output = self.get_array_cache(
            data_types.cpu_complex).get_array(len(a), a.ndim)

        self.__fft_cache(output, a, inverse=inverse)

        return output

    @core.document_operations_method
    def carange(self, n):
        return self.__fbe.arange_complex(n, data_types.cpu_float(0))

    @core.document_operations_method
    def iarange(self, n):
        return self.__fbe.arange_int(n, data_types.cpu_int(0))

    def args_to_array(self, a, dtype=data_types.cpu_float):
        if a.dtype != dtype:
            a = a.astype(dtype)
        return self.__context.to_device(a)

    @core.document_operations_method
    def ndarray_to_backend(self, a):
        return self.__context.to_device(a), len(a)

    @core.document_operations_method
    def product_by_zero_is_zero(self, f, s):
        out = self.fempty(len(f))
        gs, ls = self.__context.get_sizes(f.length)
        self.__fbe.product_by_zero_is_zero(f.length, out.ua, f.ua, s.ua,
                                           global_size=gs, local_size=ls)
        return out

    @core.document_operations_method
    def concatenate(self, arrs, maximum=None):

        # Calculate the length of the output array
        maximum = data_types.cpu_int(maximum) if maximum is not None else np.sum(
            np.fromiter(map(len, arrs), dtype=data_types.cpu_int))

        # Parse the data type
        dtype = arrs[0].dtype

        ndim = arrs[0].ndim

        if dtype == data_types.cpu_float:
            function = self.__fbe.assign_double_with_offset
            out = self.fzeros(maximum, ndim)
        elif dtype == data_types.cpu_bool:
            function = self.__fbe.assign_bool_with_offset
            out = self.bempty(maximum)
        else:
            raise NotImplementedError(
                f'Function not implemented for data type "{dtype}"')

        # Looop over the arrays till the output has the desired length
        offset = data_types.cpu_int(0)
        for a in arrs:

            l = a.length

            # how many we have to process
            m = ndim * data_types.cpu_int(
                l if l + offset <= maximum else maximum - offset)

            gs, ls = self.__context.get_sizes(a.length)

            function(m, out.ua, a.ua, ndim * offset,
                     global_size=gs, local_size=ls)

            offset += l

        return out

    @core.document_operations_method
    def count_nonzero(self, a):
        return self.__rfu.count_nonzero(a)

    @core.document_operations_method
    def bempty(self, size):
        return self.get_array_cache(data_types.cpu_bool).get_array(size)

    @core.document_operations_method
    def fempty(self, size, ndim=1):
        return self.get_array_cache(data_types.cpu_float).get_array(size, ndim)

    @core.document_operations_method
    def iempty(self, size):
        return self.get_array_cache(data_types.cpu_int).get_array(size)

    @core.document_operations_method
    def fones(self, n):
        return self.__fbe.ones_double(n)

    @core.document_operations_method
    def bones(self, n):
        return self.__fbe.ones_bool(n)

    @core.document_operations_method
    def fzeros(self, n, ndim=1):
        out = self.get_array_cache(data_types.cpu_float).get_array(n, ndim)
        gs, ls = self.__context.get_sizes(out.size)
        self.__fbe.zeros_double(
            out.size, out.ua, global_size=gs, local_size=ls)
        return out

    @core.document_operations_method
    def bzeros(self, n):
        return self.__fbe.zeros_bool(n)

    @core.document_operations_method
    def cexp(self, a):
        return self.__fbe.exponential_complex(a)

    @core.document_operations_method
    def fexp(self, a):
        return self.__fbe.exponential_double(a)

    @core.document_operations_method
    def fftconvolve(self, a, b, data):

        # Calculate the FFT of the input signals
        fa = self.reikna_fft(a.astype(data_types.cpu_complex))
        fb = self.reikna_fft(b.astype(data_types.cpu_complex))

        # Calculate the shift
        n0 = self.count_nonzero(self.lt(data, 0))
        nt = len(data)
        com = data_types.cpu_complex(+2.j * np.pi * n0 / nt)
        rng = self.carange(nt)

        shift = self.cexp(com * rng)

        fa *= shift
        fa *= fb  # avoid creating extra arrays

        # Calculate the inverse FFT
        output = self.reikna_fft(fa, inverse=True)

        output *= (data.get(1) - data.get(0))

        return self.__fbe.real(output)

    @core.document_operations_method
    def ge(self, a, v):
        return self.__fbe.ge(a, data_types.cpu_float(v))

    @core.document_operations_method
    def make_linear_interpolator(self, xp, yp):

        def wrapper(idx, x):

            idx = data_types.as_integer(idx)

            out = self.fempty(x.length)

            gs_x, ls_x, gs_y, ls_y = self.__context.get_sizes(
                xp.length, out.length)

            self.__fbe.interpolate_linear(xp.length, x.length, out.ua, x.ndim, idx, x.ua, xp.ua, yp.ua,
                                          global_size=(gs_x, gs_y), local_size=(ls_x, ls_y))

            return out

        return wrapper

    @core.document_operations_method
    def make_spline_interpolator(self, xp, yp):

        # cubic spline (must modify source code if changed)
        k = data_types.cpu_int(3)

        t, c, _ = splrep(xp.as_ndarray(), yp.as_ndarray(), k=k)

        t = arrays.darray.from_ndarray(t, self.backend)
        c = arrays.darray.from_ndarray(c, self.backend)

        def wrapper(idx, x):

            idx = data_types.as_integer(idx)

            out = self.fempty(len(x))

            gs_x, ls_x, gs_y, ls_y = self.__context.get_sizes(
                t.length, out.length)

            lt = data_types.as_integer(len(t) - k - 1)  # len(t) - k - 1

            self.__fbe.interpolate_spline(lt, out.length, out.ua, x.ndim, idx, x.ua, t.ua, c.ua,
                                          global_size=(gs_x, gs_y), local_size=(ls_x, ls_y))

            return out

        return wrapper

    @core.document_operations_method
    def is_inside(self, data, lb, ub):

        if lb.ndim == 0:
            lb = self.__context.to_device(data_types.array_float([lb]))
            ub = self.__context.to_device(data_types.array_float([ub]))
        else:
            lb = self.__context.to_device(lb)
            ub = self.__context.to_device(ub)

        lgth = data.length // data.ndim

        out = self.bempty(lgth)

        gs, ls = self.__context.get_sizes(out.length)

        self.__fbe.is_inside(lgth, out.ua, data.ndim, data.ua, lb, ub,
                             global_size=gs, local_size=ls)

        return out

    @core.document_operations_method
    def restrict_data_size(self, maximum, data):

        maximum = data_types.as_integer(maximum)

        out = self.get_array_cache(
            data_types.cpu_float).get_array(maximum, data.ndim)

        gs, ls = self.__context.get_sizes(out.length)

        self.__fbe.assign_double(maximum,
                                 out.ua, data.ndim, data.ua, global_size=gs, local_size=ls)

        return out

    @core.document_operations_method
    def lt(self, a, v):
        if np.asarray(v).dtype == object:
            return self.__fbe.alt(a, v.ua)
        else:
            return self.__fbe.lt(a, data_types.cpu_float(v))

    @core.document_operations_method
    def le(self, a, v):
        return self.__fbe.le(a, data_types.cpu_float(v))

    @core.document_operations_method
    def linspace(self, vmin, vmax, size):
        return self.__fbe.linspace(size,
                                   data_types.cpu_float(vmin),
                                   data_types.cpu_float(vmax),
                                   data_types.cpu_int(size))

    @core.document_operations_method
    def log(self, a):
        return self.__fbe.logarithm(a)

    @core.document_operations_method
    def logical_and(self, a, b, out=None):
        if out is None:
            return self.__fbe.logical_and(a, b.ua)
        else:
            return self.__fbe.logical_and_to_output(out, a.ua, b.ua)

    @core.document_operations_method
    def logical_or(self, a, b, out=None):
        if out is None:
            return self.__fbe.logical_or(a, b.ua)
        else:
            return self.__fbe.logical_or_to_output(out, a.ua, b.ua)

    @core.document_operations_method
    def max(self, a):
        return self.__rfu.amax(a)

    @core.document_operations_method
    def meshgrid(self, lb, ub, n):

        # Send to the device the lower bounds and the steps
        steps = self.__context.to_device((ub - lb) / (n - 1))
        lb = self.__context.to_device(lb)

        # Get dimension, length and the minimum size of the output array
        ndim = data_types.cpu_int(len(n))
        lgth = np.prod(n)

        # Calculate the gaps
        gaps = self.__context.to_device(
            np.cumprod(n, dtype=data_types.cpu_int) // n[0])

        # Evaluate the function
        out = self.fzeros(lgth, ndim)

        gs, ls = self.__context.get_sizes(out.length)

        self.__fbe.meshgrid(out.length, out.ua, ndim, gaps, lb, steps,
                            global_size=gs, local_size=ls)

        return out

    @core.document_operations_method
    def min(self, a):
        return self.__rfu.amin(a)

    @core.document_operations_method
    def random_grid(self, lb, ub, n):

        # Send the lower and upper bounds to the device
        lb = self.__context.to_device(lb)
        ub = self.__context.to_device(ub)

        # Calculate the dimension, length and the minimum size of the output array
        ndim = data_types.cpu_int(len(lb))

        # Get the random number generator
        dest = self.__rndm_gen.generate(n, ndim)

        # Build the output array
        out = self.fzeros(n, ndim)

        gs, ls = self.__context.get_sizes(out.length // ndim)

        self.__fbe.parse_random_grid(out.length, out.ua, ndim, lb, ub, dest.ua,
                                     global_size=gs, local_size=ls)

        return out

    @core.document_operations_method
    def random_uniform(self, vmin, vmax, size):

        a = self.__rndm_gen.generate(size)

        a *= (vmax - vmin)
        a += vmin

        return a

    @core.document_operations_method
    def set_rndm_seed(self, seed):
        self.__rndm_gen.seed(seed)

    @core.document_operations_method
    def sum(self, a):
        return self.__rfu.rsum(a)

    @core.document_operations_method
    def sum_inside(self, indices, gaps, centers, edges, values=None):

        lgth = centers.length

        gaps = self.__context.to_device(gaps)

        nbins = data_types.cpu_int(np.prod(indices[1:] - indices[:-1] - 1))

        # Get entries per bin in different blocks
        gs_data, ls_data, gs_bins, ls_bins = self.__context.get_sizes(
            lgth, nbins)

        ndata_blocks = data_types.cpu_int(gs_data // ls_data)

        partial_sum = self.fzeros(nbins * ndata_blocks)

        if values is None:
            self.__tplf_2d.get_object((ls_data, ls_bins)).sum_inside_bins(lgth, nbins, partial_sum.ua, centers.ndim, centers.ua, gaps, edges.ua,
                                                                          global_size=(gs_data, gs_bins), local_size=(ls_data, ls_bins))
        else:
            self.__tplf_2d.get_object((ls_data, ls_bins)).sum_inside_bins_with_values(lgth, nbins, partial_sum.ua, centers.ndim, centers.ua, gaps, edges.ua, values.ua,
                                                                                      global_size=(gs_data, gs_bins), local_size=(ls_data, ls_bins))

        # Sum entries in each bin
        out = self.fzeros(nbins)

        self.__fbe.stepped_sum(out.length, out.ua, nbins, ndata_blocks, partial_sum.ua,
                               global_size=gs_bins, local_size=ls_bins)

        return out

    @core.document_operations_method
    def invalid_indices(self, l):
        out = self.iempty(l)
        gs, ls = self.__context.get_sizes(out.length)
        self.__fbe.invalid_indices(
            out.length, out.ua, global_size=gs, local_size=ls)
        return out

    @core.document_operations_method
    def slice_from_boolean(self, a, valid):

        nz = data_types.cpu_int(self.count_nonzero(valid))

        if nz == 0:
            return self.fempty(0, a.ndim)  # empty array

        # Calculate the compact indices
        indices = self.__fbe.invalid_indices(len(valid))

        gs, ls = self.__context.get_sizes(indices.length)

        sizes = self.iempty(gs // ls)

        self.__tplf_1d.get_object(ls).compact_indices(indices.length, indices.ua, sizes.ua,
                                                      valid.ua, global_size=gs, local_size=ls)

        # Build the output array
        out = self.fzeros(nz, a.ndim)

        self.__fbe.take(indices.length, out.ua, a.ndim, sizes.ua,
                        indices.ua, a.ua, global_size=gs, local_size=ls)

        return out

    @core.document_operations_method
    def slice_from_integer(self, a, indices):

        l = len(indices)

        out = self.fzeros(l, a.ndim)

        gs, ls = self.__context.get_sizes(l)

        self.__fbe.slice_from_integer(out.length, out.ua, a.ndim, a.ua, indices.ua,
                                      global_size=gs, local_size=ls)

        return out

    @core.document_operations_method
    def take_column(self, a, i):

        i = data_types.as_integer(i)

        out = self.fzeros(a.length)

        gs, ls = self.__context.get_sizes(out.length)

        self.__fbe.take_each(out.length, out.ua, a.ua, a.ndim,
                             i, global_size=gs, local_size=ls)

        return out

    @core.document_operations_method
    def take_slice(self, a, start, end):

        start, end = data_types.as_integer(start, end)

        out = self.fzeros(end - start)

        gs, ls = self.__context.get_sizes(out.length)

        self.__fbe.take_slice(out.ua, a.ua, start, end,
                              global_size=gs, local_size=ls)

        return out


class RandomUniformGenerator(object):

    def __init__(self, aop, seed=DEFAULT_SEED):
        '''
        Object to manage the generation of random numbers in GPU.

        :param aop: object to do array operations.
        :type aop: ArrayOperations
        '''
        super().__init__()

        with open(os.path.join(gpu_core.GPU_SRC, 'philox.c')) as f:
            self.__module = aop.context.compile(f.read())

        self.__aop = aop

        self.seed(seed)  # set the seed and the counter

        self.rounds = 10  # default in Range123

    @property
    def rounds(self):
        '''
        Number of rounds to do in the generation.
        '''
        return self.__rounds

    @rounds.setter
    def rounds(self, r):
        self.__rounds = data_types.cpu_int(r)

    def seed(self, seed):
        '''
        Set the seed of the generator. This will compile again the source
        code using a different value for the seed.

        :param seed: new seed.
        :type seed: int
        '''
        self.__counter = data_types.cpu_int(0)
        self.__seed = data_types.cpu_int(seed)

    def generate(self, lgth, ndim=1):
        '''
        Generate random numbers between 0 and 1.

        :param lgth: length of the output array.
        :type lgth: int
        :param ndim: dimensions of the output array.
        :type ndim: int
        :return: Array with values between 0 and 1.
        :rtype: darray
        '''
        dest = self.__aop.fzeros(lgth, ndim)

        l = (dest.size // 4) + (dest.size %
                                4 != 0)  # each thread sets 4 numbers

        gs, ls = self.__aop.context.get_sizes(l)

        self.__module.philox(dest.size, dest.ua, self.__seed, self.__counter, self.__rounds,
                             global_size=gs, local_size=ls)

        self.__counter += dest.size  # update the counter to generate independent samples

        return dest
