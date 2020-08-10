########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Base classes to define PDF data_types.
'''
from . import dataset
from ..backends import autocode
from ..backends import gsl_api

from ..backends.arrays import darray
from ..backends.core import add_pdf_src, common_backend, is_gpu_backend, parse_backend
from ..base import core
from ..base import bindings
from ..base import data_types
from ..base import dependencies
from ..base import exceptions
from ..base import parameters

from scipy.optimize import curve_fit as scipy_curve_fit
from scipy.optimize import minimize as scipy_minimize

import contextlib
import functools
import logging
import numpy as np
import os
import tempfile
import textwrap
import threading
import warnings

__all__ = ['AddPDFs', 'ConvPDFs', 'FormulaPDF', 'HistPDF', 'InterpPDF', 'MultiPDF', 'pdf_from_json', 'pdf_to_json',
           'PDF', 'ProdPDFs', 'register_pdf', 'SourcePDF']

# Number of points to use to evaluate a binned sample
DEFAULT_EVB_SIZE = 100

# Default size of the grid for the convolution PDFs
DEFAULT_INTERP_SIZE = 10000

# Default number of events to generate in a single step
DEFAULT_GEN_SIZE_CPU = 10000
DEFAULT_GEN_SIZE_GPU = 100000

# Tolerance on the determination of maximum values
MAX_TOL = 1e-7
MAX_CALL = None

# Cache types
BIND = 'bind'
CONST = 'const'

CACHE_TYPES = (BIND, CONST)

# Registry of PDF classes so they can be built by name
PDF_REGISTRY = {}

logger = logging.getLogger(__name__)


def asymptotic_maximum(x, a, b, c):
    '''
    Function that represents the evolution of a maximum value as a function
    of the amount of random numbers generated.

    :param x: amount of random numbers generated.
    :type x: numpy.ndarray
    :param a: scale.
    :type a: float
    :param b: slope.
    :type b: float
    :param c: asymptote.
    :type c: float
    '''
    return a * np.exp(-b * x) + c


def register_pdf(cl):
    '''
    Decorator to register PDF classes in the PDF_REGISTRY registry.

    :param cl: decorated class.
    :type cl: class
    :returns: Class descriptor.
    :rtype: type
    '''
    if cl.__name__ not in PDF_REGISTRY:
        PDF_REGISTRY[cl.__name__] = cl
    return cl


def process_cache(cache, method, self, args, kwargs):
    '''
    Update a cache (as a dictionary) by evaluating the given method.

    :param cache: output cache
    :type cache: dict
    :param method: method to call.
    :type method: function
    :param self: class where to evaluate the method.
    :type self: class
    :param args: arguments to forward to the method call.
    :type args: tuple
    :param kwargs: keyword arguments to forward to the method call.
    :type kwargs: dict
    :returns: Whatever "method" returns.
    :rtype: type of whatever "method" returns.
    '''
    n = method.__name__
    v = cache.get(n, None)
    if v is None:
        v = method(self, *args, **kwargs)
        cache[n] = v
    return v


def allows_bind_cache(method):
    '''
    Wrap the method of a class, using a cache for calls with the same binded
    arguments.
    '''
    @functools.wraps(method)
    def __wrapper(self, *args, **kwargs):
        if self.cache_type == BIND:
            return process_cache(self.cache, method, self, args, kwargs)
        else:
            return method(self, *args, **kwargs)
    return __wrapper


def allows_const_cache(method):
    '''
    Wrap the method of a class, using a cache if the class is marked as constant.
    '''
    @functools.wraps(method)
    def __wrapper(self, *args, **kwargs):
        if self.cache_type == CONST and self.constant:
            return process_cache(self.cache, method, self, args, kwargs)
        else:
            return method(self, *args, **kwargs)
    return __wrapper


def safe_division(first, second):
    '''
    Do a safe division by a number of array that can be or contain zeros.

    :param first: input array or value.
    :type first: float, farray
    :param second: array or value to divide by.
    :type second: float, farray
    :returns: Result of the division.
    :rtype: farray
    '''
    try:
        return first / second
    except ZeroDivisionError:
        return first / np.nextafter(0, np.infty)


def _interpolation_evaluate_binned(aop, interpolator, par, data, evb_size, normalized=True):
    '''
    Evaluate on a binned sample using the given interpolator.

    :param aop: instance to do array operations.
    :type aop: ArrayOperations
    :param interpolator: instance to interpolate.
    :param par: parameter in the interpolation.
    :type par: Parameter
    :param data: data samples.
    :type data: BinnedDataSet
    :param evb_size: number of elements to consider per bin.
    :type evb_size: int
    :param normalized: whether to normalized the output.
    :type normalized: bool
    :returns: interpolated values.
    :rtype: darray
    '''
    edges = data[par.name]

    nsteps = evb_size if evb_size % 2 != 0 else evb_size + 1

    grid = dataset.adaptive_grid(aop, par, edges, nsteps)

    pdf_values = interpolator.interpolate(0, grid.values)
    pdf_values *= aop.simpson_factors(nsteps, len(edges) - 1)

    gaps = data_types.full_int(1, 1)
    # dataset.edges_indices(gaps, edges)
    indices = data_types.array_int([0, len(edges)])

    out = aop.sum_inside(indices, gaps, grid.values, edges, pdf_values)
    out *= aop.steps_from_edges(edges)
    out /= (3. * (nsteps - 1))

    if normalized:
        return safe_division(out, out.sum())
    else:
        return out


class BasePDF(object, metaclass=core.DocMeta):

    # Allow to define if a PDF object depends on other PDFs. All PDF objects
    # with this value set to "True" must have the property "pdfs" implemented.
    _dependent = False

    _has_args = False

    def __init__(self, name, data_pars, backend):

        super().__init__()

        self.name = name
        self.__data_pars = parameters.Registry(data_pars)
        self.__aop = parse_backend(backend)

    @property
    def all_pars(self):
        '''
        All the parameters associated to this class.
        This includes also any :class:`ParameterFormula` and the data parameters.

        :type: Registry(Parameter)
        '''
        return self.data_pars

    @property
    def aop(self):
        '''
        Object to do operations on arrays.

        :type: ArrayOperations
        '''
        return self.__aop

    @property
    def backend(self):
        '''
        Backend where this PDF lives.
        '''
        return self.__aop.backend

    @property
    def data_pars(self):
        '''
        Data parameters.
        '''
        return self.__data_pars

    @classmethod
    def from_json_object(cls, obj, pars, backend):
        '''
        Build a PDF from a JSON object.
        This object must represent the internal structure of the PDF.

        :param obj: JSON object.
        :type obj: dict
        :param pars: parameter to build the PDF.
        :type pars: Registry
        :param backend: backend to build the PDF.
        :type backend: Backend
        :returns: This type of PDF constructed together with its parameters.

        .. seealso:: :meth:`PDF.to_json_object`
        '''
        class_name = obj['class']
        if class_name == 'BasePDF' or class_name == 'PDF':
            raise exceptions.MethodNotDefinedError(cls, 'from_json_object')
        cl = PDF_REGISTRY.get(class_name, None)
        if cl is None:
            raise RuntimeError(
                f'Class "{class_name}" does not appear in the registry')
        # Use the correct constructor
        return cl.from_json_object(obj, pars, backend)


class PDF(BasePDF):

    _has_args = True

    def __init__(self, name, data_pars, arg_pars=None, backend=None):
        '''
        Build the class from a name, a set of data parameters and argument parameters.
        The first correspond to those that must be present on any :class:`DataSet` or
        :class:`BinnedDataSet` classes where this object is evaluated.
        The second corresponds to the parameters defining the shape of the PDF.

        :param name: name of the object.
        :type name: str
        :param data_pars: data parameters.
        :type data_pars: Registry(Parameter)
        :param arg_pars: argument parameters.
        :type arg_pars: Registry(Parameter) or None
        :param backend: backend where the PDF will operate.
        :type backend: Backend or None
        :ivar name: name of the PDF.
        :ivar evb_size: number of points to use when evaluating the PDF
           numerically on a binned data set.
        '''
        # Number of points to consider when evaluating numerically a binned sample
        self.evb_size = DEFAULT_EVB_SIZE
        self.__arg_pars = arg_pars if arg_pars is not None else parameters.Registry()

        # The cache is saved on a dictionary, and inherited classes must avoid colliding names
        self.__cache = None
        self.__cache_type = None

        super().__init__(name, data_pars, backend)

    @allows_const_cache
    def __call__(self, data, range=parameters.FULL, normalized=True):
        '''
        Call the PDF in the given set of data.

        :param data: data to evaluate.
        :type data: DataSet or BinnedDataSet
        :param range: normalization range.
        :type range: str
        :param normalized: whether to return a normalized output.
        :type normalized: bool
        :returns: Evaluation of the PDF.
        :rtype: darray
        '''
        raise exceptions.MethodNotDefinedError(self.__class__, '__call__')

    def __repr__(self):
        '''
        Attributes of the class as a string.
        '''
        out = f'{self.__class__.__name__}({self.name}):{os.linesep}'

        out += textwrap.indent(f'Data parameters:{os.linesep}' + textwrap.indent(os.linesep.join(tuple(str(p)
                                                                                                       for p in self.data_pars)), ' '), ' ')

        if len(self.args) != 0:
            out += textwrap.indent(f'{os.linesep}Arguments:{os.linesep}' + textwrap.indent(os.linesep.join(tuple(str(p)
                                                                                                                 for p in self.args)), ' '), ' ')
        return out

    def _enable_cache(self, ctype):
        '''
        Method to enable the cache for values of this PDF.

        :param ctype: cache type, it can be any of ('bind', 'const').
        :type ctype: str
        :raise ValueError: If the cache type is unknown.
        '''
        if ctype not in CACHE_TYPES:
            raise ValueError(
                f'Unknown cache type "{ctype}"; select from: {CACHE_TYPES}')
        self.__cache = {}
        self.__cache_type = ctype

    def _free_cache(self):
        '''
        Free the cache from memory.
        '''
        self.__cache = None
        self.__cache_type = None

    def _max(self, bounds, normalized, range=parameters.FULL, tol=MAX_TOL, min_points=4, max_call=MAX_CALL, sampling_size=None):
        '''
        Calculate the maximum value of the PDF in the given range.

        :param bounds: bounds of the data parameters.
        :type bounds: numpy.ndarray
        :param normalized: if set to True, the result corresponds to the
           maximum of the normalized PDF.
        :type normalized: bool
        :param range: normalization range.
        :type range: str
        :param tol: relative dispersion that is allowed for convergence.
        :type tol: float
        :param min_points: minimum number of points to estimate the maximum. It
           must be greater than four.
        :type min_points: int
        :param max_call: maximum number of iterations to perform.
        :type max_call: int
        :param sampling_size: number of elements to generate in each call.
        :type sampling_size: int
        :type scipy_fmin_opts: dict
        :returns: Maximum value.
        :rtype: float
        '''
        if sampling_size is None:
            if is_gpu_backend(self.aop.backend.btype):
                sampling_size = DEFAULT_GEN_SIZE_GPU
            else:
                sampling_size = DEFAULT_GEN_SIZE_CPU

        if min_points > 4:
            raise ValueError(
                'Minimum number of iterations must be greater than four')

        if max_call is not None and max_call < min_points:
            raise ValueError(
                'Maximum number of iterations must be greater than the minimum number of points')

        maximums = []

        new_max = 0.

        i = 0

        while True:

            # calculate a new maximum
            grid = dataset.uniform_sample(
                self.aop, self.data_pars, bounds, sampling_size)

            imax = self.__call__(grid, range, normalized).argmax()

            initials = grid.get(imax)

            # calculate the maximum from a fit
            def wrapper(values):
                for p, v in zip(self.data_pars, values):
                    p.value = v
                # we are minimizing
                return -1 * self.function(range, normalized)

            bds = np.array([p.bounds for p in self.data_pars])

            with self.restoring_state():

                res = scipy_minimize(wrapper, initials, bounds=bds)

                if not res.success:  # use only successful minimizations
                    continue

                # invert to get the maximum
                maximums.append(-1 * wrapper(res.x))

            new_max = max(maximums[-1], new_max)

            if i + 1 >= min_points:

                m = np.array(maximums)

                maxs = m[(tol * new_max + m) > new_max]

                if len(maxs) >= min_points:

                    if np.allclose(maxs[1:] / maxs[:-1], 1., rtol=tol):
                        # the fit would fail, since they are all equal
                        return new_max

                    x = np.arange(len(maxs))
                    y = np.sort(maxs)

                    popt, pcov = scipy_curve_fit(asymptotic_maximum, x, y,
                                                 bounds=([-np.infty, 0, maxs[-1]], [0, np.infty, np.infty]))

                    if np.sqrt(pcov[-1][-1]) / popt[-1] < tol:  # valid maximum
                        return popt[-1]  # asymptote

                    maximums = maxs.tolist()  # avoid large lists of values

            if max_call is not None and i == max_call:
                break
            else:
                i += 1

        warnings.warn(
            'Maximum number of function evaluations reached to calculate the maximum value', RuntimeWarning)

        return new_max

    def copy(self, backend=None):
        '''
        Create a copy of this PDF.

        :param backend: new backend.
        :type backend: Backend or None
        :returns: A copy of this PDF.
        '''
        return pdf_from_json(pdf_to_json(self), backend)

    @allows_const_cache
    def evaluate_binned(self, data, normalized=True):
        '''
        Evaluate the PDF over a binned sample.

        :param data: input data.
        :type data: BinnedDataSet
        :param normalized: whether to normalize the output array or not.
        :type normalized: bool
        :returns: Values from the evaluation.
        :rtype: farray
        '''
        raise exceptions.MethodNotDefinedError(
            self.__class__, 'evaluate_binned')

    def _generate_single_bounds(self, size, mapsize, gensize, bounds, max_opts=None):
        '''
        Generate data in a single range given the bounds of the different data parameters.

        :param size: size (or minimum size) of the output sample.
        :type size: int
        :param mapsize: number of points to consider per dimension (data parameter)
           in order to calculate the maximum value of the PDF.
        :type mapsize: int
        :param gensize: number of entries to generate per iteration.
        :type gensize: int
        :param max_opts: keyword arguments to be passed to the :meth:`PDF.max`
           function. Allowed arguments are shown below.
        :type max_opts: dict or None
        :param bounds: bounds of the different data parameters (must be sorted).
        :type bounds: numpy.ndarray
        :returns: Output sample.
        :rtype: DataSet

        Arguments that can be passed in *max_opts* are:

        * *tol*: relative tolerance allowed for the determination of the maximum.

        * *max_call*: maximum number of iterations allowed.

        * *sampling_size*: size of the samples used to calculate the maximum.

        .. seealso:: :meth:`PDF.max`
        '''
        max_opts = max_opts or {}

        m = self._max(bounds, normalized=False, **max_opts)

        samples = []

        n = 0
        while n < size:

            d = dataset.uniform_sample(
                self.aop, self.data_pars, bounds, gensize)

            f = self.__call__(d, normalized=False)

            u = self.aop.random_uniform(0, m, len(d))

            r = d.subset(u < f)

            n += len(r)

            samples.append(r)

        return dataset.DataSet.merge(samples, maximum=size)

    @property
    def all_args(self):
        '''
        All the argument parameters associated to this class.

        :type: Registry(Parameter)
        '''
        args = parameters.Registry(self.args)
        for p in filter(lambda p: p._dependent, self.args):
            args += p.all_args
        return args

    @property
    def all_pars(self):
        '''
        All the parameters associated to this class.
        This includes also any :class:`ParameterFormula` and the data parameters.

        :type: Registry(Parameter)
        '''
        return self.data_pars + self.all_args

    @property
    def all_real_args(self):
        '''
        All the argument parameters that are not :class:`ParameterFormula` instances.

        :type: Registry(Parameter)
        '''
        return parameters.Registry(filter(lambda p: not p._dependent, self.all_args))

    @property
    def args(self):
        '''
        Argument parameters this object directly depends on.

        :type: Registry(Parameter)
        '''
        return self.__arg_pars

    @property
    def real_args(self):
        '''
        Arguments that do not depend on other arguments.

        :type: Registry(Parameter)
        '''
        return parameters.Registry(filter(lambda p: not p._dependent, self.__arg_pars))

    @property
    def cache(self):
        '''
        Return the cached values for this PDF.

        :type: dict
        '''
        return self.__cache

    @property
    def cache_type(self):
        '''
        Cache type being used.

        :type: str or None
        '''
        return self.__cache_type

    @property
    def constant(self):
        '''
        Whether this is a constant PDF.

        :type: bool
        '''
        return all(p.constant for p in self.all_real_args)

    @contextlib.contextmanager
    def bind(self, range=parameters.FULL, normalized=True):
        '''
        Prepare an object that will be called many times with the same set of
        values.
        This is usefull for PDFs using a cache, to avoid creating it many times
        in successive calls to :meth:`PDF.__call__`.

        :param range: normalization range.
        :type range: str
        :param normalized: whether to return a normalized output.
        :type normalized: bool
        '''
        with self.using_cache(BIND):
            yield bindings.bind_class_arguments(self, range=range, normalized=normalized)

    def generate(self, size=10000, mapsize=1000, gensize=None, range=parameters.FULL, max_opts=None):
        '''
        Generate random data.
        A call to :meth:`PDF.bind` is implicit, since several calls will be done
        to the PDF with the same set of values.

        :param size: size (or minimum size) of the output sample.
        :type size: int
        :param mapsize: number of points to consider per dimension (data parameter)
           in order to calculate the maximum value of the PDF.
        :type mapsize: int
        :param gensize: number of entries to generate per iteration. By default it
           is set to :math:`10^4` and :math:`10^5` for CPU and GPU backends,
           respectively.
        :type gensize: int or None
        :param max_opts: keyword arguments to be passed to the :meth:`PDF.max`
           function. Allowed arguments are shown below.
        :type max_opts: dict or None
        :param range: range of the data parameters where to generate data.
        :type range: str
        :returns: Output sample.
        :rtype: DataSet

        Arguments that can be passed in *max_opts* are:

        * *tol*: relative tolerance allowed for the determination of the maximum.

        * *max_call*: maximum number of iterations allowed.

        * *sampling_size*: size of the samples used to calculate the maximum.

        .. seealso:: :meth:`PDF.max`
        '''
        if gensize is None:
            if is_gpu_backend(self.aop.backend.btype):
                gensize = DEFAULT_GEN_SIZE_GPU
            else:
                gensize = DEFAULT_GEN_SIZE_CPU

        with self.bind(range, normalized=False) as proxy:

            bounds = parameters.bounds_for_range(proxy.data_pars, range)

            if len(bounds) == 1:
                result = proxy._generate_single_bounds(
                    size, mapsize, gensize, bounds[0], max_opts)
            else:
                # Get the associated number of entries per bounds
                fracs = data_types.empty_float(len(bounds))
                # The last does not need to be calculated
                for i, b in enumerate(bounds):
                    fracs[i] = proxy._raw_integral_for_bounds(b)

                fracs /= fracs.sum()

                u = self.aop.random_uniform(0, 1, size)

                entries = data_types.empty_int(len(bounds))
                for i, f in enumerate(fracs[:-1]):
                    entries[i] = (u < f).count_nonzero()

                # The last is calculated from the required number of entries
                entries[-1] = size - entries[:-1].sum()

                # Iterate over the bounds and add data accordingly
                samples = []
                for e, b in zip(entries, bounds):
                    samples.append(proxy._generate_single_bounds(
                        e, mapsize, gensize, b, max_opts))
                result = dataset.DataSet.merge(samples)

        return result

    def get_values(self):
        '''
        Get the values of the parameters within this object.

        :returns: Dictionary with the values of the parameters.
        :rtype: dict(str, float)
        '''
        return {p.name: p.value for p in self.all_real_args}

    def integral(self, integral_range=parameters.FULL, range=parameters.FULL):
        '''
        Calculate the integral of a :class:`PDF`.

        :param integral_range: range of the integral to compute.
        :type integral_range: str
        :param range: normalization range to consider.
        :type range: str
        :returns: Integral of the PDF in the range defined by "integral_range" normalized to "range".
        :rtype: float
        '''
        return self.numerical_integral(integral_range, range)

    def max(self, range=parameters.FULL, normalized=True, tol=MAX_TOL, min_points=4, max_call=MAX_CALL, sampling_size=None):
        r'''
        Calculate the maximum value of the PDF in the given range.

        :param range: range to consider.
        :type range: str
        :param normalized: if set to True, the result corresponds to the
           maximum of the normalized PDF.
        :type normalized: bool
        :param sampling_size: size of the samples to use to determine the maximum.
        :type sampling_size: int
        :param tol: relative dispersion that is allowed for convergence.
        :type tol: float
        :param min_points: minimum number of points to estimate the maximum. It
           must be greater than four.
        :type min_points: int
        :param max_call: maximum number of iterations to perform. Must be greater
           than two.
        :type max_call: int
        :param sampling_size: number of elements to generate in each call.
        :type sampling_size: int
        :returns: Maximum value.
        :rtype: float

        The maximum is calculated from successive random samples of size
        equal to *sampling_size*. Afterwards, a fit is done in order to polish
        the minimum. Once the minimization was done at least *min_points* times,
        a final fit to the function

        .. math::
           f(n) = \alpha e^{-\beta n} + \delta

        is done, where *n* is the iteration number and :math:`f(n)` is the
        global maximum in that iteration. The value of :math:`\delta` is
        returned as long as the error satisfies the given tolerance. To improve
        the performance, only the iterations where the achieved maximum is
        within the tolerance with respect to the global maximum are considered.
        The parameter :math:`\delta` is minimized in such a way that it is
        always greater or equal to the global maximum achieved from the random
        samples.

        .. note::
           For two-dimensional PDFs, the process of calculating the maximum
           can turn really slow, so you might want to consider reducing the
           precision of the maximum value.
        '''
        return max(self._max(bds, normalized, range, tol, min_points, max_call, sampling_size) for bds in parameters.bounds_for_range(self.data_pars, range))

    def numerical_integral(self, integral_range=parameters.FULL, range=parameters.FULL):
        '''
        Calculate the integral of a :class:`PDF`.

        :param integral_range: range of the integral to compute.
        :type integral_range: str
        :param range: normalization range to consider.
        :type range: str
        :returns: Integral of the PDF in the range defined by "integral_range" normalized to "range".
        :rtype: float
        '''
        raise exceptions.MethodNotDefinedError(
            self.__class__, 'numerical_integral')

    @allows_bind_cache
    @allows_const_cache
    def norm(self, range=parameters.FULL):
        '''
        Calculate the normalization of the PDF.

        :param range: normalization range to consider.
        :type range: str
        :returns: Value of the normalization.
        :rtype: float
        '''
        raise exceptions.MethodNotDefinedError(self.__class__, 'norm')

    @allows_bind_cache
    @allows_const_cache
    def numerical_normalization(self, range=parameters.FULL):
        '''
        Calculate a numerical normalization.

        :param range: normalization range.
        :type range: str
        :returns: Normalization.
        :rtype: float
        '''
        return self._raw_integral(range)

    def set_values(self, **kwargs):
        '''
        Set the values of the parameters associated to this PDF.

        :param kwargs: keyword arguments with "name"/"value".
        :type kwargs: dict(str, float)

        .. note:: Parameters of this PDF might be shared with others.
        '''
        for a in self.all_real_args:
            a.value = kwargs.get(a.name, a.value)

    def to_backend(self, backend):
        '''
        Initialize this class in a different backend.

        :param backend: new backend.
        :type backend: Backend
        :returns: An instance of this class in the new backend.
        '''
        raise exceptions.MethodNotDefinedError(self.__class__, 'to_backend')

    def to_json_object(self):
        '''
        Dump the PDF information into a JSON object.
        The PDF can be constructed at any time by calling :meth:`PDF.from_json_object`.

        :returns: Object that can be saved into a JSON file.
        :rtype: dict

        .. seealso:: :meth:`PDF.from_json_object`
        '''
        raise exceptions.MethodNotDefinedError(
            self.__class__, 'to_json_object')

    @contextlib.contextmanager
    def using_cache(self, ctype):
        '''
        Safe method to enable a cache of the PDF. There are two types of cache:

        * *const*: evaluation with a fixed *constant* state.
          It means that we plan to call the :class:`PDF` consecutively
          in the same data set, but the value of the parameters that are not
          constant is allowed to change. This means that any :class:`PDF` that
          has all its arguments as constant is assumed to have the same value
          in each evaluation.

        * *bind*: evaluation with the same parameter values and bounds.
          This is reserved for those processes where the normalization
          range will be the same, as well as the values of the parameters of the
          :class:`PDF`. However, the function can be called in different data
          sets.

        :param ctype: cache type.
        :type ctype: str

        .. warning:: It is responsibility of the user to ensure that the
           conditions for the cache to be valid are preserved.
        '''
        self._enable_cache(ctype)
        with self.aop.using_caches():
            yield self
        self._free_cache()

    @contextlib.contextmanager
    def restoring_state(self):
        '''
        Enter a context where the attributes of the parameters will be restored
        on exit.
        '''
        with contextlib.ExitStack() as stack:
            for p in self.all_pars:
                stack.enter_context(p.restoring_state())
            yield self


class HistPDF(BasePDF):

    _dependent = False

    def __init__(self, name, data_pars, edges, gaps, values, backend=None):
        '''
        This special kind of object represents the integral of a PDF on a
        set of bins. Unlike the other PDFs, this object does not inherit
        from :class:`PDF`, since only the :meth:`HistPDF.evaluate_binned`
        method is defined. It is possible to combine this object in
        :class:`AddPDFs` and :class:`ProdPDFs` objects, as long as the
        methods called only imply the use of the :meth:`HistPDF.evaluate_binned`
        function. As a consquence, evaluating this object in unbinned data
        samples is not allowed, as well as event sampling.

        :param name: name of the PDF.
        :type name: str
        :param data_pars: data parameters.
        :type data_pars: Registry(Parameter)
        :param edges: edges of the bins.
        :type edges: darray
        :param gaps: array defining the gaps between edges.
        :type gaps: numpy.ndarray
        :param values: values of the PDF in the different bins.
        :type values: numpy.ndarray
        :param backend: backend where this PDF lives.
        :type backend: Backend or None
        '''
        super().__init__(name, data_pars, backend)

        self.__edges = edges
        self.__gaps = data_types.array_int(gaps)
        self.__values = values / values.sum()  # normalize the PDF

    def evaluate_binned(self, *args, **kwargs):
        '''
        Evaluate the PDF over a binned sample. This is the only way to evaluate
        the PDF. It is responsibility of the user to evaluate the PDF on a
        data sample with the same bin edges as for this PDF.

        :returns: Values from the evaluation.
        :rtype: farray
        '''
        return self.__values

    def norm(self, *args, **kwargs):
        '''
        This type of PDFs are always normalized, so the output is always one.
        '''
        return 1.

    @classmethod
    def from_binned_dataset(cls, name, dataset):
        '''
        Build the class from a binned data set.

        :param name: name of the PDF.
        :type name: str
        :param dataset: binned data set.
        :type dataset: BinnedDataSet
        '''
        return cls(name, dataset.data_pars, dataset.edges,  dataset.gaps, dataset.values, dataset.backend)

    @classmethod
    def from_json_object(cls, obj, pars, backend):
        '''
        Build the class from a JSON-like object.

        :param obj: JSON object.
        :type obj: dict
        :param pars: parameters to use.
        :type pars: Registry(Parameter)
        :param backend: backend where the PDF will live.
        :type backend: Backend
        '''
        aop = parse_backend(backend)

        data_pars = list(map(pars.get, obj['data_pars']))

        edges = darray.from_ndarray(np.asarray(obj['edges']), aop.backend)
        gaps = darray.from_ndarray(np.asarray(obj['gaps']), aop.backend)
        values = darray.from_ndarray(np.asarray(obj['values']), aop.backend)

        return cls(obj['name'], data_pars, edges, gaps, values, backend)

    @classmethod
    def from_ndarray(cls, name, data_par, edges, values, backend=None):
        '''
        Build the class from the array of edges and values.

        :param name: Name of the PDF.
        :type name: str
        :param data_par: data parameter.
        :type data_par: Parameter
        :param edges: edges of the bins.
        :type edges: numpy.ndarray
        :param values: values at each bin.
        :type values: numpy.ndarray
        :param backend: backend where the data set is built.
        :type backend: Backend or None
        :returns: Binned data set.
        :rtype: BinnedDataSet
        '''
        aop = parse_backend(backend)
        edges = darray.from_ndarray(edges, aop.backend)
        values = darray.from_ndarray(values, aop.backend)
        return cls(name, parameters.Registry([data_par]), edges, [1], values, backend)

    def to_backend(self, backend):
        '''
        Initialize this class in a different backend.

        :param backend: new backend.
        :type backend: Backend
        :returns: An instance of this class in the new backend.
        '''
        edges = self.__edges.to_backend(backend)
        values = self.__values.to_backend(backend)
        return self.__class__(self.name, self.data_pars, edges, self.__gaps, values, backend)

    def to_json_object(self):
        '''
        Dump the PDF information into a JSON object.
        The PDF can be constructed at any time by calling :meth:`PDF.from_json_object`.

        :returns: Object that can be saved into a JSON file.
        :rtype: dict

        .. seealso:: :meth:`PDF.from_json_object`
        '''
        return {
            'name': self.name,
            'data_pars': self.data_pars.names,
            'edges': self.__edges.as_ndarray().tolist(),
            'gaps': self.__gaps.tolist(),
            'values': self.__values.as_ndarray().tolist(),
        }


class SourcePDF(PDF):

    def __init__(self, name, data_pars, arg_pars=None, var_arg_pars=None, backend=None):
        '''
        This object defines a PDF built from source files (C++, PyOpenCL or CUDA), which depend
        on the backend to use.
        The name of the PDF is inferred from the class name, so a file with name "name.cpp" (CPU) or
        "name.c" (CUDA an OpenCL) is expected to be in PDF_PATH.

        :param name: name of the PDF.
        :type name: str
        :param data_pars: data parameters.
        :type data_pars: Registry(Parameter)
        :param arg_pars: argument parameters.
        :type arg_pars: Registry(Parameter) or None
        :param var_arg_pars: argument parameters whose number can vary.
        :type var_arg_pars: Registry(Parameter) or None
        :param backend: backend where the PDF will operate.
        :type backend: Backend or None
        '''
        arg_pars = arg_pars or []

        # Must parse it correctly, since "None" means the function can not be built with
        # a variable number of arguments
        if var_arg_pars is None:
            nvar_arg_pars = None
            var_arg_pars = []
        else:
            nvar_arg_pars = len(var_arg_pars)
            var_arg_pars = list(var_arg_pars)

        super().__init__(name, parameters.Registry(
            data_pars), parameters.Registry(arg_pars) + parameters.Registry(var_arg_pars), backend)

        # Access the PDF
        self.__function_proxies = self.aop.access_pdf(
            self.pdf_class_name, ndata_pars=len(data_pars), nvar_arg_pars=nvar_arg_pars)

        # Integral configuration (must be done after the proxies are defined)
        if len(data_pars) == 1:
            self.numint_config = {'method': 'cquad'}
        else:
            self.numint_config = {'method': 'vegas'}

    def _norm(self, values, range=parameters.FULL):
        '''
        Calculate the normalization for a set of values and a range.

        :param values: values of the parameters.
        :type values: tuple(float)
        :param range: range for the normalization.
        :type range: str
        :returns: Normalization value.
        :rtype: float
        '''
        if self.__function_proxies.integral is not None:
            # There is an analytical approach to calculate the normalization
            bounds = parameters.bounds_for_range(self.data_pars, range)
            return np.sum(data_types.fromiter_float((self.__function_proxies.integral(*b, values)
                                                     for b in bounds)))
        else:
            return self._raw_integral(range)

    def _raw_integral(self, range):
        '''
        Calculate the integral of the PDF in the given range.

        :param range: range to integrate.
        :type range: str
        :returns: Value of the integral.
        :rtype: float
        '''
        bounds = parameters.bounds_for_range(self.data_pars, range)
        return np.sum(data_types.fromiter_float((self._raw_integral_for_bounds(b) for b in bounds)))

    def _raw_integral_for_bounds(self, bounds):
        '''
        Calculate the integral of the PDF in the given bounds.

        :param bounds: bounds to integrate.
        :type bounds: tuple(numpy.ndarray, numpy.ndarray)
        :returns: Value of the integral.
        :rtype: float
        '''
        args = data_types.fromiter_float((v.value for v in self.args))

        return self.numint_config(*bounds, args)

    @property
    def pdf_class_name(self):
        '''
        Name of the PDF class.

        :type: str
        '''
        return self.__class__.__name__

    @property
    def numint_config(self):
        '''
        Configuration of the numerical integration method.

        :getter: Return a configurable object for the specified numerical
           integration method.
        :setter: Set the configuration to do a numerical integration from a
           dictionary.

        The dictionary must contain at least the key "method", and the rest of
        the keys depend on it. For unidimensional PDFs, it is possible to use
        quadrature methods to evaluate the integrals. The available methods are:

        * *qng*: use  the fixed Gauss-Kronrod-Patterson abscissae to sample the
          integrand at a maximum of 87 points. This method must be used only in
          smooth functions.
        * *qag*: determination of numerical integrals using a simple adaptive
          integration procedure of interval subdivision based on the error.

          * *limit*: maximum number of subintervals. Must be smaller than the
            workspace size.
          * *key*: Gauss-Kronrod rule to use. Keys go from 1 to 6,
            corresponding to the 15, 21, 31, 41, 51 and 61 point rules.
          * *workspace_size*: size reserved in memory to store values. The
            space is related to pairs of doubles.

        * *cquad*: this is the default method used to evaluate numerical
          integrals. It uses a doubly-adaptive general-purpose based on the
          Clenshaw-Curtis quadrature rules.

          * *workspace_size*: size reserved in memory to store values. The
            space is related to pairs of doubles.

        For multidimensional PDFs, only Monte Carlo methods are available to
        determine the integral. These are also available for unidimensional
        PDFs.

        * *plain*: use a plain Monte Carlo algorithm to evaluate the integral.
          Use the argument "calls" in order to define the number of calls to do
          in order to calculate the integral.
        * *miser*: use the MISER method of recursive stratified sampling. The
          possible configuration parameters are:

          * *calls*: number of calls to the algorithm.
          * *estimate_frac*: fraction of the currently available number of
            function calls which are allocated to estimating the variance at
            each recursive step.
          * *min_calls*: minimum number of function calls required for each
            estimate of the variance.
          * *min_calls_per_bisection*: minimum number of function calls
            required to proceed with a bisection step.
          * *alpha*: parameter to control how the estimated variances for the
            two sub-regions of a bisection are combined when allocating points.
          * *dither*: parameter that introduces a random fractional variation
            of size into each bisection.

        * *vegas*: use the VEGAS algorithm to evaluate the integral. The
          possible configuration parameters are:

          * *calls*: number of calls to the algorithm.
          * *alpha*: parameter to control the stiffness of the rebinning
            algorithm.
          * *iterations*: number of iterations to perform for each call to the
            routine.
          * *mode*: sampling method to use.

        An example of how to set VEGAS as the numerical integrator with 100
        calls is:

        .. code-block:: python

           pdf.numint_config = {'method': 'vegas': 'calls': 100}

        In addition, it is possible to define a maximum tolerance for the
        relative and absolute error on the calculation of the integral, which
        is set through the *rtol* and *atol* keys, respectively.
        For the quadrature-based algorithms, the numerical integration will
        be done till the desired tolerance is achieved.
        For more information about the algorithms please consult the sections
        about `numerical <https://www.gnu.org/software/gsl/doc/html/integration.html>`__
        and `Monte Carlo <https://www.gnu.org/software/gsl/doc/html/montecarlo.html>`__.
        integration of the GNU scientific library.
        '''
        return self.__num_intgrl_config

    @numint_config.setter
    def numint_config(self, dct):

        dct = dict(dct)

        method = dct.pop('method').upper()

        if hasattr(gsl_api, method):
            self.__num_intgrl_config = getattr(gsl_api, method)(self,
                                                                self.__function_proxies.numerical_integral, **dct)
        else:
            raise ValueError(
                f'Unknown numerical integration method "{method}"')

    @allows_const_cache
    def __call__(self, data, range=parameters.FULL, normalized=True):

        # Determine the values to use
        fvals = data_types.fromiter_float((v.value for v in self.args))

        # Prepare the data arrays I/O
        out = self.aop.dempty(len(data))

        di = data_types.fromiter_int(
            (i for i, p in enumerate(data.data_pars) if p in self.data_pars))

        # Call the real function
        self.__function_proxies.evaluate(out, di, data.values, fvals)

        # Calculate the normalization
        if normalized:
            return safe_division(out, self._norm(fvals, range))
        else:
            return out

    @allows_const_cache
    def evaluate_binned(self, data, normalized=True):

        fvals = data_types.fromiter_float((v.value for v in self.args))

        gaps_idx = data_types.array_int(
            [data.data_pars.index(p.name) for p in self.data_pars])

        out = self.aop.dempty(len(data))

        if self.__function_proxies.evaluate_binned is not None:
            self.__function_proxies.evaluate_binned(
                out, gaps_idx, data.gaps, data.edges, fvals)
        else:
            self.__function_proxies.evaluate_binned_numerical(
                out, gaps_idx, data.gaps, data.edges, self.evb_size, fvals)

        if normalized:
            return safe_division(out, out.sum())
        else:
            return out

    @classmethod
    def from_json_object(cls, obj, pars, backend=None):

        if cls.__name__ == 'SourcePDF':
            raise exceptions.MethodNotDefinedError(cls, 'from_json_object')

        data_pars = list(map(pars.get, obj['data_pars']))
        arg_pars = list(map(pars.get, obj['arg_pars']))

        return cls(obj['name'], *data_pars, *arg_pars, backend=backend)

    def function(self, range=parameters.FULL, normalized=True):
        '''
        Evaluate the function.

        :param range: normalization range.
        :type range: str
        :param normalized: whether to return a normalized value.
        :type normalized: bool
        :returns: Value of the PDF.
        :rtype: float
        '''
        dvals = data_types.fromiter_float((v.value for v in self.data_pars))
        fvals = data_types.fromiter_float((v.value for v in self.args))

        v = self.__function_proxies.function(dvals, fvals)

        if normalized:
            n = self._norm(fvals, range)
            return safe_division(v, n)
        else:
            return v

    def integral(self, integral_range=parameters.FULL, range=parameters.FULL):

        if self.__function_proxies.integral is not None:
            # There is the analytical approach to calculate the integral
            values = data_types.fromiter_float((v.value for v in self.args))
            bounds = parameters.bounds_for_range(
                self.data_pars, integral_range)
            num = np.sum(data_types.fromiter_float((self.__function_proxies.integral(*b, values)
                                                    for b in bounds)))
            bounds = parameters.bounds_for_range(self.data_pars, range)
            den = np.sum(data_types.fromiter_float((self.__function_proxies.integral(*b, values)
                                                    for b in bounds)))
            return num / den
        else:
            num = self._raw_integral(integral_range)
            den = self._raw_integral(range)
            return num / den

    @allows_bind_cache
    @allows_const_cache
    def norm(self, range=parameters.FULL):

        fvals = data_types.fromiter_float((v.value for v in self.args))

        return self._norm(fvals, range)

    def numerical_integral(self, integral_range=parameters.FULL, range=parameters.FULL):

        if integral_range == range:
            return 1.
        else:
            num = self._raw_integral(integral_range)
            den = self._raw_integral(range)
            res = num / den
            if res > 1.:
                warnings.warn(
                    f'Numerical integral for PDF "{self.name}" is out of bounds; returning one', RuntimeWarning)
                return 1.
            else:
                return res

    def to_backend(self, backend):

        self.__class__(self.name, *self.data_pars, *self.args, backend)

    def to_json_object(self):

        return {'class': self.__class__.__name__,  # Save also the class name of the PDF
                'name': self.name,
                'data_pars': self.data_pars.names,
                'arg_pars': self.args.names}


@register_pdf
class FormulaPDF(SourcePDF):

    _tmpdir = None
    _tmpdir_lock = threading.Lock()

    def __init__(self, name, formula, data_pars, arg_pars, primitive=None, backend=None):
        '''
        Create a PDF from a simple formula. The string describing the formula
        can contain any basic mathematical functions. If *primitive* is provided
        it will be used in order to calculate the normalization. The formula
        is compiled at runtime.

        :param name: name of the PDF.
        :type name: str
        :param formula: formula associated to the PDF.
        :type formula: str
        :param data_pars: data parameters.
        :type data_pars: Registry(Parameter)
        :param arg_pars: argument parameters.
        :type arg_pars: Registry(Parameter)
        :param primitive: if provided, use this formula in order to calculate
           the primitive.
        :type primitive: str or None
        :param backend: backend to build the PDF.
        :type backend: Backend or None
        '''
        with FormulaPDF._tmpdir_lock:

            if FormulaPDF._tmpdir is None:
                FormulaPDF._tmpdir = tempfile.TemporaryDirectory()
                add_pdf_src(FormulaPDF._tmpdir.name)

            # create the XML file to save the PDF
            self.__file = tempfile.NamedTemporaryFile(
                'w+t', suffix='.xml', dir=FormulaPDF._tmpdir.name, delete=False)

        # keep the arguments so we can save the class to JSON
        self.__formula = formula
        self.__primitive = primitive

        code = autocode.xml_from_formula(
            formula, data_pars, arg_pars, primitive)

        self.__file.write(code)

        self.__file.seek(0)  # rewind the file so it can be read

        # initialize the base class
        super().__init__(name, data_pars, arg_pars, backend=backend)

    @classmethod
    def from_json_object(cls, obj, pars, backend=None):

        data_pars = list(map(pars.get, obj['data_pars']))
        arg_pars = list(map(pars.get, obj['arg_pars']))

        return cls(obj['name'], obj['formula'], data_pars, arg_pars, obj['primitive'], backend=backend)

    @classmethod
    def unidimensional(cls, name, formula, data_par, arg_pars, primitive=None, backend=None):
        '''
        Build the PDF for unidimensional functions.

        :param name: name of the PDF.
        :type name: str
        :param formula: formula associated to the PDF.
        :type formula: str
        :param data_par: data parameters.
        :type data_par: Parameter
        :param arg_pars: argument parameters.
        :type arg_pars: Registry(Parameter)
        :param primitive: if provided, use this formula in order to calculate
           the primitive.
        :type primitive: str or None
        :param backend: backend to build the PDF.
        :type backend: Backend or None
        '''
        return cls(name, formula, [data_par], arg_pars, primitive, backend)

    @property
    def pdf_class_name(self):
        '''
        Name of the PDF class.

        :type: str
        '''
        name, _ = os.path.splitext(os.path.basename(self.__file.name))
        return name

    def to_json_object(self):

        return {'class': self.__class__.__name__,  # Save also the class name of the PDF
                'name': self.name,
                'formula': self.__formula,
                'primitive': self.__primitive,
                'data_pars': self.data_pars.names,
                'arg_pars': self.args.names}


class MultiPDF(PDF):

    # Allow to define if a PDF object depends on other PDFs. All PDF objects
    # with this value set to "True" must have the property "pdfs" implemented.
    _dependent = True

    def __init__(self, name, pdfs, arg_pars=None):
        '''
        Base class owing many PDFs.

        :param name: name of the PDF.
        :type name: str
        :param pdfs: :class:`PDF` objects to hold.
        :type pdfs: list(PDF)
        :param arg_pars: possible argument parameters.
        :type arg_pars: Registry(Parameter) or None
        '''
        if arg_pars is not None:
            arg_pars = parameters.Registry(arg_pars)
        else:
            arg_pars = parameters.Registry()

        self.__pdfs = parameters.Registry(pdfs)

        data_pars = parameters.Registry()
        for pdf in pdfs:
            data_pars += pdf.data_pars

        backend = common_backend(pdfs)

        super().__init__(name, data_pars, arg_pars, backend)

    def __repr__(self):
        out = f'{self.__class__.__name__}({self.name}):'
        out += os.linesep + textwrap.indent(f'Parameters:{os.linesep}' + textwrap.indent(
            os.linesep.join(tuple(str(p) for p in self.args)), ' '), ' ')
        out += os.linesep + textwrap.indent(f'PDFs:{os.linesep}' + textwrap.indent(
            os.linesep.join(tuple(str(p) for p in self.pdfs)), ' '), ' ')
        return out

    @property
    def all_args(self):

        args = parameters.Registry(super().all_args)
        for p in filter(lambda p: p._has_args, self.__pdfs):
            args += p.all_args

        return args

    @property
    def all_pars(self):

        pars = parameters.Registry(super().all_pars)
        for p in self.__pdfs:
            pars += p.all_pars

        return pars

    @property
    def all_pdfs(self):
        '''
        Recursively get all the possible PDFs belonging to this object.
        '''
        pdfs = parameters.Registry(self.__pdfs)
        for pdf in filter(lambda p: p._dependent, self.__pdfs):
            pdfs += pdf.all_pdfs
        return pdfs

    @property
    def dependencies(self):
        '''
        Registry of PDFs this instance depends on.

        :type: Registry(PDF)
        '''
        return self.__pdfs

    @property
    def pdfs(self):
        '''
        Get the registry of PDFs within this class.

        :returns: PDFs owned by this class.
        :rtype: Registry(PDF)
        '''
        return self.__pdfs

    @classmethod
    def from_json_object(cls, obj, pars, pdfs):

        class_name = obj['class']
        if class_name == 'MultiPDF':
            raise exceptions.MethodNotDefinedError(cls, 'from_json_object')
        cl = PDF_REGISTRY.get(class_name, None)
        if cl is None:
            raise RuntimeError(
                f'Class "{class_name}" does not appear in the registry')
        # Use the correct constructor
        return cl.from_json_object(obj, pars, pdfs)

    def _enable_cache(self, ctype):

        super()._enable_cache(ctype)

        for pdf in filter(lambda p: p._has_args, self.pdfs):
            pdf._enable_cache(ctype)

    def _free_cache(self):

        super()._free_cache()

        for pdf in filter(lambda p: p._has_args, self.pdfs):
            pdf._free_cache()

    def component(self, name):
        '''
        Get the :class:`PDF` object with the given name.

        :param name: name of the :class:`PDF`.
        :type name: str
        :returns: Component with the given name.
        :rtype: PDF
        '''
        for pdf in self.__pdfs:
            if pdf.name == name:
                return pdf
        raise LookupError(f'No PDF with name "{name}" hass been found')


@register_pdf
class AddPDFs(MultiPDF):

    def __init__(self, name, pdfs, yields):
        '''
        This special PDF defines the sum of many different PDFs, where each
        of them is multiplied by a factor.
        The number of factors must be equal to that of the PDFs, if one
        wants an "extended" PDF, or one smaller.
        In the latter case, the last factor is calculated from the normalization
        condition.

        :param name: name of the PDF.
        :type name: str
        :param pdfs: PDFs to add.
        :type pdfs: list(PDF)
        :param yields: factors to multiply the PDFs.
        :type yields: list(Parameter)
        '''
        assert len(pdfs) - len(yields) in (0, 1)

        super().__init__(name, pdfs, yields)

    @property
    def yields(self):
        '''
        Yields of each PDF.
        '''
        yields = data_types.empty_float(len(self.args) + (not self.extended))

        yields[:len(self.args)] = tuple(v.value for v in self.args)

        if not self.extended:
            yields[-1] = 1. - np.sum(yields[:-1])

        return yields

    @allows_const_cache
    def __call__(self, data, range=parameters.FULL, normalized=True):

        yields = self.yields

        out = self.aop.dzeros(len(data))
        for y, pdf in zip(yields, self.pdfs):
            out += y * pdf(data, range, normalized=True)

        if self.extended and normalized:
            return safe_division(out, np.sum(yields))
        else:
            return out

    @allows_const_cache
    def evaluate_binned(self, data, normalized=True):

        yields = self.yields

        out = self.aop.dzeros(len(data))
        for y, pdf in zip(yields, self.pdfs):
            out += y * pdf.evaluate_binned(data)

        if normalized:
            return safe_division(out, out.sum())
        else:
            return out

    @classmethod
    def from_json_object(cls, obj, pars, pdfs):

        pdfs = list(map(pdfs.get, obj['pdfs']))
        yields = list(map(pars.get, obj['yields']))

        return cls(obj['name'], pdfs, yields)

    @classmethod
    def two_components(cls, name, first, second, yf, ys=None):
        '''
        Build the class from two components.

        :param name: name of the class.
        :type name: str
        :param first: first PDF to use.
        :type first: PDF
        :param second: second PDF to use.
        :type second: PDF
        :param yf: yield associated to the first PDF, if both "yf" and "ys"
           are provided. If "ys" is not provided, then "yf" is the faction
           associated to the first PDF.
        :type yf: Parameter
        :param ys: possible yield for the second PDF.
        :type ys: Parameter or None
        :returns: The built class.
        :rtype: AddPDFs
        '''
        return cls(name, [first, second], [yf] if ys is None else [yf, ys])

    @property
    def extended(self):
        '''
        Whether this PDF is of "extended" type.

        :type: bool
        '''
        return len(self.pdfs) == len(self.args)

    def function(self, range=parameters.FULL, normalized=True):

        yields = self.yields

        v = np.sum(data_types.fromiter_float(
            (y * pdf.function(range, normalized=True) for y, pdf in zip(yields, self.pdfs))))

        if normalized:
            return v / np.sum(yields)
        else:
            return v

    def integral(self, integral_range=parameters.FULL, range=parameters.FULL):

        yields = self.yields
        s = np.sum(yields)

        return np.sum(data_types.fromiter_float((y * pdf.integral(integral_range, range) for y, pdf in enumerate(yields, self.pdfs)))) / s

    @allows_bind_cache
    @allows_const_cache
    def norm(self, range=parameters.FULL):

        if self.extended:
            # An extended PDF has as normalization the sum of yields
            return sum(v.value for v in self.args)
        else:
            # A non-extended PDF is always normalized
            return 1.

    def numerical_integral(self, integral_range=parameters.FULL, range=parameters.FULL):

        yields = self.yields
        s = np.sum(yields)

        return np.sum(data_types.fromiter_float((y * pdf.numerical_integral(integral_range, range) for y, pdf in enumerate(yields, self.pdfs)))) / s

    @allows_bind_cache
    @allows_const_cache
    def numerical_normalization(self, range=parameters.FULL):

        return self.norm(range)

    def to_backend(self, backend):

        new_pdfs = [pdf.to_backend(backend) for pdf in self.pdfs]

        return self.__class__(self.name, new_pdfs, self.args)

    def to_json_object(self):

        return {'class': self.__class__.__name__,  # Save also the class name of the PDF
                'name': self.name,
                'pdfs': self.pdfs.names,
                'data_pars': self.data_pars.names,
                'yields': self.args.names}


@register_pdf
class ConvPDFs(MultiPDF):

    def __init__(self, name, first, second, range=None):
        '''
        Represent the convolution of two different PDFs.

        :param name: name of the PDF.
        :type name: str
        :param first: first PDF.
        :type first: PDF
        :param second: second PDF.
        :type second: PDF
        :param range: range of the convolution. This is needed in case part of the
           PDFs lie outside the evaluation range. It is set to "full" by default.
        :type range: str or None
        :raises ValueError: If an attempt to define convolution in more than
           one dimension is done.
        '''
        if len(first.data_pars) != 1 or len(second.data_pars) != 1:
            raise ValueError(
                'Convolution is only supported in 1-dimensional PDFs')

        super().__init__(name, [first, second])

        self.interpolation_method = 'spline'

        # The convolution size and range can be changed by the user
        self.range = range or parameters.FULL
        self.evb_size = DEFAULT_EVB_SIZE
        self.conv_size = DEFAULT_INTERP_SIZE

    def __repr__(self):
        out = f'{self.__class__.__name__}({self.name}):{os.linesep}'
        out += textwrap.indent(os.linesep.join(tuple(str(p)
                                                     for p in self.pdfs)), ' ')
        return out

    def _norm(self, interpolator, range=parameters.FULL):
        '''
        Calculate the normalization using the processed convolution values
        and the integration range.

        :param interpolator: function to use in order to interpolate.
        :param range: normalization range.
        :type range: str
        :returns: Normalization.
        :rtype: float
        '''
        if range == self.range:
            return 1.  # avoid doing the convolution

        # we are using the a grid of dimension one
        s = 0
        for b in parameters.bounds_for_range(self.data_pars, range):

            grid = dataset.evaluation_grid(self.aop,
                                           self.data_pars, b, size=self.__conv_size + 1)

            pdf_values = interpolator.interpolate(0, grid.values)

            pdf_values *= (grid.values.get(1) - grid.values.get(0)
                           ) / (3. * self.__conv_size)
            pdf_values *= self.aop.simpson_factors(self.__conv_size + 1)

            s += pdf_values.sum()

        return s

    @property
    def conv_size(self):
        '''
        Size of the sample used to calculate the convolution. It must be an
        odd number.
        '''
        return self.__conv_size

    @conv_size.setter
    def conv_size(self, size):
        if size % 2 != 0:
            raise ValueError('Size must be an even number')
        self.__conv_size = size

    @property
    def interpolation_method(self):
        '''
        Function used to do the interpolation.

        :setter: Set the function to use from a string.
        '''
        return self.__interp

    @interpolation_method.setter
    def interpolation_method(self, method):
        if method == 'linear':
            self.__interp = self.aop.make_linear_interpolator
        elif method == 'spline':
            self.__interp = self.aop.make_spline_interpolator
        else:
            raise ValueError(f'Unknown interpolation method "{method}"')

    @allows_const_cache
    def __call__(self, data, range=parameters.FULL, normalized=True):

        interpolator = self.convolve(normalized)

        idx = data.data_pars.index(self.data_pars[0].name)

        pdf_values = interpolator.interpolate(idx, data.values)

        if normalized:
            return pdf_values / self._norm(interpolator, range)
        else:
            return pdf_values

    @allows_const_cache
    def evaluate_binned(self, data, normalized=True):

        interpolator = self.convolve(normalized=True)

        par = self.data_pars[0]

        return _interpolation_evaluate_binned(self.aop, interpolator, par, data, self.evb_size, normalized)

    def function(self, range=parameters.FULL, normalized=True):

        interpolator = self.convolve(normalized)

        v = interpolator.interpolate_single_value(self.data_pars[0].value)

        if normalized:
            return v / self._norm(interpolator, range)
        else:
            return v

    @classmethod
    def from_json_object(cls, obj, pars, pdfs):

        first = pdfs.get(obj['first'])
        second = pdfs.get(obj['second'])

        return cls(obj['name'], first, second, obj['range'])

    @allows_bind_cache
    @allows_const_cache
    def convolve(self, normalized=True):
        '''
        Calculate the convolution.

        :param range: normalization range.
        :type range: str
        :param normalized: whether to return a normalized output.
        :type normalized: bool
        :returns: Data and result of the evaluation.
        :rtype: numpy.ndarray, numpy.ndarray
        :raises RuntimeError: If the bounds of the parameter in the
           convolution range are disjointed.
        '''
        first, second = tuple(self.pdfs)

        bounds = parameters.bounds_for_range(self.data_pars, self.range)

        if len(bounds) != 1:
            raise RuntimeError(
                'The convolution bounds must not be disjointed')

        # Only works for the 1-dimensional case
        grid = dataset.evaluation_grid(self.aop,
                                       self.data_pars, bounds[0], size=self.__conv_size)

        fv = first(grid, self.range, normalized)
        sv = second(grid, self.range, normalized)

        dv = grid.values
        cv = self.aop.fftconvolve(fv, sv, grid.values)

        return self.__interp(dv, cv)

    @allows_bind_cache
    @allows_const_cache
    def norm(self, range=parameters.FULL):

        if self.range == range:
            return 1.  # avoid doing the convolution

        interpolator = self.convolve(normalized=True)

        return self._norm(interpolator, range)

    def numerical_integral(self, integral_range=parameters.FULL, range=parameters.FULL):

        if integral_range == range:
            return 1.

        interpolator = self.convolve(normalized=True)

        num = self._norm(interpolator, integral_range)
        den = self._norm(interpolator, range)

        return num / den

    @allows_bind_cache
    @allows_const_cache
    def numerical_normalization(self, range=parameters.FULL):

        return self.norm(range)

    def to_backend(self, backend):

        new_pdfs = [pdf.to_backend(backend) for pdf in self.pdfs]

        return self.__class__(self.name, *new_pdfs, self.range)

    def to_json_object(self):

        first, second = self.pdfs

        return {'class': self.__class__.__name__,  # Save also the class name of the PDF
                'name': self.name,
                'first': first.name,
                'second': second.name,
                'range': self.range}


@register_pdf
class ProdPDFs(MultiPDF):

    def __init__(self, name, pdfs):
        '''
        This object represents the product of many PDFs where the data parameters are
        not shared among them.

        :param name: name of the PDF.
        :type name: str
        :param pdfs: list of PDFs
        :type pdfs: list(PDF)
        '''
        super().__init__(name, pdfs, [])

    @allows_const_cache
    def __call__(self, data, range=parameters.FULL, normalized=True):

        out = self.aop.dones(len(data))
        for pdf in self.pdfs:
            out *= pdf(data, range, normalized=normalized)

        return out

    def __repr__(self):
        out = f'{self.__class__.__name__}({self.name}):{os.linesep}'
        out += textwrap.indent(os.linesep.join(tuple(str(p)
                                                     for p in self.pdfs)), ' ')
        return out

    @allows_const_cache
    def evaluate_binned(self, data, normalized=True):

        out = self.aop.dones(len(data))
        for pdf in self.pdfs:
            out *= pdf.evaluate_binned(data, normalized=False)

        if normalized:
            return safe_division(out, out.sum())
        else:
            return out

    def function(self, range=parameters.FULL, normalized=True):

        return np.prod(data_types.fromiter_float((pdf.function(range, normalized) for pdf in self.pdfs)))

    @classmethod
    def from_json_object(cls, obj, pars, pdfs):

        pdfs = list(map(pdfs.get, obj['pdfs']))

        return cls(obj['name'], pdfs)

    def integral(self, integral_range=parameters.FULL, range=parameters.FULL):

        return np.prod(data_types.fromiter_float((pdf.integral(integral_range, range) for pdf in self.pdfs)))

    @allows_bind_cache
    @allows_const_cache
    def norm(self, range=parameters.FULL):

        n = 1.
        for p in self.pdfs:
            n *= p.norm(range)

        return n

    def numerical_integral(self, integral_range=parameters.FULL, range=parameters.FULL):

        return np.prod(data_types.fromiter_float((pdf.numerical_integral(integral_range, range) for pdf in self.pdfs)))

    @allows_bind_cache
    @allows_const_cache
    def numerical_normalization(self, range=parameters.FULL):

        return np.prod(data_types.fromiter_float((pdf.numerical_normalization(range) for pdf in self.pdfs)))

    def to_backend(self, backend):

        new_pdfs = [pdf.to_backend(backend) for pdf in self.pdfs]
        return self.__class__(self.name, new_pdfs)

    def to_json_object(self):

        return {'class': self.__class__.__name__,  # Save also the class name of the PDF
                'name': self.name,
                'pdfs': self.pdfs.names}


@register_pdf
class InterpPDF(PDF):

    def __init__(self, name, data_par, centers, values, backend=None):
        '''
        Represent the convolution of two different PDFs.

        :param name: name of the PDF.
        :type name: str
        :param data_par: data parameter to consider.
        :type data_par: Parameter
        :param centers: centers to use in the interpolation process.
        :type centers: darray
        :param values: values to use in the interpolation process.
        :type values: darray
        :param backend: backend to build the PDF.
        :type backend: Backend or None
        :raises ValueError: If an attempt to define the PDF in more than
           one dimension is done.
        '''
        super().__init__(name, parameters.Registry(
            [data_par]), backend=backend)

        self.__centers = centers
        self.__values = values

        self.interpolation_method = 'spline'

        self.evb_size = DEFAULT_EVB_SIZE
        self.interp_size = DEFAULT_INTERP_SIZE

    def __repr__(self):
        vmin = self.__centers.min()
        vmax = self.__centers.max()
        size = len(self.__centers)
        return f'{self.__class__.__name__}({self.name}, range=({vmin}, {vmax}), size={size})'

    def _norm(self, range=parameters.FULL):
        '''
        Calculate the normalization using the processed interpolation values
        and the integration range.

        :param range: normalization range.
        :type range: str
        :returns: Normalization.
        :rtype: float
        '''
        # we are using the a grid of dimension one
        s = 0
        for b in parameters.bounds_for_range(self.data_pars, range):

            grid = dataset.evaluation_grid(self.aop,
                                           self.data_pars, b, size=self.__interp_size + 1)

            pdf_values = self.__interpolator.interpolate(0, grid.values)

            pdf_values *= (grid.values.get(1) - grid.values.get(0)) / 3.
            pdf_values *= self.aop.simpson_factors(self.__interp_size + 1)

            s += pdf_values.sum()

        return s

    @property
    def interp_size(self):
        '''
        Size of the sample used to calculate the convolution. It must be an
        odd number.
        '''
        return self.__interp_size

    @interp_size.setter
    def interp_size(self, size):
        if size % 2 != 0:
            raise ValueError('Size must be an even number')
        self.__interp_size = size

    @property
    def interpolation_method(self):
        '''
        Function used to do the interpolation.

        :setter: Set the function to use from a string.
        '''
        return self.__interpolator

    @interpolation_method.setter
    def interpolation_method(self, method):
        if method == 'linear':
            self.__interpolator = self.aop.make_linear_interpolator(
                self.__centers, self.__values)
        elif method == 'spline':
            self.__interpolator = self.aop.make_spline_interpolator(
                self.__centers, self.__values)
        else:
            raise ValueError(f'Unknown interpolation method "{method}"')

    @classmethod
    def from_binned_dataset(cls, name, data):
        '''
        Build the instance from a binned data set. The data set must be
        unidimensional.

        :param name: name of the PDF.
        :type name: str
        :param data: binned data set.
        :type data: BinnedDataSet
        '''
        data_par = data.data_pars[0]
        edges, values = data[data_par.name].as_ndarray(
        ), data.values.as_ndarray()

        centers = 0.5 * (edges[1:] + edges[:-1])

        return cls.from_ndarray(name, data_par, centers, values, backend=data.backend)

    @classmethod
    def from_ndarray(cls, name, data_par, centers, values, backend=None):
        '''
        Build the class from :class:`numpy.ndarray` instances.

        :param name: name of the PDF.
        :type name: str
        :param data_par: data parameter.
        :type data_par: Parameter
        :param centers: centers to use in the interpolation.
        :type centers: numpy.ndarray
        :param values: values to use in the interpolation.
        :type values: numpy.ndarray
        :param backend: backend where the PDF will operate.
        :type backend: Backend or None
        '''
        aop = parse_backend(backend)
        c = darray.from_ndarray(centers, aop.backend)
        v = darray.from_ndarray(values, aop.backend)
        return cls(name, data_par, c, v, backend)

    @allows_const_cache
    def __call__(self, data, range=parameters.FULL, normalized=True):

        idx = data.data_pars.index(self.data_pars[0].name)

        pdf_values = self.__interpolator.interpolate(idx, data.values)

        if normalized:
            return pdf_values / self._norm(range)
        else:
            return pdf_values

    @allows_const_cache
    def evaluate_binned(self, data, normalized=True):

        par = self.data_pars[0]

        return _interpolation_evaluate_binned(self.aop, self.__interpolator, par, data, self.evb_size, normalized)

    @classmethod
    def from_json_object(cls, obj, pars, backend):

        data_par = pars.get(obj['data_par']['name'])

        c, v = data_types.array_float(
            obj['centers']), data_types.array_float(obj['values'])

        return cls.from_ndarray(obj['name'], data_par, c, v, backend)

    def function(self, range=parameters.FULL, normalized=True):
        '''
        Evaluate the function.

        :param range: normalization range.
        :type range: str
        :param normalized: whether to return a normalized value.
        :type normalized: bool
        :returns: Value of the PDF.
        :rtype: float
        '''
        v = self.__interpolator.interpolate_single_value(
            self.data_pars[0].value)

        if normalized:
            n = self._norm(range)
            return safe_division(v, n)
        else:
            return v

    @allows_bind_cache
    @allows_const_cache
    def norm(self, range=parameters.FULL):
        return self._norm(range)

    def numerical_integral(self, integral_range=parameters.FULL, range=parameters.FULL):

        if integral_range == range:
            return 1.

        num = self._norm(integral_range)
        den = self._norm(range)

        return num / den

    @allows_bind_cache
    @allows_const_cache
    def numerical_normalization(self, range=parameters.FULL):

        return self.norm(range)

    def to_backend(self, backend):

        c, v = self.__centers.to_backend(
            backend), self.__values.to_backend(backend)

        return self.__class__(self.name, self.data_pars, c, v, backend)

    def to_json_object(self):

        return {'class': self.__class__.__name__,  # Save also the class name of the PDF
                'name': self.name,
                # there is only one element
                'data_par': self.data_pars[0].to_json_object(),
                'centers': self.__centers.as_ndarray().tolist(),
                'values': self.__values.as_ndarray().tolist()}


def pdf_from_json(obj, backend=None):
    '''
    Load a PDF from a JSON object.

    :param obj: JSON-like object.
    :type obj: dict
    :param backend: backend to build the PDF.
    :type backend: Backend or None
    :returns: A PDF with the configuration from the object.
    :rtype: PDF

    .. seealso:: :meth:`pdf_to_json`
    '''
    # Parse parameters
    pars = parameters.Registry(
        [parameters.Parameter.from_json_object(o) for o in obj['ipars']])

    for o in obj['dpars']:
        pars.append(parameters.Formula.from_json_object(o, pars))

    # Parse PDFs
    pdfs = parameters.Registry(
        [PDF.from_json_object(o, pars, backend) for o in obj['ipdfs']])

    for o in obj['dpdfs']:
        pdfs.append(MultiPDF.from_json_object(o, pars, pdfs))

    # Last PDF to be constructed is the main PDF
    return pdfs[-1]


def pdf_to_json(pdf):
    '''
    Dump a PDF to a JSON-like object.

    :param pdf: :class:`PDF` to dump.
    :type pdf: PDF
    :returns: JSON-like object.
    :rtype: dict

    .. seealso:: :meth:`pdf_from_json`
    '''
    # Solve dependencies for parameters
    ipars, dpars = dependencies.split_dependent_objects_with_resolution_order(
        pdf.all_pars)

    # Solve dependencies for PDF objects
    if pdf._dependent:
        ipdfs, dpdfs = dependencies.split_dependent_objects_with_resolution_order(
            pdf.pdfs)
        dpdfs.insert(0, pdf)  # must include this PDF
    else:
        ipdfs, dpdfs = [pdf], []

    return {'dpdfs': [pdf.to_json_object() for pdf in dpdfs],
            'ipdfs': [pdf.to_json_object() for pdf in ipdfs],
            'dpars': [p.to_json_object() for p in dpars],
            'ipars': [p.to_json_object() for p in ipars]}
