'''
Base classes to define PDF types.
'''
from . import accessors
from . import bindings
from .core import aop
from . import dataset
from . import parameters
from .operations import types

import contextlib
import functools
import logging
import numpy as np

__all__ = ['AddPDFs', 'ConvPDFs', 'pdf_from_json', 'pdf_to_json',
           'PDF', 'ProdPDFs', 'register_pdf', 'SourcePDF']

# Default size of the samples to be used during numerical normalization
# and calculation of integrals
NORM_SIZE = 1000000

# Default size of the grid for the convolution PDFs
CONV_SIZE = 10000

# Registry of PDF classes so they can be built by name
PDF_REGISTRY = {}

logger = logging.getLogger(__name__)


def register_pdf(cl):
    '''
    Decorator to register PDF classes in the PDF_REGISTRY registry.

    :param cl: decorated class.
    :type cl: class
    :returns: class descriptor.
    :rtype: class
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
    :returns: whatever "method" returns.
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
    Wrap the method of a class.

    :param method: method of the class.
    :type method: function
    :returns: wrapper around the method.
    :rtype: function
    '''
    @functools.wraps(method)
    def __wrapper(self, *args, **kwargs):
        '''
        Internal wrapper.
        '''
        if self.cache_type == PDF.BIND:
            return process_cache(self.cache, method, self, args, kwargs)
        else:
            return method(self, *args, **kwargs)
    return __wrapper


def allows_const_cache(method):
    '''
    Wrap the method of a class, using a cache if the class is marked as constant.

    :param method: method of the class.
    :type method: function
    :returns: wrapper around the method.
    :rtype: function
    '''
    @functools.wraps(method)
    def __wrapper(self, *args, **kwargs):
        '''
        Internal wrapper.
        '''
        if self.cache_type == PDF.CONST and self.constant:
            return process_cache(self.cache, method, self, args, kwargs)
        else:
            return method(self, *args, **kwargs)
    return __wrapper


def allows_bind_or_const_cache(method):
    '''
    Wrap the method of a class, using a cache if the class is marked as constant.

    :param method: method of the class.
    :type method: function
    :returns: wrapper around the method.
    :rtype: function
    '''
    @functools.wraps(method)
    def __wrapper(self, *args, **kwargs):
        '''
        Internal wrapper.
        '''
        if self.cache_type == PDF.BIND or (self.cache_type == PDF.CONST and self.constant):
            return process_cache(self.cache, method, self, args, kwargs)
        else:
            return method(self, *args, **kwargs)
    return __wrapper


class PDF(object):
    '''
    Object defining the properties of a PDF.
    '''
    # Cache types
    BIND = 'bind'
    CONST = 'const'

    CACHE_TYPES = (BIND, CONST)

    # Allow to define if a PDF object depends on other PDFs. All PDF objects
    # with this value set to "True" must have the property "pdfs" implemented.
    dependent = False

    def __init__(self, name, data_pars, args_pars):
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
        :type arg_pars: Registry(Parameter)
        '''
        self.name = name
        self.__data_pars = data_pars
        self.__arg_pars = args_pars
        self.norm_size = NORM_SIZE

        # The cache is saved on a dictionary, and inherited classes must avoid colliding names
        self.__cache = None
        self.__cache_type = None

        super(PDF, self).__init__()

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
        '''
        raise NotImplementedError(
            'Classes inheriting from "minkit.PDF" must define the "__call__" operator')

    @classmethod
    def from_json_object(cls, obj, pars):
        '''
        Build a PDF from a JSON object.
        This object must represent the internal structure of the PDF.

        :param obj: JSON object.
        :type obj: dict
        :returns: newly constructed PDF and parameters.
        :rtype: PDF
        '''
        class_name = obj['class']
        if class_name == 'PDF':
            raise NotImplementedError(
                'Classes inheriting from "minkit.PDF" must define the "to_json_object" method')
        cl = PDF_REGISTRY.get(class_name, None)
        if cl is None:
            raise RuntimeError(
                f'Class "{class_name}" does not appear in the registry')
        return cl.from_json_object(obj, pars)  # Use the correct constructor

    def _generate_single_bounds(self, size, mapsize, gensize, safe_factor, bounds):
        '''
        Generate data in a single range given the bounds of the different data parameters.

        :param size: size (or minimum size) of the output sample.
        :type size: int
        :param mapsize: number of points to consider per dimension (data parameter) \
        in order to calculate the maximum value of the PDF.
        :type mapsize: int
        :param gensize: number of entries to generate per iteration.
        :type gensize: int
        :param safe_factor: additional factor to multiply the numerically calculated \
        maximum of the function. In general this must be modified if the function is \
        not well-behaved.
        :type safe_factor: float
        :param bounds: bounds of the different data parameters (must be sorted).
        :type bounds: numpy.ndarray
        :returns: output sample.
        :rtype: DataSet
        '''
        grid = dataset.evaluation_grid(self.data_pars, bounds, mapsize)

        m = safe_factor * \
            aop.max(self.__call__(grid, normalized=False))

        samples = []

        n = 0
        while n < size:

            d = dataset.uniform_sample(self.data_pars, bounds, gensize)
            f = self.__call__(d, normalized=False)
            u = aop.random_uniform(0, m, len(d))

            r = d.subset(aop.ale(u, f))

            n += len(r)

            samples.append(r)

        return dataset.DataSet.merge(samples, maximum=size)

    def _integral_bin_area(self, bounds, size):
        '''
        Calculate the area of a bin used to calculate the numerical normalization.

        :param bounds: bounds defining the range.
        :type bounds: numpy.ndarray
        :returns: area of the normalization bin.
        :rtype: float
        '''
        return np.prod(bounds[1::2] - bounds[0::2]) * 1. / size

    def _integral_single_bounds(self, bounds, range):
        '''
        Calculate the integral of the PDF on a single set of bounds.

        :param bounds: bounds of the data parameters.
        :type bounds: numpy.ndarray
        :param range: normalization range to consider.
        :type range: str
        :returns: integral of the PDF in the range defined by "bounds" normalized to "range".
        :rtype: float
        '''
        g = dataset.evaluation_grid(self.data_pars, bounds, self.norm_size)
        a = self._integral_bin_area(bounds, len(g))
        i = self.__call__(g, range=range)
        return aop.sum(i) * a

    def _numerical_normalization_single_bounds(self, bounds):
        '''
        Calculate the normalization value for a set of bounds.

        :param bounds: bounds of the data parameters.
        :type bounds: numpy.ndarray
        :returns: normalization value.
        :rtype: float
        '''
        g = dataset.evaluation_grid(self.data_pars, bounds, self.norm_size)
        a = self._integral_bin_area(bounds, len(g))
        i = self.__call__(g, normalized=False)
        return aop.sum(i) * a

    @property
    def all_args(self):
        '''
        Get all the argument parameters associated to this class.

        :returns: all the argument parameters associated to this class.
        :rtype: Registry(Parameter)
        '''
        return self.args

    @property
    def all_pars(self):
        '''
        Get all the parameters associated to this class.
        This includes also any :class:`ParameterFormula` and the data parameters.

        :returns: all the argument parameters associated to this class.
        :rtype: Registry(Parameter)
        '''
        return self.data_pars + self.__arg_pars

    @property
    def all_real_args(self):
        return parameters.Registry(filter(lambda p: not p.dependent, self.all_args))

    @property
    def args(self):
        '''
        Get the argument parameters this object directly depends on.

        :returns: parameters this object directly depends on.
        :rtype: Registry(Parameter)
        '''
        return self.__arg_pars

    @property
    def cache(self):
        '''
        Return the cached values for this PDF.

        :returns: cache
        :rtype: dict
        '''
        return self.__cache

    @property
    def cache_type(self):
        '''
        Get whether there are elements in the cache or not.

        :returns: whether there are elements in the cache or not.
        :rtype: bool
        '''
        return self.__cache_type

    @property
    def constant(self):
        '''
        Get whether this is a constant :class:`PDF`.

        :returns: whether this object can be marked as constant.
        :rtype: bool
        '''
        return all(p.constant for p in self.all_real_args)

    @property
    def data_pars(self):
        '''
        Get the data parameters this object directly depends on.

        :returns: parameters this object directly depends on.
        :rtype: Registry(Parameter)
        '''
        return self.__data_pars

    @contextlib.contextmanager
    def bind(self, range=parameters.FULL, normalized=True):
        '''
        Prepare an object that will be called many times with the same set of
        values.
        This is usefull for PDFs using a cache, to avoid creating it many times
        in sucessive calls to :meth:`PDF.__call__`.

        :param range: normalization range.
        :type range: str
        :param normalized: whether to return a normalized output.
        :type normalized: bool
        '''
        self.enable_cache(PDF.BIND)
        yield bindings.bind_class_arguments(self, range=range, normalized=normalized)
        self.free_cache()

    def enable_cache(self, ctype):
        '''
        Method to enable the cache for values of this PDF.

        :param ctype: cache type, it can be any of ('bind', 'const').
        :type ctype: str

        .. warning:: This method is not meant to be utilized by users since \
        it can turn really harmful if not handled properly.
        '''
        if ctype not in PDF.CACHE_TYPES:
            raise ValueError(
                f'Unknown cache type "{ctype}"; select from: {PDF.CACHE_TYPES}')
        self.__cache = {}
        self.__cache_type = ctype

    def free_cache(self):
        '''
        Free the cache from memory.

        .. warning:: This method is not meant to be utilized by users since \
        it can turn really harmful if not handled properly.
        '''
        self.__cache = None
        self.__cache_type = None

    def generate(self, size=10000, mapsize=100, gensize=10000, safe_factor=1.1, range=parameters.FULL):
        '''
        Generate random data.
        A call to :meth:`PDF.bind` is implicit, since several calls will be done
        to the PDF with the same sets of values.

        :param size: size (or minimum size) of the output sample.
        :type size: int
        :param mapsize: number of points to consider per dimension (data parameter) \
        in order to calculate the maximum value of the PDF.
        :type mapsize: int
        :param gensize: number of entries to generate per iteration.
        :type gensize: int
        :param safe_factor: additional factor to multiply the numerically calculated \
        maximum of the function. In general this must be modified if the function is \
        not well-behaved.
        :type safe_factor: float
        :param range: range of the data parameters where to generate data.
        :type range: str
        :returns: output sample.
        :rtype: DataSet
        '''
        with self.bind(range, normalized=False) as proxy:

            bounds = parameters.bounds_for_range(proxy.data_pars, range)
            if len(bounds.shape) == 1:
                result = proxy._generate_single_bounds(
                    size, mapsize, gensize, safe_factor, bounds)
            else:
                # Get the associated number of entries per bounds
                fracs = []
                for b in bounds[:-1]:  # The last does not need to be calculated
                    fracs.append(
                        proxy._integral_single_bounds(b, range))

                u = aop.random_uniform(0, 1, size)

                entries = []
                for f in fracs:
                    entries.append(aop.count_nonzero(aop.le(u, f)))
                # The last is calculated from the required number of entries
                entries.append(
                    size - np.sum(np.array(entries, dtype=types.cpu_int)))

                # Iterate over the bounds and add data accordingly
                samples = []
                for e, b in zip(entries, bounds):
                    samples.append(proxy._generate_single_bounds(
                        e, mapsize, gensize, safe_factor, b))
                result = dataset.DataSet.merge(samples)

        return result

    def get_values(self):
        '''
        Get the values of the parameters within this object.

        :returns: dictionary with the values of the parameters.
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
        :returns: integral of the PDF in the range defined by "integral_range" normalized to "range".
        :rtype: float
        '''
        bounds = parameters.bounds_for_range(self.data_pars, integral_range)
        if len(bounds.shape) == 1:
            return self._integral_single_bounds(bounds, range)
        else:
            return np.sum(np.fromiter((self._integral_single_bounds(b, range) for b in bounds), dtype=types.cpu_type))

    @allows_bind_or_const_cache
    def norm(self, range=parameters.FULL):
        '''
        Calculate the normalization of the PDF.

        :param range: normalization range to consider.
        :type range: str
        :returns: value of the normalization.
        :rtype: float
        '''
        raise NotImplementedError(
            'Classes inheriting from "minkit.PDF" must define the "norm" method')

    @allows_bind_or_const_cache
    def numerical_normalization(self, range=parameters.FULL):
        '''
        Calculate a numerical normalization.

        :param range: normalization range.
        :type range: str
        :returns: normalization.
        :rtype: float
        '''
        bounds = parameters.bounds_for_range(self.data_pars, range)
        if len(bounds.shape) == 1:
            return self._numerical_normalization_single_bounds(bounds)
        else:
            return np.sum(np.fromiter((self._numerical_normalization_single_bounds(b) for b in bounds), dtype=types.cpu_type))

    def set_values(self, **kwargs):
        '''
        Set the values of the parameters associated to this PDF.

        :param kwargs: keyword arguments with "name"/"value".
        :type kwargs: dict(str, float)

        .. note:: Any PDF sharing parameters with this will also change its behaviour.
        '''
        for a in self.all_real_args:
            a.value = kwargs.get(a.name, a.value)

    def to_json_object(self):
        '''
        Dump the PDF information into a JSON object.
        The PDF can be constructed at any time by calling :meth:`PDF.from_json_object`.

        :returns: object that can be saved into a JSON file.
        :rtype: dict
        '''
        raise NotImplementedError(
            'Classes inheriting from "minkit.PDF" must define the "to_json_object" method')


class SourcePDF(PDF):
    '''
    A PDF created from source files.
    '''

    def __init__(self, name, data_pars, arg_pars=None, var_arg_pars=None):
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
        :type arg_pars: Registry(Parameter)
        :param var_arg_pars: argument parameters whose number can vary.
        :type var_arg_pars: Registry(Parameter)
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

        # Access the PDF
        function, pdf, normalization = accessors.access_pdf(
            self.__class__.__name__, ndata_pars=len(data_pars), narg_pars=len(arg_pars), nvar_arg_pars=nvar_arg_pars)

        self.__function = function
        self.__pdf = pdf
        self.__norm = normalization

        super(SourcePDF, self).__init__(name, parameters.Registry(
            data_pars), parameters.Registry(arg_pars + var_arg_pars))

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
        '''
        # Determine the values to use
        fvals = tuple(v.value for v in self.args)

        # Prepare the data arrays I/O
        out = aop.zeros(len(data))

        data_arrs = tuple(data[p.name] for p in self.data_pars)

        # Call the real function
        self.__pdf(out, *data_arrs, *fvals)

        # Calculate the normalization
        if normalized:
            if self.__norm is not None:
                # There is an analytical approach to calculate the normalization
                nr = parameters.bounds_for_range(self.data_pars, range)
                if len(nr.shape) == 1:
                    n = self.__norm(*fvals, *nr)
                else:
                    n = np.sum(np.fromiter((self.__norm(*fvals, *inr)
                                            for inr in nr), dtype=types.cpu_type))
                return out / n
            else:
                # Must use a numerical normalization
                return out / self.numerical_normalization(range)
        else:
            return out

    @classmethod
    def from_json_object(cls, obj, pars):
        '''
        Build a PDF from a JSON object.
        This object must represent the internal structure of the PDF.

        :param obj: JSON object.
        :type obj: dict
        :returns: newly constructed PDF and parameters.
        :rtype: PDF
        '''
        if cls.__name__ == 'SourcePDF':
            raise NotImplementedError(
                'SourcePDF is an abstract class; do not use it directly')

        data_pars = list(map(pars.get, obj['data_pars']))
        arg_pars = list(map(pars.get, obj['arg_pars']))

        return cls(obj['name'], *data_pars, *arg_pars)

    def function(self, range=parameters.FULL, normalized=True):
        '''
        Evaluate the function, where the data values are provided by the user as single numbers.

        :param range: normalization range.
        :type range: str
        :param normalized: whether to return a normalized value.
        :type normalized: bool
        :returns: value of the PDF.
        :rtype: float
        '''
        dvals = tuple(v.value for v in self.data_pars)
        fvals = tuple(v.value for v in self.args)
        v = self.__function(*dvals, *fvals)
        if normalized:
            nr = parameters.bounds_for_range(self.data_pars, range)
            if len(nr.shape) == 1:
                n = self.__norm(*fvals, *nr)
            else:
                n = np.sum(np.fromiter((self.__norm(*fvals, *inr)
                                        for inr in nr), dtype=types.cpu_type))
            return v / n
        else:
            return v

    @allows_bind_or_const_cache
    def norm(self, range=parameters.FULL):
        '''
        Calculate the normalization of the PDF.
        If the normalization is not defined in the source file, then a
        numerical integration is done in order to calculate it.

        :param range: normalization range to consider.
        :type range: str
        :returns: value of the normalization.
        :rtype: float
        '''
        if self.__norm is not None:
            fvals = tuple(v.value for v in self.args)
            nr = parameters.bounds_for_range(self.data_pars, range)
            if len(nr.shape) == 1:
                return self.__norm(*fvals, *nr)
            else:
                return np.sum(np.fromiter((self.__norm(*fvals, *inr) for inr in nr), dtype=types.cpu_type))
        else:
            return self.numerical_normalization(range)

    def to_json_object(self):
        '''
        Dump the PDF information into a JSON object.
        The PDF can be constructed at any time by calling :meth:`PDF.from_json_object`.

        :returns: object that can be saved into a JSON file.
        :rtype: dict
        '''
        return {'class': self.__class__.__name__,  # Save also the class name of the PDF
                'name': self.name,
                'data_pars': self.data_pars.names,
                'arg_pars': self.args.names}


class MultiPDF(PDF):
    '''
    Base class for subclasses managing multiple PDFs.
    '''

    # Allow to define if a PDF object depends on other PDFs. All PDF objects
    # with this value set to "True" must have the property "pdfs" implemented.
    dependent = True

    def __init__(self, name, pdfs, arg_pars=None):
        '''
        Base class owing many PDFs.

        :param name: name of the PDF.
        :type name: str
        :param pdfs: :class:`PDF` objects to hold.
        :type pdfs: list(PDF)
        :param arg_pars: possible argument parameters.
        :type arg_pars: Registry(Parameter)
        '''
        if arg_pars is not None:
            arg_pars = parameters.Registry(arg_pars)
        else:
            arg_pars = parameters.Registry()

        self.__pdfs = parameters.Registry(pdfs)

        data_pars = parameters.Registry()
        for pdf in pdfs:
            data_pars += pdf.data_pars

        super(MultiPDF, self).__init__(name, data_pars, arg_pars)

    @property
    def all_args(self):
        '''
        Get all the argument parameters associated to this class.
        If this object is composed by many :class:`PDF`, a recursion is done in order
        to get all of them.

        :returns: all the argument parameters associated to this class.
        :rtype: Registry(Parameter)
        '''
        args = parameters.Registry(super(MultiPDF, self).all_args)
        for p in self.__pdfs:
            args += p.all_args
        return args

    @property
    def all_pars(self):
        '''
        Get all the parameters associated to this class.
        This includes also any :class:`ParameterFormula`.
        If this object is composed by many :class:`PDF`, a recursion is done in order
        to get all of them.

        :returns: all the argument parameters associated to this class.
        :rtype: Registry(Parameter)
        '''
        pars = parameters.Registry(super(MultiPDF, self).all_pars)
        for p in self.__pdfs:
            pars += p.all_pars
        return pars

    @property
    def all_pdfs(self):
        '''
        Recursively get all the possible PDFs belonging to this object.
        '''
        pdfs = parameters.Registry(self.__pdfs)
        for pdf in filter(lambda p: p.dependent, self.__pdfs):
            pdfs += pdf.all_pdfs
        return pdfs

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
        '''
        Build a PDF from a JSON object.
        This object must represent the internal structure of the PDF.

        :param obj: JSON object.
        :type obj: dict
        :returns: newly constructed PDF and parameters.
        :rtype: PDF
        '''
        class_name = obj['class']
        if class_name == 'MultiPDF':
            raise NotImplementedError(
                'Classes inheriting from "minkit.MultiPDF" must define the "to_json_object" method')
        cl = PDF_REGISTRY.get(class_name, None)
        if cl is None:
            raise RuntimeError(
                f'Class "{class_name}" does not appear in the registry')
        # Use the correct constructor
        return cl.from_json_object(obj, pars, pdfs)

    @contextlib.contextmanager
    def bind(self, range=parameters.FULL, normalized=True):
        '''
        Prepare an object that will be called many times with the same set of
        values.
        This is usefull for PDFs using a cache, to avoid creating it many times
        in sucessive calls to :meth:`PDF.__call__`.

        :param range: normalization range.
        :type range: str
        :param normalized: whether to return a normalized output.
        :type normalized: bool
        '''
        for pdf in self.pdfs:
            pdf.enable_cache(PDF.BIND)
        with super(MultiPDF, self).bind(range, normalized) as base:
            yield base

    def component(self, name):
        '''
        Get the :class:`PDF` object with the given name.

        :param name: name of the :class:`PDF`.
        :type name: str
        :returns: component with the given name.
        :rtype: PDF
        '''
        for pdf in self.__pdfs:
            if pdf.name == name:
                return pdf
        raise LookupError(f'No PDF with name "{name}" hass been found')

    def enable_cache(self, ctype):
        '''
        Method to enable the cache for values of this PDF.

        :param ctype: cache type, it can be any of ('bind', 'const').
        :type ctype: str

        .. warning:: This method is not meant to be utilized by users since \
        it can turn really harmful if not handled properly.
        '''
        super(MultiPDF, self).enable_cache(ctype)
        for pdf in self.pdfs:
            pdf.enable_cache(ctype)

    def free_cache(self):
        '''
        Free the cache from memory.

        .. warning:: This method is not meant to be utilized by users since \
        it can turn really harmful if not handled properly.
        '''
        super(MultiPDF, self).free_cache()
        for pdf in self.pdfs:
            pdf.free_cache()


@register_pdf
class AddPDFs(MultiPDF):
    '''
    Definition of the addition of many PDFs.
    '''

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

        super(AddPDFs, self).__init__(name, pdfs, yields)

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
        '''
        yields = list(v.value for v in self.args)

        if not self.extended:
            yields.append(1. - sum(yields))

        out = aop.zeros(len(data))
        for y, pdf in zip(yields, self.pdfs):
            out += y * pdf(data, range, normalized=True)

        if self.extended and normalized:
            return out / sum(yields)
        else:
            return out

    @classmethod
    def from_json_object(cls, obj, pars, pdfs):
        '''
        Build a PDF from a JSON object.
        This object must represent the internal structure of the PDF.

        :param obj: JSON object.
        :type obj: dict
        :returns: newly constructed PDF and parameters.
        :rtype: PDF
        '''
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
        :param yf: yield associated to the first PDF, if both "yf" and "ys" \
        are provided. If "ys" is not provided, then "yf" is the faction \
        associated to the first PDF.
        :type yf: Parameter
        :param ys: possible yield for the second PDF.
        :type ys: Parameter
        :returns: the built class.
        :rtype: AddPDFs
        '''
        return cls(name, [first, second], [yf] if ys is None else [yf, ys])

    @property
    def extended(self):
        '''
        Get whether this PDF is of "extended" type.

        :returns: whether this PDF is of "extended" type.
        :rtype: bool
        '''
        return len(self.pdfs) == len(self.args)

    @allows_bind_or_const_cache
    def norm(self, range=parameters.FULL):
        '''
        Calculate the normalization of the PDF.

        :param range: normalization range to consider.
        :type range: str
        :returns: value of the normalization.
        :rtype: float
        '''
        if self.extended:
            # An extended PDF has as normalization the sum of yields
            return sum(v.value for v in self.args)
        else:
            # A non-extended PDF is always normalized
            return 1.

    def to_json_object(self):
        '''
        Dump the PDF information into a JSON object.
        The PDF can be constructed at any time by calling :meth:`PDF.from_json_object`.

        :returns: object that can be saved into a JSON file.
        :rtype: dict
        '''
        return {'class': self.__class__.__name__,  # Save also the class name of the PDF
                'name': self.name,
                'pdfs': self.pdfs.names,
                'data_pars': self.data_pars.names,
                'yields': self.args.names}


@register_pdf
class ConvPDFs(MultiPDF):
    '''
    Definition of a convolution of two PDFs.
    '''

    def __init__(self, name, first, second, range=None):
        '''
        Represent the convolution of two different PDFs.

        :param name: name of the PDF.
        :type name: str
        :param first: first PDF.
        :type first: PDF
        :param second: second PDF.
        :type second: PDF
        :param range: range of the convolution. This is needed in case part of the \
        PDFs lie outside the evaluation range. It is set to "full" by default.
        :type range: str
        '''
        error = ValueError(
            f'Convolution is only supported in 1-dimensional PDFs')
        if len(first.data_pars) != 1:
            raise error

        if len(second.data_pars) != 1:
            raise error

        # The convolution size and range can be changed by the user
        self.range = range or parameters.FULL
        self.conv_size = CONV_SIZE

        super(ConvPDFs, self).__init__(name, [first, second])

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
        '''
        dv, cv = self.convolve(range, normalized)

        par = tuple(self.data_pars)[0]

        pdf_values = aop.interpolate_linear(data[par.name], dv, cv)

        return pdf_values

    @classmethod
    def from_json_object(cls, obj, pars, pdfs):
        '''
        Build a PDF from a JSON object.
        This object must represent the internal structure of the PDF.

        :param obj: JSON object.
        :type obj: dict
        :returns: newly constructed PDF and parameters.
        :rtype: PDF
        '''
        first = pdfs.get(obj['first'])
        second = pdfs.get(obj['second'])
        return cls(obj['name'], first, second, obj['range'])

    @allows_bind_or_const_cache
    def convolve(self, range=parameters.FULL, normalized=True):
        '''
        Calculate the convolution.

        :param range: normalization range.
        :type range: str
        :param normalized: whether to return a normalized output.
        :type normalized: bool
        :returns: data and result of the evaluation.
        :rtype: numpy.ndarray, numpy.ndarray
        '''
        first, second = tuple(self.pdfs)

        bounds = parameters.bounds_for_range(first.data_pars, self.range)

        if bounds.shape != (2,):
            raise RuntimeError(
                f'The convolution bounds must not be disjointed')

        # Only works for the 1-dimensional case
        par = tuple(first.data_pars)[0]

        grid = dataset.evaluation_grid(
            first.data_pars, bounds, size=self.conv_size)

        fv = first(grid, self.range, normalized)
        sv = second(grid, self.range, normalized)

        return grid[par.name], aop.real(aop.fftconvolve(fv, sv, grid[par.name]))

    @allows_bind_or_const_cache
    def norm(self, range=parameters.FULL):
        '''
        Calculate the normalization of the PDF.
        In this case, a numerical normalization is used, so the effect
        is equivalent to :meth:`ConvPDFs.numerical_normalization`.

        :param range: normalization range to consider.
        :type range: str
        :returns: value of the normalization.
        :rtype: float
        '''
        return self.numerical_normalization(range)

    @allows_bind_or_const_cache
    def numerical_normalization(self, range=parameters.FULL):
        '''
        Calculate a numerical normalization.

        :param range: normalization range.
        :type range: str
        :returns: normalization.
        :rtype: float
        '''
        # A convolutions is always normalized
        return 1.

    def to_json_object(self):
        '''
        Dump the PDF information into a JSON object.
        The PDF can be constructed at any time by calling :meth:`PDF.from_json_object`.

        :returns: object that can be saved into a JSON file.
        :rtype: dict
        '''
        first, second = self.pdfs
        return {'class': self.__class__.__name__,  # Save also the class name of the PDF
                'name': self.name,
                'first': first.name,
                'second': second.name,
                'range': self.range}


@register_pdf
class ProdPDFs(MultiPDF):
    '''
    Definition of the product of two different PDFs with different data parameters.
    '''

    def __init__(self, name, pdfs):
        '''
        This object represents the product of many PDFs where the data parameters are
        not shared among them.

        :param name: name of the PDF.
        :type name: str
        :param pdfs: list of PDFs
        :type pdfs: list(PDF)
        '''
        super(ProdPDFs, self).__init__(name, pdfs, [])

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
        '''
        out = aop.ones(len(data))
        for pdf in self.pdfs:
            out *= pdf(data, range, normalized=True)

        return out

    @classmethod
    def from_json_object(cls, obj, pars, pdfs):
        '''
        Build a PDF from a JSON object.
        This object must represent the internal structure of the PDF.

        :param obj: JSON object.
        :type obj: dict
        :returns: newly constructed PDF and parameters.
        :rtype: PDF
        '''
        pdfs = list(map(pdfs.get, obj['pdfs']))
        return cls(obj['name'], pdfs)

    @allows_bind_or_const_cache
    def norm(self, range=parameters.FULL):
        '''
        Calculate the normalization of the PDF.

        :param range: normalization range to consider.
        :type range: str
        :returns: value of the normalization.
        :rtype: float
        '''
        n = 1.
        for p in self.pdfs:
            n *= p.norm(range)
        return n

    @allows_bind_or_const_cache
    def numerical_normalization(self, range=parameters.FULL):
        '''
        Calculate a numerical normalization.

        :param range: normalization range.
        :type range: str
        :returns: normalization.
        :rtype: float
        '''
        return np.prod(np.fromiter((pdf.numerical_normalization(range) for pdf in self.pdfs), dtype=types.cpu_type))

    def to_json_object(self):
        '''
        Dump the PDF information into a JSON object.
        The PDF can be constructed at any time by calling :meth:`PDF.from_json_object`.

        :returns: object that can be saved into a JSON file.
        :rtype: dict
        '''
        return {'class': self.__class__.__name__,  # Save also the class name of the PDF
                'name': self.name,
                'pdfs': self.pdfs.names}


def pdf_from_json(obj):
    '''
    Load a PDF from a JSON object.

    :param obj: JSON-like object.
    :type obj: dict
    :returns: a PDF with the configuration from the object.
    :rtype: PDF
    '''
    # Parse parameters
    pars = parameters.Registry(
        [parameters.Parameter.from_json_object(o) for o in obj['ipars']])

    for o in obj['dpars']:
        pars.append(parameters.Formula.from_json_object(o, pars))

    # Parse PDFs
    pdfs = parameters.Registry(
        [PDF.from_json_object(o, pars) for o in obj['ipdfs']])

    for o in obj['dpdfs']:
        pdfs.append(MultiPDF.from_json_object(o, pars, pdfs))

    # Last PDF to be constructed is the main PDF
    return pdfs[-1]


def _solve_dependencies(objects):
    '''
    Get the resolution order for the given set of dependencies.

    :param objects: names of the objects with their dependencies.
    :type objects: list(tuple(str, list(str)))
    :returns: resolution order.
    :rtype: list(str)
    '''
    ntot = len(objects)

    result = []

    objects = list(objects)
    while len(result) != ntot:
        new_objects = []
        for name, lst in objects:
            l = set(lst).difference(result)
            if len(l) == 0:
                result.append(name)
            else:
                new_objects.append((name, l))
        objects = new_objects
    return result


def pdf_to_json(pdf):
    '''
    Dump a PDF to a JSON-like object.

    :param pdf: :class:`PDF` to dump.
    :type pdf: PDF
    :returns: JSON-like object.
    :rtype: dict
    '''
    # Declare some lambda functions to make code more readable
    def is_dependent(p): return p.dependent
    def not_dependent(p): return not p.dependent

    # Solve dependencies for ParameterFormula objects
    pars = pdf.all_pars
    ipars = parameters.Registry(filter(not_dependent, pars))
    dpars = parameters.Registry(filter(is_dependent, pars))
    for p in filter(is_dependent, pars):

        pars = p.all_args
        ipars += parameters.Registry(filter(not_dependent, pars))
        dpars += parameters.Registry(filter(is_dependent, pars))

    ro = _solve_dependencies(
        [(d.name, set(d.all_args.names).difference(ipars.names)) for d in dpars])

    dpars = parameters.Registry([dpars.get(n) for n in ro])

    # Solve dependencies for PDF objects
    if pdf.dependent:
        pdfs = pdf.all_pdfs
        ipdfs = parameters.Registry(filter(not_dependent, pdfs))
        dpdfs = parameters.Registry(filter(is_dependent, pdfs))

        dpdfs.append(pdf)  # Do not forget to add itself

        ro = _solve_dependencies(
            [(d.name, set(d.all_pdfs.names).difference(ipdfs.names)) for d in dpdfs])

        dpdfs = parameters.Registry([dpdfs.get(n) for n in ro])
    else:
        ipdfs = parameters.Registry([pdf])
        dpdfs = parameters.Registry()

    return {'dpdfs': [pdf.to_json_object() for pdf in dpdfs],
            'ipdfs': [pdf.to_json_object() for pdf in ipdfs],
            'dpars': [p.to_json_object() for p in dpars],
            'ipars': [p.to_json_object() for p in ipars]}
