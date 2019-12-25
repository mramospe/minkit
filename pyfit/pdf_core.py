'''
Base classes to define PDF types.
'''

from . import core
from . import data
from . import parameters
from . import types

import itertools
import logging
import numpy as np

__all__ = ['AddPDFs', 'ProdPDFs']

# Default size of the samples to be used during numerical normalization
NORM_SIZE = 1000000

logger = logging.getLogger(__name__)


class PDF(object):
    '''
    Object defining the properties of a PDF.
    '''
    def __init__( self, name, data_pars, args_pars ):
        '''
        Build the class from a name, a set of data parameters and argument parameters.
        The first correspond to those that must be present on any :class:`DataSet` or
        :class:`BinnedDataSet` classes where this object is evaluated.
        The second corresponds to the parameters defining the shape of the PDF.

        :param name: name of the object.
        :type name: str
        :param data_pars: data parameters.
        :type data_pars: Registry(str, Parameter)
        :param arg_pars: argument parameters.
        :type arg_pars: Registry(str, Parameter)
        '''
        self.name        = name
        self.__data_pars = data_pars
        self.__arg_pars  = args_pars
        self.norm_size   = NORM_SIZE
        super(PDF, self).__init__()

    def _integral_bin_area( self, norm_range = parameters.FULL ):
        '''
        Calculate the area of a bin used to calculate the numerical normalization.

        :param norm_range: normalization range.
        :type norm_range: str
        :returns: area of the normalization bin.
        :rtype: float
        '''
        a = 1.
        for p in self.data_pars.values():
            pmin, pmax = p.ranges[norm_range]
            a *= (pmax - pmin) * 1. / self.norm_size
        return a

    def _process_values( self, values = None ):
        '''
        Process the input values.
        If "values" is set to None, then the values from the argument 
        parameters are used.

        :param values: possible values for the parameters to use.
        :type values: Registry(str, float)
        :returns: processed values, sorted following the argument parameters.
        :rtype: tuple(float)
        '''
        if values is None:
            return tuple(v.value for v in self.args.values())
        else:
            return tuple(values[n] for n in self.args.keys())

    def _process_norm_range( self, norm_range = parameters.FULL ):
        '''
        Process a normalization range, returning the associated bounds for each
        data parameter.

        :param norm_range: normalization range.
        :type norm_range: str
        :returns: bounds for each data parameter.
        :rtype: tuple(tuple(float, float))
        '''
        return tuple(itertools.chain.from_iterable(p.ranges[norm_range] for p in self.data_pars.values()))

    def __call__( self, data, values = None, norm_range = parameters.FULL, normalized = True ):
        '''
        Call the PDF in the given set of data.

        :param data: data to evaluate.
        :type data: DataSet or BinnedDataSet
        :param values: values for the argument parameters to use.
        :type values: Registry(str, float)
        :param norm_range: normalization range.
        :type norm_range: str
        :param normalized: whether to return a normalized output.
        :type normalized: boole
        '''
        raise NotImplementedError('Classes inheriting from "pyfit.PDF" must define the "__call__" operator')

    @property
    def all_args( self ):
        '''
        Get all the argument parameters associated to this class.
        If this object is composed by many :class:`PDF`, a recursion is done in order
        to get all of them.

        :returns: all the argument parameters associated to this class.
        :rtype: Registry(str, Parameter)
        '''
        return self.args

    @property
    def args( self ):
        '''
        Get the argument parameters this object directly depends on.

        :returns: parameters this object directly depends on.
        :rtype: Registry(str, Parameter)
        '''
        return self.__arg_pars

    @property
    def constant( self ):
        '''
        Get whether this is a constant :class:`PDF`.

        :returns: whether this object can be marked as constant.
        :rtype: bool
        '''
        return all(p.constant for p in self.all_args.values())

    @property
    def data_pars( self ):
        '''
        Get the data parameters this object directly depends on.

        :returns: parameters this object directly depends on.
        :rtype: Registry(str, Parameter)
        '''
        return self.__data_pars

    def frozen( self, data, norm_range = parameters.FULL ):
        '''
        Check whether this PDF is constant or not.
        If so, it returns a cached version of it, and itself otherwise.
        This is meant to be used before successive possible expensive calls.

        :param data: data where this class is supposed to be evaluated.
        :type data: DataSet or BinnedDataSet
        :param norm_range: normalization range.
        :type norm_range: str
        :returns: a cached version of this object, if it is constant, itself otherwise.
        :rtype: PDF
        '''
        if self.constant:
            # This PDF is constant, so save itself into a cache
            logger.info(f'Function "{self.name}" marked as constant; will precalculate values and save them in a cache')
            return CachePDF(self, data, norm_range)
        else:
            return self

    def generate( self, size = 10000, values = None, mapsize = 100, safe_factor = 1.1, drop_excess = True, eval_range = parameters.FULL ):
        '''
        Generate random data.

        :param size: size (or minimum size) of the output sample.
        :type size: int
        :param values: values of the argument parameters to be used.
        :type values: Registry(str, float)
        :param mapsize: number of points to consider per dimension (data parameter) \
        in order to calculate the maximum value of the PDF.
        :type mapsize: int
        :param safe_factor: additional factor to multiply the numerically calculated \
        maximum of the function. In general this must be modified if the function is \
        not well-behaved.
        :type safe_factor: float
        :param drop_excess: since the generation is done in chunks, after processing \
        the size of the output data will be bigger than requested. If set to True, \
        then the excess is dropped.
        :type drop_excess: bool
        :param eval_range: range of the data parameters where to generate data.
        :type eval_range: str
        :returns: output sample.
        :rtype: DataSet
        '''
        grid = data.evaluation_grid(self.data_pars, mapsize, eval_range)

        m = safe_factor * core.max(self.__call__(grid, values))

        result = None

        while result is None or len(result) < size:

            d = data.uniform_sample(self.data_pars, size)
            f = self.__call__(d, values)
            u = core.random_uniform(0, m, len(d))

            if result is None:
                result = d.subset(u < f)
            else:
                result.add(d.subset(u < f), inplace=True)

        if drop_excess:
            return result.subset(slice(size))
        else:
            return result

    def integral( self, values = None, eval_range = parameters.FULL ):
        '''
        Calculate the integral of a :class:`PDF`.
        '''
        g = data.evaluation_grid(self.data_pars, self.norm_size**len(self.data_pars), eval_range)
        a = self._integral_bin_area(norm_range)
        i = self.__call__(g, values)
        return core.sum(i) * a

    def norm( self, values = None, norm_range = parameters.FULL ):
        '''
        Calculate the normalization of the PDF.

        :param values: values of the parameters to use.
        :type values: Registry(str, float)
        :param norm_range: normalization range to consider.
        :type norm_range: str
        :returns: value of the normalization.
        :rtype: float
        '''
        raise NotImplementedError('Classes inheriting from "pyfit.PDF" must define the "norm" method')

    def numerical_normalization( self, values = None, norm_range = parameters.FULL ):
        '''
        Calculate a numerical normalization.

        :param values: values to use.
        :type values: Registry(str, float)
        :param norm_range: normalization range.
        :type norm_range: str
        :returns: normalization.
        :rtype: float
        '''
        g = data.evaluation_grid(self.data_pars, self.norm_size**len(self.data_pars), norm_range)
        a = self._integral_bin_area(norm_range)
        i = self.__call__(g, values, normalized=False)
        return core.sum(i) * a


class CachePDF(PDF):
    '''
    Cache object to replace constant PDFs.
    '''
    def __init__( self, pdf, data, norm_range = parameters.FULL ):
        '''
        A CachePDF is an object that replaces the an usual PDF when it is marked as
        constant in a minimization process.

        :param pdf: function to process.
        :type pdf: PDF
        :param data: data where the :class:`PDF` is being evaluated.
        :type data: DataSet or BinnedDataSet.
        :param norm_range: normalization range.
        :type norm_range: str
        '''
        self.__cache = pdf(data, norm_range=norm_range)
        self.__norm  = pdf.norm(norm_range=norm_range)

        super(CachePDF, self).__init__(pdf.name, pdf.data_pars, pdf.args)

    def __call__( self, *args, normalized = True, **kwargs ):
        '''
        Return the evaluation of the PDF in the data.
        Only the "normalized" keyword is considered for this class, since the other
        input arguments do not make any effect.

        :param normalized: whether to return a normalized output or not.
        :type normalized: bool
        :returns: values of the PDF in the data.
        :rtype: numpy.ndarray
        '''
        if normalized:
            return self.__cache / self.__norm
        else:
            return self.__cache

    def norm( self, *args, **kwargs ):
        '''
        Return the normalization of the PDF.
        Since the information has been saved in the cache, it simply returns
        the normalization stored.

        :returns: normalization.
        :rtype: float
        '''
        return self.__norm


class SourcePDF(PDF):
    '''
    A PDF created from source files.
    '''
    def __init__( self, name, function, pdf, normalization, data_pars, arg_pars = None, var_arg_pars = None ):
        '''
        This object defines a PDF built from source files (C++, PyOpenCL or CUDA), which depend
        on the backend to use.

        :param name: name of the PDF.
        :type name: str
        :param function: function where the input data is a simple float.
        :type function: function
        :param pdf: function where the input data is considered as an array.
        :type pdf: function
        :param norm: normalization function.
        :type norm: function
        :param data_pars: data parameters.
        :type data_pars: Registry(str, Parameter)
        :param arg_pars: argument parameters.
        :type arg_pars: Registry(str, Parameter)
        :param var_arg_pars: argument parameters whose number can vary.
        :type var_arg_pars: Registry(str, Parameter)
        '''
        arg_pars = arg_pars or []

        if var_arg_pars is not None:
            var_arg_pars = list(var_arg_pars)
            self.__var_args = parameters.Registry.from_list(var_arg_pars)
        else:
            var_arg_pars = []
            self.__var_args = None

        self.__function = function
        self.__pdf      = pdf
        self.__norm     = normalization

        super(SourcePDF, self).__init__(name, parameters.Registry.from_list(data_pars), parameters.Registry.from_list(arg_pars + var_arg_pars))

    def __call__( self, data, values = None, norm_range = parameters.FULL, normalized = True ):
        '''
        Call the PDF in the given set of data.

        :param data: data to evaluate.
        :type data: DataSet or BinnedDataSet
        :param values: values for the argument parameters to use.
        :type values: Registry(str, float)
        :param norm_range: normalization range.
        :type norm_range: str
        :param normalized: whether to return a normalized output.
        :type normalized: boole
        '''
        # Determine the values to use
        fvals = self._process_values(values)

        # If there is a variable number of arguments, must be at the end
        if self.__var_args is not None:
            nvar_args = len(self.__var_args)
            var_args_array = np.array(fvals[-nvar_args:], dtype=types.c_double)
            fvals = fvals[:-nvar_args] + (var_args_array,)

        # Prepare the data arrays I/O
        out = core.zeros(len(data))
        data_arrs = tuple(data[n] for n in self.data_pars)

        # Call the real function
        self.__pdf(out, *data_arrs, *fvals)

        # Calculate the normalization
        if normalized:
            if self.__norm is not None:
                # There is an analytical approach to calculate the normalization
                nr = self._process_norm_range(norm_range)
                n = self.__norm(*fvals, *nr)
                return out / n
            else:
                # Must use a numerical normalization
                return out / self.numerical_normalization(values, norm_range)
        else:
            return out

    def function( self, *data_values, values = None, norm_range = parameters.FULL, normalized = True ):
        '''
        Evaluate the function, where the data values are provided by the user as single numbers.

        :param data_values: values of the data parameters.
        :type dat_values: tuple(float)
        :param values: values of the argument parameters.
        :type values: Registry(str, float)
        :param norm_range: normalization range.
        :type norm_range: str
        :param normalized: whether to return a normalized value.
        :type normalized: bool
        :returns: value of the PDF.
        :rtype: float
        '''
        fvals = self._process_values(values)
        v = self.__function(*data_values, *fvals)
        if normalized:
            nr = self._process_norm_range(norm_range)
            n = self.__norm(*fvals, *nr)
            return v / n
        else:
            return v

    def norm( self, values = None, norm_range = parameters.FULL ):
        '''
        Calculate the normalization of the PDF.
        If the normalization is not defined in the source file, then a
        numerical integration is done in order to calculate it.

        :param values: values of the parameters to use.
        :type values: Registry(str, float)
        :param norm_range: normalization range to consider.
        :type norm_range: str
        :returns: value of the normalization.
        :rtype: float
        '''
        if self.__norm is not None:
            fvals = self._process_values(values)
            nr = self._process_norm_range(norm_range)
            return self.__norm(*fvals, *nr)
        else:
            return self.numerical_normalization(values, norm_range)


class MultiPDF(PDF):
    '''
    Base class for subclasses managing multiple PDFs.
    '''
    def __init__( self, name, pdfs, arg_pars = None ):
        '''
        Base class owing many PDFs.

        :param name: name of the PDF.
        :type name: str
        :param pdfs: :class:`PDF` objects to hold.
        :type pdfs: list(PDF)
        :param arg_pars: possible argument parameters.
        :type arg_pars: Registry(str, Parameter)
        '''
        if arg_pars is not None:
            arg_pars = parameters.Registry.from_list(arg_pars)
        else:
            arg_pars = parameters.Registry()

        self.__pdfs = parameters.Registry.from_list(pdfs)

        data_pars = parameters.Registry()
        for pdf in pdfs:
            data_pars.update(pdf.data_pars)

        super(MultiPDF, self).__init__(name, data_pars, arg_pars)

    @property
    def all_args( self ):
        '''
        Get all the argument parameters associated to this class.
        If this object is composed by many :class:`PDF`, a recursion is done in order
        to get all of them.

        :returns: all the argument parameters associated to this class.
        :rtype: Registry(str, Parameter)
        '''
        args = parameters.Registry(self.args)
        for p in self.__pdfs.values():
            args.update(p.all_args)
        return args

    def component( self, name ):
        '''
        Get the :class:`PDF` object with the given name.

        :param name: name of the :class:`PDF`.
        :type name: str
        :returns: component with the given name.
        :rtype: PDF
        '''
        for pdf in self.__pdfs.values():
            if pdf.name == name:
                return pdf
        raise LookupError(f'No PDF with name "{name}" hass been found')

    def frozen( self, data, norm_range = parameters.FULL ):
        '''
        Check whether this PDF is constant or not.
        If so, it returns a cached version of it, and itself otherwise.
        This is meant to be used before successive possible expensive calls.

        :param data: data where this class is supposed to be evaluated.
        :type data: DataSet or BinnedDataSet
        :param norm_range: normalization range.
        :type norm_range: str
        :returns: a cached version of this object, if it is constant, itself otherwise.
        :rtype: PDF
        '''
        if not any(pdf.constant for pdf in self.__pdfs.values()):
            # There is no constant PDF within this class. Return itself.
            return self
        elif self.constant:
            # This PDF is constant, call the base class method
            return super(AddPDFs, self).frozen(data, norm_range)
        else:
            # At least one of the contained PDFs is constant, must create a new instance
            pdfs = list(pdf.frozen(data, norm_range) for pdf in self.__pdfs.values())
            return self.__class__(self.name, pdfs, self.args.to_list())

    @property
    def pdfs( self ):
        '''
        Get the registry of PDFs within this class.

        :returns: PDFs owned by this class.
        :rtype: Registry(str, PDF)
        '''
        return self.__pdfs


class AddPDFs(MultiPDF):
    '''
    Definition of the addition of many PDFs.
    '''
    def __init__( self, name, pdfs, yields ):
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

    def __call__( self, data, values = None, norm_range = parameters.FULL, normalized = True ):
        '''
        Call the PDF in the given set of data.

        :param data: data to evaluate.
        :type data: DataSet or BinnedDataSet
        :param values: values for the argument parameters to use.
        :type values: Registry(str, float)
        :param norm_range: normalization range.
        :type norm_range: str
        :param normalized: whether to return a normalized output.
        :type normalized: boole
        '''
        yields = list(self._process_values(values))
        if not self.extended:
            yields.append(1. - sum(yields))

        out = core.zeros(len(data))
        for y, pdf in zip(yields, self.pdfs.values()):
            out += y * pdf(data, values, norm_range, normalized=True)

        if self.extended and normalized:
            return out / sum(yields)
        else:
            return out

    @property
    def extended( self ):
        '''
        Get whether this PDF is of "extended" type.

        :returns: whether this PDF is of "extended" type.
        :rtype: bool
        '''
        return len(self.pdfs) == len(self.args)

    def norm( self, values = None, norm_range = parameters.FULL ):
        '''
        Calculate the normalization of the PDF.

        :param values: values of the parameters to use.
        :type values: Registry(str, float)
        :param norm_range: normalization range to consider.
        :type norm_range: str
        :returns: value of the normalization.
        :rtype: float
        '''
        if self.extended:
            # An extended PDF has a normalization equal to the sum of yields
            return sum(y.value for y in self.args.values())
        else:
            # A non-extended PDF is always normalized
            return 1.

    @classmethod
    def two_components( cls, name, first, second, yf, ys = None ):
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


class ProdPDFs(MultiPDF):
    '''
    Definition of the product of two different PDFs with different data parameters.
    '''
    def __init__( self, name, pdfs ):
        '''
        This object represents the product of many PDFs where the data parameters are
        not shared among them.

        :param name: name of the PDF.
        :type name: str
        :param pdfs: list of PDFs
        :type pdfs: list(PDF)
        '''
        super(ProdPDFs, self).__init__(name, pdfs, [])

    def __call__( self, data, values = None, norm_range = parameters.FULL, normalized = True ):
        '''
        Call the PDF in the given set of data.

        :param data: data to evaluate.
        :type data: DataSet or BinnedDataSet
        :param values: values for the argument parameters to use.
        :type values: Registry(str, float)
        :param norm_range: normalization range.
        :type norm_range: str
        :param normalized: whether to return a normalized output.
        :type normalized: boole
        '''
        out = core.ones(len(data))
        for pdf in self.pdfs.values():
            out *= pdf(data, values, norm_range, normalized=True)

        return out

    def norm( self, values = None, norm_range = parameters.FULL ):
        '''
        Calculate the normalization of the PDF.

        :param values: values of the parameters to use.
        :type values: Registry(str, float)
        :param norm_range: normalization range to consider.
        :type norm_range: str
        :returns: value of the normalization.
        :rtype: float
        '''
        n = 1.
        for p in self.pdfs.values():
            n *= p.norm(values, norm_range)
        return n


class EvaluatorProxy(object):
    '''
    Definition of a proxy class to evaluate an FCN with a PDF.
    '''
    def __init__( self, fcn, pdf, data, norm_range = parameters.FULL, constraints = None ):
        '''
        :param fcn: FCN to be used during minimization.
        :type fcn: str
        :param data: data sample to process.
        :type data: DataSet or BinnedDataSet
        :param norm_range: normalization range.
        :type norm_range: str
        :param constraints: set of constraints for the argument parameters.
        :type constraints: list(PDF)
        '''
        super(EvaluatorProxy, self).__init__()

        self.__constraints = constraints or []
        self.__fcn  = fcn
        self.__pdf  = pdf.frozen(data, norm_range)
        self.__data = data
        self.__norm_range = norm_range

    def __call__( self, *values ):
        '''
        Evaluate the FCN.
        Values must be provided sorted as :method:`PDF.args`.

        :param values: set of values to evaluate the FCN.
        :type values: tuple(float)
        :returns: value of the FCN.
        :rtype: float
        '''
        r = parameters.Registry()
        for i, n in enumerate(self.__pdf.all_args):
            r[n] = values[i]

        return self.__fcn(self.__pdf, self.__data, r, self.__norm_range, self.__constraints)
