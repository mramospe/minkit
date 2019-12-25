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


NORM_SIZE = 1000000

logger = logging.getLogger(__name__)


class PDF(object):

    def __init__( self, name, data_pars, args_pars ):
        '''
        '''
        self.name        = name
        self.__data_pars = data_pars
        self.__arg_pars  = args_pars
        super(PDF, self).__init__()

    def _process_values( self, values = None ):
        '''
        '''
        if values is None:
            return tuple(v.value for v in self.args.values())
        else:
            return tuple(values[n] for n in self.args.keys())

    def _process_norm_range( self, norm_range = parameters.FULL ):
        '''
        '''
        return tuple(itertools.chain.from_iterable(p.ranges[norm_range] for p in self.data_pars.values()))

    def __call__( self, data, values = None, norm_range = parameters.FULL, normalized = True ):
        '''
        '''
        raise NotImplementedError('Classes inheriting from "pyfit.PDF" must define the "__call__" operator')

    @property
    def all_args( self ):
        return self.args

    @property
    def args( self ):
        return self.__arg_pars

    @property
    def data_pars( self ):
        return self.__data_pars

    def frozen( self, data, norm_range = parameters.FULL ):

        if self.constant:
            # This PDF is constant, so save itself into a cache
            logger.info(f'Function "{self.name}" marked as constant; will precalculate values and save them in a cache')
            return CachePDF(self, data, norm_range)
        else:
            return self

    def generate( self, size = 10000, values = None, mapsize = 100, safe_factor = 1.1, drop_excess = True ):
        '''
        '''
        grid = data.evaluation_grid(self.data_pars, mapsize)

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

    @property
    def constant( self ):
        return all(p.constant for p in self.all_args.values())

    def norm( self, values = None, norm_range = parameters.FULL ):
        '''
        '''
        raise NotImplementedError('Classes inheriting from "pyfit.PDF" must define the "norm" method')


class CachePDF(PDF):

    def __init__( self, pdf, data, norm_range = parameters.FULL ):

        self.__cache = pdf(data, norm_range=norm_range)
        self.__norm  = pdf.norm(norm_range=norm_range)

        super(CachePDF, self).__init__(pdf.name, pdf.data_pars, pdf.args)

    def __call__( self, data, values = None, norm_range = parameters.FULL, normalized = True ):
        '''
        '''
        if normalized:
            return self.__cache / self.__norm
        else:
            return self.__cache

    def norm( self, values = None, norm_range = parameters.FULL ):
        return self.__norm


class SourcePDF(PDF):

    def __init__( self, name, function, pdf, normalization, data_pars, arg_pars = None, var_arg_pars = None ):
        '''
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
        self.norm_size  = NORM_SIZE

        super(SourcePDF, self).__init__(name, parameters.Registry.from_list(data_pars), parameters.Registry.from_list(arg_pars + var_arg_pars))

    def __call__( self, data, values = None, norm_range = parameters.FULL, normalized = True ):
        '''
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

    def numerical_normalization( self, values = None, norm_range = parameters.FULL ):
        '''
        '''
        grid = data.evaluation_grid(self.data_pars, self.norm_size**len(self.data_pars))

        a = 1.
        for p in self.data_pars.values():
            pmin, pmax = p.ranges[norm_range]
            a *= (pmax - pmin) * 1. / self.norm_size

        i = self.__call__(grid, values, normalized=False)

        return core.sum(i) * a

    def function( self, *data_values, values = None, norm_range = parameters.FULL, normalized = True ):
        '''
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
        '''
        if self.__norm is not None:
            fvals = self._process_values(values)
            nr = self._process_norm_range(norm_range)
            return self.__norm(*fvals, *nr)
        else:
            return self.numerical_normalization(values, norm_range)


class MultiPDF(PDF):

    def __init__( self, name, pdfs, arg_pars = None ):

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
        args = parameters.Registry(self.args)
        for p in self.__pdfs.values():
            args.update(p.all_args)
        return args

    def component( self, name ):
        for pdf in self.__pdfs.values():
            if pdf.name == name:
                return pdf
        raise LookupError(f'No PDF with name "{name}" hass been found')

    def frozen( self, data, norm_range = parameters.FULL ):

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
        return self.__pdfs



class AddPDFs(MultiPDF):

    def __init__( self, name, pdfs, yields ):
        '''
        '''
        assert len(pdfs) - len(yields) in (0, 1)

        super(AddPDFs, self).__init__(name, pdfs, yields)

    def __call__( self, data, values = None, norm_range = parameters.FULL, normalized = True ):

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
        return len(self.pdfs) == len(self.args)

    def norm( self, values = None, norm_range = parameters.FULL ):
        '''
        '''
        if self.extended:
            # An extended PDF has a normalization equal to the sum of yields
            return sum(y.value for y in self.args.values())
        else:
            # A non-extended PDF is always normalized
            return 1.

    @classmethod
    def two_components( cls, name, first, second, yf, ys = None ):
        return cls(name, [first, second], [yf] if ys is None else [yf, ys])


class ProdPDFs(MultiPDF):

    def __init__( self, name, pdfs ):
        '''
        
        '''
        super(ProdPDFs, self).__init__(name, pdfs, [])

    def __call__( self, data, values = None, norm_range = parameters.FULL, normalized = True ):

        out = core.ones(len(data))
        for pdf in self.pdfs.values():
            out *= pdf(data, values, norm_range, normalized=True)

        return out

    def norm( self, values = None, norm_range = parameters.FULL ):
        '''
        '''
        n = 1.
        for p in self.pdfs.values():
            n *= p.norm(values, norm_range)
        return n


class EvaluatorProxy(object):

    def __init__( self, fcn, pdf, data, norm_range = parameters.FULL, constraints = None ):

        super(EvaluatorProxy, self).__init__()

        self.__constraints = constraints or []
        self.__fcn  = fcn
        self.__pdf  = pdf.frozen(data, norm_range)
        self.__data = data
        self.__norm_range = norm_range

    def __call__( self, *values ):
        '''
        Values must be provided sorted as :method:`PDF.args`.
        '''
        r = parameters.Registry()
        for i, n in enumerate(self.__pdf.all_args):
            r[n] = values[i]

        return self.__fcn(self.__pdf, self.__data, r, self.__norm_range, self.__constraints)
