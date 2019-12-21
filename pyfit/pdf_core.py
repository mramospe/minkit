'''
Base classes to define PDF types.
'''

from . import core
from . import parameters

import itertools

__all__ = ['AddPDFs']


class PDF(object):

    def __init__( self, name, data_pars, args_pars ):
        '''
        '''
        self.name        = name
        self.__data_pars = parameters.Registry(*data_pars)
        self.__arg_pars  = parameters.Registry(*args_pars)
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

    def norm( self, values = None, norm_range = parameters.FULL ):
        '''
        '''
        raise NotImplementedError('Classes inheriting from "pyfit.PDF" must define the "norm" method')


class SourcePDF(PDF):

    def __init__( self, name, function, normalization, data_pars, args_pars ):
        '''
        '''
        self.__function  = function
        self.__norm      = normalization
        super(SourcePDF, self).__init__(name, data_pars, args_pars)

    def __call__( self, data, values = None, norm_range = parameters.FULL, normalized = True ):
        '''
        '''
        # Determine the values to use
        fvals = self._process_values(values)

        # Prepare the data arrays I/O
        out = core.zeros(len(data))
        data_arrs = tuple(data[n] for n in self.data_pars)

        # Call the real function
        self.__function(out, *data_arrs, *fvals)

        # Calculate the normalization
        if normalized:
            nr = self._process_norm_range(norm_range)
            n = self.__norm(*fvals, *nr)
            return out / n
        else:
            return out

    def norm( self, values = None, norm_range = parameters.FULL ):
        '''
        '''
        fvals = self._process_values(values)
        nr = self._process_norm_range(norm_range)
        return self.__norm(*fvals, *nr)


class AddPDFs(PDF):

    def __init__( self, name, pdfs, yields ):
        '''
        '''
        self.__pdfs = parameters.Registry(*pdfs)
        self.__yields = parameters.Registry(*yields)

        assert len(self.__pdfs) - len(self.__yields) in (0, 1)

        data_pars = parameters.Registry()
        for p in pdfs:
            data_pars.update(p.data_pars)

        super(AddPDFs, self).__init__(name, data_pars.to_list(), yields)

    def __call__( self, data, values = None, norm_range = parameters.FULL, normalized = True ):

        yields = [y.value for y in self.__yields.values()]
        if not self.extended:
            yields.append(1. - sum(yields))

        out = core.zeros(len(data))
        for y, pdf in zip(yields, self.__pdfs.values()):
            out += y * pdf(data, values, norm_range, normalized=True)

        if self.extended and normalized:
            return out / sum(yields)
        else:
            return out

    @property
    def all_args( self ):
        args = self.args
        for p in self.__pdfs.values():
            args.update(p.all_args)
        return args

    def component( self, name ):
        for pdf in self.__pdfs.values():
            if pdf.name == name:
                return pdf
        raise LookupError(f'No PDF with name "{name}" hass been found')

    @property
    def extended( self ):
        return len(self.__pdfs) == len(self.__yields)

    def norm( self, values = None, norm_range = parameters.FULL ):
        '''
        '''
        if self.extended:
            # An extended PDF has a normalization equal to the sum of yields
            return sum(y.value for y in self.__yields.values())
        else:
            # A non-extended PDF is always normalized
            return 1.

    @property
    def pdfs( self ):
        return self.__pdfs

    @classmethod
    def two_components( cls, name, first, second, yf, ys = None ):
        return cls(name, [first, second], [yf] if ys is None else [yf, ys])

    @property
    def yields( self ):
        return self.__yields


class EvaluatorProxy(object):

    def __init__( self, fcn, pdf, data, norm_range = parameters.FULL ):

        super(EvaluatorProxy, self).__init__()

        self.__fcn  = fcn
        self.__pdf  = pdf
        self.__data = data
        self.__norm_range = norm_range

    def __call__( self, *values ):
        '''
        Values must be provided sorted as :method:`PDF.args`.
        '''
        r = parameters.Registry()
        for i, n in enumerate(self.__pdf.all_args):
            r[n] = values[i]

        return self.__fcn(self.__pdf, self.__data, r, self.__norm_range)
