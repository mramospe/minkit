'''
Base classes to define PDF types.
'''

from . import core
from . import parameters

import itertools

__all__ = []


class PDF(object):

    def __init__( self, name, function, normalization, data_pars, args_pars ):
        '''
        '''
        self.name        = name
        self.__function  = function
        self.__norm      = normalization
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

    @property
    def args( self ):
        return self.__arg_pars

    @property
    def data_pars( self ):
        return self.__data_pars

    def norm( self, values = None, norm_range = parameters.FULL ):
        '''
        '''
        fvals = self._process_values(values)
        nr = self._process_norm_range(norm_range)
        return self.__norm(*fvals, *nr)


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
        for i, n in enumerate(self.__pdf.args):
            r[n] = values[i]

        return self.__fcn(self.__pdf, self.__data, r, self.__norm_range)
