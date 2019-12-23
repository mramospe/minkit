'''
Functions and classes to handle sets of data.
'''
from . import core
from . import parameters
from . import types

import logging

__all__ = ['DataSet', 'BinnedDataSet']

logger = logging.getLogger(__name__)


class DataSet(object):

    def __init__( self, data, pars, weights = None ):
        '''
        '''
        self.__data = {name: core.array(arr) for name, arr in dict(data).items()}

        self.__weights = weights if weights is None else core.array(weights)

        assert data.keys() == self.__data.keys()

        valid = None
        for p in pars.values():
            if p.bounds is None:
                raise ValueError(f'Must define the bounds for data parameter "{p.name}"')

            iv = (data[p.name] >= p.bounds[0])*(data[p.name] <= p.bounds[1])

            if valid is None:
                valid = iv
            else:
                valid *= iv

        # Remove out of range points, if necessary
        diff = len(valid) - core.sum(valid)
        if diff != 0:
            logger.info(f'Removing "{diff}" out of range points')

        if self.__weights is not None:
            self.__weights = self.__weights[valid]

        for name, array in self.__data.items():
            self.__data[name] = array[valid]

    @property
    def weights():
        return self.__weights

    @classmethod
    def from_array( cls, arr, data_par, weights = None ):
        '''
        '''
        return cls({data_par.name: arr}, parameters.Registry([(data_par.name, data_par)]), weights)

    @classmethod
    def from_records( cls, arr, data_pars, weights = None ):
        '''
        '''
        dct = {}
        for p in data_pars:
            if p not in arr.dtype.names:
                raise RuntimeError(f'No data for parameter "{p.name}" has been found in the input array')
            dct[p] = arr[p]
        return cls(dct, data_pars, weights=weights)

    def to_records():
        '''
        '''
        out = np.zeros(len(self), dtype=[(n, types.cpu_type) for n in self.__data])
        for n, v in self.__data.items():
            out[n] = core.extract_ndarray(v)
        return out

    def __getitem__( self, var ):
        return self.__data[var]

    def __len__( self ):
        return len(self.__data[tuple(self.__data.keys())[0]])


class BinnedDataSet(object):

    def __init__( self, centers, values ):
        '''
        '''
        super(BinnedDataSet, self).__init__()

        self.__centers = {name: core.array(arr) for name, arr in dict(centers).items()}
        self.__values = values

        assert centers.keys() == self.__centers.keys()

    @property
    def values( self ):
        return self.__values

    def __getitem__( self, var ):
        return self.__centers[var]

    def __len__( self ):
        return len(self.__centers[tuple(self.__centers.keys())[0]])
