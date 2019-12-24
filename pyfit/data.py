'''
Functions and classes to handle sets of data.
'''
from . import core
from . import parameters
from . import types

import logging
import numpy as np

__all__ = ['DataSet', 'BinnedDataSet']

logger = logging.getLogger(__name__)


class DataSet(object):

    def __init__( self, data, pars, weights = None ):
        '''
        '''
        self.__data = {p.name: core.array(data[p.name]) for p in pars.values()}
        self.__data_pars = pars
        self.__weights = weights if weights is None else core.array(weights)

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

    def subset( self, cond ):
        '''
        '''
        if self.weights is None:
            weights = None
        else:
            weights = self.weights[cond]
        return self.__class__({p.name: self[p.name][cond] for p in self.data_pars.values()}, self.data_pars, weights)

    def add( self, other, inplace = False ):
        '''
        '''
        dct = {}
        for n, p in self.__data.items():
            dct[n] = core.concatenate(p, other[n])

        if self.weights is not None:
            if other.weights is None:
                raise RuntimeError('Attempt to merge samples with and without weihts')
            weights = core.concatenate(self.weights, other.weights)
        else:
            weights = None

        if inplace:
            for n in self.__data:
                self.__data[n] = dct[n]
            return self
        else:
            return self.__class__(dct, self.__data_pars, weights)

    @property
    def data_pars( self ):
        return self.__data_pars
    

    @property
    def weights( self ):
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


def evaluation_grid( data_pars, size ):
    '''
    '''
    values = []
    for p in data_pars.values():
        values.append(np.linspace(*p.bounds, size))

    data = {p.name: a.flatten() for p, a in zip(data_pars.values(), np.meshgrid(*values))}

    return DataSet(data, data_pars)


def uniform_sample( data_pars, size ):
    '''
    '''
    values = []
    for p in data_pars.values():
        values.append(np.random.uniform(*p.bounds, size)) # Need a core.meshgrid

    data = {p.name: a.flatten() for p, a in zip(data_pars.values(), np.meshgrid(*values))}

    return DataSet(data, data_pars)
