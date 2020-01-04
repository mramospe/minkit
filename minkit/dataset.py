'''
Functions and classes to handle sets of data.
'''
from .core import aop
from . import parameters
from .operations import types

import logging
import numpy as np

__all__ = ['DataSet', 'BinnedDataSet']

# Names of different data types
BINNED = 'binned'
UNBINNED = 'unbinned'

logger = logging.getLogger(__name__)


class DataSet(object):
    '''
    Definition of a set of data.
    '''

    def __init__(self, data, pars, weights=None, copy=True, convert=True):
        '''
        Build the class from a data sample which can be indexed as a dictionary, the data parameters and a possible set of weights.

        :param data: data to load.
        :type data: dict, numpy.ndarray
        :param pars: data parameters.
        :type pars: Registry(str, Parameter)
        :param weights: possible set of weights.
        :type weights: numpy.ndarray or None
        '''
        self.__data = {p.name: aop.array(
            data[p.name], copy=copy, convert=convert) for p in pars.values()}
        self.__data_pars = pars
        self.__weights = weights if weights is None else aop.array(
            weights, copy=copy, convert=convert)

        valid = None
        for p in pars.values():
            if p.bounds is None:
                raise ValueError(
                    f'Must define the bounds for data parameter "{p.name}"')

            iv = aop.logical_and(
                aop.geq(self.__data[p.name], p.bounds[0]),
                aop.leq(self.__data[p.name], p.bounds[1]))

            if valid is None:
                valid = iv
            else:
                valid *= iv

        # Remove out of range points, if necessary
        diff = len(valid) - aop.sum(valid)
        if diff != 0:
            logger.info(f'Removing "{diff}" out of range points')

        if self.__weights is not None:
            self.__weights = aop.slice_from_boolean(
                self.__weights, valid)

        for name, array in self.__data.items():
            self.__data[name] = aop.slice_from_boolean(
                array, valid)

    def __getitem__(self, var):
        '''
        Get the array of data for the given parameter.

        :returns: data array.
        :rtype: numpy.ndarray
        '''
        return self.__data[var]

    def __len__(self):
        '''
        Get the size of the sample.

        :returns: size of the sample.
        :rtype: int
        '''
        return len(self.__data[tuple(self.__data.keys())[0]])

    def add(self, other, inplace=False):
        '''
        Add a new data set to this one.
        If "inplace" is set to True, then the data is directly added to
        the existing class.

        :param other: data set to add.
        :type other: DataSet
        :param inplace: whether to do the operation in-place or not.
        :type inplace: bool
        :returns: new data set if "inplace" is set to False or otherwise this class, \
        in both cases with the data arrays concatenated.
        :rtype: DataSet
        '''
        dct = {}
        for n, p in self.__data.items():
            dct[n] = aop.concatenate(p, other[n])

        if self.weights is not None:
            if other.weights is None:
                raise RuntimeError(
                    'Attempt to merge samples with and without weihts')
            weights = aop.concatenate(self.weights, other.weights)
        else:
            weights = None

        if inplace:
            for n in self.__data:
                self.__data[n] = dct[n]
            return self
        else:
            return self.__class__(dct, self.__data_pars, weights)

    @property
    def data_pars(self):
        '''
        Get the data parameters associated to this sample.

        :returns: data parameters associated to this sample.
        :rtype: Registry(str, Parameter)
        '''
        return self.__data_pars

    @classmethod
    def from_array(cls, arr, data_par, weights=None, copy=True, convert=True):
        '''
        Build the class from a single array.

        :param arr: array of data.
        :type arr: numpy.ndarray
        :param data_pars: data parameters.
        :type data_pars: Registry(name, Parameter)
        :param weights: possible weights to use.
        :type weights: numpy.ndarray or None
        '''
        return cls({data_par.name: arr}, parameters.Registry([(data_par.name, data_par)]), weights, copy=copy, convert=convert)

    @classmethod
    def from_records(cls, arr, data_pars, weights=None):
        '''
        Build the class from a :class:`numpy.ndarray` object.

        :param arr: array of data.
        :type arr: numpy.ndarray
        :param data_pars: data parameters.
        :type data_pars: Registry(name, Parameter)
        :param weights: possible weights to use.
        :type weights: numpy.ndarray or None
        '''
        dct = {}
        for p in data_pars:
            if p not in arr.dtype.names:
                raise RuntimeError(
                    f'No data for parameter "{p.name}" has been found in the input array')
            dct[p] = arr[p]
        return cls(dct, data_pars, weights=weights)

    def shuffle(self, inplace=False):
        '''
        '''
        index = aop.shuffling_index(len(self))

        if inplace:

            for k, v in self.__data.items():
                self.__data[k] = aop.slice_from_integer(v, index)
            if self.__weights is not None:
                self.__weights = aop.slice_from_integer(
                    self.__weights, index)

            return self
        else:
            data = {n: aop.slice_from_integer(
                self.__data[n], index) for n in self.data_pars}

            if self.__weights is None:
                weights = None
            else:
                weights = self.__weights[index]

            return self.__class__(data, self.data_pars, weights, copy=False, convert=False)

    def subset(self, cond=None, range=None, copy=True, rescale_weights=False):
        '''
        Get a subset of this data set.

        :param cond: condition to apply to the data arrays to build the new data set.
        :type cond: numpy.ndarray or slice
        :returns: new data set.
        :rtype: DataSet
        '''
        if cond is None:
            cond = aop.ones(len(self), dtype=types.cpu_bool)

        if range is not None:
            for n, p in self.data_pars.items():
                r = p.get_range(range)
                a = self[n]
                if r.disjoint:
                    c = aop.zeros(len(cond), dtype=types.cpu_bool)
                    for vmin, vmax in r.bounds:
                        i = aop.logical_and(aop.geq(a, vmin),
                                            aop.leq(a, vmax))
                        c = aop.logical_or(c, i)
                    cond = aop.logical_and(cond, c)
                else:
                    i = aop.logical_and(
                        aop.geq(a, r.bounds[0]),
                        aop.leq(a, r.bounds[1]))
                    cond = aop.logical_and(cond, i)

        if cond.dtype == types.cpu_int:
            data = {p.name: aop.slice_from_integer(
                self[p.name], cond) for p in self.data_pars.values()}
        else:
            data = {p.name: aop.slice_from_boolean(
                self[p.name], cond) for p in self.data_pars.values()}

        if self.__weights is not None:

            if cond.dtype == types.cpu_int:
                weights = aop.slice_from_integer(
                    self.__weights, cond)
            else:
                weights = aop.slice_from_boolean(
                    self.__weights, cond)

            if rescale_weights:
                weights = weights * \
                    aop.sum(weights) / \
                    aop.sum(weights**2)
        else:
            weights = self.__weights

        return self.__class__(data, self.data_pars, weights, copy=copy, convert=False)

    def to_records():
        '''
        Convert this class into a :class:`numpy.ndarray` object.

        :returns: this object as a a :class:`numpy.ndarray` object.
        :rtype: numpy.ndarray
        '''
        out = np.zeros(len(self), dtype=[
                       (n, types.cpu_type) for n in self.__data])
        for n, v in self.__data.items():
            out[n] = aop.extract_ndarray(v)
        return out

    @property
    def weights(self):
        '''
        Get the weights of the sample.

        :returns: weights of the sample.
        :rtype: numpy.ndarray or None
        '''
        return self.__weights

    @weights.setter
    def weights(self, weights):
        '''
        Set the weights of the sample.

        :param weights: input weights.
        :type weights: numpy.ndarray
        '''
        if len(weights) != len(self):
            raise ValueError(
                'Length of the provided weights does not match that of the DataSet')
        self.__weights = aop.array(weights)


class BinnedDataSet(object):
    '''
    A binned data set.
    '''

    def __init__(self, centers, data_pars, values, copy=True, convert=True):
        '''
        A binned data set.

        :param centers: centers of the bins.
        :type centers: numpy.ndarray
        :param data_pars: data parameters.
        :type data_pars: Registry(name, Parameter)
        :param values: values of the data for each center.
        :type values: numpy.ndarray
        '''
        super(BinnedDataSet, self).__init__()

        self.__centers = {name: aop.array(arr, copy=copy, convert=convert)
                          for name, arr in dict(centers).items()}
        self.__data_pars = data_pars
        self.__values = aop.array(
            values, copy=copy, convert=convert)

        assert centers.keys() == self.__centers.keys()

    def __getitem__(self, var):
        '''
        Get the centers of the bins for the given parameter.

        :returns: centers of the bins.
        :rtype: numpy.ndarray
        '''
        return self.__centers[var]

    def __len__(self):
        '''
        Get the size of the sample.

        :returns: size of the sample.
        :rtype: int
        '''
        return len(self.__centers[tuple(self.__centers.keys())[0]])

    @property
    def data_pars(self):
        '''
        Get the data parameters of this sample.

        :returns: data parameters of this sample.
        :rtype: Registry(str, Parameter)
        '''
        return self.__data_pars

    @classmethod
    def from_array(cls, centers, data_par, values, copy=True, convert=True):
        '''
        Build the class from the array of centers and values.

        :param centers: centers of the bins.
        :type centers: numpy.ndarray
        :param data_par: data parameter.
        :type data_par: Parameter
        :param values: values at each bin.
        :type values: numpy.ndarray
        :returns: binned data set.
        :rtype: BinnedDataSet
        '''
        return cls({data_par.name: centers}, parameters.Registry([(data_par.name, data_par)]), values, copy=copy, convert=convert)

    @property
    def values(self):
        '''
        Get the values of the data set.

        :returns: values of the data set.
        :rtype: numpy.ndarray
        '''
        return self.__values


def evaluation_grid(data_pars, bounds, size):
    '''
    Create a grid of points to evaluate a :class:`PDF` object.

    :param data_pars: data parameters.
    :type data_pars: Registry(str, Parameter)
    :param size: number of entries in the output sample per set of bounds. \
    This means that "size" entries will be generated for each pair of (min, max) \
    provided, that is, per data parameter.
    :type size: int
    :param bounds: bounds of the different data parameters. Even indices for \
    the lower bounds, and odd indices for the upper bounds.
    :type bounds: numpy.ndarray
    :returns: uniform sample.
    :rtype: DataSet
    '''
    bounds = np.array(bounds)

    if bounds.shape == (2,):
        assert len(data_pars) == 1
        data = {tuple(data_pars.values())[
            0].name: aop.linspace(*bounds, size)}
    else:
        values = []
        for p, vmin, vmax in zip(data_pars.values(), bounds[0::2], bounds[1::2]):
            values.append(aop.linspace(vmin, vmax, size))
        data = {p.name: a for p, a in zip(
            data_pars.values(), aop.meshgrid(*values))}

    return DataSet(data, data_pars, copy=False)


def uniform_sample(data_pars, bounds, size):
    '''
    Generate a sample following an uniform distribution in all data parameters.

    :param data_pars: data parameters.
    :type data_pars: Registry(name, Parameter)
    :param size: number of entries in the output sample.
    :type size: int
    :param bounds: bounds where to create the sample.
    :type bounds: tuple(float, float)
    :returns: uniform sample.
    :rtype: DataSet
    '''
    data = {n: aop.random_uniform(l, h, size)
            for n, l, h in zip(data_pars, bounds[0::2], bounds[1::2])}
    return DataSet(data, data_pars)
