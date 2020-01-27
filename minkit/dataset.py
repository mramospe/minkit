'''
Functions and classes to handle sets of data.
'''
from .core import aop
from . import parameters
from .operations import types

import logging
import numpy as np

__all__ = ['evaluation_grid', 'DataSet', 'BinnedDataSet']

# Names of different data types
BINNED = 'binned'
UNBINNED = 'unbinned'

logger = logging.getLogger(__name__)


class DataSet(object):
    '''
    Definition of a set of data.
    '''

    def __init__(self, data, pars, weights=None, copy=True, convert=True, trim=False):
        '''
        Definition of an unbinned data set to evaluate PDFs.

        :param data: data to load.
        :type data: dict, numpy.ndarray
        :param pars: data parameters.
        :type pars: Registry(Parameter)
        :param weights: possible set of weights.
        :type weights: numpy.ndarray
        :param copy: whether to do copies of arrays.
        :type copy: bool
        :param convert: in case of working in GPUs, the input arrays \
        are assumed to be on the host and are copied to the device.
        :type convert: bool
        :param trim: whether to check the bounds of the data parameters \
        and remove the data points out of them.
        :type trim: bool
        '''
        self.__data = {p.name: aop.data_array(
            data[p.name], copy=copy, convert=convert) for p in pars}
        self.__data_pars = pars
        self.__weights = weights if weights is None else aop.data_array(
            weights, copy=copy, convert=convert)

        if trim:
            # Remove the data that lies outside the parameter bounds
            valid = None
            for p in pars:
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
            diff = len(valid) - aop.count_nonzero(valid)
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
            dct[n] = aop.concatenate((p, other[n]))

        if self.weights is not None:
            if other.weights is None:
                raise RuntimeError(
                    'Attempt to merge samples with and without weihts')
            weights = aop.concatenate((self.weights, other.weights))
        else:
            weights = None

        if inplace:
            self.__weights = weights
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
        :rtype: Registry(Parameter)
        '''
        return self.__data_pars

    @property
    def weights(self):
        '''
        Get the weights of the sample.

        :returns: weights of the sample.
        :rtype: numpy.ndarray
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
        self.__weights = aop.data_array(weights)

    @classmethod
    def from_array(cls, arr, data_par, weights=None, copy=True, convert=True):
        '''
        Build the class from a single array.

        :param arr: array of data.
        :type arr: numpy.ndarray
        :param data_pars: data parameters.
        :type data_pars: Registry(Parameter)
        :param weights: possible weights to use.
        :type weights: numpy.ndarray
        '''
        return cls({data_par.name: arr}, parameters.Registry([data_par]), weights, copy=copy, convert=convert)

    @classmethod
    def from_records(cls, arr, data_pars, weights=None):
        '''
        Build the class from a :class:`numpy.ndarray` object.

        :param arr: array of data.
        :type arr: numpy.ndarray
        :param data_pars: data parameters.
        :type data_pars: Registry(Parameter)
        :param weights: possible weights to use.
        :type weights: numpy.ndarray
        '''
        dct = {}
        for p in data_pars:
            if p not in arr.dtype.names:
                raise RuntimeError(
                    f'No data for parameter "{p.name}" has been found in the input array')
            dct[p] = arr[p]
        return cls(dct, data_pars, weights=weights)

    def make_binned(self, bins=100):
        '''
        Make a binned version of this sample.

        :param bins: number of bins per data parameter.
        :type bins: int
        :returns: binned data sample.
        :rtype: BinnedDataSet
        '''
        edges = {p.name: aop.linspace(*p.bounds, bins + 1)
                 for p in self.data_pars}

        centers = [self[p.name] for p in self.data_pars]

        e = [edges[p.name] for p in self.data_pars]

        values = aop.sum_inside(centers, e)

        return BinnedDataSet(edges, self.data_pars, values, copy=False, convert=False)

    @classmethod
    def merge(cls, samples, maximum=None, shuffle=False, trim=False):
        '''
        Merge many samples into one. If "maximum" is specified, then the last elements will
        be dropped. If "shuffle" is specified, then it the result is shuffled before the "maximum"
        condition is applied, if needed.

        :param samples: samples to merge.
        :type samples: tuple(DataSet)
        :param maximum: maximum number of entries for the final sample.
        :type maximum: int
        :param shuffle: whether to shuffle the sample before trimming it.
        :type shuffle: bool
        :param trim: whether to check the bounds of the data parameters \
        and remove the data points out of them.
        :type trim: bool
        :returns: merged sample.
        :rtype: DataSet
        '''
        ns = np.sum(np.fromiter(map(len, samples), dtype=types.cpu_int))
        if maximum is None:
            maximum = ns
        else:
            if maximum > ns:
                logger.warning(
                    'Specified a maximum length that exceeds the sum of lengths of the samples to merge; set to the latter')
                maximum = ns

        data_pars = samples[0].data_pars

        data = {p.name: aop.concatenate(
            tuple(s[p.name] for s in samples), maximum=maximum) for p in data_pars}

        if samples[0].weights is not None:
            weights = aop.concatenate(tuple(s.weights for s in samples))
        else:
            weights = None

        result = cls(data, data_pars, weights, copy=False,
                     convert=False, trim=trim)

        if shuffle:
            result.shuffle(inplace=True)

        return result

    def shuffle(self, inplace=False):
        '''
        Shuffle the data set, reordering the data with random numbers.

        :param inplace: if set to True, the current data set is modified. Otherwise a copy is returned.
        :type inplace: bool
        :returns: the shuffled data set.
        :rtype: DataSet
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
                self.__data[p.name], index) for p in self.data_pars}

            if self.__weights is None:
                weights = None
            else:
                weights = self.__weights[index]

            return self.__class__(data, self.data_pars, weights, copy=False, convert=False)

    def subset(self, cond=None, range=None, copy=True, trim=False, rescale_weights=False):
        '''
        Get a subset of this data set.

        :param cond: condition to apply to the data arrays to build the new data set.
        :type cond: numpy.ndarray or slice
        :param range: range to consider for the subset.
        :type range: str
        :param copy: whether to copy the arrays.
        :type copy: bool
        :param trim: whether to check the bounds of the data parameters \
        and remove the data points out of them.
        :type trim: bool
        :param rescale_weights: if set to True, the weights are rescaled, so the FCN makes sense.
        :type rescale_weights: bool
        :returns: new data set.
        :rtype: DataSet
        '''
        if cond is None:
            cond = aop.ones(len(self), dtype=types.cpu_bool)

        if range is not None:

            if cond.dtype == types.cpu_int:
                raise NotImplementedError(
                    'Creating a subset by index and by range at the same time is not supported')

            for p in self.data_pars:
                n = p.name
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
                self[p.name], cond) for p in self.data_pars}
        else:
            data = {p.name: aop.slice_from_boolean(
                    self[p.name], cond) for p in self.data_pars}

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


class BinnedDataSet(object):
    '''
    A binned data set.
    '''

    def __init__(self, edges, data_pars, values, copy=True, convert=True):
        '''
        A binned data set.

        :param edges: centers of the bins.
        :type edges: dict
        :param data_pars: data parameters.
        :type data_pars: Registry(Parameter)
        :param values: values of the data for each center.
        :type values: numpy.ndarray
        '''
        super(BinnedDataSet, self).__init__()

        self.__data_pars = data_pars

        self.__edges = {name: aop.data_array(arr, copy=copy, convert=convert)
                        for name, arr in dict(edges).items()}

        self.__values = aop.data_array(
            values, copy=copy, convert=convert)

    def __getitem__(self, var):
        '''
        Get the centers of the bins for the given parameter.

        :returns: centers of the bins.
        :rtype: numpy.ndarray
        '''
        return self.__edges[var]

    def __len__(self):
        '''
        Get the size of the sample.

        :returns: size of the sample.
        :rtype: int
        '''
        return len(self.__values)

    @property
    def bounds(self):
        '''
        Bounds of each data parameter.
        '''
        bounds = np.empty(2*len(self.__data_pars), dtype=types.cpu_type)
        bounds[0::2] = [aop.extract_ndarray(
            self.__edges[p.name][0]) for p in self.__data_pars]
        bounds[1::2] = [aop.extract_ndarray(
            self.__edges[p.name][-1]) for p in self.__data_pars]
        return bounds

    @property
    def data_pars(self):
        '''
        Get the data parameters of this sample.

        :returns: data parameters of this sample.
        :rtype: Registry(Parameter)
        '''
        return self.__data_pars

    @classmethod
    def from_array(cls, edges, data_par, values, copy=True, convert=True):
        '''
        Build the class from the array of edges and values.

        :param edges: edges of the bins.
        :type edges: numpy.ndarray
        :param data_par: data parameter.
        :type data_par: Parameter
        :param values: values at each bin.
        :type values: numpy.ndarray
        :returns: binned data set.
        :rtype: BinnedDataSet
        '''
        return cls({data_par.name: edges}, parameters.Registry([data_par]), values, copy=copy, convert=convert)

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
    :type data_pars: list(Parameter)
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
        data = {data_pars[0].name: aop.linspace(*bounds, size)}
    else:
        values = []
        for p, vmin, vmax in zip(data_pars, bounds[0::2], bounds[1::2]):
            values.append(aop.linspace(vmin, vmax, size))
        data = {p.name: a for p, a in zip(
            data_pars, aop.meshgrid(*values))}

    return DataSet(data, data_pars, copy=False, convert=False)


def uniform_sample(data_pars, bounds, size):
    '''
    Generate a sample following an uniform distribution in all data parameters.

    :param data_pars: data parameters.
    :type data_pars: Registry(Parameter)
    :param size: number of entries in the output sample.
    :type size: int
    :param bounds: bounds where to create the sample.
    :type bounds: tuple(float, float)
    :returns: uniform sample.
    :rtype: DataSet
    '''
    data = {p.name: aop.random_uniform(l, h, size)
            for p, l, h in zip(data_pars, bounds[0::2], bounds[1::2])}
    return DataSet(data, data_pars, copy=False, convert=False)
