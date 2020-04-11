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

    def __init__(self, data, pars, weights=None, copy=True, convert=True):
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
        '''
        self.__data = aop.data_array(data, copy=copy, convert=convert)
        self.__data_pars = parameters.Registry(pars)
        self.__weights = weights if weights is None else aop.data_array(
            weights, copy=copy, convert=convert)

    def __getitem__(self, var):
        '''
        Get the array of data for the given parameter.

        :returns: data array.
        :rtype: numpy.ndarray
        '''
        i = self.__data_pars.index(var)
        l = len(self)
        return self.__data[i * l:(i + 1) * l]

    def __len__(self):
        '''
        Get the size of the sample.

        :returns: size of the sample.
        :rtype: int
        '''
        if self.__data is not None:
            return len(self.__data) // len(self.__data_pars)
        else:
            return 0

    @property
    def data_pars(self):
        '''
        Data parameters associated to this sample.
        '''
        return self.__data_pars

    @property
    def ndim(self):
        '''
        Number of dimensions.
        '''
        return len(self.__data_pars)

    @property
    def values(self):
        '''
        Values of the data set.
        '''
        return self.__data

    @property
    def weights(self):
        '''
        Weights of the sample.
        '''
        return self.__weights

    @weights.setter
    def weights(self, weights):
        '''
        Set the weights of the sample.
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
        return cls(arr, parameters.Registry([data_par]), weights, copy=copy, convert=convert)

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
        arrs = []
        for p in data_pars:
            if p.name not in arr.dtype.names:
                raise RuntimeError(
                    f'No data for parameter "{p.name}" has been found in the input array')
            arrs.append(arr[p.name])
        data = np.concatenate(arrs)
        return cls(data, data_pars, weights=weights)

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
        :raises RuntimeError: If the samples have different data parameters or \
        if one has weights and the other has not.
        '''
        if len(self.__data_pars) != len(other.data_pars) or not all(p in other.data_pars for p in self.__data_pars):
            raise RuntimeError(
                'Attempt to merge samples with different data parameters')

        # None is set when the data set is the result of an operation involving "slice_from_boolean"
        if self.__data is None:
            return other
        if other.values is None:
            return self

        data = aop.concatenate(
            tuple(s[p.name] for p in self.__data_pars for s in (self, other)))

        if self.weights is not None:
            if other.weights is None:
                raise RuntimeError(
                    'Attempt to merge samples with and without weihts')
            weights = aop.concatenate((self.__weights, other.weights))
        else:
            weights = None

        if inplace:
            self.__weights = weights
            self.__data = data
            return self
        else:
            return self.__class__(data, self.__data_pars, weights)

    def make_binned(self, bins=100):
        '''
        Make a binned version of this sample.

        :param bins: number of bins per data parameter.
        :type bins: int or tuple(int, ...)
        :returns: binned data sample.
        :rtype: BinnedDataSet
        '''
        bins = np.array(bins, dtype=types.cpu_int)
        if bins.ndim > 0:
            edges = aop.concatenate(
                tuple(aop.linspace(*p.bounds, b + 1) for p, b in zip(self.__data_pars, bins)))
            gaps = bins
        else:
            edges = aop.concatenate(
                tuple(aop.linspace(*p.bounds, bins + 1) for p in self.__data_pars))
            gaps = np.full(len(self.__data_pars), bins)

        gaps[1:] = np.cumsum(gaps[:-1], dtype=types.cpu_int)
        gaps[0] = 1

        indices = edges_indices(gaps, edges)

        values = aop.sum_inside(
            indices, gaps, self.__data, edges, self.__weights)

        return BinnedDataSet(edges, gaps, self.__data_pars, values, copy=False, convert=False)

    @classmethod
    def merge(cls, samples, maximum=None, shuffle=False):
        '''
        Merge many samples into one. If "maximum" is specified, then the last elements will
        be dropped. If "shuffle" is specified, then it the result is shuffled before the "maximum"
        condition is applied, if needed.

        :param samples: samples to merge.
        :type samples: tuple(DataSet)
        :param maximum: maximum number of entries for the final sample.
        :type maximum: int
        :param shuffle: whether to shuffle the sample.
        :type shuffle: bool
        :returns: merged sample.
        :rtype: DataSet
        '''
        ns = np.sum(np.fromiter(map(len, samples), dtype=types.cpu_int))
        if maximum is not None and maximum > ns:
            logger.warning(
                'Specified a maximum length that exceeds the sum of lengths of the samples to merge; set to the latter')
            maximum = None

        data_pars = samples[0].data_pars

        for s in samples[1:]:
            if len(data_pars) != len(s.data_pars) or not all(p in s.data_pars for p in data_pars):
                raise RuntimeError(
                    'Attempt to merge samples with different data parameters')

        mw = (samples[0].weights is None)
        if any(map(lambda s: (s.weights is None) != mw, samples[1:])):
            raise RuntimeError(
                'Attempt to merge samples with and without weihts')

        data = aop.concatenate(
            tuple(s[p.name] for p in data_pars for s in filter(lambda s: s.values is not None, samples)))

        if not mw:
            weights = aop.concatenate(tuple(s.weights for s in filter(
                lambda s: s.weights is not None, samples)))
        else:
            weights = None

        ndim = samples[0].ndim

        # Shuffle the data
        if shuffle:
            sidx = aop.shuffling_index(len(data))
            data = aop.slice_from_integer(data, sidx, dim=ndim)
            if weights is not None:
                weights = aop.slice_from_integer(weights, sidx, sim=ndim)

        if maximum is not None:

            l = len(data) // ndim

            data = aop.keep_to_limit(maximum, ndim, l, data)

            if weights is not None:
                weights = weights[:maximum]

        result = cls(data, data_pars, weights, copy=False, convert=False)

        return result

    def subset(self, arg, copy=True, rescale_weights=False):
        '''
        Get a subset of this data set.

        :param cond: condition to apply to the data arrays to build the new data set.
        :type cond: numpy.ndarray or slice
        :param range: range to consider for the subset.
        :type range: str
        :param copy: whether to copy the arrays.
        :type copy: bool
        :param rescale_weights: if set to True, the weights are rescaled, so the FCN makes sense.
        :type rescale_weights: bool
        :returns: new data set.
        :rtype: DataSet
        '''
        if np.array(arg).dtype.kind == np.dtype(str).kind:
            use_range = True
        else:
            use_range = False

        if use_range:
            cond = aop.zeros(len(self), dtype=types.cpu_bool)
        elif arg is None:
            cond = aop.ones(len(self), dtype=types.cpu_bool)
        else:
            cond = arg

        if use_range:

            bounds = parameters.bounds_for_range(self.data_pars, arg)

            if len(bounds) == 1:
                cond = aop.logical_or(
                    cond, aop.is_inside(self.__data, *bounds[0]))
            else:
                for lb, ub in bounds:
                    c = aop.is_inside(self.__data, lb, ub)
                    cond = aop.logical_or(cond, c)

        if cond.dtype == types.cpu_int:
            data = aop.slice_from_integer(self.__data, cond, dim=self.ndim)
        else:
            data = aop.slice_from_boolean(self.__data, cond, dim=self.ndim)

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
        l = len(self)

        out = np.zeros(l, dtype=[(p.name, types.cpu_type)
                                 for p in self.__data_pars])
        data = aop.extract_ndarray(self.__data)
        for i, p in self.__data_pars:
            out[p.name] = data[i * l:(i + 1) * l]
        return out


class BinnedDataSet(object):

    def __init__(self, edges, gaps, pars, values, copy=True, convert=True):
        '''
        A binned data set.

        :param edges: centers of the bins.
        :type edges: dict
        :param gaps: gaps between edges belonging to different parameters.
        :type gaps: numpy.ndarray
        :param data_pars: data parameters.
        :type data_pars: Registry(Parameter)
        :param values: values of the data for each center.
        :type values: numpy.ndarray
        '''
        super(BinnedDataSet, self).__init__()

        self.__edges = aop.data_array(edges, copy=copy, convert=convert)
        self.__gaps = np.array(gaps, dtype=types.cpu_int)
        self.__data_pars = parameters.Registry(pars)
        self.__values = aop.data_array(
            values, copy=copy, convert=convert)

    def __getitem__(self, var):
        '''
        Get the centers of the bins for the given parameter.

        :returns: centers of the bins.
        :rtype: numpy.ndarray or reikna.cluda.Array
        '''
        i = self.__data_pars.index(var)
        e = self.edges_indices
        return self.__edges[e[i]:e[i + 1]]

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
        # The gaps refer to the bins, not to the edges
        g = self.edges_indices

        bounds = np.empty(2 * len(self.__data_pars), dtype=types.cpu_type)
        bounds[0::2] = [aop.extract_ndarray(self.__edges[i]) for i in g[:-1]]
        bounds[1::2] = [aop.extract_ndarray(
            self.__edges[i - 1]) for i in g[1:]]

        return bounds

    @property
    def data_pars(self):
        '''
        Get the data parameters of this sample.
        '''
        return self.__data_pars

    @property
    def edges(self):
        '''
        Edges of the histogram.
        '''
        return self.__edges

    @property
    def edges_indices(self):
        '''
        Indices to access the edges.
        '''
        return edges_indices(self.__gaps, self.__edges)

    @property
    def gaps(self):
        '''
        Gaps among the different edges.
        '''
        return self.__gaps

    @property
    def ndim(self):
        '''
        Number of dimensions.
        '''
        return len(self.__data_pars)

    @property
    def values(self):
        '''
        Get the values of the data set.

        :returns: values of the data set.
        :rtype: numpy.ndarray
        '''
        return self.__values

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
        return cls(edges, [1], parameters.Registry([data_par]), values, copy=copy, convert=convert)


def edges_indices(gaps, edges):
    '''
    Calculate the indices to access the first element and
    that following to the last of a list of edges.

    :param gaps: gaps used to address the correct edges from \
    a common array.
    :type gaps: numpy.ndarray
    :param edges: common array of edges.
    :type edges: numpy.ndarray
    :returns: array of indices.
    :rtype: numpy.ndarray
    '''
    l = len(gaps)
    g = np.empty(l + 1, dtype=types.cpu_int)
    g[1:-1] = gaps[1:] + np.arange(1, l)
    g[0], g[-1] = 0, len(edges)
    return g


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
    if bounds.shape == (2,):
        data = aop.linspace(*bounds, size)
    else:
        values = []
        lb, ub = bounds
        for p, l, u in zip(data_pars, lb, ub):
            values.append(aop.linspace(l, u, size))

        data = aop.concatenate(aop.meshgrid(*values))

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
    data = aop.concatenate(tuple(aop.random_uniform(l, u, size)
                                 for l, u in zip(*bounds)))
    return DataSet(data, data_pars, copy=False, convert=False)
