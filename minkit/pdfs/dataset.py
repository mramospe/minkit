########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Functions and classes to handle sets of data.
'''
from ..base import core
from ..base import data_types
from ..base import exceptions
from ..base import parameters
from ..backends.arrays import darray
from ..backends.core import parse_backend

import logging
import numpy as np

__all__ = ['DataSet', 'BinnedDataSet']

# Names of different data types
BINNED = 'binned'
UNBINNED = 'unbinned'

logger = logging.getLogger(__name__)


class DataObject(object, metaclass=core.DocMeta):

    sample_type = None  # must be defined for each class

    def __init__(self, pars, backend):
        '''
        Abstract class for data objects.

        :param pars: data parameters.
        :type pars: Registry(Parameter)
        :param backend: backend where this object is built.
        :type backend: Backend
        '''
        self.__data_pars = pars
        self.__aop = backend.aop

        super().__init__()

    @property
    def aop(self):
        '''
        Object to do operations on arrays.

        :type: ArrayOperations
        '''
        return self.__aop

    @property
    def backend(self):
        '''
        Backend interface.

        :type: Backend
        '''
        return self.__aop.backend

    @property
    def data_pars(self):
        '''
        Data parameters associated to this sample.

        :type: Registry(Parameter)
        '''
        return self.__data_pars

    @property
    def ndim(self):
        '''
        Number of dimensions.

        :type: int
        '''
        return data_types.cpu_int(len(self.__data_pars))

    def to_backend(self, backend):
        '''
        Initialize this class in a different backend.

        :param backend: new backend.
        :type backend: Backend
        :returns: This class in the new backend.
        '''
        raise exceptions.MethodNotDefinedError(self.__class__, 'to_backend')


class DataSet(DataObject):

    sample_type = UNBINNED

    def __init__(self, data, pars, weights=None):
        '''
        Definition of an unbinned data set to evaluate PDFs.

        :param data: data in the provided backend.
        :type data: darray
        :param pars: data parameters.
        :type pars: Registry(Parameter)
        :param weights: possible set of weights.
        :type weights: darray
        '''
        super().__init__(pars, data.aop.backend)

        self.__data = data
        self.__weights = weights

    def __getitem__(self, var):
        '''
        Get the array of data for the given parameter.

        :returns: Data array.
        :rtype: numpy.ndarray
        '''
        return self.__data.take_column(self.data_pars.index(var))

    def __len__(self):
        '''
        Get the size of the sample.

        :returns: Size of the sample.
        :rtype: int
        '''
        if self.__data is not None:
            return len(self.__data)
        else:
            return data_types.cpu_int(0)

    @property
    def values(self):
        '''
        Values of the data set.

        :type: darray
        '''
        return self.__data

    @property
    def weights(self):
        '''
        Weights of the sample.

        :type: darray or None
        '''
        return self.__weights

    @classmethod
    def from_ndarray(cls, arr, data_par, weights=None, backend=None):
        '''
        Build the class from a single array.

        :param arr: array of data.
        :type arr: numpy.ndarray
        :param data_pars: data parameters.
        :type data_pars: Registry(Parameter)
        :param weights: possible weights to use.
        :type weights: numpy.ndarray
        '''
        aop = parse_backend(backend)

        data = darray.from_ndarray(arr, aop.backend)

        if weights is not None:
            weights = darray.from_ndarray(weights, aop.backend)

        if data.ndim > 1:
            return cls(data, parameters.Registry(data_par), weights)
        else:
            return cls(data, parameters.Registry([data_par]), weights)

    @classmethod
    def from_records(cls, arr, data_pars, weights=None, backend=None):
        '''
        Build the class from a :class:`numpy.ndarray` object.

        :param arr: array of data.
        :type arr: numpy.ndarray
        :param data_pars: data parameters.
        :type data_pars: Registry(Parameter)
        :param weights: possible weights to use.
        :type weights: numpy.ndarray
        '''
        aop = parse_backend(backend)
        arrs = np.empty((len(arr), len(data_pars)))
        for i, p in enumerate(data_pars):
            if p.name not in arr.dtype.names:
                raise RuntimeError(
                    f'No data for parameter "{p.name}" has been found in the input array')
            arrs[:, i] = arr[p.name]
        data = darray.from_ndarray(arrs, aop.backend)
        if weights is not None:
            weights = darray.from_ndarray(weights, aop.backend)
        return cls(data, data_pars, weights)

    def make_binned(self, bins=100):
        '''
        Make a binned version of this sample.

        :param bins: number of bins per data parameter.
        :type bins: int or tuple(int, ...)
        :returns: Binned data sample.
        :rtype: BinnedDataSet
        '''
        bins = data_types.array_int(bins)
        if bins.ndim > 0:
            edges = self.aop.concatenate(
                tuple(self.aop.linspace(*p.bounds, b + 1) for p, b in zip(self.data_pars, bins)))
            gaps = bins
        else:
            edges = self.aop.concatenate(
                tuple(self.aop.linspace(*p.bounds, bins + 1) for p in self.data_pars))
            gaps = data_types.full_int(len(self.data_pars), bins)

        gaps = np.cumprod(gaps) // gaps[0]

        indices = edges_indices(gaps, edges)

        values = self.aop.sum_inside(
            indices, gaps, self.__data, edges, self.__weights)

        return BinnedDataSet(edges, gaps, self.data_pars, values)

    @classmethod
    def merge(cls, samples, maximum=None):
        '''
        Merge many samples into one. If *maximum* is specified, then the last elements will
        be dropped.

        :param samples: samples to merge.
        :type samples: tuple(DataSet)
        :param maximum: maximum number of entries for the final sample.
        :type maximum: int
        :returns: Merged sample.
        :rtype: DataSet

        ... warning:: If *maximum* is specified, the last elements corresponding to the
            last samples might be dropped.
        '''
        ns = np.sum(data_types.fromiter_int(map(len, samples)))
        if maximum is not None and maximum > ns:
            logger.warning(
                'Specified a maximum length that exceeds the sum of lengths of the samples to merge; set to the latter')
            maximum = None

        data_pars = samples[0].data_pars
        aop = samples[0].aop

        for s in samples[1:]:
            if len(data_pars) != len(s.data_pars) or not all(p in s.data_pars for p in data_pars):
                raise RuntimeError(
                    'Attempt to merge samples with different data parameters')

        mw = (samples[0].weights is None)
        if any(map(lambda s: (s.weights is None) != mw, samples[1:])):
            raise RuntimeError(
                'Attempt to merge samples with and without weihts')

        data = aop.concatenate(
            tuple(s.values for s in filter(lambda s: s.values is not None, samples)))

        if not mw:
            weights = aop.concatenate(tuple(s.weights for s in filter(
                lambda s: s.weights is not None, samples)))
        else:
            weights = None

        if maximum is not None:

            data = aop.restrict_data_size(maximum, data)

            if weights is not None:
                weights = weights[:maximum]

        return cls(data, data_pars, weights)

    def subset(self, arg, rescale_weights=False):
        r'''
        Get a subset of this data set. If *arg* is a string, it will be
        considered as a range. In case it is a :class:`barray`, then it is
        considered to be a mask array. If *rescale_weights* is set to True, then
        the weights are rescaled so their statistical weight in minimization
        processes is proportional to the event weights:

        .. math::
           \omega^\prime_i = \omega_i \times \frac{\sum_{j = 0}^n \omega_j}{\sum_{j = 0}^n \omega_j^2}

        :param arg: argument to create the subset.
        :type arg: str or barray
        :param rescale_weights: whether to rescale the sample weights.
        :type rescale_weights: bool
        :returns: New data set.
        :rtype: DataSet
        '''
        if np.asarray(arg).dtype.kind == np.dtype(str).kind:
            use_range = True
        else:
            use_range = False

        if use_range:
            cond = self.aop.bzeros(len(self))
        elif arg is None:
            cond = self.aop.bones(len(self))
        else:
            cond = arg

        if use_range:

            bounds = parameters.bounds_for_range(self.data_pars, arg)

            if len(bounds) == 1:
                cond |= self.aop.is_inside(self.__data, *bounds[0])
            else:
                for lb, ub in bounds:
                    cond |= self.aop.is_inside(self.__data, lb, ub)

        data = self.__data.slice(cond)

        if self.__weights is not None:
            weights = self.__weights.slice(cond)
        else:
            weights = self.__weights

        if weights is not None and rescale_weights:
            weights = rescale_weights_sw2(weights)

        return self.__class__(data, self.data_pars, weights)

    def to_backend(self, backend):

        data = self.__data.to_backend(backend)

        if self.__weights is not None:
            weights = self.__weights.to_backend(backend)
        else:
            weights = None

        return self.__class__(data, self.data_pars, weights)

    def to_records(self):
        '''
        Convert this class into a :class:`numpy.ndarray` object.

        :returns: This object as a a :class:`numpy.ndarray` object.
        :rtype: numpy.ndarray
        '''
        out = np.empty(len(self), dtype=[(p.name, data_types.cpu_float)
                                         for p in self.data_pars])
        data = self.__data.as_ndarray()
        for i, p in enumerate(self.data_pars):
            out[p.name] = data[i::self.ndim]
        return out


class BinnedDataSet(DataObject):

    sample_type = BINNED

    def __init__(self, edges, gaps, pars, values):
        '''
        A binned data set.

        :param edges: edges of the bins.
        :type edges: darray
        :param gaps: gaps between edges belonging to different parameters.
        :type gaps: numpy.ndarray
        :param data_pars: data parameters.
        :type data_pars: Registry(Parameter)
        :param values: values of the data for each center.
        :type values: darray
        '''
        super().__init__(pars, edges.aop.backend)

        self.__edges = edges
        self.__gaps = data_types.array_int(gaps)
        self.__values = values

        # The gaps refer to the bins, not to the edges
        g = self.edges_indices

        bounds = data_types.empty_float(2 * len(pars))
        bounds[0::2] = [self.__edges.get(i) for i in g[:-1]]
        bounds[1::2] = [self.__edges.get(i - 1) for i in g[1:]]

        self.__bounds = bounds

    def __getitem__(self, var):
        '''
        Get the centers of the bins for the given parameter.

        :returns: Centers of the bins.
        :rtype: darray
        '''
        i = self.data_pars.index(var)
        e = self.edges_indices
        return self.__edges.take_slice(e[i], e[i + 1])

    def __len__(self):
        '''
        Get the size of the sample.

        :returns: Size of the sample.
        :rtype: int
        '''
        return len(self.__values)

    @property
    def bounds(self):
        '''
        Bounds of each data parameter.

        :type: numpy.ndarray
        '''
        return self.__bounds

    @property
    def edges(self):
        '''
        Edges of the histogram.

        :type: darray
        '''
        return self.__edges

    @property
    def edges_indices(self):
        '''
        Indices to access the edges.

        :type: numpy.ndarray
        '''
        return edges_indices(self.__gaps, self.__edges)

    @property
    def gaps(self):
        '''
        Gaps among the different edges.

        :type: numpy.ndarray
        '''
        return self.__gaps

    @property
    def values(self):
        '''
        Values of the data set.

        :type: darray
        '''
        return self.__values

    @classmethod
    def from_ndarray(cls, edges, data_par, values, backend=None):
        '''
        Build the class from the array of edges and values.

        :param edges: edges of the bins.
        :type edges: numpy.ndarray
        :param data_par: data parameter.
        :type data_par: Parameter
        :param values: values at each bin.
        :type values: numpy.ndarray
        :returns: Binned data set.
        :rtype: BinnedDataSet
        '''
        aop = parse_backend(backend)
        edges = darray.from_ndarray(edges, aop.backend)
        values = darray.from_ndarray(values, aop.backend)
        return cls(edges, [1], parameters.Registry([data_par]), values)

    def to_backend(self, backend):

        edges = self.__edges.to_backend(backend)
        values = self.__values.to_backend(backend)

        return self.__class__(edges, self.__gaps, self.__data_pars, values)


def edges_indices(gaps, edges):
    '''
    Calculate the indices to access the first element and
    that following to the last of a list of edges.

    :param gaps: gaps used to address the correct edges from \
    a common array.
    :type gaps: numpy.ndarray
    :param edges: common array of edges.
    :type edges: numpy.ndarray
    :returns: Array of indices.
    :rtype: numpy.ndarray
    '''
    g = data_types.empty_int(len(gaps) + 1)
    g[0], g[-1] = 0, len(edges)
    g[1:-1] = np.cumsum(np.arange(1, len(gaps)) + gaps[1:] // gaps[:-1])
    return g


def evaluation_grid(aop, data_pars, bounds, size):
    '''
    Create a grid of points to evaluate a :class:`PDF` object.

    :param data_pars: data parameters.
    :type data_pars: list(Parameter)
    :param size: number of entries in the output sample per set of bounds. \
    This means that *size* entries will be generated for each pair of (min, max) \
    provided, that is, per data parameter.
    :type size: int or tuple(int)
    :param bounds: bounds of the different data parameters. Even indices for \
    the lower bounds, and odd indices for the upper bounds.
    :type bounds: numpy.ndarray
    :returns: Uniform sample.
    :rtype: DataSet
    '''
    if bounds.shape == (2,):
        data = aop.linspace(*bounds, size)
    else:
        if np.asarray(size).ndim == 0:
            size = data_types.full_int(len(bounds[0]), size)
        data = aop.meshgrid(*bounds, size)

    return DataSet(data, data_pars)


def rescale_weights_sw2(weights):
    r'''
    Rescale the given weights so the statistical power is treated properly in
    :math:`\log\mathcal{L}` FCNs. This is done using the following equation:

    .. math:: \omega^\prime_i = \omega_i \times \frac{\sum_{j = 0}^n \omega_j}{\sum_{j = 0}^n \omega_j^2}

    :param weights: weights to rescale.
    :type weights: numpy.ndarray
    :returns: rescaled weights.
    :rtype: numpy.ndarray
    '''
    return weights * weights.sum() / (weights**2).sum()


def uniform_sample(aop, data_pars, bounds, size):
    '''
    Generate a sample following an uniform distribution in all data parameters.

    :param data_pars: data parameters.
    :type data_pars: Registry(Parameter)
    :param size: number of entries in the output sample.
    :type size: int
    :param bounds: bounds where to create the sample.
    :type bounds: tuple(float, float)
    :returns: Uniform sample.
    :rtype: DataSet
    '''
    if bounds.shape == (2,):
        data = aop.random_uniform(*bounds, size)
    else:
        data = aop.random_grid(*bounds, size)

    return DataSet(data, data_pars)
