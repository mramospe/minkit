'''
Define classes and functions to work with parameters.
'''
from . import types
import collections
import json
import logging
import numpy as np

__all__ = ['Parameter', 'Range', 'Registry']

# Default range
FULL = 'full'

logger = logging.getLogger(__name__)


class Parameter(object):

    def __init__( self, name, value = None, bounds = None, ranges = None, error = 0., constant = False ):
        '''
        '''
        self.name     = name
        self.value    = value
        self.error    = error
        self.__ranges = {}
        self.bounds   = bounds # This sets the FULL range
        self.constant = (bounds is None) or constant

        # Set the ranges skipping the FULL range, since it is determined by the bounds
        for n, r in (ranges or {}).items():
            if n != FULL:
                self.__ranges[n] = Range(r)

    def __repr__( self ):
        return '{}(name={}, value={}, bounds={}, error={}, constant={})'.format(
            self.__class__.__name__, self.name, self.value, self.bounds, self.error, self.constant)

    @property
    def bounds( self ):
        return self.__bounds

    @bounds.setter
    def bounds( self, values ):
        self.__bounds = values
        self.__ranges[FULL] = Range(self.__bounds)

    @classmethod
    def from_json_object( cls, obj ):
        '''
        '''
        obj = dict(obj)
        obj['ranges'] = {n: Range(o) for n, o in obj['ranges'].items()}
        return cls(**obj)

    def get_range( self, name ):
        return self.__ranges[name]

    def set_range( self, name, values ):
        if name == FULL:
            raise ValueError(f'Range name "{name}" is protected; can not be used')
        self.__ranges[name] = Range(values)

    def to_json_object( self ):
        '''
        '''
        return {'name': self.name,
                'value': self.value,
                'bounds': self.bounds,
                'ranges': {n: r.bounds.tolist() for n, r in self.__ranges.items()},
                'error': self.error,
                'constant': self.constant}


class Range(object):

    def __init__( self, bounds ):
        '''
        '''
        self.__bounds = np.array(bounds)

    def __len__( self ):
        return len(self.__bounds)

    @property
    def bounds( self ):
        return self.__bounds

    @property
    def disjoint( self ):
        return len(self.__bounds.shape) > 1

    @property
    def size( self ):
        if len(self.__bounds.shape) == 1:
            return self.__bounds[1] - self.__bounds[0]
        else:
            return np.sum(np.fromiter(((s - f) for f, s in self.__bounds), dtype=types.cpu_type))


class Registry(collections.OrderedDict):

    def __init__( self, *args, **kwargs ):
        '''
        '''
        super(Registry, self).__init__(*args, **kwargs)

    @classmethod
    def from_list( cls, args ):
        c = cls()
        for a in args:
            if a.name in c:
                raise KeyError(f'A parameter with name "{a.name}" already exists in the registry')
            c[a.name] = a
        return c

    def to_list( self ):
        return list(self.values())

    @classmethod
    def from_json_object( cls, obj ):
        '''
        '''
        return cls.from_list(map(Parameter.from_json_object, obj))

    def to_json_object( self ):
        '''
        '''
        return [p.to_json_object() for p in self.values()]


def bounds_for_range( data_pars, range ):
    '''
    Get the bounds associated to a given range, and return it as a single array.

    :param data_pars: data parameters.
    :type data_pars: Registry(str, Parameter)
    :param range: range to evaluate.
    :type range: str
    :returns: bounds for the given range.
    :rtype: numpy.ndarray
    '''
    single_bounds = Registry()
    multi_bounds  = Registry()
    for p in data_pars.values():
        r = p.get_range(range)
        if r.disjoint:
            multi_bounds[p.name] = r
        else:
            single_bounds[p.name] = r

    if len(multi_bounds) == 0:
        # Simple case, all data parameters have only one set of bounds
        # for this normalization range
        return np.array([r.bounds for r in single_bounds.values()]).flatten()
    else:
        # Must calculate all the combinations of normalization ranges
        # for every data parameter.
        mins = Registry()
        maxs = Registry()
        for n in data_pars:
            if n in single_bounds:
                mins[n], maxs[n] = single_bounds[n].bounds.T
            elif p.name in multi_bounds:
                mins[n], maxs[n] = multi_bounds[n].bounds.T
            else:
                raise RuntimeError('Internal error detected; please report the bug')

        # Get all the combinations of minimum and maximum values for the bounds of each variable
        mmins = [m.flatten() for m in np.meshgrid(*[b for b in mins.values()])]
        mmaxs = [m.flatten() for m in np.meshgrid(*[b for b in maxs.values()])]

        return np.concatenate([np.array([mi, ma]).T for mi, ma in zip(mmins, mmaxs)], axis=1)

