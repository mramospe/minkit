'''
Define classes and functions to work with parameters.
'''
import collections
import numpy as np

__all__ = ['Parameter', 'Range', 'Registry']

# Default range
FULL = 'full'


class Parameter(object):

    def __init__( self, name, value = None, bounds = None, ranges = None, error = 0., constant = False ):
        '''
        '''
        if ranges is not None and FULL in ranges:
            logger.error(f'Range with name "{FULL}" is protected; use another name')

        self.name     = name
        self.value    = value
        self.error    = error
        self.__ranges = {}
        self.bounds   = bounds # This sets the FULL range
        self.constant = (bounds is None) or constant

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

    def get_range( self, name ):
        return self.__ranges[name]

    def set_range( self, name, values ):
        if name == FULL:
            raise ValueError(f'Range name "{name}" is protected; can not be used')
        self.__ranges[name] = Range(values)


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
            return np.sum((s - f) for f, s in self.__bounds)


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
