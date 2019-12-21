'''
Define classes and functions to work with parameters.
'''
import collections

__all__ = ['Parameter', 'Registry']

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
        self.ranges   = ranges or {}
        self.bounds   = bounds # This sets the FULL range
        self.constant = (bounds is None) or constant

    @property
    def bounds( self ):
        return self.__bounds

    @bounds.setter
    def bounds( self, values ):
        self.__bounds = values
        self.ranges[FULL] = self.__bounds


class Registry(collections.OrderedDict):

    def __init__( self, *args ):
        '''
        '''
        super(Registry, self).__init__()
        for a in args:
            if a.name in self:
                raise KeyError(f'A parameter with name "{a.name}" already exists in the registry')
            self[a.name] = a

    def to_list( self ):
        return list(self.values())

    def clone( self ):
        return self.__class__(*self.to_list())
