'''
Define classes and functions to work with parameters.
'''
from .operations import types
import collections
import json
import logging
import numpy as np

__all__ = ['Parameter', 'Formula', 'Range', 'Registry']

# Default range
FULL = 'full'

logger = logging.getLogger(__name__)


class Parameter(object):

    # Flag to determine if this parameter class depends on other parameters
    # or not. Any object with dependent = True must have the property "args"
    # defined.
    dependent = False

    def __init__(self, name, value=None, bounds=None, ranges=None, error=0., constant=False):
        '''
        Object to represent a parameter for a PDF.

        :param name: name of the parameter.
        :type name: str
        :param value: initial value.
        :type value: float
        :param bounds: bounds for the parameter. This defines the "full" range.
        :type bounds: tuple or tuple(tuple, ...)
        :param ranges: possible ranges
        :type ranges: dict(str, tuple) or dict(str, tuple(tuple, ...))
        :param error: error of the parameter.
        :type error: float
        :param constant: whether to initialize this parameter as constant.
        :type constant: bool
        :ivar name: name of the parameter.
        :ivar error: error of the parameter.
        '''
        self.name = name
        self.value = value
        self.error = error
        self.__ranges = {}
        self.bounds = bounds  # This sets the FULL range
        self.__constant = constant

        # Set the ranges skipping the FULL range, since it is determined by the bounds
        for n, r in (ranges or {}).items():
            if n != FULL:
                self.__ranges[n] = Range(r)

    def __repr__(self):
        '''
        Rerpresent this class as a string, showing its attributes.

        :returns: this class as a string.
        :rtype: str
        '''
        return '{}(name={}, value={}, bounds={}, error={}, constant={})'.format(
            self.__class__.__name__, self.name, self.value, self.bounds, self.error, self.constant)

    @property
    def bounds(self):
        '''
        Bounds of the parameter, defining the "full" range.
        '''
        return self.__bounds

    @bounds.setter
    def bounds(self, values):
        '''
        Set the bounds of the parameter, which also modifies the "full" range.

        :param values: bounds to set.
        :type values: tuple or tuple(tuple, ...)
        '''
        if values is None:
            self.__bounds = values
        else:
            self.__bounds = np.array(values, dtype=types.cpu_type)
        self.__ranges[FULL] = Range(self.__bounds)

    @property
    def constant(self):
        '''
        Return whether this parameter is marked as constant.
        Any parameter forced to constant or with no bounds is considered to be constant.

        :returns: whether this parameter is constant or not.
        :rtype: bool
        '''
        return self.__constant or (self.__bounds is None)

    @constant.setter
    def constant(self, v):
        '''
        Define the "constant" status of the class.

        :param v: value determining whether this object must be considered as constant or not.
        :type v: bool
        '''
        self.__constant = v

    @classmethod
    def from_json_object(cls, obj):
        '''
        Build the parameter from a JSON object (a dictionary).
        This is meant to be used together with the :mod:`json` module.

        :param obj: object to use to construct the class.
        :type obj: dict
        :returns: parameter created from the JSON object.
        :rtype: Parameter
        '''
        obj = dict(obj)
        obj['ranges'] = {n: o for n, o in obj['ranges'].items()}
        return cls(**obj)

    def get_range(self, name):
        '''
        Get the range with the given name.

        :param name: name of the range.
        :type name: str
        :returns: attached range.
        :rtype: Range
        '''
        return self.__ranges[name]

    def set_range(self, name, values):
        '''
        Define the range with name "name".
        Must not be "full".

        :param name: name of the range.
        :type name: str
        :param values: bounds of the range.
        :type values: tuple or tuple(tuple, ...)
        '''
        if name == FULL:
            raise ValueError(
                f'Range name "{name}" is protected; can not be used')
        self.__ranges[name] = Range(values)

    def to_json_object(self):
        '''
        Represent this class as a JSON-like object.

        :returns: this class as a JSON-like object.
        :rtype: dict
        '''
        if self.bounds is None:
            bounds = self.bounds
        else:
            bounds = self.bounds.tolist()

        return {'name': self.name,
                'value': self.value,
                'bounds': bounds,
                'ranges': {n: r.bounds.tolist() for n, r in self.__ranges.items() if n != FULL},
                'error': self.error,
                'constant': self.constant}

    @property
    def value(self):
        '''
        Get the value of the parameter.
        '''
        return self.__value

    @value.setter
    def value(self, value):
        '''
        Set the value of the parameter.
        '''
        if value is not None:
            self.__value = types.cpu_type(value)
        else:
            self.__value = value


class Formula(object):

    # Flag to determine if this parameter class depends on other parameters
    # or not. Any object with dependent = True must have the property "args"
    # defined.
    dependent = True

    def __init__(self, name, formula, pars):
        '''
        Parameter representing an operation of many parameters.

        :param name: name of the parameter.
        :type name: str
        :param formula: formula to apply. Any function defined in :py:mod:`math` is allowed.
        :type formula: str
        :param pars: input parameters.
        :type pars: Registry
        '''
        self.name = name
        self.__pars = Registry(pars)

        # Replace the names of the parameters with brackets
        names = list(
            reversed(sorted(zip(map(len, self.__pars.names), self.__pars.names))))
        for i, (_, n) in enumerate(names):
            formula = formula.replace(n, '{' + str(i) + '}')

        # Avoid partially replacing similar names
        for i, (_, n) in enumerate(names):
            formula = formula.replace('{' + str(i) + '}', '{' + n + '}')

        self.__formula = formula

        super(Formula, self).__init__()

    def __repr__(self):
        '''
        Rerpresent this class as a string, showing its attributes.

        :returns: this class as a string.
        :rtype: str
        '''
        return '{}(name={}, formula=\'{}\', parameters={})'.format(
            self.__class__.__name__, self.name, self.__formula, self.__pars.names)

    @property
    def all_args(self):
        '''
        Get the parameters this object depends on.
        '''
        pars = Registry(self.__pars)
        for p in filter(lambda p: p.dependent, pars):
            pars += p.all_args
        return pars

    @property
    def args(self):
        '''
        Get the parameters this object depends on.
        '''
        return self.__pars

    @property
    def value(self):
        '''
        Value, evaluated from the values of the other parameters.
        '''
        import math
        values = {p.name: p.value for p in self.args}
        return eval(self.__formula.format(**values))

    @classmethod
    def from_json_object(cls, obj, pars):
        '''
        Build the parameter from a JSON object (a dictionary).
        This is meant to be used together with the :mod:`json` module.

        :param obj: object to use to construct the class.
        :type obj: dict
        :returns: parameter created from the JSON object.
        :rtype: Parameter
        '''
        pars = Registry(pars.get(n) for n in obj['pars'])
        return cls(obj['name'], obj['formula'], pars)

    def to_json_object(self):
        '''
        Represent this class as a JSON-like object.

        :returns: this class as a JSON-like object.
        :rtype: dict
        '''
        return {'name': self.name, 'formula': self.__formula, 'pars': self.args.names}


class Range(object):

    def __init__(self, bounds):
        '''
        Object to define bounds for a parameter.

        :param bounds: bounds of the range.
        :type bounds: tuple or tuple(tuple, ...)
        '''
        self.__bounds = np.array(bounds, dtype=types.cpu_type)

    def __len__(self):
        '''
        Return the number of bounds.

        :return: number of bounds.
        :rtype: int
        '''
        return len(self.__bounds)

    @property
    def bounds(self):
        '''
        Bounds of the range.

        :returns: bounds of the range.
        :rtype: numpy.ndarray
        '''
        return self.__bounds

    @property
    def disjoint(self):
        '''
        Return whether this range is composed by more than one subrange (with no
        common borders.

        :returns: whether the range is disjoint.
        :rtype: bool
        '''
        return len(self.__bounds.shape) > 1

    @property
    def size(self):
        '''
        Return the size of the range.
        This corresponds to the sum of the areas of any subrange.

        :returns: size of the range.
        :rtype: float
        '''
        if len(self.__bounds.shape) == 1:
            return self.__bounds[1] - self.__bounds[0]
        else:
            return np.sum(np.fromiter(((s - f) for f, s in self.__bounds), dtype=types.cpu_type))


class Registry(list):

    def __init__(self, *args, **kwargs):
        '''
        Extension of list to hold information used in :py:mod:`minkit`.
        It represents a collection of objects with the attribute "name", providing a unique
        identifier (each object is assumed to be identified by its name).
        Any attempt to add a new element to the registry with the same name as one already
        existing will skip the process, as long as the two objects are the same.
        If they are not, then an error is raised.
        Constructor is directly forwarded to :class:`list`.
        '''
        super(Registry, self).__init__(*args, **kwargs)

    def __add__(self, other):
        '''
        Add elements from another registry inplace.
        Only elements with different names to those in the registry are added.

        :param other: registry to take elements from.
        :type other: Registry
        :returns: a registry with the new elements added.
        :rtype: Registry
        '''
        res = self.__class__(self)
        return res.__iadd__(other)

    def __iadd__(self, other):
        '''
        Add elements from another registry inplace.
        Only elements with different names to those in the registry are added.

        :param other: registry to take elements from.
        :type other: Registry
        :returns: this object with the new elements added.
        :rtype: Registry
        '''
        for el in filter(lambda p: p.name in self.names, other):
            self._raise_if_not_same(el)
        return super(Registry, self).__iadd__(filter(lambda p: p.name not in self.names, other))

    def _raise_if_not_same(self, el):
        '''
        Raise an error saying that an object that is trying to be added with given name
        is not the same as that in the registry.
        The name of the element is assumed to be already in the registry.
        '''
        curr = self.get(el.name)
        if curr is not el:
            raise ValueError(
                f'Attempt to add an element with name "{el.name}" ({hex(id(el))}) to a registry with a different object associated to that name ({hex(id(curr))})')

    @property
    def names(self):
        '''
        Get the names in the current registry.

        :returns: names in the current registry.
        :rtype: list(str)
        '''
        return [p.name for p in self]

    @classmethod
    def from_json_object(cls, obj):
        '''
        Build the parameter from a JSON object (a dictionary).
        This is meant to be used together with the :py:mod:`json` module.

        :param obj: object to use to construct the class.
        :type obj: dict
        :returns: parameter created from the JSON object.
        :rtype: Registry
        '''
        return cls(map(Parameter.from_json_object, obj))

    def append(self, el):
        '''
        Append a new element to the registry.

        :param el: new element to add.
        :type el: object
        '''
        if el.name not in self.names:
            super(Registry, self).append(el)
        else:
            self._raise_if_not_same(el)

    def get(self, name):
        '''
        Return the object with name "name" in this registry.

        :param name: name of the object.
        :type name: str
        :returns: object with the specified name.
        :raises LookupError: If no object is found with the given name.
        '''
        for e in self:
            if e.name == name:
                return e
        raise LookupError(f'Object with name "{name}" has not been found')

    def index(self, name):
        '''
        Get the position in the registry of the parameter with the given name.

        :param name: name of the parameter.
        :type name: str
        :returns: position.
        :rtype: int
        :raises LookupError: If no object is found with the given name.
        '''
        for i, p in enumerate(self):
            if p.name == name:
                return i
        raise LookupError(f'Object with name "{name}" has not been found')

    def insert(self, i, p):
        '''
        Insert an object before index "i".

        :param i: index where to insert the object.
        :type i: int
        :param p: object to insert.
        :type p: object
        '''
        if p.name in self.names:
            self._raise_if_not_same(p)
            return self
        else:
            return super(Registry, self).insert(i, p)

    def reduce(self, names):
        '''
        Create a new :class:`Registry` object keeping only the given names.

        :param names: names
        :type names: tuple(str)
        :returns: new registry keeping only the provided names.
        :rype: Registry
        '''
        return self.__class__(filter(lambda p: p.name in names, self))

    def to_json_object(self):
        '''
        Represent this class as a JSON-like object.

        :returns: this class as a JSON-like object.
        :rtype: dict
        '''
        return [p.to_json_object() for p in self]


def bounds_for_range(data_pars, range):
    '''
    Get the bounds associated to a given range, and return it as a single array.

    :param data_pars: data parameters.
    :type data_pars: Registry(Parameter)
    :param range: range to evaluate.
    :type range: str
    :returns: bounds for the given range.
    :rtype: numpy.ndarray
    '''
    single_bounds = collections.OrderedDict()
    multi_bounds = collections.OrderedDict()
    for p in data_pars:
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
        mins = collections.OrderedDict()
        maxs = collections.OrderedDict()
        for n in data_pars.names:
            if n in single_bounds:
                mins[n], maxs[n] = single_bounds[n].bounds.T
            elif p.name in multi_bounds:
                mins[n], maxs[n] = multi_bounds[n].bounds.T
            else:
                raise RuntimeError(
                    'Internal error detected; please report the bug')

        # Get all the combinations of minimum and maximum values for the bounds of each variable
        mmins = [m.flatten() for m in np.meshgrid(*[b for b in mins.values()])]
        mmaxs = [m.flatten() for m in np.meshgrid(*[b for b in maxs.values()])]

        return np.concatenate([np.array([mi, ma]).T for mi, ma in zip(mmins, mmaxs)], axis=1)
