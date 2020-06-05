########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Define classes and functions to work with parameters.
'''
from . import core
from . import data_types
from . import dependencies
from . import exceptions

import collections
import contextlib
import logging
import numpy as np

__all__ = ['ParameterBase', 'Parameter', 'Formula', 'Registry']

# Default range
FULL = 'full'

logger = logging.getLogger(__name__)


class ParameterBase(object, metaclass=core.DocMeta):

    # Flag to determine if this parameter class depends on other parameters
    # or not. Any object with dependent = True must have the property "args"
    # defined.
    dependent = False

    def __init__(self):
        '''
        Abstract class for parameter objects.
        '''
        super().__init__()

    def copy(self):
        '''
        Create a copy of this instance.

        .. warning::

           Avoid calling this method directly for sets of parameters and use the
           :meth:`Registry.copy` method instead, so the possible dependencies
           among parameters are correctly solved.
        '''
        raise exceptions.MethodNotDefinedError(self.__class__, 'copy')

    @property
    def value(self):
        '''
        Value of the parameter.

        :type: float
        '''
        raise exceptions.PropertyNotDefinedError(self.__class__, 'value')

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
        raise exceptions.MethodNotDefinedError(cls, 'from_json_object')

    @contextlib.contextmanager
    def restoring_state(self):
        '''
        Enter a context where the attributes of the parameter will be restored
        on exit.
        '''
        yield self

    def to_json_object(self):
        '''
        Represent this class as a JSON-like object.

        :returns: this class as a JSON-like object.
        :rtype: dict
        '''
        raise exceptions.MethodNotDefinedError(
            self.__class__, 'to_json_object')


class Parameter(ParameterBase):

    def __init__(self, name, value=None, bounds=None, ranges=None, error=0., constant=False, asym_errors=None):
        '''
        Object to represent a parameter for a PDF.

        :param name: name of the parameter.
        :type name: str
        :param value: initial value.
        :type value: float
        :param bounds: bounds for the parameter. This defines the *full* range.
        :type bounds: tuple or tuple(tuple, ...)
        :param ranges: possible ranges
        :type ranges: dict(str, tuple) or dict(str, tuple(tuple, ...))
        :param error: error of the parameter.
        :type error: float
        :param constant: whether to initialize this parameter as constant.
        :type constant: bool
        :param asym_errors: asymmetric errors.
        :type asym_errors: tuple(float, float)
        :ivar name: name of the parameter.
        :ivar error: error of the parameter.
        '''
        super().__init__()

        self.__constant = constant
        self.__ranges = {}

        self.name = name
        self.value = value
        self.bounds = bounds  # This sets the FULL range
        self.error = error
        self.asym_errors = asym_errors if asym_errors is None else tuple(
            asym_errors)

        if ranges is not None:
            for n, r in ranges.items():
                self.set_range(n, r)

    def __repr__(self):
        '''
        Rerpresent this class as a string, showing its attributes.

        :returns: this class as a string.
        :rtype: str
        '''
        return '{}(name={}, value={}, bounds={}, error={}, asym_errors={}, constant={})'.format(
            self.__class__.__name__, self.name, self.value, self.bounds, self.error, self.asym_errors, self.constant)

    @property
    def bounds(self):
        '''
        Bounds of the parameter, defining the *full* range.

        :type: numpy.ndarray
        '''
        return self.__bounds

    @bounds.setter
    def bounds(self, values):
        '''
        Set the bounds of the parameter, which also modifies the *full* range.

        :param values: bounds to set.
        :type values: tuple or tuple(tuple, ...)
        '''
        if values is None:
            self.__bounds = values
        else:
            self.__bounds = data_types.array_float(values)
        self.__ranges[FULL] = self.__bounds

    @property
    def constant(self):
        '''
        Whether this parameter is marked as constant.

        :type: bool
        '''
        return self.__constant or (self.__bounds is None)

    @constant.setter
    def constant(self, v):
        self.__constant = v

    def copy(self):
        # a copy is created during __init__ for bounds, ranges, and asym_errors
        ranges = {r: n for r, n in self.__ranges.items() if r != FULL}
        return self.__class__(self.name, self.value, self.__bounds, ranges, self.error, self.__constant, self.asym_errors)

    @property
    def ranges(self):
        '''
        Names of the parameter ranges.
        '''
        return list(self.__ranges.keys())

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
        :rtype: numpy.ndarray
        '''
        return self.__ranges[name]

    def set_range(self, name, values):
        '''
        Define the range with name *name*.
        Must not be *full*.

        :param name: name of the range.
        :type name: str
        :param values: bounds of the range.
        :type values: tuple or tuple(tuple, ...)
        '''
        if name == FULL:
            raise ValueError(
                f'Range name "{name}" is protected; can not be used')
        self.__ranges[name] = data_types.array_float(values)

    @contextlib.contextmanager
    def restoring_state(self):
        ranges = {n: r for n, r in self.__ranges.items() if n != FULL}
        vals = (self.name, self.value, self.bounds, ranges,
                self.error, self.constant, self.asym_errors)
        yield self
        self.__init__(*vals)

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
                'ranges': {n: r.tolist() for n, r in self.__ranges.items() if n != FULL},
                'error': self.error,
                'constant': self.constant}

    @property
    def value(self):
        '''
        Value of the parameter.

        :type: float
        '''
        return self.__value

    @value.setter
    def value(self, value):
        if value is not None:
            self.__value = data_types.cpu_float(value)
        else:
            self.__value = value


class Formula(ParameterBase):

    dependent = True

    def __init__(self, name, formula, pars):
        '''
        Parameter representing an operation of many parameters.
        The formula can be expressed as a function of parameter names, as a function
        of indices or a mixture of the two, like:

        * *Parameter names*: "a * b" will multiply *a* and *b*.
        * *Indices*: "{0} * {1}" will multiply the first and second elements in *pars*.
        * *Mixed*: "{a} * {b} + {1}" will multiply *a* and *b* and sum the second element \
        in *pars*.

        :param name: name of the parameter.
        :type name: str
        :param formula: formula to apply. Any function defined in :py:mod:`math` is allowed.
        :type formula: str
        :param pars: input parameters.
        :type pars: Registry
        '''
        super().__init__()

        self.name = name
        self.__formula = formula
        self.__pars = Registry(pars)

    def __repr__(self):
        '''
        Rerpresent this class as a string, showing its attributes.

        :returns: this class as a string.
        :rtype: str
        '''
        return '{}(name={}, formula=\'{}\', parameters={})'.format(
            self.__class__.__name__, self.name, self.__formula, self.__pars.names)

    @property
    def args(self):
        '''
        Argument parameters this object directly depends on.

        :type: Registry(Parameter)
        '''
        return self.__pars

    @property
    def dependencies(self):
        '''
        Registry of parameters this instance depends on.

        :type: Registry(Parameter)
        '''
        return self.__pars

    @property
    def all_args(self):
        '''
        Argument parameters this object depends on.

        :type: Registry(Parameter)
        '''
        args = Registry(self.__pars)
        for p in filter(lambda p: p.dependent, args):
            args += p.all_args
        return args

    @property
    def value(self):
        '''
        Value, evaluated from the values of the other parameters.

        :type: float
        '''
        values = {p.name: p.value for p in self.args}
        return core.eval_math_expression(self.__formula.format(**values))

    def copy(self, pars):
        '''
        Create a copy of this instance.

        :param pars: parameter to build the class.
        :type pars: Registry
        :returns: A copy of this instance.
        :rtype: Formula

        .. warning::

           Avoid calling this method directly for sets of parameters and use the
           :meth:`Registry.copy` method instead, so the possible dependencies
           among parameters are correctly solved.
        '''
        return self.__class__(self.name, self.__formula, pars)

    @classmethod
    def from_json_object(cls, obj, pars):
        '''
        Build the parameter from a JSON object (a dictionary).
        This is meant to be used together with the :mod:`json` module.

        :param obj: object to use to construct the class.
        :type obj: dict
        :param pars: registry with the parameters this object depends on.
        :type pars: Registry
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


def range_is_disjoint(r):
    '''
    Return whether this range is composed by more than one subrange (with no
    common borders.

    :param r: range to process.
    :type r: numpy.ndarray
    :returns: whether the range is disjoint or not.
    :rtype: bool
    '''
    return len(r.shape) > 1


class Registry(list):

    def __init__(self, *args, **kwargs):
        '''
        Extension of list to hold information used in :py:mod:`minkit`.
        It represents a collection of objects with the attribute *name*, providing a unique
        identifier (each object is assumed to be identified by its name).
        Any attempt to add a new element to the registry with the same name as one already
        existing will skip the process, as long as the two objects are the same.
        If they are not, then an error is raised.
        Constructor is directly forwarded to :class:`list`.
        '''
        super().__init__(*args, **kwargs)

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
        return super().__iadd__(filter(lambda p: p.name not in self.names, other))

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
        Names in the current registry.

        :type: list(str)
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
            super().append(el)
        else:
            self._raise_if_not_same(el)

    def copy(self, contained_type='parameter'):
        '''
        Create a copy of this instance.

        :returns: Copy of this instance.
        :rtype: Registry
        '''
        iobjs, dobjs = dependencies.split_dependent_objects_with_resolution_order(
            self)

        objs = Registry([o.copy() for o in iobjs])

        for o in dobjs:
            objs.append(o.copy(objs))

        result = Registry(len(self) * [None])
        for i, p in enumerate(self):
            result[i] = objs.get(p.name)

        return result

    def get(self, name):
        '''
        Return the object with name *name* in this registry.

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
        Insert an object before index *i*.

        :param i: index where to insert the object.
        :type i: int
        :param p: object to insert.
        :type p: object
        '''
        if p.name in self.names:
            self._raise_if_not_same(p)
            return self
        else:
            return super().insert(i, p)

    def reduce(self, names):
        '''
        Create a new :class:`Registry` object keeping only the given names.

        :param names: names
        :type names: tuple(str)
        :returns: new registry keeping only the provided names.
        :rype: Registry
        '''
        return self.__class__(filter(lambda p: p.name in names, self))

    @contextlib.contextmanager
    def restoring_state(self):
        '''
        Enter a context where the attributes of the parameter will be restored
        on exit.
        '''
        with contextlib.ExitStack() as stack:
            for p in self:
                stack.enter_context(p.restoring_state())
            yield self

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
        if range_is_disjoint(r):
            multi_bounds[p.name] = r
        else:
            single_bounds[p.name] = r

    if len(multi_bounds) == 0:
        # Simple case, all data parameters have only one set of bounds
        # for this normalization range
        t = data_types.array_float([r for r in single_bounds.values()]).T
        return data_types.array_float([t])
    else:
        # Must calculate all the combinations of normalization ranges
        # for every data parameter.
        mins = collections.OrderedDict()
        maxs = collections.OrderedDict()
        for n in data_pars.names:
            if n in single_bounds:
                mins[n], maxs[n] = single_bounds[n]
            elif p.name in multi_bounds:
                mins[n], maxs[n] = multi_bounds[n].T
            else:
                raise RuntimeError(
                    'Internal error detected; please report the bug')

        # Get all the combinations of minimum and maximum values for the bounds of each variable
        mmins = data_types.array_float([m.flatten()
                                        for m in np.meshgrid(*[b for b in mins.values()])]).T
        mmaxs = data_types.array_float([m.flatten()
                                        for m in np.meshgrid(*[b for b in maxs.values()])]).T

        return data_types.array_float([(lb, ub) for lb, ub in zip(mmins, mmaxs)])
