from .base.core import get_exposed_package_objects
from .backends import core as backends_core
from .backends import aop

import functools
import inspect
import os
import types

PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))

__all__ = ['Backend']

minkit_api = get_exposed_package_objects(PACKAGE_PATH)

globals().update(minkit_api)

__all__ += list(sorted(minkit_api.keys()))


class Backend(object):

    objects_backend_dependent_with_init = ('Amoroso', 'Chebyshev',
                                           'CrystalBall', 'Exponential', 'Gaussian', 'Polynomial', 'PowerLaw', 'PDF',
                                           'SourcePDF')

    objects_backend_no_init = ('BinnedDataSet', 'DataSet')

    objects_backend_dependent = objects_backend_dependent_with_init + objects_backend_no_init

    def __init__(self, btype=backends_core.CPU, **kwargs):
        '''
        Object used in order to do operations with objects of the :mod:`minkit`
        module. Any object depending on a backend can be directly built using
        this class, which will forward itself during its construction.

        :param btype: backend type ('cpu', 'cuda', 'opencl').
        :type btype: str
        :param kwargs: arguments forwarded to the backend constructor \
        (cuda and opencl backends only). It can contain any of the following \
        keys: \
        * device: \
        * interactive: \
        :type kwargs: dict
        '''
        super(Backend, self).__init__()
        self.__btype = btype

        self.__aop = aop.array_operations(self, **kwargs)

        for n in Backend.objects_backend_no_init:
            setattr(self, n, object_wrapper(minkit_api[n], self))

        for n in Backend.objects_backend_dependent_with_init:
            setattr(self, n, iobject_wrapper(minkit_api[n], self))

    @property
    def aop(self):
        '''
        Object to do operations on arrays.
        '''
        return self.__aop

    @property
    def btype(self):
        '''
        Backend type.
        '''
        return self.__btype


class object_wrapper(object):

    members_backend_dependent = {o: {n: v for n, v in inspect.getmembers(
        minkit_api[o], inspect.ismethod)} for o in Backend.objects_backend_dependent}

    def __init__(self, cls, backend):
        '''
        Object to wrap the members of other objects so the backend is always
        set to that provided to this class.

        :param cls: class to wrap.
        :type cls: class
        :param backend: backend to use when calling the members.
        :type backend: Backend
        '''
        super(object_wrapper, self).__init__()
        self.__cls = cls
        self.__backend = backend
        self.__members = object_wrapper.members_backend_dependent[cls.__name__]

    def __call__(self, *args, **kwargs):
        '''
        Initialize the wrapped class.

        :param args: arguments forwarded to the __init__ function.
        :type args: tuple
        :param kwargs: arguments forwarded to the __init__ function.
        :type kwargs: dict
        :returns: wrapped object.
        '''
        return self.__cls(*args, **kwargs)

    def __getattr__(self, name):
        '''
        Get the given member of the object.

        :param name: name of the member.
        :type name: str
        :returns: wrapper function.
        :rtype: function
        '''
        def wrapper(*args, **kwargs):
            return self.__members[name](*args, backend=self.__backend, **kwargs)
        wrapper.__name__ = name
        wrapper.__doc__ = f'''
Wrapper around the "{name}" function, which automatically sets the backend.
'''
        return wrapper

    def __repr__(self):
        '''
        Represent this class as a string.

        :returns: this class as a string.
        :rtype: str
        '''
        return f'object_wrapper({self.__cls.__name__}, {tuple(self.__members.keys())})'


class iobject_wrapper(object_wrapper):

    def __init__(self, cls, backend):
        '''
        Object to wrap the members of other objects (including initialization methods)
        so the backend is always set to that provided to this class.

        :param cls: class to wrap.
        :type cls: class
        :param backend: backend to use when calling the members.
        :type backend: Backend
        '''
        super(iobject_wrapper, self).__init__(cls, backend)

    def __call__(self, *args, **kwargs):
        '''
        Initialize the wrapped class.

        :param args: arguments forwarded to the __init__ function.
        :type args: tuple
        :param kwargs: arguments forwarded to the __init__ function.
        :type kwargs: dict
        :returns: wrapped object.
        '''
        return self.__cls(*args, backend=self.__backend, **kwargs)

    def __repr__(self):
        '''
        Represent this class as a string.

        :returns: this class as a string.
        :rtype: str
        '''
        return f'iobject_wrapper({self.__cls.__name__}, {tuple(self.__members.keys())})'


# Determine the default backend
DEFAULT_BACKEND_TYPE = os.environ.get(
    'MINKIT_BACKEND', backends_core.CPU).lower()

if DEFAULT_BACKEND_TYPE == backends_core.CPU:
    DEFAULT_BACKEND = Backend(DEFAULT_BACKEND_TYPE)
else:
    dev = os.environ.get('MINKIT_DEVICE', 0)
    itv = os.environ.get('MINKIT_INTERACTIVE', True)
    DEFAULT_BACKEND = Backend(DEFAULT_BACKEND_TYPE,
                              device=dev, interactive=itv)

backends_core.set_default_aop(DEFAULT_BACKEND.aop)
