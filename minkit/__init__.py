from .base.core import get_exposed_package_objects
from .backends import core as backends_core
from .backends import aop

import inspect
import os

PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))

__all__ = ['Backend']

minkit_api = get_exposed_package_objects(PACKAGE_PATH)

globals().update(minkit_api)

__all__ += list(minkit_api.keys())

__all__.sort()


class Backend(object):

    _objects_backend_dependent_with_init = ('Amoroso', 'Chebyshev',
                                            'CrystalBall', 'Exponential', 'Gaussian', 'Polynomial', 'PowerLaw', 'PDF',
                                            'SourcePDF')

    _objects_backend_no_init = ('BinnedDataSet', 'DataSet')

    _objects_backend_dependent = _objects_backend_dependent_with_init + \
        _objects_backend_no_init

    def __init__(self, btype=backends_core.CPU, **kwargs):
        '''
        Object used in order to do operations with objects of the :mod:`minkit`
        module. Any object depending on a backend can be directly built using
        this class, which will forward itself during its construction.

        :param btype: backend type ('cpu', 'cuda', 'opencl').
        :type btype: str
        :param kwargs: arguments forwarded to the backend constructor \
        (cuda and opencl backends only). See below for more details.
        :type kwargs: dict

        The keyword arguments can contain any of the following:
        * device
        * interactive
        These arguments are only available in *cuda* and *opencl* backends only.
        '''
        super(Backend, self).__init__()
        self.__btype = btype.lower()

        self.__aop = aop.ArrayOperations(self, **kwargs)

        for n in Backend._objects_backend_no_init:
            setattr(self, n, object_wrapper(minkit_api[n], self))

        for n in Backend._objects_backend_dependent_with_init:
            setattr(self, n, iobject_wrapper(minkit_api[n], self))

    @property
    def aop(self):
        '''
        Object to do operations on arrays.

        :type: ArrayOperations
        '''
        return self.__aop

    @property
    def btype(self):
        '''
        Backend type.

        :type: str
        '''
        return self.__btype


class object_wrapper(object):

    members_backend_dependent = {o: {n: v for n, v in inspect.getmembers(
        minkit_api[o], inspect.ismethod)} for o in Backend._objects_backend_dependent}

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
        :returns: Wrapped object.
        '''
        return self.__cls(*args, **kwargs)

    def __getattr__(self, name):
        '''
        Get the given member of the object.

        :param name: name of the member.
        :type name: str
        :returns: Wrapped function.
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

        :returns: This class as a string.
        :rtype: str
        '''
        return f'object_wrapper({self.__cls.__name__}, {tuple(self.__members.keys())})'


class iobject_wrapper(object_wrapper):

    def __init__(self, cls, backend):
        '''
        Object to wrap the members of other objects (including initialization
        methods) so the backend is always set to that provided to this class.

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
        :returns: Wrapped object.
        '''
        return self.__cls(*args, backend=self.__backend, **kwargs)

    def __repr__(self):
        '''
        Represent this class as a string.

        :returns: This class as a string.
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
