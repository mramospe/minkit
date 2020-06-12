########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Define classes to blind parameters.
'''
from . import core
from . import exceptions

import functools

__all__ = []

# Blinding methods
FULL = 'full'
OFFSET = 'offset'
SCALE = 'scale'

BLINDING_METHODS = {}


def check_return_none(method):
    '''
    Decorator for methods with a single argument that will return None if
    the input argument is None.
    '''
    @functools.wraps(method)
    def wrapper(self, arg):
        if arg is None:
            return arg
        else:
            return method(self, arg)
    return wrapper


def register_blinding_method(name):
    '''
    Register a new blinding method with the given name.
    '''
    def wrapper(cls):
        BLINDING_METHODS[name] = cls
        return cls
    return wrapper


def build_blinding(method, **state):
    '''
    Build the instance from the state
    (returned by the :meth:`Blinding.state` method).

    :param method: blinding method.
    :type method: str
    :param state: state of the blinding.
    :type state: tuple
    '''
    constructor = BLINDING_METHODS.get(method, None)

    if method is None:
        raise ValueError(f'Unknown blinding method "{method}"')
    else:
        return constructor.build(**state)


class Blinding(object):

    def __init__(self):
        '''
        Base class for instances blinding parameters.
        '''
        super().__init__()

    @property
    def state(self):
        '''
        State of the blinding transformation as a dictionary. It always contains
        the key *method* specifying the blinding method. The rest of keys
        correspond to the arguments to the :meth:`Blinding.build` method.
        '''
        raise exceptions.PropertyNotDefinedError(self.__class__, 'state')

    @classmethod
    def build(cls, **state):
        '''
        Build the instance from the state
        (returned by the :meth:`Blinding.state` method).

        :param state: state of the blinding.
        :type state: tuple
        '''
        raise exceptions.MethodNotDefinedError(cls, 'build')

    @check_return_none
    def blind(self, value):
        '''
        Get the blinded value associated to that given.

        :param value: value to blind.
        :type value: float
        :returns: Blinded value.
        :rtype: float
        '''
        raise exceptions.MethodNotDefinedError(self.__class__, 'blind')

    @check_return_none
    def blind_error(self, error):
        '''
        Get the blinded value of the given error.

        :param error: error to blind.
        :type error: float
        :returns: Blinded error.
        :rtype: float
        '''
        raise exceptions.MethodNotDefinedError(self.__class__, 'blind_error')

    @check_return_none
    def unblind(self, value):
        '''
        Get the unblinded value associated to that given.

        :param value: value to unblind.
        :type value: float
        :returns: Unblinded value.
        :rtype: float
        '''
        raise exceptions.MethodNotDefinedError(self.__class__, 'unblind')

    @check_return_none
    def unblind_error(self, error):
        '''
        Get the unblinded value for the given error.

        :param error: error to unblind.
        :type error: float
        :returns: Unblinded error.
        :rtype: float
        '''
        raise exceptions.MethodNotDefinedError(self.__class__, 'unblind_error')


@register_blinding_method(FULL)
class BlindingFull(Blinding):

    def __init__(self, offset, scale):
        '''
        Object to fully blind the value of a parameter.

        :param offset: offset of the transformation.
        :type offset: float
        :param scale: scale of the transformation.
        :type scale: float
        '''
        super().__init__()

        self.__offset = offset
        self.__scale = scale

    @property
    def state(self):
        return {'method': FULL, 'offset': self.__offset, 'scale': self.__scale}

    @classmethod
    def build(cls, offset, scale):
        '''
        Build the instance using a offset and a scale. The actual value of
        these attributes will be obtained from two pseudorandom numbers.

        :param offset: prototype for the offset.
        :type offset: float
        :param scale: prototype for the scale.
        :type scale:
        '''
        c = offset * core.GLOBAL_RND_GEN.uniform(-1, +1)
        s = core.GLOBAL_RND_GEN.uniform(1, scale)
        return cls(c, s)

    @check_return_none
    def blind(self, value):
        return self.__scale * (value + self.__offset)

    @check_return_none
    def blind_error(self, error):
        return abs(self.__scale) * error

    @check_return_none
    def unblind(self, value):
        return value / self.__scale - self.__offset

    @check_return_none
    def unblind_error(self, error):
        return error / abs(self.__scale)


@register_blinding_method(OFFSET)
class BlindingOffset(Blinding):

    def __init__(self, offset):
        '''
        Blinding class that simply adds an offset to the true value of a
        parameter. If this blinding method is used, the error of the blinded
        parameter is the same as that for the true value.

        :param offset: offset to consider.
        :type offset: float
        '''
        super().__init__()

        self.__offset = offset

    @property
    def state(self):
        return {'method': OFFSET, 'offset': self.__offset}

    @classmethod
    def build(cls, offset):
        '''
        Build the instance using an offset prototype. The actual value will be
        contained in [-offset, +offset].

        :param offset: prototype for the offset.
        :type offset: float
        '''
        c = offset * core.GLOBAL_RND_GEN.uniform(-1, +1)
        return cls(c)

    @check_return_none
    def blind(self, value):
        return value + self.__offset

    @check_return_none
    def blind_error(self, error):
        return error

    @check_return_none
    def unblind(self, value):
        return value - self.__offset

    @check_return_none
    def unblind_error(self, error):
        return error


@register_blinding_method(SCALE)
class BlindingScale(Blinding):

    def __init__(self, scale):
        '''
        Blinding class that multiplies the true value of a parameter. If this
        blinding method is used, the relative error of the blinded and true
        values are the same.

        :param scale: scale to use.
        :type scale: float
        '''
        super().__init__()

        self.__scale = scale

    @property
    def state(self):
        return {'method': SCALE, 'scale': self.__scale}

    @classmethod
    def build(cls, scale):
        '''
        Build the instance using an offset prototype. The actual value will be
        contained in [1, scale].

        :param scale: prototype for the scale.
        :type scale: float
        '''
        c = core.GLOBAL_RND_GEN.uniform(1, scale)
        return cls(c)

    @check_return_none
    def blind(self, value):
        return self.__scale * value

    @check_return_none
    def blind_error(self, error):
        return self.__scale * error

    @check_return_none
    def unblind(self, value):
        return value / self.__scale

    @check_return_none
    def unblind_error(self, error):
        return error / self.__scale
