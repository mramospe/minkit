########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Definition of exceptions that are used within Minkit.
'''

__all__ = []


class MethodNotDefinedError(NotImplementedError):

    def __init__(self, cls, name):
        '''
        Exception to be raised calls to abstract class methods.
        '''
        super().__init__(
            f'Attempt to call abstract class method {cls.__name__}.{name}')


class PropertyNotDefinedError(NotImplementedError):

    def __init__(self, cls, name):
        '''
        Exception to be raised calls to abstract class properties.
        '''
        super().__init__(
            f'Attempt to obtain an abstract class property {cls.__name__}.{name}')
