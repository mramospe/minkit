'''
Basic functions and classes to do minimizations.
'''
from ..base.core import DocMeta
from ..base import exceptions

__all__ = ['Minimizer']


class Minimizer(object, metaclass=DocMeta):

    def __init__(self):
        '''
        Abstract class to serve as an API between Minkit and the different
        minimization methods.
        '''
        super().__init__()

    def result_to_registry(self, result):
        '''
        Transform the parameters from a call to Minuit into a :class:`Registry`.

        :param result: result from a call to the minimizer.
        :returns: independent registry of parameters with the results.
        :rtype: Registry(Parameter)

        .. warning::
           The parameters in the returned registry do not belong to the
           minimized object(s).
        '''
        raise exceptions.MethodNotDefinedError(
            self.__class__, 'result_to_registry')
