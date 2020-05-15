'''
Definition of the interface functions and classes with :mod:`iminuit`.
'''
from ..base import parameters
from . import core

import functools
import iminuit

__all__ = ['MinuitMinimizer']

# Definition of the errors. This is given from the nature of the FCNs. If this is
# changed the output of the FCNs must change accordingly. A value of 1 means
# that the output of the FCNs is a chi-square-like function.
ERRORDEF = 1.


def use_const_cache(method):
    '''
    Use the constant cache of the evaluator when calling the method.
    '''
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.evaluator.using_caches():
            return method(self, *args, **kwargs)
    return wrapper


def registry_to_minuit_input(registry):
    '''
    Transform a registry of parameters into a dictionary to be parsed by Minuit.

    :param registry: registry of parameters.
    :type registry: Registry(Parameter)
    :returns: Minuit configuration dictionary.
    :rtype: dict
    '''
    values = {v.name: v.value for v in registry}
    errors = {f'error_{v.name}': v.error for v in registry}
    limits = {f'limit_{v.name}': v.bounds for v in registry}
    const = {f'fix_{v.name}': v.constant for v in registry}
    return dict(errordef=ERRORDEF, **values, **errors, **limits, **const)


class MinuitMinimizer(core.Minimizer):

    def __init__(self, evaluator, **minimizer_config):
        '''
        Interface with the :class:`iminuit.Minuit` class. In the calls to the
        different methods of this class it is ensured that the parameters of
        the PDF(s) from the evaluator instance have their value set to that of
        the minimum after a minimization process.

        :param evaluator: evaluator to be used in the minimization.
        :type evaluator: UnbinnedEvaluator, BinnedEvaluator or SimultaneousEvaluator
        '''
        super().__init__()

        self.__eval = evaluator
        self.__args = evaluator.args
        self.__minuit = iminuit.Minuit(evaluator,
                                       forced_parameters=self.__args.names,
                                       pedantic=False,
                                       **minimizer_config,
                                       **registry_to_minuit_input(self.__args))

    def _update_args(self, params):
        '''
        Update the parameters using the information from the given parameters.

        :param params: list of parameters.
        :type params: iminuit.util.Params
        '''
        for p in params:
            a = self.__args.get(p.name)
            # Update the fields
            a.value = p.value
            if p.has_limits:
                a.bounds = p.lower_limit, p.upper_limit
            a.error = p.error
            a.constant = p.is_fixed

    @property
    def evaluator(self):
        '''
        Evaluator of the minimizer.
        '''
        return self.__eval

    @property
    def minuit(self):
        '''
        Underlying :class:`iminuit.Minuit` object.
        '''
        return self.__minuit

    @use_const_cache
    def hesse(self, *args, **kwargs):
        '''
        Arguments are forwarded to the :py:meth:`iminuit.Minuit.hesse` function,
        and the values of the parameters are set to those from the minimization
        result.

        :returns: output from :py:meth:`iminuit.Minuit.hesse`.
        '''
        result = self.__minuit.hesse(*args, *kwargs)

        self._update_args(result)

        return result

    @use_const_cache
    def migrad(self, *args, **kwargs):
        '''
        Arguments are forwarded to the :py:meth:`iminuit.Minuit.migrad` function,
        and the values of the parameters are set to those from the minimization
        result.

        :returns: output from :py:meth:`iminuit.Minuit.migrad`.
        '''
        result = self.__minuit.migrad(*args, *kwargs)

        self._update_args(result.params)

        return result

    @use_const_cache
    def minos(self, *args, **kwargs):
        '''
        Arguments are forwarded to the :py:meth:`iminuit.Minuit.minos` function,
        and the values of the parameters are set to those from the minimization
        result.

        :returns: output from :py:meth:`iminuit.Minuit.minos`.
        '''
        result = self.__minuit.minos(*args, *kwargs)

        for k, v in result.items():
            a = self.__args.get(k)
            a.asym_errors = v.lower, v.upper

        return result

    @staticmethod
    def result_to_registry(result):
        '''
        Transform the parameters from a call to Minuit into a :class:`Registry`.

        :param result: result from a migrad call.
        :type result: iminuit.util.Params
        :returns: registry of parameters with the result from Migrad.
        :rtype: Registry(Parameter)

        .. warning::
           The parameters in the returned registry do not belong to the
           minimized object(s).
        '''
        reg = parameters.Registry()
        for p in result:
            limits = (p.lower_limit, p.upper_limit) if p.has_limits else None
            reg.append(parameters.Parameter(
                p.name, p.value, bounds=limits, error=p.error, constant=p.is_fixed))
        return reg
