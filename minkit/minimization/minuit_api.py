########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Definition of the interface functions and classes with :mod:`iminuit`.
'''
from ..base import data_types
from ..base import parameters
from . import core

import contextlib
import functools
import iminuit

__all__ = ['MinuitMinimizer']

# Definition of the errors. This is given from the nature of the FCNs. If this is
# changed the output of the FCNs must change accordingly. A value of 1 means
# that the output of the FCNs is a chi-square-like function.
ERRORDEF = 1.

MINUIT = 'minuit'


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
    # 0 for Minuit, None for MinKit
    errors = {f'error_{v.name}': v.error for v in registry if v.error is not None}
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
        super().__init__(evaluator)

        self.__args = evaluator.args
        self.__minuit = iminuit.Minuit(evaluator,
                                       forced_parameters=self.__args.names,
                                       pedantic=False,
                                       **minimizer_config,
                                       **registry_to_minuit_input(self.__args))

    def _asym_error(self, par, bound, cov, var=1, atol=core.DEFAULT_ASYM_ERROR_ATOL, rtol=core.DEFAULT_ASYM_ERROR_RTOL, maxcall=None):
        with self._restore_minuit():
            return super()._asym_error(par, bound, cov, var, atol, rtol, maxcall)

    @contextlib.contextmanager
    def _restore_minuit(self):
        '''
        Create a new context where the class holds a new Minuit instance,
        with the same parameters as the previous but with no information
        about the minimization state.
        '''
        previous = self.__minuit
        self.__minuit = iminuit.Minuit(self.evaluator,
                                       forced_parameters=self.__args.names,
                                       pedantic=False,
                                       print_level=previous.print_level,
                                       **self.__minuit.fitarg)
        yield self
        self.__minuit = previous

    def _set_parameter_state(self, par, value=None, error=None, fixed=None):

        super()._set_parameter_state(par, value, error, fixed)

        if value is not None:
            self.__minuit.values[par.name] = value

        if error is not None:
            self.__minuit.errors[par.name] = error

        if fixed is not None:
            self.__minuit.fixed[par.name] = fixed

    def _update_args(self, params):
        '''
        Update the parameters using the information from the Minuit result.

        :param params: list of parameters.
        :type params: iminuit.util.Params
        '''
        for p in params:
            a = self.__args.get(p.name)
            a.value, a.error = p.value, p.error

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

        :returns: Output from :py:meth:`iminuit.Minuit.hesse`.
        '''
        with self.__args.restoring_state():
            params = self.__minuit.hesse(*args, *kwargs)

        self._update_args(params)

        return params

    @use_const_cache
    def migrad(self, *args, **kwargs):
        '''
        Arguments are forwarded to the :py:meth:`iminuit.Minuit.migrad` function,
        and the values of the parameters are set to those from the minimization
        result.

        :returns: output from :py:meth:`iminuit.Minuit.migrad`.

        .. seealso:: :meth:`MinuitMinimizer.minimize`
        '''
        result, params = self.__minuit.migrad(*args, *kwargs)

        self._update_args(params)

        return result, params

    def minimization_profile(self, wa, values, minimization_results=False, minimizer_config=None):
        with self._restore_minuit():
            return super().minimization_profile(wa, values, minimization_results, minimizer_config)

    @use_const_cache
    def minimize(self, *args, **kwargs):
        '''
        Same as :meth:`MinuitMinimizer.migrad`, but offering a common interface
        for all the :class:`Minimizer` objects. It returns a tuple with the
        information whether the minimization succeded and the covariance matrix.

        .. seealso:: :meth:`MinuitMinimizer.migrad`
        '''
        res, _ = self.migrad(*args, **kwargs)

        if res.is_valid:

            cov = data_types.empty_float((len(self.__args), len(self.__args)))

            non_fixed = parameters.Registry(
                filter(lambda p: not p.constant, self.__args))

            for i, p1 in enumerate(non_fixed):
                for j, p2 in enumerate(non_fixed[i:]):
                    cov[i][j] = cov[j][i] = self.__minuit.covariance[(
                        p1.name, p2.name)]
        else:
            cov = None

        return core.MinimizationResult(res.is_valid, res.fval, cov)

    @use_const_cache
    def minos(self, *args, **kwargs):
        '''
        Arguments are forwarded to the :py:meth:`iminuit.Minuit.minos` function,
        and the values of the parameters are set to those of the MINOS result.
        If a new minimum is found, the value of the parameter is set accordingly.

        :returns: output from :py:meth:`iminuit.Minuit.minos`.

        .. seealso:: :meth:`MinuitMinimizer.asymmetric_errors`
        '''
        with self.__args.restoring_state():  # to preserve the information of the minimum
            result = self.__minuit.minos(*args, *kwargs)

        for k, v in result.items():
            a = self.__args.get(k)
            a.value = v.min
            a.asym_errors = abs(v.lower), v.upper  # lower error is negative

        return result

    def minos_profile(self, *args, **kwargs):
        '''
        Arguments are forwarded to the :py:meth:`iminuit.Minuit.mnprofile` function.
        '''
        return self.__minuit.mnprofile(*args, **kwargs)

    @contextlib.contextmanager
    def restoring_state(self):
        '''
        Method to ensure that modifications of parameters within a minimizer
        context are reset properly. Sadly, the :class:`iminuit.Minuit` class
        is not stateless, so each time a parameter is modified it must be
        notified of the change.

        .. warning:: This does not preserve the minimization state of MIGRAD.
        '''
        with super().restoring_state():  # restores the parameters
            values, errors, fixed = zip(
                *[(p.value, p.error, p.constant) for p in self.evaluator.args])
            yield self
            for p, v, e, f in zip(self.evaluator.args, values, errors, fixed):
                self.__minuit.values[p.name] = v
                # 0 for Minuit, None for MinKit
                self.__minuit.errors[p.name] = e or 0.
                self.__minuit.fixed[p.name] = f
