'''
Definition of the interface functions and classes with :mod:`scipy`.
'''
from . import core
from ..base import data_types
from ..base import parameters

import scipy.optimize as scipyopt

import numdifftools
import numpy as np
import uncertainties
import warnings

__all__ = ['SciPyMinimizer']

# Choices and default method to minimize with SciPy
SCIPY_CHOICES = ('L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr')
SCIPY_DEFAULT = SCIPY_CHOICES[0]


class SciPyMinimizer(core.Minimizer):

    def __init__(self, evaluator, **minimizer_config):
        '''
        Interface with the :func:`scipy.optimize.minimize` function.

        :param evaluator: evaluator to be used in the minimization.
        :type evaluator: UnbinnedEvaluator, BinnedEvaluator or SimultaneousEvaluator
        '''
        super().__init__()

        self.__eval = evaluator
        self.__varids = []
        self.__values = data_types.empty_float(len(evaluator.args))

        for i, a in enumerate(evaluator.args):
            if a.constant:
                self.__values[i] = a.value
            else:
                self.__varids.append(i)

    def _evaluate(self, *args):
        '''
        Evaluate the FCN, parsing the values provided by SciPy.

        :param args: arguments from SciPy.
        :type args: tuple(float, ...)
        :returns: value of the FCN.
        :rtype: float
        '''
        self.__values[self.__varids] = args
        return self.__eval(*self.__values)

    @property
    def evaluator(self):
        '''
        Evaluator of the minimizer.
        '''
        return self.__eval

    def minimize(self, method=SCIPY_DEFAULT, tol=None):
        '''
        Minimize the PDF using the provided method and tolerance.
        Only the methods ('L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr') are allowed.

        :param method: method parsed by :func:`scipy.optimize.minimize`.
        :type method: str
        :param tol: tolerance to be used in the minimization.
        :type tol: float
        :returns: result of the minimization.
        :rtype: scipy.optimize.OptimizeResult
        '''
        if method not in SCIPY_CHOICES:
            raise ValueError(
                f'Unknown minimization method "{method}"; choose from {SCIPY_CHOICES}')

        varargs = parameters.Registry(
            filter(lambda v: not v.constant, self.__eval.args))

        initials = tuple(a.value for a in varargs)

        bounds = tuple(a.bounds for a in varargs)

        with self.__eval.using_caches(), warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, module=r'.*_hessian_update_strategy')
            warnings.filterwarnings('once', category=RuntimeWarning)
            return scipyopt.minimize(self._evaluate, initials, method=method, bounds=bounds, tol=tol)

    def result_to_registry(self, result, ignore_warnings=True, **kwargs):
        '''
        Transform the output of a minimization call done with any of the SciPy methods
        to a :class:`minkit.Registry`.
        This function uses :class:`numdifftools.Hessian` in order to calculate the Hessian
        matrix of the FCN.
        Uncertainties are extracted from the inverse of the Hessian, taking into account
        the correlation among the variables.

        :param result: result of the minimization.
        :type result: scipy.optimize.OptimizeResult
        :param ignore_warnings: whether to ignore the warnings during the evaluation \
        of the Hessian or not.
        :type ignore_warnings: bool
        :param kwargs: keyword arguments to :class:`numdifftools.Hessian`
        :type kwargs: dict
        :returns: registry with new parameters with the values and errors defined.
        :rtype: Registry
        '''
        # Disable warnings, since "numdifftools" does not allow to set bounds
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hessian = numdifftools.Hessian(
                lambda a: self._evaluate(*a), **kwargs)
            cov = 2. * np.linalg.inv(hessian(result.x))

        values = uncertainties.correlated_values(result.x, cov)

        reg = parameters.Registry()
        for i, p in enumerate(self.__eval.args):
            if i in self.__varids:
                reg.append(parameters.Parameter(
                    p.name, values[i].nominal_value, bounds=p.bounds, error=values[i].std_dev, constant=p.constant))
            else:
                reg.append(parameters.Parameter(
                    p.name, p.value, bounds=p.bounds, error=p.error, constant=p.constant))

        return reg
