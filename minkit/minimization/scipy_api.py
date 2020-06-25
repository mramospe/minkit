########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Definition of the interface functions and classes with :mod:`scipy`.
'''
from . import core

import scipy.optimize as scipyopt
import warnings

__all__ = ['SciPyMinimizer']

# Choices and default method to minimize with SciPy
SCIPY_CHOICES = ('L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr')


class SciPyMinimizer(core.Minimizer):

    def __init__(self, method, evaluator, **minimizer_config):
        '''
        Interface with the :func:`scipy.optimize.minimize` function.

        :param method: method parsed by :func:`scipy.optimize.minimize`.
        :type method: str
        :param evaluator: evaluator to be used in the minimization.
        :type evaluator: UnbinnedEvaluator, BinnedEvaluator or SimultaneousEvaluator
        :param minimizer_config: configuration for the minimizer.
        :type minimizer_config: dict
        :raises ValueError: If the minimization method is unknown.
        '''
        super().__init__(evaluator)

        if method not in SCIPY_CHOICES:
            raise ValueError(f'Unknown minimization method "{method}"')

        self.__method = method

    def minimize(self, *args, **kwargs):
        '''
        Minimize the FCN for the stored PDF and data sample. It returns a tuple
        with the information whether the minimization succeded and the
        covariance matrix.

        .. seealso:: :meth:`SciPyMinimizer.scipy_minimize`
        '''
        res, cov = self.scipy_minimize(*args, **kwargs)
        return core.MinimizationResult(res.success, res.fun, cov)

    def scipy_minimize(self, tol=None, hessian_opts=None):
        '''
        Minimize the PDF.

        :param tol: tolerance to be used in the minimization.
        :type tol: float or None
        :param hessian_opts: options to be passed to :class:`numdifftools.core.Hessian`.
        :type hessian_opts: dict or None
        :returns: Result of the minimization and covariance matrix.
        :rtype: scipy.optimize.OptimizeResult, numpy.ndarray

        .. seealso:: :meth:`SciPyMinimizer.minimize`
        '''
        varargs, values, varids = core.determine_varargs_values_varids(
            self.evaluator.args)

        initials = tuple(a.value for a in varargs)

        def _evaluate(*args):  # set the non-constant argument values
            values[varids] = args
            return self.evaluator(*values)

        bounds = [p.bounds for p in varargs]

        with self.evaluator.using_caches(), warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, module=r'.*_hessian_update_strategy')
            warnings.filterwarnings('once', category=RuntimeWarning)
            result = scipyopt.minimize(
                _evaluate, initials, method=self.__method, bounds=bounds, tol=tol)

        with self.restoring_state():
            errors, cov = core.errors_and_covariance_matrix(
                _evaluate, result.x)

        # Update the values and errors of the parameters
        for p, v, e in zip(varargs, result.x, errors):
            p.value, p.error = v, e

        return result, cov
