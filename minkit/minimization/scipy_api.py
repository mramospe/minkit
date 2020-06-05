########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
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
        super().__init__(evaluator)

    def minimize(self, *args, **kwargs):
        '''
        Minimize the FCN for the stored PDF and data sample. It returns a tuple
        with the information whether the minimization succeded and the
        covariance matrix.

        .. seealso:: scipy_minimize
        '''
        res, cov = self.scipy_minimize(*args, **kwargs)
        return core.MinimizationResult(res.success, res.fun, cov)

    def scipy_minimize(self, method=SCIPY_DEFAULT, tol=None, hessian_opts=None):
        '''
        Minimize the PDF using the provided method and tolerance.
        Only the methods ('L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr') are allowed.

        :param method: method parsed by :func:`scipy.optimize.minimize`.
        :type method: str
        :param tol: tolerance to be used in the minimization.
        :type tol: float
        :param hessian_opts: options to be passed to :meth:`numdifftools.core.Hessian`.
        :type hessian_opts: dict
        :returns: result of the minimization and covariance matrix.
        :rtype: scipy.optimize.OptimizeResult, numpy.ndarray
        '''
        if method not in SCIPY_CHOICES:
            raise ValueError(
                f'Unknown minimization method "{method}"; choose from {SCIPY_CHOICES}')

        hessian_opts = hessian_opts or {}

        varargs = parameters.Registry(
            filter(lambda v: not v.constant, self.evaluator.args))

        initials = tuple(a.value for a in varargs)

        bounds = tuple(a.bounds for a in varargs)

        # We must create an array with all the values
        varids = []
        values = data_types.empty_float(len(self.evaluator.args))

        for i, a in enumerate(self.evaluator.args):
            if a.constant:
                values[i] = a.value
            else:
                varids.append(i)

        def _evaluate(*args):  # set the non-constant argument values
            values[varids] = args
            return self.evaluator(*values)

        with self.evaluator.using_caches(), warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, module=r'.*_hessian_update_strategy')
            warnings.filterwarnings('once', category=RuntimeWarning)
            result = scipyopt.minimize(
                _evaluate, initials, method=method, bounds=bounds, tol=tol)

        # Disable warnings, since "numdifftools" does not allow to set bounds
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hessian = numdifftools.Hessian(
                lambda a: _evaluate(*a), **hessian_opts)
            cov = 2. * np.linalg.inv(hessian(result.x))

        values = uncertainties.correlated_values(result.x, cov)

        # Update the values and errors of the parameters
        for i, p in enumerate(self.evaluator.args):
            if i in varids:
                p.value, p.error = values[i].nominal_value, values[i].std_dev

        return result, cov
