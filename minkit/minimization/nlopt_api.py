########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Definition of the interface functions and classes with :mod:`nlopt`.
'''
from . import core

import nlopt

__all__ = ['NLoptMinimizer']

# Choices and default method to minimize with NLopt
NLOPT_DICT = {
    'COBYLA': nlopt.LN_COBYLA,
    'BOBYQA': nlopt.LN_BOBYQA,
    'NEWUOA': nlopt.LN_NEWUOA,
    'PRAXIS': nlopt.LN_PRAXIS,
    'NELDERMEAD': nlopt.LN_NELDERMEAD,
    'SBPLX': nlopt.LN_SBPLX,
}
NLOPT_CHOICES = tuple(NLOPT_DICT.keys())


class NLoptMinimizer(core.Minimizer):

    def __init__(self, method, evaluator, **minimizer_config):
        '''
        Interface with the :mod:`nlopt` minimizers.

        :param method: method name in :mod:`nlopt`.
        :type method: str
        :param evaluator: evaluator to be used in the minimization.
        :type evaluator: UnbinnedEvaluator, BinnedEvaluator or SimultaneousEvaluator
        :param minimizer_config: configuration for the minimizer.
        :type minimizer_config: dict
        :raises ValueError: If the minimization method is unknown.
        '''
        super().__init__(evaluator)

        if method not in NLOPT_CHOICES:
            raise ValueError(f'Unknown minimization method "{method}"')

        self.__method = method

    def minimize(self, *args, **kwargs):
        '''
        Minimize the FCN for the stored PDF and data sample. It returns a tuple
        with the information whether the minimization succeded and the
        covariance matrix.

        .. seealso:: :meth:`NLoptMinimizer.nlopt_minimize`
        '''
        status, fcn, cov = self.nlopt_minimize(*args, **kwargs)
        return core.MinimizationResult(status, fcn, cov)

    def nlopt_minimize(self, tol=1e-7, hessian_opts=None):
        '''
        Minimize the PDF.

        :param tol: tolerance to be used in the minimization.
        :type tol: float
        :param hessian_opts: options to be passed to :class:`numdifftools.core.Hessian`.
        :type hessian_opts: dict or None
        :returns: Result of the minimization, covariance matrix and FCN at the minimum.
        :rtype: int, numpy.ndarray, float

        .. seealso:: :meth:`NLoptMinimizer.minimize`
        '''
        varargs, values, varids = core.determine_varargs_values_varids(
            self.evaluator.args)

        initials = tuple(a.value for a in varargs)

        def _evaluate(*args):  # set the non-constant argument values
            values[varids] = args
            return self.evaluator(*values)

        # Build and call the minimizer
        minimizer = nlopt.opt(NLOPT_DICT[self.__method], len(varargs))
        minimizer.set_lower_bounds([p.bounds[0] for p in varargs])
        minimizer.set_upper_bounds([p.bounds[1] for p in varargs])
        minimizer.set_min_objective(lambda args, grad: _evaluate(*args))

        minimizer.set_xtol_rel(tol)

        with self.evaluator.using_caches():
            result = minimizer.optimize(initials)

        fcn = minimizer.last_optimum_value()

        with self.restoring_state():
            values, cov = core.errors_and_covariance_matrix(_evaluate, result)

        # Update the values and errors of the parameters
        for p, v in zip(varargs, values):
            p.value, p.error = v.nominal_value, v.std_dev

        status = minimizer.last_optimize_result() > 0  # positive values are OK

        return status, fcn, cov
