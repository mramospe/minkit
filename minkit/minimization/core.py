########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Basic functions and classes to do minimizations.
'''
from ..base import data_types
from ..base import exceptions
from ..base import parameters
from ..base.core import DocMeta

import collections
import contextlib
import numpy as np
import warnings

DEFAULT_ASYM_ERROR_ATOL = 1e-8  # same as numpy.allclose
DEFAULT_ASYM_ERROR_RTOL = 1e-5  # same as numpy.allclose


__all__ = ['Minimizer']


MinimizationResult = collections.namedtuple(
    'MinimizationResult', ['valid', 'fcn', 'cov'])


def _process_parameters(args, wa, values):
    '''
    Process an input set of parameters, converting it to a :class:`Registry`,
    if only one parameter is provided.

    :param args: whole set of parameters.
    :type args: Registry
    :param wa: input arguments to process.
    :type wa: str or list(str)
    :param values: values to calculate a profile.
    :type values: numpy.ndarray
    :returns: registry of parameters and values.
    :rtype: Registry, numpy.ndarray
    '''
    values = np.asarray(values)

    if np.asarray(wa).ndim == 0:
        wa = [wa]
        values = np.array([values])

    if len(wa) != len(values):
        raise RuntimeError(
            'Length of the profile values must coincide with profile parameters')

    pars = parameters.Registry(args.get(n) for n in wa)

    return pars, values


class Minimizer(object, metaclass=DocMeta):

    def __init__(self, evaluator):
        '''
        Abstract class to serve as an API between Minkit and the different
        minimization methods.

        :param evaluator: evaluator to be used in the minimization.
        :type evaluator: UnbinnedEvaluator, BinnedEvaluator or SimultaneousEvaluator
        '''
        super().__init__()

        self.__eval = evaluator

    def _asym_error(self, par, bound, cov, nsigma=1, atol=DEFAULT_ASYM_ERROR_ATOL, rtol=DEFAULT_ASYM_ERROR_RTOL, maxcall=None):
        '''
        Calculate the asymmetric error using the variation of the FCN from
        *value* to *bound*.

        :param par: parameter to calculate the asymmetric errors.
        :type par: float
        :param bound: bound of the parameter.
        :type bound: float
        :param cov: covariance matrix. If provided, the initial values of the \
        parameters will be obtained from them.
        :type cov: numpy.ndarray
        :param atol: absolute tolerance for the error.
        :type atol: float
        :param rtol: relative tolerance for the error.
        :type rtol: float
        :param maxcall: maximum number of calls to calculate each error bound.
        :type maxcall: int or None
        :returns: Absolute value of the error.
        :rtype: float
        '''
        with self.restoring_state():

            initial = par.value

            l, r = par.value, bound  # consider the minimum on the left

            fcn_l = ref_fcn = self._minimize_check_minimum(par)

            self._set_parameter_state(par, bound, fixed=True)

            if cov is not None:  # set initial values using the covariance matrix
                s = data_types.empty_float(len(self.__eval.args))
                s[self.__eval.args.index(par.name)] = bound
                vals = np.matmul(cov, s)
                for p, v in zip(self.__eval.args, vals):
                    self._set_parameter_state(p, v)

            fcn_r = self._minimize_check_minimum(par)

            if fcn_r - ref_fcn < nsigma:
                raise RuntimeError(
                    'Parameter bounds are smaller than the provided number of standard deviations')
            elif fcn_r - ref_fcn == nsigma:
                return abs(par.value - bound)

            closest_fcn = fcn_r

            i = 0
            while not np.allclose(abs(closest_fcn - ref_fcn), nsigma, atol=atol, rtol=rtol) and (True if maxcall is None else i < maxcall):

                i += 1  # increase the internal counter (for maxcall)

                self._set_parameter_state(par, 0.5 * (l + r))

                fcn = self._minimize_check_minimum(par)

                if abs(fcn - ref_fcn) < 1:
                    l, fcn_l = par.value, fcn
                else:
                    r, fcn_r = par.value, fcn

                if nsigma - (fcn_l - ref_fcn) < (fcn_r - ref_fcn) - nsigma:
                    bound, closest_fcn = l, fcn_l
                else:
                    bound, closest_fcn = r, fcn_r

            if maxcall is not None and i > maxcall:
                warnings.warn(
                    'Reached maximum number of minimization calls', RuntimeWarning, stacklevel=1)

            return abs(initial - bound)

    def _minimize_check_minimum(self, *pars):
        '''
        Check the minimum of a minimization and warn if it is not valid.

        :returns: Value of the FCN.
        :rtype: float
        '''
        m = self.minimize()
        if not m.valid:
            warnings.warn('Minimum is not valid during FCN scan',
                          RuntimeWarning, stacklevel=2)
        return m.fcn

    def _set_parameter_state(self, par, value=None, error=None, fixed=None):
        '''
        Set the state of the parameter.

        :param par: parameter to modify.
        :type par: Parameter
        :param value: new value of a parameter.
        :type value: float or None
        :param error: error of the parameter.
        :type error: float or None
        :param fixed: whether to fix the parameter.
        :type fixed: bool or None
        '''
        if value is not None:
            par.value = value
        if error is not None:
            par.error = error
        if fixed is not None:
            par.constant = fixed

    @property
    def evaluator(self):
        '''
        Evaluator of the minimizer.
        '''
        return self.__eval

    def asymmetric_errors(self, name, cov=None, nsigma=1, atol=DEFAULT_ASYM_ERROR_ATOL, rtol=DEFAULT_ASYM_ERROR_RTOL, maxcall=None):
        '''
        Calculate the asymmetric errors for the given parameter. This is done
        by subdividing the bounds of the parameter into two till the variation
        of the FCN is one. Unlike MINOS, this method does not treat new
        minima.

        :param name: name of the parameter.
        :type name: str
        :param cov: covariance matrix. If provided, the initial values of the \
        parameters will be obtained from them.
        :type cov: numpy.ndarray
        :param nsigma: number of standard deviations to compute.
        :type nsigma: float
        :param atol: absolute tolerance for the error.
        :type atol: float
        :param rtol: relative tolerance for the error.
        :type rtol: float
        :param maxcall: maximum number of calls to calculate each error bound.
        :type maxcall: int or None
        '''
        par = self.__eval.args.get(name)

        lb, ub = par.bounds

        lo = self._asym_error(par, lb, cov, nsigma, atol, rtol, maxcall)
        hi = self._asym_error(par, ub, cov, nsigma, atol, rtol, maxcall)

        par.asym_errors = lo, hi

    def fcn_profile(self, wa, values):
        '''
        Evaluate the profile of an FCN for a set of parameters and values.

        :param wa: single variable or set of variables.
        :type wa: str or list(str).
        :param values: values for each parameter specified in *wa*.
        :type values: numpy.ndarray
        :returns: Profile of the FCN for the given values.
        :rtype: numpy.ndarray
        '''
        profile_pars, values = _process_parameters(
            self.__eval.args, wa, values)

        fcn_values = data_types.empty_float(len(values[0]))

        with self.restoring_state():

            for p in self.__eval.args:
                if p in profile_pars:
                    self._set_parameter_state(p, fixed=False)
                else:
                    self._set_parameter_state(p, fixed=True)

            with self.__eval.using_caches():
                for i, vals in enumerate(values.T):
                    for p, v in zip(profile_pars, vals):
                        p.value = v
                    fcn_values[i] = self.__eval.fcn()

        return fcn_values

    def minimize(self, *args, **kwargs):
        '''
        Minimize the FCN for the stored PDF and data sample. Arguments depend
        on the specific minimizer to use. It returns a tuple with the
        information whether the minimization succeded and the covariance matrix.
        '''
        raise exceptions.MethodNotDefinedError(
            self.__class__, 'minimize')

    def minimization_profile(self, wa, values, minimizer_config=None):
        '''
        Minimize a PDF an calculate the FCN for each set of parameters and values.

        :param wa: single variable or set of variables.
        :type wa: str or list(str).
        :param values: values for each parameter specified in *wa*.
        :type values: numpy.ndarray
        :param minimizer_config: arguments passed to :meth:`Minimizer.minimize`.
        :type minimizer_config: dict or None
        :returns: Profile of the FCN for the given values.
        :rtype: numpy.ndarray
        '''
        profile_pars, values = _process_parameters(
            self.__eval.args, wa, values)

        fcn_values = data_types.empty_float(len(values[0]))

        minimizer_config = minimizer_config or {}

        with self.restoring_state():

            for p in profile_pars:
                self._set_parameter_state(p, fixed=True)

            for i, vals in enumerate(values.T):
                for p, v in zip(profile_pars, vals):
                    self._set_parameter_state(p, value=v)

                res = self.minimize(**minimizer_config)

                if not res.valid:
                    warnings.warn(
                        'Minimum in FCN scan is not valid', RuntimeWarning)

                fcn_values[i] = res.fcn

        return fcn_values

    @contextlib.contextmanager
    def restoring_state(self):
        '''
        Method to ensure that modifications of parameters within a minimizer
        context are reset properly. Sadly, the :class:`iminuit.Minuit` class
        is not stateless, so each time a parameter is modified it must be
        notified of the change.

        .. warning:: For the :class:`MinuitMinimizer` class, a call to this
           method does not preserve the minimization state of MIGRAD.
        '''
        with self.__eval.args.restoring_state():
            yield self

    def set_parameter_state(self, name, value=None, error=None, fixed=None):
        '''
        Method to ensure that a modification of a parameter within a minimizer
        context is treated properly. Sadly, the :class:`iminuit.Minuit` class
        is not stateless, so each time a parameter is modified it must be
        notified of the change.

        :param name: name of the parameter.
        :type name: str
        :param value: new value of a parameter.
        :type value: float or None
        :param error: error of the parameter.
        :type error: float or None
        :param fixed: whether to fix the parameter.
        :type fixed: bool or None
        '''
        par = self.__eval.args.get(name)
        return self._set_parameter_state(par, value, error, fixed)
