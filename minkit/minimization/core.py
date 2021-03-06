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
import numdifftools
import numpy as np
import warnings

DEFAULT_ASYM_ERROR_ATOL = 1e-8  # same as numpy.allclose
DEFAULT_ASYM_ERROR_RTOL = 1e-5  # same as numpy.allclose


__all__ = ['Minimizer']


MinimizationResult = collections.namedtuple(
    'MinimizationResult', ['valid', 'fcn', 'cov'])


def errors_and_covariance_matrix(evaluate, result, hessian_opts=None):
    '''
    Calculate the covariance matrix given a function to evaluate the FCN
    and the values of the parameters at the minimum.

    :param evaluate: function used to evaluate the FCN. It must take all the
       parameters that are not constant.
    :param result: values at the FCN minimum.
    :type result: numpy.ndarray
    :param hessian_opts: options to be passed to :class:`numdifftools.core.Hessian`.
    :type hessian_opts: dict or None
    :returns: values with the errors and covariance matrix.
    :rtype: numpy.ndarray(uncertainties.core.Variable), numpy.ndarray
    '''
    hessian_opts = hessian_opts or {}

    # Disable warnings, since "numdifftools" does not allow to set bounds
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hessian = numdifftools.Hessian(
            lambda a: evaluate(*a), **hessian_opts)
        cov = 2. * np.linalg.inv(hessian(result))

    errors = np.sqrt(np.diag(cov))

    return errors, cov


def determine_varargs_values_varids(args):
    '''
    Check the arguments that are constant and create three collections: one
    containing the arguments that are constant, the values of these parameters
    and the position in the list.

    :param args: collection of arguments.
    :type args: Registry(Parameter)
    :returns: constant parameters, values and indices in the registry.
    :rtype: Registry(Parameter), numpy.ndarray, numpy.ndarray
    '''
    varargs = parameters.Registry(filter(lambda v: not v.constant, args))

    # We must create an array with all the values
    varids = []
    values = data_types.empty_float(len(args))

    for i, a in enumerate(args):
        if a.constant:
            values[i] = a.value
        else:
            varids.append(i)

    return varargs, values, varids


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
    :raises RuntimeError: If the number of provided sets of values is differet
       to the number of profile parameters.
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
        Abstract class to serve as an API between MinKit and the different
        minimization methods.

        :param evaluator: evaluator to be used in the minimization.
        :type evaluator: UnbinnedEvaluator, BinnedEvaluator or SimultaneousEvaluator
        '''
        super().__init__()

        self.__eval = evaluator

    def _asym_error(self, par, bound, var=1, atol=DEFAULT_ASYM_ERROR_ATOL, rtol=DEFAULT_ASYM_ERROR_RTOL, max_call=None):
        '''
        Calculate the asymmetric error using the variation of the FCN from
        *value* to *bound*.

        :param par: parameter to calculate the asymmetric errors.
        :type par: float
        :param bound: bound of the parameter.
        :type bound: float
        :param var: squared number of standard deviations.
        :type var: float
        :param atol: absolute tolerance for the error.
        :type atol: float
        :param rtol: relative tolerance for the error.
        :type rtol: float
        :param max_call: maximum number of calls to calculate each error bound.
        :type max_call: int or None
        :returns: Absolute value of the error.
        :rtype: float
        '''
        with self.restoring_state():

            initial = par.value

            l, r = par.value, bound  # consider the minimum on the left

            fcn_l = ref_fcn = self.__eval.fcn()  # it must have been minimized

            self._set_parameter_state(par, value=bound, fixed=True)

            fcn_r = self._minimize_check_minimum(par)

            if np.allclose(fcn_r - ref_fcn, var, atol=atol, rtol=rtol):
                return abs(par.value - bound)

            closest_fcn = fcn_r

            i = 0
            while (True if max_call is None else i < max_call) and not np.allclose(abs(closest_fcn - ref_fcn), var, atol=atol, rtol=rtol):

                i += 1  # increase the internal counter (for max_call)

                self._set_parameter_state(par, value=0.5 * (l + r))

                fcn = self._minimize_check_minimum(par)

                if abs(fcn - ref_fcn) < var:
                    l, fcn_l = par.value, fcn
                else:
                    r, fcn_r = par.value, fcn

                if var - (fcn_l - ref_fcn) < (fcn_r - ref_fcn) - var:
                    bound, closest_fcn = l, fcn_l
                else:
                    bound, closest_fcn = r, fcn_r

            if max_call is not None and i == max_call:
                warnings.warn(
                    'Reached maximum number of minimization calls', RuntimeWarning, stacklevel=1)

            return abs(initial - bound)

    def _minimize_check_minimum(self, par):
        '''
        Check the minimum of a minimization and warn if it is not valid.

        :param par: parameter to work with.
        :type par: Parameter
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

    def asymmetric_errors(self, name, sigma=1, atol=DEFAULT_ASYM_ERROR_ATOL, rtol=DEFAULT_ASYM_ERROR_RTOL, max_call=None):
        '''
        Calculate the asymmetric errors for the given parameter. This is done
        by subdividing the bounds of the parameter into two till the variation
        of the FCN is one. Unlike MINOS, this method does not treat new
        minima. Remember that the PDF must have been minimized before a call
        to this function.

        :param name: name of the parameter.
        :type name: str
        :param sigma: number of standard deviations to compute.
        :type sigma: float
        :param atol: absolute tolerance for the error.
        :type atol: float
        :param rtol: relative tolerance for the error.
        :type rtol: float
        :param max_call: maximum number of calls to calculate each error bound.
        :type max_call: int or None
        '''
        par = self.__eval.args.get(name)

        lb, ub = par.bounds

        var = sigma * sigma

        lo = self._asym_error(par, lb, var, atol, rtol, max_call)
        hi = self._asym_error(par, ub, var, atol, rtol, max_call)

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
                        self._set_parameter_state(p, value=v)
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

    def minimization_profile(self, wa, values, minimization_results=False, minimizer_config=None):
        '''
        Minimize a PDF an calculate the FCN for each set of parameters and values.

        :param wa: single variable or set of variables.
        :type wa: str or list(str).
        :param values: values for each parameter specified in *wa*.
        :type values: numpy.ndarray
        :param minimization_results: if set to True, then the results for each
           step are returned.
        :type minimization_results: bool
        :param minimizer_config: arguments passed to :meth:`Minimizer.minimize`.
        :type minimizer_config: dict or None
        :returns: Profile of the FCN for the given values.
        :rtype: numpy.ndarray, (list(MinimizationResult))
        '''
        profile_pars, values = _process_parameters(
            self.__eval.args, wa, values)

        fcn_values = data_types.empty_float(len(values[0]))

        minimizer_config = minimizer_config or {}

        results = []

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

                results.append(res)

        if minimization_results:
            return fcn_values, res
        else:
            return fcn_values

    @contextlib.contextmanager
    def restoring_state(self):
        '''
        Method to ensure that modifications of parameters within a minimizer
        context are reset properly.

        .. seealso:: :meth:`MinuitMinimizer.restoring_state`, :meth:`SciPyMinimizer.restoring_state`
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
