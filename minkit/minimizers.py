'''
API for minimizers.
'''
from . import dataset
from . import fcns
from . import parameters
from . import pdf_core
from .operations import types

import collections
import contextlib
import functools
import iminuit
import logging
import numdifftools
import numpy as np
import scipy.optimize as scipyopt
import uncertainties
import warnings

__all__ = ['Category', 'minimizer', 'minuit_to_registry',
           'simultaneous_minimizer', 'SciPyMinimizer']

logger = logging.getLogger(__name__)

# Names for the minimizers
MINUIT = 'minuit'
SCIPY = 'scipy'

# Choices and default method to minimize with SciPy
SCIPY_CHOICES = ('L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr')
SCIPY_DEFAULT = SCIPY_CHOICES[0]

# Object to help users to create simultaneous minimizers.
Category = collections.namedtuple('Category', ['fcn', 'pdf', 'data'])
Category.__doc__ = '''\
Object serving as a proxy for an FCN to be evaluated using a PDF on a data set.
The type of data (binned/unbinned) is assumed from the FCN.'''


class BinnedEvaluator(object):
    '''
    Definition of a proxy class to evaluate an FCN with a PDF on a BinnedDataSet object.
    '''

    def __init__(self, fcn, pdf, data, constraints=None):
        '''
        :param fcn: FCN to be used during minimization.
        :type fcn: str
        :param pdf: PDF to minimize.
        :type pdf: PDF
        :param data: data sample to process.
        :type data: BinnedDataSet
        :param constraints: set of constraints to consider in the minimization.
        :type constraints: list(PDF)
        '''
        self.__data = data
        self.__fcn = fcn
        self.__pdf = pdf
        self.__constraints = constraints or []

        super(BinnedEvaluator, self).__init__()

    def __call__(self, *values):
        '''
        Evaluate the FCN.
        Values must be provided sorted as :meth:`PDF.args`.

        :param values: set of values to evaluate the FCN.
        :type values: tuple(float)
        :returns: value of the FCN.
        :rtype: float
        '''
        for i, p in enumerate(self.args):
            p.value = values[i]
        return self.__fcn(self.__pdf, self.__data, self.__constraints)

    @property
    def args(self):
        '''
        Return all the arguments of the PDF.

        :returns: all the arguments of the PDF.
        :rtype: Registry
        '''
        return self.__pdf.all_real_args


class UnbinnedEvaluator(object):
    '''
    Definition of a proxy class to evaluate an FCN with a PDF.
    '''

    def __init__(self, fcn, pdf, data, range=parameters.FULL, constraints=None, rescale_weights=True):
        '''
        :param fcn: FCN to be used during minimization.
        :type fcn: str
        :param pdf: PDF to minimize.
        :type pdf: PDF
        :param data: data sample to process.
        :type data: DataSet
        :param range: range of data to minimize.
        :type range: str
        :param constraints: set of constraints to consider in the minimization.
        :type constraints: list(PDF)
        :param rescale_weights: whether to rescale the weights, so the statistical power remains constant.
        :type rescale_weights: bool
        '''
        if data.weights is not None:
            if rescale_weights:
                logger.info('Rescaling weights for the fit')
                self.__data = data.subset(
                    range=range, copy=False, rescale_weights=rescale_weights, trim=True)
        else:
            self.__data = data.subset(range=range, copy=False, trim=True)

        self.__fcn = fcn
        self.__pdf = pdf
        self.__range = range
        self.__constraints = constraints or []

        super(UnbinnedEvaluator, self).__init__()

    def __call__(self, *values):
        '''
        Evaluate the FCN.
        Values must be provided sorted as :meth:`PDF.args`.

        :param values: set of values to evaluate the FCN.
        :type values: tuple(float)
        :returns: value of the FCN.
        :rtype: float
        '''
        for i, p in enumerate(self.args):
            p.value = values[i]
        return self.__fcn(self.__pdf, self.__data, self.__range, self.__constraints)

    @property
    def args(self):
        '''
        Return all the arguments of the PDF.

        :returns: all the arguments of the PDF.
        :rtype: Registry
        '''
        return self.__pdf.all_real_args


class SimultaneousEvaluator(object):
    '''
    Definition of an evaluator of PDFs for simultaneous fits.
    '''

    def __init__(self, evaluators):
        '''
        Build an object to evaluate PDFs on independent data samples.
        This class is not meant to be used by users.

        :param evaluators: list of evaluators to use.
        :type evaluators: list(UnbinnedEvaluator or BinnedEvaluator)
        '''
        self.__evaluators = evaluators

        super(SimultaneousEvaluator, self).__init__()

    def __call__(self, *values):
        '''
        Call each PDF in the corresponding data sample.

        :param data: data samples to process.
        :type data: list(DataSet or BinnedDataSet)
        :param args: forwarded to :meth:`PDF.__call__`
        :type args: tuple
        :param kwargs: forwarded to :meth:`PDF.__call__`
        :type kwargs: dict
        '''
        args = self.args
        sfcn = 0.
        for e in self.__evaluators:
            sfcn += e.__call__(*(values[args.index(a.name)] for a in e.args))
        return sfcn

    @property
    def args(self):
        '''
        Get the arguments for the evaluator.
        '''
        args = parameters.Registry(self.__evaluators[0].args)
        for e in self.__evaluators[1:]:
            args += e.args
        return args


def minuit_to_registry(params):
    '''
    Transform the parameters from a call to Minuit into a :class:`Registry`.

    :param result: result from a migrad call.
    :type result: iminuit.util.Params
    :returns: registry of parameters with the result from Migrad.
    :rtype: Registry(str, Parameter)
    '''
    reg = parameters.Registry()
    for p in params:
        limits = (p.lower_limit, p.upper_limit) if p.has_limits else None
        reg.append(parameters.Parameter(
            p.name, p.value, bounds=limits, error=p.error, constant=p.is_fixed))
    return reg


def registry_to_minuit_input(registry, errordef=fcns.ERRORDEF):
    '''
    Transform a registry of parameters into a dictionary to be parsed by Minuit.

    :param registry: registry of parameters.
    :type registry: Registry(Parameter)
    :param errordef: error definition for Minuit.
    :type errorder: float
    :returns: Minuit configuration dictionary.
    :rtype: dict
    '''
    values = {v.name: v.value for v in registry}
    errors = {f'error_{v.name}': v.error for v in registry}
    limits = {f'limit_{v.name}': v.bounds for v in registry}
    const = {f'fix_{v.name}': v.constant for v in registry}
    return dict(errordef=errordef, **values, **errors, **limits, **const)


class SciPyMinimizer(object):

    def __init__(self, evaluator):
        '''
        Wrapper around the :func:`scipy.optimize.minimize` function.

        :param evaluator: evaluator to be used in the minimization.
        :type evaluator: UnbinnedEvaluator, BinnedEvaluator or SimultaneousEvaluator
        '''
        self.__eval = evaluator
        self.__varids = []
        self.__values = np.empty(len(evaluator.args), dtype=types.cpu_type)

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


@contextlib.contextmanager
def minimizer(fcn, pdf, data, minimizer=MINUIT, minimizer_config=None, **kwargs):
    '''
    Create a new minimizer to be used within a context.
    This represents a "constant" object, that is, parameters defining
    the PDFs must keep their constness during the context.

    :param fcn: type of FCN to use for the minimization.
    :type fcn: str
    :param pdf: function to minimize.
    :type pdf: PDF
    :param data: data set to use.
    :type data: UnbinnedDataSet or BinnedDataSet
    :param minimizer: name of the minimizer to use.
    :type minimizer: str
    :param minimizer_config: any extra configuration to be passed to the minimizer. For \
    "minuit", the argument "forced_parameters" is unavailable.
    :type minimizer_config: dict
    :param kwargs: extra arguments to the evaluator to use. Current allowed values are: \
    * constraints: (:class:`list`(:class:`PDF`)) constraints for the FCN.\
    * rescale_weights: (:class:`bool`) (unbinned data sets only) whether to rescale the weights, so the \
    statistical power remains constant.
    :type kwargs: dict

    .. warning: Do not change any attribute of the parameters defining the \
    PDFs, since it will not be properly reflected during the minimization \
    calls.
    '''
    pdf.enable_cache(pdf_core.PDF.CONST)  # Enable the cache

    mf = fcns.fcn_from_name(fcn)

    if fcns.data_type_for_fcn(fcn) == dataset.BINNED:
        evaluator = BinnedEvaluator(mf, pdf, data, **kwargs)
    else:
        evaluator = UnbinnedEvaluator(mf, pdf, data, **kwargs)

    minimizer_config = minimizer_config or {}

    if minimizer == MINUIT:
        yield iminuit.Minuit(evaluator,
                             forced_parameters=pdf.all_real_args.names,
                             **minimizer_config,
                             **registry_to_minuit_input(pdf.all_real_args))
    elif minimizer == SCIPY:
        yield SciPyMinimizer(evaluator)
    else:
        raise ValueError(f'Unknown minimizer "{minimizer}"')

    pdf.free_cache()  # Very important


@contextlib.contextmanager
def simultaneous_minimizer(categories, minimizer=MINUIT, minimizer_config=None, rescale_weights=True, range=parameters.FULL, constraints=None):
    '''
    Create a new instance of :class:`iminuit.Minuit`.
    This represents a "constant" object, that is, parameters defining
    the PDFs are assumed to remain constant during all its lifetime.

    :param categories: categories to process.
    :type categories: list(Category)
    :param minimizer: minimizer to use (only "minuit" is available for the moment).
    :type minimizer: str
    :param minimizer_config: (Minuit only) extra configuration passed to the minimizer.
    :type minimizer_config: dict
    :param range: range of data to minimize.
    :type range: str
    :param constraints: set of constraints to consider in the minimization.
    :type constraints: list(PDF)
    :returns: minimizer to call.
    :rtype: depends on the "minimizer" argument

    .. warning: Do not change any attribute of the parameters defining the \
    PDFs, since it will not be properly reflected during the minimization \
    calls.
    '''
    # Build the simultaneous evaluator
    evaluators = []
    for cat in categories:

        cat.pdf.enable_cache(pdf_core.PDF.CONST)  # Enable the cache

        fcn = fcns.fcn_from_name(cat.fcn)

        if fcns.data_type_for_fcn(cat.fcn) == dataset.BINNED:
            e = BinnedEvaluator(fcn, cat.pdf, cat.data, constraints)
        else:
            e = UnbinnedEvaluator(
                fcn, cat.pdf, cat.data, range, constraints, rescale_weights)

        evaluators.append(e)

    evaluator = SimultaneousEvaluator(evaluators)

    minimizer_config = minimizer_config or {}

    # Return the minimizer
    if minimizer == MINUIT:
        yield iminuit.Minuit(evaluator,
                             forced_parameters=evaluator.args.names,
                             pedantic=False,
                             **minimizer_config,
                             **registry_to_minuit_input(evaluator.args))
    elif minimizer == SCIPY:
        yield SciPyMinimizer(evaluator)
    else:
        raise ValueError(f'Unknown minimizer "{minimizer}"')

    for c in categories:
        c.pdf.free_cache()  # Very important
