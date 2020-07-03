########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Interface with the GNU scientific library.
'''
from ..base import core
from ..base import data_types
from ..base import exceptions
from ..base.data_types import c_double, c_double_p, py_object

import collections
import warnings

# To store the compiled functions
NumericalIntegration = collections.namedtuple(
    'NumericalIntegration', ['qng', 'qag', 'cquad', 'plain', 'miser', 'vegas'])


class NumInt(object, metaclass=core.DocMeta):

    def __init__(self, pdf, function, rtol, atol):
        '''
        Base class for the classes to do numerical integrals.

        :param pdf: PDF this instance will be related to.
        :param pdf: PDF
        :param function: function to call to integrate a PDF.
        :param rtol: tolerance for the relative error.
        :type rtol: float
        :param atol: tolerance for the integration value.
        :type atol: float
        '''
        super().__init__()

        self.__pdf = pdf  # so we can track the name of the PDF
        self.__function = function
        self.rtol = rtol
        self.atol = atol

    def __call__(self, lb, ub, args):
        '''
        Calculate the integral in the given bounds.

        :param lb: lower bounds.
        :type lb: numpy.ndarray
        :param ub: upper bounds.
        :type ub: numpy.ndarray
        :param args: arguments for the function call.
        :type args: numpy.ndarray
        :returns: Value of the integral.
        :rtype: float
        '''
        raise exceptions.MethodNotDefinedError(self.__class__, '__call__')

    @property
    def function(self):
        '''
        Function to call to integrate the PDF.
        '''
        return self.__function

    @property
    def pdf_name(self):
        '''
        Name of the PDF this instance is related to.
        '''
        return self.__pdf.name


class NumIntQ(NumInt):

    def __init__(self, pdf, function, rtol, atol):
        '''
        Base class for methods based on (non) adaptive size steps.

        :param pdf: PDF this instance will be related to.
        :param pdf: PDF
        :param function: function to call to integrate a PDF.
        :param rtol: tolerance for the relative error.
        :type rtol: float
        :param atol: tolerance for the integration value.
        :type atol: float
        '''
        f = numerical_integral_wrapper(self, function)
        super().__init__(pdf, f, rtol, atol)


class NumIntMonte(NumInt):

    def __init__(self, pdf, function, rtol, atol):
        '''
        Base class for Monte Carlo methods to calculate numerical integrals.

        :param pdf: PDF this instance will be related to.
        :param pdf: PDF
        :param function: function to call to integrate a PDF.
        :param rtol: tolerance for the relative error.
        :type rtol: float
        :param atol: tolerance for the integration value.
        :type atol: float
        '''
        f = monte_numerical_integral_wrapper(self, function)
        super().__init__(pdf, f, rtol, atol)


class QNG(NumIntQ):

    def __init__(self, pdf, proxy, rtol=1e-5, atol=0):
        '''
        Use the fixed Gauss-Kronrod-Patterson abscissae to sample the integrand
        at a maximum of 87 points. This method must be used only in smooth
        functions.

        :param pdf: PDF this instance will be related to.
        :param pdf: PDF
        :param proxy: proxy of numerical integration functions.
        :type proxy: NumericalIntegration
        :param rtol: tolerance for the relative error.
        :type rtol: float
        :param atol: tolerance for the integration value.
        :type atol: float
        '''
        super().__init__(pdf, proxy.qng, rtol, atol)

    def __call__(self, lb, ub, args):

        if len(lb) > 1:
            raise RuntimeError(
                'QNG integration method is only available for 1-dimensional PDFs')

        config = (self.atol, self.rtol)

        return self.function(lb.item(), ub.item(), config, args)


class QAG(NumIntQ):

    def __init__(self, pdf, proxy, rtol=1e-5, atol=0, limit=1000, key=1, workspace_size=1000):
        '''
        Determination of numerical integrals using a simple adaptive integration
        procedure of interval subdivision based on the error.

        :param pdf: PDF this instance will be related to.
        :param pdf: PDF
        :param proxy: proxy of numerical integration functions.
        :type proxy: NumericalIntegration
        :param rtol: tolerance for the relative error.
        :type rtol: float
        :param atol: tolerance for the integration value.
        :type atol: float
        :param limit: maximum number of subintervals. Must be smaller than the
           workspace size.
        :type limit: int
        :param key: Gauss-Kronrod rule to use. Keys go from 1 to 6,
           corresponding to the 15, 21, 31, 41, 51 and 61 point rules.
        :type key: int
        :param workspace_size: size reserved in memory to store values. The
           space is related to pairs of doubles.
        :type workspace_size: int
        '''
        super().__init__(pdf, proxy.qag, rtol, atol)

        if limit > workspace_size:
            raise ValueError(
                'Limit of subintervals must be smaller than the workspace size')

        if key < 1 or key > 6:
            raise ValueError('Integration rules go from 1 to 6')

        self.__limit = limit
        self.__key = key
        self.__workspace_size = workspace_size

    def __call__(self, lb, ub, args):

        if len(lb) > 1:
            raise RuntimeError(
                'QNG integration method is only available for 1-dimensional PDFs')

        config = (self.atol, self.rtol, self.__limit,
                  self.__key, self.__workspace_size)

        return self.function(lb.item(), ub.item(), config, args)


class CQUAD(NumIntQ):

    def __init__(self, pdf, proxy, rtol=1e-5, atol=0, workspace_size=1000):
        '''
        Use a doubly-adaptive general-purpose based on the Clenshaw-Curtis
        quadrature rules.

        :param pdf: PDF this instance will be related to.
        :param pdf: PDF
        :param proxy: proxy of numerical integration functions.
        :type proxy: NumericalIntegration
        :param rtol: tolerance for the relative error.
        :type rtol: float
        :param atol: tolerance for the integration value.
        :type atol: float
        :param workspace_size: size reserved in memory to store values. The
           space is related to pairs of doubles.
        :type workspace_size: int
        '''
        super().__init__(pdf, proxy.cquad, rtol, atol)

        self.__workspace_size = workspace_size

    def __call__(self, lb, ub, args):

        if len(lb) > 1:
            raise RuntimeError(
                'QNG integration method is only available for 1-dimensional PDFs')

        config = (self.atol, self.rtol, self.__workspace_size)

        return self.function(lb.item(), ub.item(), config, args)


class PLAIN(NumIntMonte):

    def __init__(self, pdf, proxy, calls=100000, rtol=1e-2, atol=0):
        '''
        Configurable class for the Plain algorithm.

        :param pdf: PDF this instance will be related to.
        :param pdf: PDF
        :param proxy: proxy of numerical integration functions.
        :type proxy: NumericalIntegration
        :param calls: number of calls per iteration.
        :type calls: int
        :param rtol: tolerance for the relative error.
        :type rtol: float
        :param atol: tolerance for the integration value.
        :type atol: float
        '''
        super().__init__(pdf, proxy.plain, rtol, atol)

        self.calls = calls

    def __call__(self, lb, ub, args):

        config = (self.calls,)

        return self.function(lb, ub, config, args)


class MISER(NumIntMonte):

    def __init__(self, pdf, proxy, calls=100000, estimate_frac=0.1, min_calls=None, min_calls_per_bisection=None, alpha=2, dither=0, rtol=1e-4, atol=0):
        r'''
        Configurable class for the MISER algorithm.

        :param pdf: PDF this instance will be related to.
        :param pdf: PDF
        :param proxy: proxy of numerical integration functions.
        :type proxy: NumericalIntegration
        :param calls: number of calls per iteration.
        :type calls: int
        :param estimate_frac: parameter to specify the fraction of the
           currently available number of function calls which are allocated to
           estimating the variance at each recursive step.
        :type estimate_frac: float
        :param min_calls: minimum number of function calls required to proceed
           with a bisection step. The default value is set to :math:`16 \times \text{ndim}`.
        :type min_calls: int or None
        :param min_calls_per_bisection: this parameter specifies the minimum
           number of function calls required to proceed with a bisection step.
           The default value is set to :math:`32 \times \text{min_calls}`.
        :type min_calls_per_bisection: int or None
        :param alpha: parameter to control how the estimated variances for the
           two sub-regions of a bisection are combined when allocating points.
        :type alpha: float
        :param dither: parameter introduces a random fractional variation of
           into each bisection
        :type dither: float
        :param rtol: tolerance for the relative error.
        :type rtol: float
        :param atol: tolerance for the integration value.
        :type atol: float
        '''
        super().__init__(pdf, proxy.miser, rtol, atol)

        self.calls = calls
        self.estimate_frac = estimate_frac
        self.min_calls = min_calls
        self.min_calls_per_bisection = min_calls_per_bisection
        self.alpha = alpha
        self.dither = dither

    def __call__(self, lb, ub, args):

        min_calls = self.min_calls or 16 * len(lb)  # default value by GSL
        min_calls_per_bisection = self.min_calls_per_bisection or 32 * \
            min_calls  # default value by GSL

        config = (self.calls, self.estimate_frac, min_calls,
                  min_calls_per_bisection, self.alpha, self.dither)

        return self.function(lb, ub, config, args)


class VEGAS(NumIntMonte):

    _importance = 'importance'
    _stratified = 'stratified'
    _importance_only = 'importance_only'

    def __init__(self, pdf, proxy, calls=10000, alpha=1.5, iterations=100, mode=_importance, rtol=1e-5, atol=0):
        '''
        Configurable class to do numerical integration with the VEGAS algorithm.

        :param pdf: PDF this instance will be related to.
        :param pdf: PDF
        :param proxy: proxy of numerical integration functions.
        :type proxy: NumericalIntegration
        :param alpha: controls the stiffness of the rebinning algorithm. It is
           typically set between one and two. A value of zero prevents rebinning
           of the grid.
        :type alpha: float
        :param calls: number of calls per iteration.
        :type calls: int
        :param iterations: number of iterations to perform for each call to
           the routine.
        :type iterations: int
        :param mode: whether the algorithm must use importance sampling or
           stratified sampling, or whether it can pick on its own.
        :param rtol: tolerance for the relative error.
        :type rtol: float
        :param atol: tolerance for the integration value.
        :type atol: float
        '''
        super().__init__(pdf, proxy.vegas, rtol, atol)

        self.mode = mode

        self.calls = calls
        self.alpha = alpha
        self.iterations = iterations

    def __call__(self, lb, ub, args):

        config = (self.calls, self.alpha, self.iterations, self.mode)

        return self.function(lb, ub, config, args)

    @property
    def mode(self):
        '''
        Sampling mode to use.

        :getter: Returns the internal code for the sampling mode.
        :setter: Sets the sampling mode by name.
        '''
        return self.__mode

    @mode.setter
    def mode(self, m):
        if m == VEGAS._importance:
            self.__mode = +1  # GSL_VEGAS_MODE_IMPORTANCE
        elif m == VEGAS._stratified:
            self.__mode = -1  # GSL_VEGAS_MODE_STRATIFIED
        elif m == VEGAS._importance_only:
            self.__mode = 0  # GSL_VEGAS_MODE_IMPORTANCE_ONLY
        else:
            raise ValueError(f'Unknown sampling type "{m}"')


def check_integration_result(numint, res, err):
    '''
    Check the result of a integration process.

    :param numint: numerical integral class.
    :type numint: NumInt
    :param res: value of the integral.
    :type res: float
    :param err: error of the integral.
    :type err: float
    '''
    if numint.atol > 0:
        if err > numint.atol:
            warnings.warn(
                f'Integral ("{numint.__class__.__name__.lower()}") error for PDF "{numint.pdf_name}" exceeds the absolute tolerance', RuntimeWarning, stacklevel=3)

    if res == 0. and numint.atol == 0:
        warnings.warn(
            f'Integral ("{numint.__class__.__name__.lower()}") returns zero for PDF "{numint.pdf_name}" and no absolute tolerance has been specified', RuntimeWarning, stacklevel=3)
    elif err / res > numint.rtol:
        warnings.warn(
            f'Integral ("{numint.__class__.__name__.lower()}") relative error for PDF "{numint.pdf_name}" exceeds the tolerance ({err / res:.2e} > {numint.rtol})', RuntimeWarning, stacklevel=3)


def numerical_integral_wrapper(obj, cfunction):
    '''
    Wrapper around numerical integration functions in a library.
    '''
    cfunction.argtypes = [c_double, c_double, py_object, c_double_p]
    cfunction.restype = py_object

    def wrapper(lb, ub, config, args):

        l, u = data_types.as_double(lb, ub)
        a = data_types.data_as_c_double(args)
        c = data_types.as_py_object(config)

        res, err = cfunction(l, u, c, a)[:2]  # skip "neval" for some methods

        check_integration_result(obj, res, err)

        return res

    return wrapper


def monte_numerical_integral_wrapper(obj, cfunction):
    '''
    Wrapper around numerical integration functions using Monte Carlo in a
    library.
    '''
    cfunction.argtypes = [c_double_p, c_double_p, py_object, c_double_p]
    cfunction.restype = py_object

    def wrapper(lb, ub, config, args):

        l, u, a = data_types.data_as_c_double(lb, ub, args)
        c = data_types.as_py_object(config)

        res, err = cfunction(l, u, c, a)

        check_integration_result(obj, res, err)

        return res

    return wrapper


def parse_functions(module, ndim):
    '''
    Parse the given module and define the functions to use for numerical
    integration.

    :param module: module where to get the functions from.
    :type module: module
    :param ndim: number of dimensions of the PDF.
    :type ndim: int
    :returns: functions to do numerical integration.
    :rtype: NumericalIntegration
    '''
    if ndim == 1:
        qng = module.integrate_qng
        qag = module.integrate_qag
        cquad = module.integrate_cquad
    else:
        qng = qag = cquad = None

    plain = module.integrate_plain
    miser = module.integrate_miser
    vegas = module.integrate_vegas

    return NumericalIntegration(qng, qag, cquad, plain, miser, vegas)
