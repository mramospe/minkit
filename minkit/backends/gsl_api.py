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
from ..base.data_types import c_int, c_double_p, py_object

import collections
import warnings

# Integration methods
PLAIN = 'plain'
MISER = 'miser'
VEGAS = 'vegas'

# To store the compiled functions
NumericalIntegration = collections.namedtuple(
    'NumericalIntegration', [PLAIN, MISER, VEGAS])


class NumIntConfig(object, metaclass=core.DocMeta):

    def __init__(self, function, tolerance):
        '''
        Base class for the classes to do numerical integrals.

        :param function: function to call to integrate a PDF.
        :param tolerance: maximum relative error allowed in the integral.
        :type tolerance: float
        '''
        super().__init__()

        self.__function = numerical_integral_wrapper(self, function)

        self.tolerance = tolerance

    def __call__(self, ndim, lb, ub, args):
        '''
        Calculate the integral in the given bounds.

        :param ndim: number of dimensions.
        :type ndim: int
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


class PlainConfig(NumIntConfig):

    method_name = PLAIN

    def __init__(self, proxy, calls=10000, tolerance=1e-3):
        '''
        Configurable class for the Plain algorithm.

        :param proxy: proxy of numerical integration functions.
        :type proxy: NumericalIntegration
        :param calls: number of calls per iteration.
        :type calls: int
        :param tolerance: maximum relative error allowed in the integral.
        :type tolerance: float
        '''
        super().__init__(proxy.plain, tolerance)

        self.calls = calls

    def __call__(self, ndim, lb, ub, args):

        config = (self.calls,)

        return self.function(ndim, lb, ub, config, args)


class MiserConfig(NumIntConfig):

    method_name = MISER

    def __init__(self, proxy, calls=10000, estimate_frac=0.1, min_calls=None, min_calls_per_bisection=None, alpha=2, dither=0, tolerance=1e-4):
        r'''
        Configurable class for the MISER algorithm.

        :param proxy: proxy of numerical integration functions.
        :type proxy: NumericalIntegration
        :param calls: number of calls per iteration.
        :type calls: int
        :param estimate_frac: parameter to specify the fraction of the \
        currently available number of function calls which are allocated to \
        estimating the variance at each recursive step.
        :type estimate_frac: float
        :param min_calls: minimum number of function calls required to proceed \
        with a bisection step. The default value is set to :math:`16 \times \text{ndim}`.
        :type min_calls: int
        :param min_calls_per_bisection: this parameter specifies the minimum \
        number of function calls required to proceed with a bisection step. \
        The default value is set to :math:`32 \times \text{min_calls}`.
        :type min_calls_per_bisection: int
        :param alpha: parameter to control how the estimated variances for the \
        two sub-regions of a bisection are combined when allocating points.
        :type alpha: float
        :param dither: parameter introduces a random fractional variation of \
        into each bisection
        :type dither: float
        :param tolerance: maximum relative error allowed in the integral.
        :type tolerance: float
        '''
        super().__init__(proxy.miser, tolerance)

        self.calls = calls
        self.estimate_frac = estimate_frac
        self.min_calls = min_calls
        self.min_calls_per_bisection = min_calls_per_bisection
        self.alpha = alpha
        self.dither = dither

    def __call__(self, ndim, lb, ub, args):

        min_calls = self.min_calls or 16 * ndim  # default value by GSL
        min_calls_per_bisection = self.min_calls_per_bisection or 32 * \
            min_calls  # default value by GSL

        config = (self.calls, self.estimate_frac, min_calls,
                  min_calls_per_bisection, self.alpha, self.dither)

        return self.function(ndim, lb, ub, config, args)


class VegasConfig(NumIntConfig):

    method_name = VEGAS

    _importance = 'importance'
    _stratified = 'stratified'
    _importance_only = 'importance_only'

    def __init__(self, proxy, calls=10000, alpha=1.5, iterations=5, mode=_importance, tolerance=1e-5):
        '''
        Configurable class to do numerical integration with the VEGAS algorithm.

        :param proxy: proxy of numerical integration functions.
        :type proxy: NumericalIntegration
        :param alpha: controls the stiffness of the rebinning algorithm. It is \
        typically set between one and two. A value of zero prevents rebinning \
        of the grid.
        :type alpha: float
        :param calls: number of calls per iteration.
        :type calls: int
        :param iterations: number of iterations to perform for each call to\
        the routine.
        :type iterations: int
        :param mode: whether the algorithm must use importance sampling or \
        stratified sampling, or whether it can pick on its own.
        :param tolerance: maximum relative error allowed in the integral.
        :type tolerance: float
        '''
        super().__init__(proxy.vegas, tolerance)

        self.mode = mode

        self.calls = calls
        self.alpha = alpha
        self.iterations = iterations

    def __call__(self, ndim, lb, ub, args):

        config = (self.calls, self.alpha, self.iterations, self.mode)

        return self.function(ndim, lb, ub, config, args)

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
        if m == VegasConfig._importance:
            self.__mode = +1  # GSL_VEGAS_MODE_IMPORTANCE
        elif m == VegasConfig._stratified:
            self.__mode = -1  # GSL_VEGAS_MODE_STRATIFIED
        elif m == VegasConfig._importance_only:
            self.__mode = 0  # GSL_VEGAS_MODE_IMPORTANCE_ONLY
        else:
            raise ValueError(f'Unknown sampling type "{m}"')


def numerical_integral_wrapper(obj, cfunction):
    '''
    Wrapper around numerical integration functions in a library.
    '''
    cfunction.argtypes = [c_int, c_double_p, c_double_p, py_object, c_double_p]
    cfunction.restype = py_object

    def wrapper(dim, lb, ub, config, args):

        l, u, a = data_types.data_as_c_double(lb, ub, args)
        c = data_types.as_py_object(config)
        d = data_types.as_integer(dim)

        res, err = cfunction(d, l, u, c, a)

        if err / res > obj.tolerance:
            warnings.warn(
                f'Numerical integration for method "{obj.method_name}" exceeds the tolerance ({err / res:.2e} > {obj.tolerance})', RuntimeWarning, stacklevel=2)

        return res

    return wrapper


def parse_functions(module):
    '''
    Parse the given module and define the functions to use for numerical
    integration.

    :param module: module where to get the functions from.
    :type module: module
    :returns: functions to do numerical integration.
    :rtype: NumericalIntegration
    '''
    plain = module.integrate_plain
    miser = module.integrate_miser
    vegas = module.integrate_vegas
    return NumericalIntegration(plain, miser, vegas)
