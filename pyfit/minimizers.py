'''
API for minimizers.
'''
from . import dataset
from . import fcns
from . import parameters
from . import types

import collections
import contextlib
import functools
import iminuit
import numpy as np

__all__ = ['Category', 'binned_minimizer', 'unbinned_minimizer', 'migrad_output_to_registry', 'simultaneous_minimizer']

# Names for the minimizers
MINUIT = 'minuit'

# Object to help users to create simultaneous minimizers.
Category = collections.namedtuple('Category', ['fcn', 'pdf', 'data'])
Category.__doc__ = '''\
Object serving as a proxy for an FCN to be evaluated using a PDF on a data set.
The type of data (binned/unbinned) is assumed from the FCN.'''


def parse_fcn( data_type ):
    '''
    Parser the "fcn" argument of a function, checking its validity for the given data type.
    '''
    def __wrapper( function ):
        '''
        Internal wrapper.
        '''
        @functools.wraps(function)
        def __wrapper( fcn, *args, **kwargs ):
            '''
            Internal wrapper.
            '''
            if fcns.data_type_for_fcn(fcn) != data_type:
                raise NotImplementedError(f'FCN with name "{fcn}" is not available for "{data_type}" data type')
            return function(fcns.fcn_from_name(fcn), *args, **kwargs)
        return __wrapper
    return __wrapper


class BinnedEvaluatorProxy(object):
    '''
    Definition of a proxy class to evaluate an FCN with a PDF on a BinnedDataSet object.
    '''
    def __init__( self, fcn, pdf, data, constraints = None ):
        '''
        :param fcn: FCN to be used during minimization.
        :type fcn: str
        :param data: data sample to process.
        :type data: BinnedDataSet
        :param kwargs: arguments to be forwarded to the FCN function.
        :type kwargs: dict
        '''
        self.__data        = data
        self.__fcn         = fcn
        self.__pdf         = pdf.frozen(self.__data)
        self.__constraints = constraints or []

        super(BinnedEvaluatorProxy, self).__init__()

    def __call__( self, *values ):
        '''
        Evaluate the FCN.
        Values must be provided sorted as :method:`PDF.args`.

        :param values: set of values to evaluate the FCN.
        :type values: tuple(float)
        :returns: value of the FCN.
        :rtype: float
        '''
        r = parameters.Registry()
        for i, n in enumerate(self.__pdf.all_args):
            r[n] = values[i]
        return self.call_from_dict(r)

    def call_from_dict( self, values ):
        '''
        '''
        return self.__fcn(self.__pdf, self.__data, values, self.__constraints)

    @property
    def args( self ):
        '''
        '''
        return self.__pdf.all_args


class UnbinnedEvaluatorProxy(object):
    '''
    Definition of a proxy class to evaluate an FCN with a PDF.
    '''
    def __init__( self, fcn, pdf, data, range = parameters.FULL, constraints = None ):
        '''
        :param fcn: FCN to be used during minimization.
        :type fcn: str
        :param data: data sample to process.
        :type data: DataSet
        :param kwargs: arguments to be forwarded to the FCN function.
        :type kwargs: dict
        '''
        self.__data        = data.subset(range=range, copy=False)
        self.__fcn         = fcn
        self.__pdf         = pdf.frozen(self.__data, range=range)
        self.__range       = range
        self.__constraints = constraints or []

        super(UnbinnedEvaluatorProxy, self).__init__()

    def __call__( self, *values ):
        '''
        Evaluate the FCN.
        Values must be provided sorted as :method:`PDF.args`.

        :param values: set of values to evaluate the FCN.
        :type values: tuple(float)
        :returns: value of the FCN.
        :rtype: float
        '''
        r = parameters.Registry()
        for i, n in enumerate(self.__pdf.all_args):
            r[n] = values[i]
        return self.call_from_dict(r)

    def call_from_dict( self, values ):
        '''
        '''
        return self.__fcn(self.__pdf, self.__data, values, self.__range, self.__constraints)

    @property
    def args( self ):
        '''
        '''
        return self.__pdf.all_args


class SimultaneousEvaluator(object):
    '''
    Definition of an evaluator of PDFs for simultaneous fits.
    '''
    def __init__( self, evaluators ):
        '''
        Build an object to evaluate PDFs on independent data samples.
        This class is not meant to be used by users.

        :param evaluators: list of evaluators to use.
        :type evaluators: list(UnbinnedEvaluator or BinnedEvaluator)
        '''
        self.__evaluators = evaluators

        super(SimultaneousEvaluator, self).__init__()

    def __call__( self, *values ):
        '''
        Call each PDF in the corresponding data sample.

        :param data: data samples to process.
        :type data: list(DataSet or BinnedDataSet)
        :param args: forwarded to :method:`PDF.__call__`
        :type args: tuple
        :param kwargs: forwarded to :method:`PDF.__call__`
        :type kwargs: dict
        '''
        r = parameters.Registry()
        for i, n in enumerate(self.args):
            r[n] = values[i]
        return np.sum(np.fromiter((e.call_from_dict(r) for e in self.__evaluators), dtype=types.cpu_type))

    @property
    def args( self ):
        '''
        Get the arguments for the evaluator.
        '''
        args = parameters.Registry(self.__evaluators[0].args)
        for e in self.__evaluators[1:]:
            args.update(e.args)
        return args


def migrad_output_to_registry( result ):
    '''
    Transform the output from a call to Migrade into a :class:`Registry`.

    :param result: result from a migrad call.
    :type result: iminuit.Migrad
    :returns: registry of parameters with the result from Migrad.
    :rtype: Registry(str, Parameter)
    '''
    r = parameters.Registry()
    for p in result.params:
        limits = (p.lower_limit, p.upper_limit) if p.has_limits else None
        r[p.name] = parameters.Parameter(p.name, p.value, bounds=limits, error=p.error, constant=p.is_fixed)
    return r


def registry_to_minuit_input( registry, errordef = 1. ):
    '''
    Transform a registry of parameters into a dictionary to be parsed by Minuit.

    :param registry: registry of parameters.
    :type registry: Registry(str, Parameter)
    :param errordef: error definition for Minuit.
    :type errorder: float
    :returns: Minuit configuration dictionary.
    :rtype: dict
    '''
    values = {v.name: v.value for v in registry.values()}
    errors = {f'error_{v.name}': v.error for v in registry.values()}
    limits = {f'limit_{v.name}': v.bounds for v in registry.values()}
    const  = {f'fix_{v.name}': v.constant for v in registry.values()}
    return dict(errordef=errordef, **values, **errors, **limits, **const)


@contextlib.contextmanager
@parse_fcn('unbinned')
def unbinned_minimizer( fcn, pdf, data, minimizer = MINUIT, **kwargs ):
    '''
    Create a new instance of :class:`iminuit.Minuit`.
    This represents a "frozen" object, that is, parameters defining
    the PDFs are assumed to remain constant during all its lifetime.

    .. warning: Do not change any attribute of the parameters defining the \
    PDFs, since it will not be properly reflected during the minimization \
    calls.
    '''
    evaluator = UnbinnedEvaluatorProxy(fcn, pdf, data, **kwargs)

    if minimizer == MINUIT:
        yield iminuit.Minuit(evaluator,
                             forced_parameters=tuple(pdf.all_args.keys()),
                             pedantic=False,
                             **registry_to_minuit_input(pdf.all_args))
    else:
        raise ValueError(f'Unknown minimizer "{minimizer}"')


@contextlib.contextmanager
@parse_fcn('binned')
def binned_minimizer( fcn, pdf, data, minimizer = MINUIT, **kwargs ):
    '''
    Create a new instance of :class:`iminuit.Minuit`.
    This represents a "frozen" object, that is, parameters defining
    the PDFs are assumed to remain constant during all its lifetime.

    .. warning: Do not change any attribute of the parameters defining the \
    PDFs, since it will not be properly reflected during the minimization \
    calls.
    '''
    evaluator = BinnedEvaluatorProxy(fcn, pdf, data, **kwargs)

    if minimizer == MINUIT:
        yield iminuit.Minuit(evaluator,
                             forced_parameters=tuple(pdf.all_args.keys()),
                             pedantic=False,
                             **registry_to_minuit_input(pdf.all_args))
    else:
        raise ValueError(f'Unknown minimizer "{minimizer}"')


@contextlib.contextmanager
@parse_fcn('unbinned')
def simultaneous_unbinned_minimizer( fcn, pdfs, data, minimizer = MINUIT, **kwargs ):
    '''
    Create a new instance of :class:`iminuit.Minuit`.
    This represents a "frozen" object, that is, parameters defining
    the PDFs are assumed to remain constant during all its lifetime.

    .. warning: Do not change any attribute of the parameters defining the \
    PDFs, since it will not be properly reflected during the minimization \
    calls.
    '''
    if minimizer == MINUIT:
        cfg = registry_to_minuit_input(pdf.all_args)

        evaluator = UnbinnedEvaluatorProxy(fcn, pdf, data, **kwargs)

        yield iminuit.Minuit(evaluator,
                             forced_parameters=tuple(pdf.all_args.keys()),
                             pedantic=False,
                             **cfg)
    else:
        raise ValueError(f'Unknown minimizer "{minimizer}"')


@contextlib.contextmanager
def simultaneous_minimizer( categories, minimizer = MINUIT, range = parameters.FULL, constraints = None ):
    '''
    Create a new instance of :class:`iminuit.Minuit`.
    This represents a "frozen" object, that is, parameters defining
    the PDFs are assumed to remain constant during all its lifetime.

    .. warning: Do not change any attribute of the parameters defining the \
    PDFs, since it will not be properly reflected during the minimization \
    calls.
    '''
    # Build the simultaneous evaluator
    evaluators = []
    for cat in categories:

        fcn = fcns.fcn_from_name(cat.fcn)

        if fcns.data_type_for_fcn(cat.fcn) == dataset.BINNED:
            e = BinnedEvaluatorProxy(fcn, cat.pdf, cat.data, constraints)
        else:
            e = UnbinnedEvaluatorProxy(fcn, cat.pdf, cat.data, range, constraints)
        evaluators.append(e)

    evaluator = SimultaneousEvaluator(evaluators)

    # Return the minimizer
    if minimizer == MINUIT:
        yield iminuit.Minuit(evaluator,
                             forced_parameters=tuple(evaluator.args.keys()),
                             pedantic=False,
                             **registry_to_minuit_input(evaluator.args))
    else:
        raise ValueError(f'Unknown minimizer "{minimizer}"')
