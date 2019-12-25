'''
API for minimizers.
'''
from . import dataset
from . import fcns
from . import parameters
from . import pdf_core

import functools
import iminuit
import contextlib

__all__ = ['create_minuit_binned', 'create_minuit_unbinned', 'migrad_output_to_registry']


def parse_fcn( data_type ):
    '''
    '''
    def __wrapper( function ):
        '''
        '''
        @functools.wraps(function)
        def __wrapper( fcn, pdf, data, *args, **kwargs ):
            '''
            '''
            notimpl = NotImplementedError(f'FCN with name "{fcn}" is not available for "{data_type}" data type')

            if data_type == dataset.BINNED:
                if fcn == 'bml':
                    fcn = fcns.binned_maximum_likelihood
                else:
                    raise notimpl
            else:
                if fcn == 'ueml':
                    fcn = fcns.unbinned_extended_maximum_likelihood
                elif fcn == 'uml':
                    fcn = fcns.unbinned_maximum_likelihood
                else:
                    raise notimpl
            return function(fcn, pdf, data, *args, **kwargs)
        return __wrapper
    return __wrapper


def migrad_output_to_registry( result ):
    '''
    '''
    r = parameters.Registry()
    for p in result.params:
        limits = (p.lower_limit, p.upper_limit) if p.has_limits else None
        r[p.name] = parameters.Parameter(p.name, p.value, bounds=limits, error=p.error, constant=p.is_fixed)
    return r


def registry_to_minuit_input( registry, errordef = 1. ):
    '''
    '''
    values = {v.name: v.value for v in registry.values()}
    errors = {f'error_{v.name}': v.error for v in registry.values()}
    limits = {f'limit_{v.name}': v.bounds for v in registry.values()}
    const  = {f'fix_{v.name}': v.constant for v in registry.values()}
    return dict(errordef=errordef, **values, **errors, **limits, **const)


@contextlib.contextmanager
@parse_fcn('unbinned')
def create_minuit_unbinned( fcn, pdf, data, **kwargs ):
    '''
    Create a new instance of :class:`iminuit.Minuit`.
    This represents a "frozen" object, that is, parameters defining
    the PDFs are assumed to remain constant during all its lifetime.

    .. warning: Do not change any attribute of the parameters defining the \
    PDFs, since it will not be properly reflected during the minimization \
    calls.
    '''
    all_args = pdf.all_args

    cfg = registry_to_minuit_input(all_args)

    evaluator = pdf_core.UnbinnedEvaluatorProxy(fcn, pdf, data, **kwargs)

    yield iminuit.Minuit(evaluator,
                         forced_parameters=tuple(all_args.keys()),
                         pedantic=False,
                         **cfg)


@contextlib.contextmanager
@parse_fcn('binned')
def create_minuit_binned( fcn, pdf, data, **kwargs ):
    '''
    Create a new instance of :class:`iminuit.Minuit`.
    This represents a "frozen" object, that is, parameters defining
    the PDFs are assumed to remain constant during all its lifetime.

    .. warning: Do not change any attribute of the parameters defining the \
    PDFs, since it will not be properly reflected during the minimization \
    calls.
    '''
    all_args = pdf.all_args

    cfg = registry_to_minuit_input(all_args)

    evaluator = pdf_core.BinnedEvaluatorProxy(fcn, pdf, data, **kwargs)

    yield iminuit.Minuit(evaluator,
                         forced_parameters=tuple(all_args.keys()),
                         pedantic=False,
                         **cfg)
