'''
Definition of different minimization functions.
'''
from . import parameters
from . import core

import functools

__all__ = ['binned_maximum_likelihood', 'unbinned_extended_maximum_likelihood', 'unbinned_maximum_likelihood']


def binned_maximum_likelihood( pdf, data, values = None, norm_range = parameters.FULL ):
    '''
    '''
    return pdf.norm(data, values, norm_range) - core.sum(data.values * core.log(pdf(data, values, norm_range, normalized=False)))


def unbinned_extended_maximum_likelihood( pdf, data, values = None, norm_range = parameters.FULL ):
    '''
    '''
    return pdf.norm(data, values, norm_range) - core.sum(core.log(pdf(data, values, norm_range, normalized=False)))


def unbinned_maximum_likelihood( pdf, data, values = None, norm_range = parameters.FULL ):
    '''
    '''
    return - core.sum(core.log(pdf(data, values, norm_range)))
