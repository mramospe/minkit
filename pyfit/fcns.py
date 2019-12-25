'''
Definition of different minimization functions.
'''
from . import parameters
from . import core

import functools
import numpy as np

__all__ = ['binned_maximum_likelihood', 'unbinned_extended_maximum_likelihood', 'unbinned_maximum_likelihood']


def evaluate_constraints( values = None, constraints = None ):
    '''
    '''
    res = 1.

    if constraints is None:
        return res

    for c in constraints:
        # Must extract the values for the data parameters
        fvals = []
        for n, p in c.data_pars.items():
            if values is None or n not in values:
                fvals.append(p.value)
            else:
                fvals.append(values[n])

        if values is not None:
            values = parameters.Registry(values)
            for n, p in c.all_args.items():
                if n not in values:
                    values[n] = p.value

        res *= c.function(*fvals, values=values, normalized=False)

    return res


def binned_maximum_likelihood( pdf, data, values = None, norm_range = parameters.FULL, constraints = None ):
    '''
    '''
    c = evaluate_constraints(values, constraints)
    return pdf.norm(values, norm_range) - core.sum(data.values * core.log(c * pdf(data, values, norm_range)))


def unbinned_extended_maximum_likelihood( pdf, data, values = None, norm_range = parameters.FULL, constraints = None ):
    '''
    '''
    c = evaluate_constraints(values, constraints)
    return pdf.norm(values, norm_range) - core.sum(core.log(pdf(data, values, norm_range))) - len(data) * np.log(c)


def unbinned_maximum_likelihood( pdf, data, values = None, norm_range = parameters.FULL, constraints = None ):
    '''
    '''
    c = evaluate_constraints(values, constraints)
    return - core.sum(core.log(pdf(data, values, norm_range))) - len(data) * np.log(c)
