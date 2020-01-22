'''
Definition of different minimization functions.
'''
from .core import aop
from . import dataset
from . import parameters

import functools
import numpy as np

__all__ = ['binned_maximum_likelihood', 'binned_chisquare',
           'unbinned_extended_maximum_likelihood', 'unbinned_maximum_likelihood']

# Names of different FCNs
BINNED_CHISQUARE = 'chi2'
BINNED_MAXIMUM_LIKELIHOOD = 'bml'
UNBINNED_MAXIMUM_LIKELIHOOD = 'uml'
UNBINNED_EXTENDED_MAXIMUM_LIKELIHOOD = 'ueml'


def data_type_for_fcn(fcn):
    '''
    Get the associated data type for a given FCN.

    :param fcn: FCN to consider.
    :type fcn: str
    :returns: data type associated to the FCN.
    :rtype: str
    :raises ValueError: if the FCN is unknown.
    '''
    if fcn in (BINNED_CHISQUARE, BINNED_MAXIMUM_LIKELIHOOD):
        return dataset.BINNED
    elif fcn in (UNBINNED_MAXIMUM_LIKELIHOOD,
                 UNBINNED_EXTENDED_MAXIMUM_LIKELIHOOD):
        return dataset.UNBINNED
    else:
        raise ValueError(f'Unknown FCN type "{fcn}"')


def evaluate_constraints(constraints=None):
    '''
    Calculate the values of the constraints, if any.

    :param constraints: functions defining constraints to different parameters.
    :type contraints: list(PDF) or None
    :returns: evaluation of the product of constraints.
    :rtype: float
    '''
    if constraints is None:
        return 1.

    res = 1.
    for c in constraints:
        res *= c.function(normalized=False)

    return res


def nll_to_chi2(function):
    '''
    Decorate the a function that returns a negative logarithm of likelihood into
    a chi-square, by multiplying it by two.
    '''
    @functools.wraps(function)
    def __wrapper(*args, **kwargs):
        return 2. * function(*args, **kwargs)
    return __wrapper


def binned_chisquare(pdf, data, range=parameters.FULL, constraints=None):
    '''
    Definition of the binned chi-square FCN.

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: BinnedDataSet
    :param range: normalization range of the PDF.
    :type range: str
    :param constraints: PDFs with the constraints for paramters in "pdf".
    :type constraints: list(PDF)
    :returns: value of the FCN.
    :rtype: float
    '''
    c = evaluate_constraints(constraints)
    f = c * pdf.evaluate_binned(data)
    return aop.sum((data.values - f)**2 / f)


@nll_to_chi2
def binned_maximum_likelihood(pdf, data, constraints=None):
    '''
    Definition of the binned maximum likelihood FCN.

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: BinnedDataSet
    :param constraints: PDFs with the constraints for paramters in "pdf".
    :type constraints: list(PDF)
    :returns: value of the FCN.
    :rtype: float
    '''
    c = evaluate_constraints(constraints)
    return pdf.norm() - aop.sum(data.values * aop.log(c * pdf.evaluate_binned(data)))


@nll_to_chi2
def unbinned_extended_maximum_likelihood(pdf, data, range=parameters.FULL, constraints=None):
    '''
    Definition of the unbinned extended maximum likelihood FCN.
    In this case, entries in data are assumed to follow a Poissonian distribution.
    The given :class:`PDF` must be of "extended" type.

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: DataSet
    :param range: normalization range of the PDF.
    :type range: str
    :param constraints: PDFs with the constraints for paramters in "pdf".
    :type constraints: list(PDF)
    :returns: value of the FCN.
    :rtype: float
    '''
    c = evaluate_constraints(constraints)
    lf = aop.log(pdf(data, range, normalized=False))
    if data.weights is not None:
        lf *= data.weights
    return pdf.norm(range) - aop.sum(lf) - len(data) * np.log(c)


@nll_to_chi2
def unbinned_maximum_likelihood(pdf, data, range=parameters.FULL, constraints=None):
    '''
    Definition of the unbinned maximum likelihood FCN.
    The given :class:`PDF` must not be of "extended" type.

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: DataSet
    :param range: normalization range of the PDF.
    :type range: str
    :param constraints: PDFs with the constraints for paramters in "pdf".
    :type constraints: list(PDF)
    :returns: value of the FCN.
    :rtype: float
    '''
    c = evaluate_constraints(constraints)
    lf = aop.log(pdf(data, range))
    if data.weights is not None:
        lf *= data.weights
    return - aop.sum(lf) - len(data) * np.log(c)


def fcn_from_name(name):
    '''
    Return the FCN associated to the given name.

    :param name: name of the FCN.
    :type name: str
    :returns: associated function.
    :rtype: function
    '''
    if name == BINNED_CHISQUARE:
        return binned_chisquare
    elif name == BINNED_MAXIMUM_LIKELIHOOD:
        return binned_maximum_likelihood
    elif name == UNBINNED_MAXIMUM_LIKELIHOOD:
        return unbinned_maximum_likelihood
    elif name == UNBINNED_EXTENDED_MAXIMUM_LIKELIHOOD:
        return unbinned_extended_maximum_likelihood
    else:
        raise ValueError(f'Unknown FCN type "{name}"')
