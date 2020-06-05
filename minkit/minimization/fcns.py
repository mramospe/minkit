########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Definition of different minimization functions.
'''
from ..base import parameters
from ..pdfs import dataset

import functools
import numpy as np
import warnings

__all__ = ['binned_chisquare', 'binned_maximum_likelihood',
           'binned_extended_chisquare', 'binned_extended_maximum_likelihood',
           'unbinned_maximum_likelihood', 'unbinned_extended_maximum_likelihood']

# Names of different FCNs
BINNED_CHISQUARE = 'chi2'
BINNED_EXTENDED_CHISQUARE = 'echi2'
BINNED_MAXIMUM_LIKELIHOOD = 'bml'
BINNED_EXTENDED_MAXIMUM_LIKELIHOOD = 'beml'
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
    if fcn in (BINNED_CHISQUARE, BINNED_EXTENDED_CHISQUARE, BINNED_MAXIMUM_LIKELIHOOD, BINNED_EXTENDED_MAXIMUM_LIKELIHOOD):
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
    :type contraints: list(PDF)
    :returns: evaluation of the product of constraints.
    :rtype: float
    '''
    if constraints is None:
        return 0.

    res = 1.
    for c in constraints:
        res *= c.function(normalized=True)

    return -2. * np.log(res)


def warn_if_nan(function):
    '''
    Decorate an FCN so a warning is displayed if the result is nan.
    This happens when the probability is zero or negative for at least one value.
    '''
    @functools.wraps(function)
    def __wrapper(*args, **kwargs):

        with warnings.catch_warnings():
            # Get rid of the warnings for the calls to "log" with zero or negative
            # values and the multiplication of the result by an array
            warnings.filterwarnings('ignore', category=RuntimeWarning)

            v = function(*args, **kwargs)

        if np.isnan(v):
            warnings.warn(
                'Evaluation of the PDF is zero or negative', RuntimeWarning, stacklevel=2)

        return v

    return __wrapper


@warn_if_nan
def binned_chisquare(pdf, data, range=parameters.FULL):
    r'''
    Definition of the binned chi-square FCN. The returned values are directly
    the :math:`\chi^2` of the PDF in the data sample, and it is computed as

    .. math::
       \text{FCN} = \sum_{b = 0}^M \frac{\left(n_b - f_b(\vec{p})\right)^2}{f_b(\vec{p})},

    where :math:`n_b` is the number of entries in bin *b*, :math:`f(b;\vec{\theta})`
    is the integral of the function to minimize in the bin *b*
    (without normalization), :math:`\vec{\theta}` are the function parameters

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: BinnedDataSet
    :param range: normalization range of the PDF.
    :type range: str
    :returns: value of the FCN.
    :rtype: float
    '''
    f = pdf.evaluate_binned(data)
    f *= data.values.sum()
    return ((data.values - f)**2 / f).sum()


@warn_if_nan
def binned_extended_chisquare(pdf, data):
    r'''
    Definition of the binned chi-square FCN for extended PDFs. The returned
    value is directly the :math:`\chi^2` of the PDF in the data sample, and it
    is computed as

    .. math::
       \text{FCN} = \sum_{b = 0}^M \frac{\left(n_b - f_b(\vec{p})\right)^2}{f_b(\vec{p})},

    where :math:`n_b` is the number of entries in bin *b*, :math:`f(b;\vec{\theta})`
    is the integral of the function to minimize in the bin *b*
    (without normalization), :math:`\vec{\theta}` are the function parameters

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: BinnedDataSet
    :param range: normalization range of the PDF.
    :type range: str
    :returns: value of the FCN.
    :rtype: float
    '''
    f = pdf.evaluate_binned(data, normalized=False)
    return ((data.values - f)**2 / f).sum()


@warn_if_nan
def binned_maximum_likelihood(pdf, data):
    r'''
    Definition of the binned maximum likelihood FCN.
    The output is two times the logarithm of the likelihood, and is computed as

    .. math::
       \text{FCN} = -2 \times \left[\sum_{b=0}^M n_b \log\frac{n_b}{f(b;\vec{\theta})} + f(b;\vec{\theta}) - n_b\right],

    where :math:`n_b` is the number of entries in bin *b*,
    :math:`f(b;\vec{\theta})` is the integral of the function to minimize
    in the bin *b* (without normalization), :math:`\vec{\theta}` are the function
    parameters.

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: BinnedDataSet
    :returns: value of the FCN.
    :rtype: float
    '''
    f = pdf.evaluate_binned(data)
    f *= data.values.sum()
    return 2. * (pdf.aop.product_by_zero_is_zero(data.values, pdf.aop.log(data.values / f)) + f - data.values).sum()


@warn_if_nan
def binned_extended_maximum_likelihood(pdf, data):
    r'''
    Definition of the binned extended maximum likelihood FCN.
    The output is two times the logarithm of the likelihood, and is computed as

    .. math::
       \text{FCN} = -2 \times \left[\sum_{b=0}^M n_b \log\frac{n_b}{f(b;\vec{\theta})} + f(b;\vec{\theta}) - n_b\right],

    where :math:`n_b` is the number of entries in bin *b*,
    :math:`f(b;\vec{\theta})` is the integral of the function to minimize
    in the bin *b* (without normalization), :math:`\vec{\theta}` are the function
    parameters.

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: BinnedDataSet
    :returns: value of the FCN.
    :rtype: float
    '''
    f = pdf.evaluate_binned(data, normalized=False)
    return 2. * (pdf.aop.product_by_zero_is_zero(data.values, pdf.aop.log(data.values / f)) + f - data.values).sum()


@warn_if_nan
def unbinned_extended_maximum_likelihood(pdf, data, range=parameters.FULL):
    r'''
    Definition of the unbinned extended maximum likelihood FCN.
    In this case, entries in data are assumed to follow a Poissonian distribution.
    The given :class:`PDF` must be of *extended* type.
    The output is two times the logarithm of the likelihood, and is computed as

    .. math::
       \text{FCN} = -2 \times \left[\sum_{i=0}^N \log f(\vec{x}_i;\vec{\theta}) - A(\vec{\theta})\right],

    where :math:`f(\vec{x}_i;\vec{\theta})` is the function to minimize (without
    normalization), :math:`\vec{x}_i` are the data points, :math:`\vec{\theta}`
    are the function parameters and :math:`A(\vec{\theta})` is the normalization
    value.

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: DataSet
    :param range: normalization range of the PDF.
    :type range: str
    :returns: value of the FCN.
    :rtype: float
    '''
    lf = pdf.aop.log(pdf(data, range, normalized=False))
    if data.weights is not None:
        lf *= data.weights
    return 2. * (pdf.norm(range) - lf.sum())


@warn_if_nan
def unbinned_maximum_likelihood(pdf, data, range=parameters.FULL):
    r'''
    Definition of the unbinned maximum likelihood FCN.
    The given :class:`PDF` must not be of *extended* type.
    The output is two times the logarithm of the likelihood, and is computed as

    .. math::
       \text{FCN} = -2 \times \left[\sum_{i=0}^N \log \frac{f(\vec{x}_i;\vec{\theta})}{A(\vec{\theta})}\right],

    where :math:`f(\vec{x}_i;\vec{\theta})` is the function to minimize (without
    normalization), :math:`\vec{x}_i` are the data points, :math:`\vec{\theta}`
    are the function parameters and :math:`A(\vec{\theta})` is the normalization
    value.

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: DataSet
    :param range: normalization range of the PDF.
    :type range: str
    :returns: value of the FCN.
    :rtype: float
    '''
    lf = pdf.aop.log(pdf(data, range))
    if data.weights is not None:
        lf *= data.weights
    return -2. * lf.sum()


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
    elif name == BINNED_EXTENDED_CHISQUARE:
        return binned_extended_chisquare
    elif name == BINNED_EXTENDED_MAXIMUM_LIKELIHOOD:
        return binned_extended_maximum_likelihood
    elif name == BINNED_MAXIMUM_LIKELIHOOD:
        return binned_maximum_likelihood
    elif name == UNBINNED_MAXIMUM_LIKELIHOOD:
        return unbinned_maximum_likelihood
    elif name == UNBINNED_EXTENDED_MAXIMUM_LIKELIHOOD:
        return unbinned_extended_maximum_likelihood
    else:
        raise ValueError(f'Unknown FCN type "{name}"')
