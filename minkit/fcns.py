'''
Definition of different minimization functions.
'''
from .core import aop
from . import dataset
from . import parameters

import functools
import numpy as np

__all__ = ['binned_maximum_likelihood',
           'unbinned_extended_maximum_likelihood', 'unbinned_maximum_likelihood']

# Names of different FCNs
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
    if fcn in (BINNED_MAXIMUM_LIKELIHOOD,):
        return dataset.BINNED
    elif fcn in (UNBINNED_MAXIMUM_LIKELIHOOD,
                 UNBINNED_EXTENDED_MAXIMUM_LIKELIHOOD):
        return dataset.UNBINNED
    else:
        raise ValueError(f'Unknown FCN type "{fcn}"')


def evaluate_constraints(values=None, constraints=None):
    '''
    Calculate the values of the constraints, if any.

    :param values: values of the parameters in the fit.
    :type values: Registry
    :param constraints: functions defining constraints to different parameters.
    :type contraints: list(PDF)
    :returns: evaluation of the product of constraints.
    :rtype: float
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


def binned_chisquare(pdf, data, values=None, range=parameters.FULL, constraints=None):
    '''
    Definition of the binned chi-square FCN.

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: BinnedDataSet
    :param values: values to be passed to the :meth:`PDF.__call__` method.
    :type values: Registry
    :param range: normalization range of the PDF.
    :type range: str
    :param constraints: PDFs with the constraints for paramters in "pdf".
    :type constraints: list(PDF)
    :returns: value of the FCN.
    :rtype: float
    '''
    raise NotImplementedError('Function not implemented yet')


def binned_maximum_likelihood(pdf, data, values=None, constraints=None):
    '''
    Definition of the binned maximum likelihood FCN.

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: BinnedDataSet
    :param values: values to be passed to the :meth:`PDF.__call__` method.
    :type values: Registry
    :param constraints: PDFs with the constraints for paramters in "pdf".
    :type constraints: list(PDF)
    :returns: value of the FCN.
    :rtype: float
    '''
    c = evaluate_constraints(values, constraints)
    return pdf.norm(values) - aop.sum(data.values * aop.log(c * pdf(data, values)))


def unbinned_extended_maximum_likelihood(pdf, data, values=None, range=parameters.FULL, constraints=None):
    '''
    Definition of the unbinned extended maximum likelihood FCN.
    In this case, entries in data are assumed to follow a Poissonian distribution.
    The given :class:`PDF` must be of "extended" type.

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: DataSet
    :param values: values to be passed to the :meth:`PDF.__call__` method.
    :type values: Registry
    :param range: normalization range of the PDF.
    :type range: str
    :param constraints: PDFs with the constraints for paramters in "pdf".
    :type constraints: list(PDF)
    :returns: value of the FCN.
    :rtype: float
    '''
    c = evaluate_constraints(values, constraints)
    lf = aop.log(pdf(data, values, range, normalized=False))
    if data.weights is not None:
        lf *= data.weights
    return pdf.norm(values, range) - aop.sum(lf) - len(data) * np.log(c)


def unbinned_maximum_likelihood(pdf, data, values=None, range=parameters.FULL, constraints=None):
    '''
    Definition of the unbinned maximum likelihood FCN.
    The given :class:`PDF` must not be of "extended" type.

    :param pdf: function to evaluate.
    :type pdf: PDF
    :param data: data to evaluate.
    :type data: DataSet
    :param values: values to be passed to the :meth:`PDF.__call__` method.
    :type values: Registry
    :param range: normalization range of the PDF.
    :type range: str
    :param constraints: PDFs with the constraints for paramters in "pdf".
    :type constraints: list(PDF)
    :returns: value of the FCN.
    :rtype: float
    '''
    c = evaluate_constraints(values, constraints)
    lf = aop.log(pdf(data, values, range))
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
    if name == BINNED_MAXIMUM_LIKELIHOOD:
        return binned_maximum_likelihood
    elif name == UNBINNED_MAXIMUM_LIKELIHOOD:
        return unbinned_maximum_likelihood
    elif name == UNBINNED_EXTENDED_MAXIMUM_LIKELIHOOD:
        return unbinned_extended_maximum_likelihood
    else:
        raise ValueError(f'Unknown FCN type "{name}"')
