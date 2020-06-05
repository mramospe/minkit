########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Functions and classes related to the s-plot technique.
More information at: https://arxiv.org/abs/physics/0402083
'''
from ..base import parameters
from ..base import data_types

import numpy as np

__all__ = ['sweights', 'sweights_u']


def sweights(pdfs, yields, data, range=parameters.FULL, return_covariance=False):
    '''
    Calculate the s-weights for the different provided species.

    :param pdfs: registry of PDFs to use.
    :type pdfs: Registry
    :param yields: yields of the PDFs.
    :type yields: Registry
    :param data: data to evaluate.
    :type data: DataSet
    :param range: range to consider for evaluating the PDFs.
    :type range: str
    :param return_covariance: if set to True, it also returns the covariance matrix.
    :type return_covariance: bool
    :returns: Values of the s-weights for each specie and possible covariance matrix.
    :rtype: list(darray), (numpy.ndarray)
    '''
    l = len(yields)

    if l == len(pdfs):
        yields = np.array([p.value for p in yields])
    elif l + 1 == len(pdfs):
        yields = data_types.empty_float(l + 1)
        yields[:l] = [p.value for p in yields]
        yields[-1] = 1. - yields[:l].sum()
    else:
        raise ValueError(
            'Number of provided yields must be equal than number of PDFs, or at least only one less')

    fvals = [pdf(data, range) for pdf in pdfs]

    aop = pdfs[0].aop

    yf = aop.sum_arrays(tuple(y * f for y, f in zip(yields, fvals)))

    # Calculate the inverse of the covariance matrix
    den = yf**2

    iV = data_types.empty_float((l, l))
    for i, fi in enumerate(fvals):

        iV[i, i] = aop.sum(fi * fi / den)

        for j, fj in enumerate(fvals[i + 1:], i + 1):
            iV[i, j] = iV[j, i] = aop.sum(fi * fj / den)

    V = np.linalg.inv(iV)

    # Calculate the s-weights
    w = [aop.sum_arrays(tuple(v * f for v, f in zip(V[i], fvals))) /
         yf for i in np.arange(l)]

    if return_covariance:
        return w, V
    else:
        return w


def sweights_u(a, sweights, bins=10, range=None):
    r'''
    Get the uncertainty associated to the s-weights related to sample *a*.
    Arguments are similar to those of :func:`numpy.histogram`.
    By definition, the uncertainty on the s-weights (for plotting), is defined
    as the sum of the squares of the weights in that bin, like

    .. math:: \sigma = \sqrt{\sum_{b \in \delta x} \omega^2}

    :param a: array of data.
    :type a: numpy.ndarray
    :param sweights: array of weights.
    :type sweights: numpy.ndarray
    :param bins: bins for the histogram.
    :type bins: int or numpy.ndarray
    :param range: range of the histogram.
    :type range: tuple(float, float) or None
    :returns: Uncertainties for each bin of the histogram.
    :rtype: numpy.ndarray
    '''
    return np.sqrt(np.histogram(a, bins=bins, weights=sweights*sweights)[0])
