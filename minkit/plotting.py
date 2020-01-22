'''
Some utilities to plot data and PDFs using matplotlib.
'''
import numpy as np
from . import parameters
from .core import aop
from .operations import types

__all__ = ['scale_pdf_values']


def bin_area(edges):
    '''
    Calculate the area of the bins given a set of edges.

    :param edges: bounds of the bins
    :type edges: numpy.ndarray or tuple(numpy.ndarray, ...)
    :returns: area of the bins.
    :rtype: float
    '''
    edges = np.array(edges)
    if len(edges.shape) == 1:
        return edges[1] - edges[0]
    else:
        return np.prod(np.fromiter((e[1] - e[0] for e in edges), dtype=types.cpu_type))


def scale_pdf_values(pdf, grid, data_values, edges, range=parameters.FULL, component=None):
    '''
    Scale the PDF values given the data values that have already been
    plotted using a defined set of edges.

    :param pdf: PDF to work with.
    :type pdf: PDF
    :param grid: evaluation grid.
    :type grid: numpy.ndarray or reikna.cluda.Array
    :param data_values: data points to use for normalization
    :type data_values: numpy.ndarray
    :param edges: bounds of the bins
    :type edges: numpy.ndarray or tuple(numpy.ndarray, ...)
    :param range: normalization range.
    :type range: str
    :param component: if provided, then "pdf" is assumed to be a :class:`AddPDFs` \
    class, and the values associated to the given component will be calculated.
    :type component: str
    :returns: normalized values of the PDF
    :rtype: numpy.ndarray
    '''
    if component is None:
        return aop.extract_ndarray(pdf(grid, range=range) * np.sum(data_values) * bin_area(edges))
    else:
        i = pdf.pdfs.index(component)
        c = pdf.pdfs[i]
        if pdf.extended:
            return aop.extract_ndarray(pdf.args[i].value * c(grid, range=range) * bin_area(edges))
        else:
            if i == len(pdf.pdfs) - 1:
                y = 1. - \
                    np.sum(np.fromiter(
                        (a.value for a in pdf.args), dtype=types.cpu_type))
            else:
                y = pdf.args[i]
            return y * scale_pdf_values(c(grid, range=range), data_values, edges)
