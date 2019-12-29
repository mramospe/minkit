'''
Some utilities to plot data and PDFs using matplotlib.
'''
import numpy as np

__all__ = ['scale_pdf_values']


def scale_pdf_values(pdf_values, data_values, edges):
    '''
    Scale the PDF values given the data values that have already been
    plotted using a defined set of edges.

    :param pdf_values: values of the PDF
    :type pdf_values: numpy.ndarray or sequence
    :param data_values: data points to use for normalization
    :type data_values: numpy.ndarray or sequence
    :param edges: bounds of the bins
    :type edges: numpy.ndarray or sequence
    :returns: normalized values of the PDF
    :rtype: numpy.ndarray
    '''
    return pdf_values * np.sum(data_values) * (edges[1] - edges[0])
