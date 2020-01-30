'''
Some utilities to plot data and PDFs using matplotlib.
'''
import numpy as np
from . import parameters
from .core import aop
from .dataset import evaluation_grid
from .operations import types

__all__ = ['pdf_centers_values']


def bin_area(edges):
    '''
    Calculate the area associated to a given set of edges.

    :param edges: bounds of the bins
    :type edges: numpy.ndarray or tuple(numpy.ndarray, ...)
    :returns: area of the bins.
    :rtype: float
    '''
    edges = np.array(edges)
    if len(edges.shape) > 1:
        m = tuple(c.flatten()
                  for c in np.meshgrid(*tuple(e[1:] - e[:-1] for e in edges)))
        area = np.prod(m, axis=0)
    else:
        area = edges[1:] - edges[:-1]

    # For the moment we do not allow non-uniform binning
    return area[0]


def scaled_pdf_values(pdf, grid, data_values, edges, range=parameters.FULL, component=None):
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
        return aop.extract_ndarray(pdf(grid, range=range) * np.sum(data_values)) * bin_area(edges)
    else:
        i = pdf.pdfs.index(component)
        c = pdf.pdfs[i]
        if pdf.extended:
            return aop.extract_ndarray(pdf.args[i].value * c(grid, range=range)) * bin_area(edges)
        else:
            if i == len(pdf.pdfs) - 1:
                y = 1. - \
                    np.sum(np.fromiter(
                        (a.value for a in pdf.args), dtype=types.cpu_type))
            else:
                y = pdf.args[i]
            return y * scaled_pdf_values(c(grid, range=range), data_values, edges)


def calculate_projection(grid, pdf_values, edges, projection, size):
    '''
    Calculate the values of a projection after the evaluation of a PDF on a grid.

    :param grid: evaluation grid.
    :type gird: DataSet
    :param pdf_values: values of the PDF.
    :type pdf_values: numpy.ndarray
    :param edges: edges of the data sample in every dimension.
    :type edges: numpy.ndarray
    :param projection: name of the parameter for the projection.
    :type projection: str
    :param size: size used for the evaluation grid.
    :type size: int
    :returns: centers and values of the PDF.
    :rtype: numpy.ndarray, numpy.ndarray
    '''
    i = grid.data_pars.index(projection)

    pdf_values = pdf_values.reshape(np.full(len(grid.data_pars), size))

    # Product of the ratio of bin areas between data and grid
    r = np.prod(np.fromiter(((len(e) - 1) / size for j,
                             e in enumerate(edges) if j != i), dtype=types.cpu_type))

    c = aop.extract_ndarray(grid[projection])[::size**i][:size]
    v = r * np.sum(pdf_values, axis=i)

    return c, v


def pdf_centers_values(pdf, data_values, edges, range=parameters.FULL, component=None, projection=None, size=1000):
    '''
    Scale the PDF values given the data values that have already been
    plotted using a defined set of edges.

    :param pdf: PDF to work with.
    :type pdf: PDF
    :param data_values: data points to use for normalization
    :type data_values: numpy.ndarray
    :param edges: bounds of the bins
    :type edges: numpy.ndarray or tuple(numpy.ndarray, ...)
    :param range: normalization range.
    :type range: str
    :param component: if provided, then "pdf" is assumed to be a :class:`AddPDFs` \
    class, and the values associated to the given component will be calculated.
    :type component: str
    :param projection: calculate the projection on a given variable.
    :type projection: str
    :param size: number of points to evaluate. If the range is disjoint, then this \
    size corresponds to the number of points per subrange.
    :type size: int
    :returns: normalized values of the PDF and tuple with the centers in each \
    dimension. In the 1D case, the array of centers is directly returned. \
    If the range is disjoint, the result is a tuple of the previously mentioned \
    quantities, one per subrange.
    :rtype: numpy.ndarray or tuple(numpy.ndarray, ...), numpy.ndarray

    .. note:: The input edges must be consistent with the range for plotting.
    '''
    bounds = parameters.bounds_for_range(pdf.data_pars, range)

    if len(bounds.shape) == 1:

        grid = evaluation_grid(pdf.data_pars, bounds, size)

        pdf_values = scaled_pdf_values(
            pdf, grid, data_values, edges, range, component)
        centers = aop.extract_ndarray(grid[pdf.data_pars[0].name])

        if projection is None:
            return centers, pdf_values
        else:
            return calculate_projection(grid, pdf_values, edges, projection, size)
    else:
        centers, values = [], []

        for bds in bounds:

            # Copy the array of values and edges
            dv = np.array(data_values)
            ed = np.array(edges)

            # Get only the elements that are inside the bounds
            for i, (l, r) in zip(bds[0::2], bds[1::2]):

                c = 0.5 * (ed[i][1:] + ed[i][:-1])

                cond = np.logical_and(c >= l, c <= r)

                dv = dv[cond]

                me = np.ones(len(ed[i]), dtype=np.bool)

                me[0], me[-1] = cond[0], cond[-1]

                me[1:-1] = np.logical_and(cond[1:], cond[:-1])

                ed[i] = ed[i][me]

            grid = evaluation_grid(pdf.data_pars, bds, size)

            pdf_values = scaled_pdf_values(pdf, grid, dv, ed, range, component)

            # Reduce the centers and modify the values if asking for a projection
            if projection is None:
                centers.append(tuple(aop.extract_ndarray(
                    grid[p.name]) for p in grid.data_pars))
                values.append(p)
            else:
                c, v = calculate_projection(
                    grid, pdf_values, ed, projection, size)
                centers.append(c)
                values.append(v)

        return tuple(centers), tuple(values)
