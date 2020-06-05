########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Some utilities to plot data and PDFs using matplotlib.
'''
from ..base import parameters
from ..base import data_types
from ..pdfs import dataset

import numpy as np

__all__ = ['data_plotting_arrays', 'pdf_plotting_arrays']


def bin_area(edges):
    '''
    Calculate the area associated to a given set of edges.

    :param edges: bounds of the bins
    :type edges: numpy.ndarray or tuple(numpy.ndarray, ...)
    :returns: Area of the bins.
    :rtype: float
    '''
    edges = data_types.array_float(edges)
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
    :type grid: DataSet
    :param data_values: data points to use for normalization
    :type data_values: numpy.ndarray
    :param edges: bounds of the bins
    :type edges: numpy.ndarray or tuple(numpy.ndarray, ...)
    :param range: normalization range.
    :type range: str
    :param component: if provided, then *pdf* is assumed to be a :class:`AddPDFs` \
    class, and the values associated to the given component will be calculated.
    :type component: str
    :returns: Normalized values of the PDF
    :rtype: numpy.ndarray
    '''
    if component is None:
        return (pdf(grid, range=range) * np.sum(data_values)).as_ndarray() * bin_area(edges)
    else:
        i = pdf.pdfs.index(component)
        c = pdf.pdfs[i]
        if pdf.extended:
            return (pdf.args[i].value * c(grid, range=range)).as_ndarray() * bin_area(edges)
        else:
            if i == len(pdf.pdfs) - 1:
                y = 1. - \
                    np.sum(data_types.fromiter_float(
                        (a.value for a in pdf.args)))
            else:
                y = pdf.args[i].value
            return y * scaled_pdf_values(c, grid, data_values, edges, range)


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
    :returns: Centers and values of the PDF.
    :rtype: numpy.ndarray, numpy.ndarray
    '''
    i = grid.data_pars.index(projection)

    pdf_values = pdf_values.reshape(
        data_types.full_int(grid.ndim, size))

    # Product of the ratio of bin areas between data and grid
    r = np.prod(data_types.fromiter_float(
        ((len(e) - 1) / size for j, e in enumerate(edges) if j != i)))

    c = grid[projection].as_ndarray()[::size**i][:size]
    v = r * np.sum(pdf_values, axis=i)

    return c, v


def data_plotting_arrays(data, **kwargs):
    '''
    Get the values from a data sample for plotting.
    The possibilities for *kwargs* differ from the binned and unbinned cases:

    **Unbinned**

    - *bins* (int or tuple(int, ...)): number of bins per \
    dimension of data.

    - *projection* (str): project the output data in the given dimension.

    - *sw2* (bool): if the sample has weights and is set to True, return also the errors \
    calculated as the square root of the sum of weights per bin: \
    :math:`\\sigma_j = \\sqrt{\\sum_i^n (\\omega_j^i)^2}`

    **Binned**

    - *rebin* (int or tuple(int, ...)): change the bins \
    by mergin *rebin* bins together.

    - *projection* (str): project the output data in the given dimension.

    :param data: data sample.
    :type data: DataSet or BinnedDataSet
    :param kwargs: keyword arguments.
    :type kwargs: dict
    :returns: In the unbinned case, the values, list of edges, and the errors if *sw2* is set \
    to True. In the binned case, the values and the list of edges. If the has only one \
    dimension, the edges are returned as a single array.
    :rtype: numpy.ndarray, (numpy.ndarray or list(numpy.ndarray)), (numpy.ndarray)
    '''
    projection = kwargs.get('projection', None)

    ndim = len(data.data_pars)

    if projection:
        sa = tuple(i for i, p in enumerate(
            data.data_pars) if p.name != projection)

    if data.sample_type == dataset.UNBINNED:

        # Exclusive options for unbinned samples
        bins = kwargs.get('bins', None)
        sw2 = kwargs.get('sw2', None)

        # Make the histogram of values (common for any case)
        if data.weights is not None:
            weights = data.weights.as_ndarray()
        else:
            weights = None

        values, edges = np.histogramdd(tuple(data[p.name].as_ndarray() for p in data.data_pars),
                                       bins=bins,
                                       range=tuple(
                                           p.bounds for p in data.data_pars),
                                       weights=weights)

        if projection:
            values = np.sum(values, axis=sa)
            edges = edges[data.data_pars.index(projection)]

        # Return a single value if we are in 1D
        if ndim == 1:
            edges = edges[0]

        if data.weights is not None and sw2:

            # Must calculate the errors per bin
            clone = dataset.DataSet(
                data.data, data.data_pars, data.weights**2)

            if projection:
                errors = np.sqrt(np.sum(np.histogramdd(tuple(clone[p.name].as_ndarray() for p in clone.data_pars),
                                                       bins=bins,
                                                       range=tuple(
                                                           p.bounds for p in clone.data_pars),
                                                       weights=clone.weights.as_ndarray())[0], axis=sa))
            else:
                errors = np.sqrt(clone.weights.as_ndarray())

            return values, edges, errors
        else:
            return values.T.flatten(), edges
    else:

        # Exclusive options for binned samples
        rebin = kwargs.get('rebin', None)

        if rebin is not None:
            # Rebin the data

            rebin = data_types.array_int(rebin)

            if rebin.ndim == 0:
                # Only one rebinning argument; all the variables will have the same
                kwargs['rebin'] = data_types.full_int(ndim, rebin)
                return data_plotting_arrays(data, **kwargs)
            else:
                if ndim != len(rebin):
                    raise ValueError(
                        'Number of rebinning arguments must be equal to the number of dimensions in data')

                edges = tuple(data[p.name].as_ndarray()
                              for p in data.data_pars)

                centers = tuple(a.flatten() for a in np.meshgrid(
                    *(0.5 * (e[1:] + e[:-1]) for e in edges)))

                for e, b in zip(edges, rebin):
                    if (len(e) - 1) % b:
                        raise ValueError(
                            'Rebinning must be a proper divisor of the number of bins')

                edges = tuple(e[::b] for e, b in zip(edges, rebin))

                v = data.values.as_ndarray()

                values = np.histogramdd(centers, bins=edges, weights=v)[0]
        else:
            edges = tuple(data[p.name].as_ndarray() for p in data.data_pars)
            values = data.values.as_ndarray().reshape(
                tuple(len(e) - 1 for e in edges))

        if projection:
            values = np.sum(values.T, axis=sa)
            edges = edges[data.data_pars.index(projection)]
        else:
            values = values.flatten()

        # Return a single value if we are in 1D
        if ndim == 1:
            edges = edges[0]

        return values, edges


def pdf_plotting_arrays(pdf, data_values, edges, range=parameters.FULL, component=None, projection=None, size=1000):
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
    :param component: if provided, then *pdf* is assumed to be a :class:`AddPDFs` \
    class, and the values associated to the given component will be calculated.
    :type component: str
    :param projection: calculate the projection on a given variable.
    :type projection: str
    :param size: number of points to evaluate. If the range is disjoint, then this \
    size corresponds to the number of points per subrange.
    :type size: int
    :returns: Normalized values of the PDF and tuple with the centers in each \
    dimension. In the 1D case, the array of centers is directly returned. \
    If the range is disjoint, the result is a tuple of the previously mentioned \
    quantities, one per subrange.
    :rtype: numpy.ndarray or tuple(numpy.ndarray, ...), numpy.ndarray
    :raises RuntimeError: If the number of data parameters is greater than one \
    and no projection is specified.

    .. note:: The input edges must be consistent with the range for plotting.
    '''
    bounds = parameters.bounds_for_range(pdf.data_pars, range)

    if len(bounds) == 1:

        grid = dataset.evaluation_grid(
            pdf.aop, pdf.data_pars, bounds[0], size)

        pdf_values = scaled_pdf_values(
            pdf, grid, data_values, edges, range, component)

        if projection is None:
            if grid.ndim > 1:
                raise RuntimeError(
                    'Number of dimensions is greater than one and no projection has been specified')
            centers = grid[grid.data_pars[0].name].as_ndarray()
            return centers, pdf_values
        else:
            return calculate_projection(grid, pdf_values, edges, projection, size)
    else:
        centers, values = [], []

        for bds in bounds:

            # Copy the array of values and edges
            dv = data_types.array_float(data_values)
            ed = data_types.array_float(edges)

            # Get only the elements that are inside the bounds
            for i, (l, u) in zip(*bds):

                c = 0.5 * (ed[i][1:] + ed[i][:-1])

                cond = np.logical_and(c >= l, c <= u)

                dv = dv[cond]

                me = np.ones(len(ed[i]), dtype=np.bool)

                me[0], me[-1] = cond[0], cond[-1]

                me[1:-1] = np.logical_and(cond[1:], cond[:-1])

                ed[i] = ed[i][me]

            grid = dataset.evaluation_grid(pdf.aop, pdf.data_pars, bds, size)

            v = scaled_pdf_values(pdf, grid, dv, ed, range, component)

            # Reduce the centers and modify the values if asking for a projection
            if projection is None:
                centers.append(tuple(grid[p.name].as_ndarray()
                                     for p in grid.data_pars))
                values.append(v)
            else:
                c, v = calculate_projection(
                    grid, pdf_values, ed, projection, size)
                centers.append(c)
                values.append(v)

        return tuple(centers), tuple(values)
