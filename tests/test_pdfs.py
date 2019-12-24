'''
Test the "pdfs" module.
'''
import numpy as np
import pyfit

pyfit.initialize()

# For reproducibility
np.random.seed(98953)


def _compare_with_numpy( pdf, numpy_data, data_par ):
    '''
    '''
    # Create the data
    values, edges = np.histogram(numpy_data, bins=100, range=data_par.bounds)

    centers = pyfit.DataSet.from_array(0.5 * (edges[1:] + edges[:-1]), data_par)

    pv = pyfit.extract_ndarray(pdf(centers))

    pdf_values = pyfit.scale_pdf_values(pv, values, edges)

    assert np.allclose(np.sum(pdf_values), np.sum(values))


def test_exponential():
    '''
    Test the "Exponential" PDF
    '''
    m = pyfit.Parameter('m', bounds=(-5, +5))
    k = pyfit.Parameter('k', -0.05, bounds=(-0.1, 0))
    e = pyfit.Exponential('exponential', m, k)

    data = np.random.exponential(-1. / k.value, 100000)

    _compare_with_numpy(e, data, m)


def test_gaussian():
    '''
    Test the "Gaussian" PDF.
    '''
    m = pyfit.Parameter('m', bounds=(-5, +5))
    c = pyfit.Parameter('c', 0., bounds=(-2, +2))
    s = pyfit.Parameter('s', 1., bounds=(-3, +3))
    g = pyfit.Gaussian('gaussian', m, c, s)

    data = np.random.normal(c.value, s.value, 100000)

    _compare_with_numpy(g, data, m)


def test_polynomial():
    '''
    Test the "Polynomial" PDF.
    '''
    m = pyfit.Parameter('m', bounds=(-5, +5))
    p1 = pyfit.Parameter('p1', 1.)
    p2 = pyfit.Parameter('p2', 2.)
    p3 = pyfit.Parameter('p3', 3.)

    # Test constant PDF
    pol0 = pyfit.Polynomial('pol0', m)

    data = np.random.uniform(-5, 5, 100000)

    _compare_with_numpy(pol0, data, m)

    # Test straight line
    pol1 = pyfit.Polynomial('pol1', m, p1)

    # Test a parabola
    pol2 = pyfit.Polynomial('pol2', m, p1, p2)

    # Test a three-degree polynomial
    pol2 = pyfit.Polynomial('pol2', m, p1, p2, p3)
