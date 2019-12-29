'''
Test the "pdfs" module.
'''
import numpy as np
import minkit

minkit.initialize()

# For reproducibility
np.random.seed(98953)


def _compare_with_numpy(pdf, numpy_data, data_par):
    '''
    '''
    # Create the data
    values, edges = np.histogram(numpy_data, bins=100, range=data_par.bounds)

    centers = minkit.DataSet.from_array(
        0.5 * (edges[1:] + edges[:-1]), data_par)

    pv = minkit.extract_ndarray(pdf(centers))

    pdf_values = minkit.scale_pdf_values(pv, values, edges)

    assert np.allclose(np.sum(pdf_values), np.sum(values))


def test_amoroso():
    '''
    Test the "Amoroso" PDF.
    '''
    # This is actually the chi-square distribution with one degree of freedom
    m = minkit.Parameter('m', bounds=(0, 10))
    a = minkit.Parameter('a', 0)
    theta = minkit.Parameter('theta', 2)
    alpha = minkit.Parameter('alpha', 0.5)
    beta = minkit.Parameter('beta', 2)
    pdf = minkit.Amoroso('amoroso', m, a, theta, alpha, beta)

    data = np.random.chisquare(2, 100000)

    _compare_with_numpy(pdf, data, m)


def test_exponential():
    '''
    Test the "Exponential" PDF
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    k = minkit.Parameter('k', -0.05, bounds=(-0.1, 0))
    e = minkit.Exponential('exponential', m, k)

    data = np.random.exponential(-1. / k.value, 100000)

    _compare_with_numpy(e, data, m)

    assert np.allclose(e.numerical_normalization(), e.norm())


def test_gaussian():
    '''
    Test the "Gaussian" PDF.
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    c = minkit.Parameter('c', 0., bounds=(-2, +2))
    s = minkit.Parameter('s', 1., bounds=(-3, +3))
    g = minkit.Gaussian('gaussian', m, c, s)

    data = np.random.normal(c.value, s.value, 100000)

    _compare_with_numpy(g, data, m)

    assert np.allclose(g.numerical_normalization(), g.norm())


def test_polynomial():
    '''
    Test the "Polynomial" PDF.
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    p1 = minkit.Parameter('p1', 1.)
    p2 = minkit.Parameter('p2', 2.)
    p3 = minkit.Parameter('p3', 3.)

    # Test constant PDF
    pol0 = minkit.Polynomial('pol0', m)

    data = np.random.uniform(-5, 5, 100000)

    _compare_with_numpy(pol0, data, m)

    rndm = pol0.generate(1000)

    assert np.allclose(pol0.numerical_normalization(), pol0.norm())

    # Test straight line
    pol1 = minkit.Polynomial('pol1', m, p1)

    # Test a parabola
    pol2 = minkit.Polynomial('pol2', m, p1, p2)

    # Test a three-degree polynomial
    pol2 = minkit.Polynomial('pol2', m, p1, p2, p3)


def test_chebyshev():
    '''
    Test the "Chebyshev" PDF.
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    p1 = minkit.Parameter('p1', 1.)
    p2 = minkit.Parameter('p2', 2.)
    p3 = minkit.Parameter('p3', 3.)

    # Test constant PDF
    pol0 = minkit.Chebyshev('pol0', m)

    data = np.random.uniform(-5, 5, 100000)

    _compare_with_numpy(pol0, data, m)

    # Test straight line
    pol1 = minkit.Chebyshev('pol1', m, p1)

    # Test a parabola
    pol2 = minkit.Chebyshev('pol2', m, p1, p2)

    # Test a three-degree polynomial
    pol2 = minkit.Chebyshev('pol2', m, p1, p2, p3)
