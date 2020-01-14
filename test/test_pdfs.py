'''
Test the "pdfs" module.
'''
from helpers import compare_with_numpy
import helpers
import minkit
import numpy as np
import pytest

helpers.configure_logging()
minkit.initialize()

# For reproducibility
np.random.seed(98953)


@pytest.mark.pdfs
@pytest.mark.source_pdf
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

    compare_with_numpy(pdf, data, m)


@pytest.mark.pdfs
@pytest.mark.source_pdf
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

    compare_with_numpy(pol0, data, m)

    # Test straight line
    pol1 = minkit.Chebyshev('pol1', m, p1)

    # Test a parabola
    pol2 = minkit.Chebyshev('pol2', m, p1, p2)

    # Test a three-degree polynomial
    pol2 = minkit.Chebyshev('pol2', m, p1, p2, p3)


@pytest.mark.pdfs
@pytest.mark.source_pdf
def test_crystalball():
    '''
    Test the "Crystal-Ball" PDF.
    '''
    m = minkit.Parameter('m', bounds=(460, 540))
    c = minkit.Parameter('c', 500)
    s = minkit.Parameter('s', 5)
    a = minkit.Parameter('a', 10000)
    n = minkit.Parameter('n', 2)
    cb = minkit.CrystalBall('crystal-ball', m, c, s, a, n)

    data = np.random.normal(500, 5, 100000)

    # For a very large value of "a", it behaves as a Gaussian
    compare_with_numpy(cb, data, m)

    # The same stands if the tail is flipped
    a.value = - a.value
    compare_with_numpy(cb, data, m)

    # Test the normalization
    assert np.allclose(cb.integral(), 1)
    a.value = +1
    assert np.allclose(cb.numerical_normalization(), cb.norm())
    a.value = -1
    assert np.allclose(cb.numerical_normalization(), cb.norm())


@pytest.mark.pdfs
@pytest.mark.source_pdf
def test_exponential():
    '''
    Test the "Exponential" PDF
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    k = minkit.Parameter('k', -0.05, bounds=(-0.1, 0))
    e = minkit.Exponential('exponential', m, k)

    data = np.random.exponential(-1. / k.value, 100000)

    compare_with_numpy(e, data, m)

    assert np.allclose(e.numerical_normalization(), e.norm())


@pytest.mark.pdfs
@pytest.mark.source_pdf
def test_gaussian():
    '''
    Test the "Gaussian" PDF.
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    c = minkit.Parameter('c', 0., bounds=(-2, +2))
    s = minkit.Parameter('s', 1., bounds=(-3, +3))
    g = minkit.Gaussian('gaussian', m, c, s)

    data = np.random.normal(c.value, s.value, 100000)

    compare_with_numpy(g, data, m)

    assert np.allclose(g.numerical_normalization(), g.norm())


@pytest.mark.pdfs
@pytest.mark.source_pdf
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

    compare_with_numpy(pol0, data, m)

    rndm = pol0.generate(1000)

    assert np.allclose(pol0.integral(), 1)
    assert np.allclose(pol0.numerical_normalization(), pol0.norm())

    # Test straight line
    pol1 = minkit.Polynomial('pol1', m, p1)

    assert np.allclose(pol1.integral(), 1)
    assert np.allclose(pol1.numerical_normalization(), pol1.norm())

    # Test a parabola
    pol2 = minkit.Polynomial('pol2', m, p1, p2)

    assert np.allclose(pol2.integral(), 1)
    assert np.allclose(pol2.numerical_normalization(), pol2.norm())

    # Test a three-degree polynomial
    pol3 = minkit.Polynomial('pol3', m, p1, p2, p3)

    assert np.allclose(pol3.integral(), 1)
    assert np.allclose(pol3.numerical_normalization(), pol3.norm())


@pytest.mark.pdfs
@pytest.mark.source_pdf
def test_powerlaw():
    '''
    Test the "PowerLaw" PDF.
    '''
    m = minkit.Parameter('m', bounds=(460, 540))
    c = minkit.Parameter('c', 400)
    n = minkit.Parameter('n', 2)
    pl = minkit.PowerLaw('power-law', m, c, n)

    # Test the normalization
    assert np.allclose(pl.integral(), 1)
    assert np.allclose(pl.numerical_normalization(), pl.norm())
