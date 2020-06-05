########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Test the "pdfs" module.
'''
from helpers import compare_with_numpy, setting_seed
import helpers
import minkit
import numpy as np
import pytest

helpers.configure_logging()


@pytest.mark.pdfs
@pytest.mark.source_pdf
@setting_seed
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

    assert np.allclose(pdf.integral(), 1)

    data = helpers.rndm_gen.chisquare(2, 100000)

    compare_with_numpy(pdf, data, m, rtol=0.015)


@pytest.mark.pdfs
@pytest.mark.source_pdf
@setting_seed
def test_argus():
    '''
    Test the "Argus" PDF.
    '''
    # This is actually the chi-square distribution with one degree of freedom
    m = minkit.Parameter('m', bounds=(0, 1))
    mu = minkit.Parameter('mu', 0.9, bounds=(0.5, 1))
    c = minkit.Parameter('c', 0.2, bounds=(0.01, 2))
    p = minkit.Parameter('p', 0.5, bounds=(0.1, 1))
    pdf = minkit.Argus('argus', m, mu, c, p)

    assert np.allclose(pdf.integral(), 1)

    m.set_range('reduced', (0, mu.value))

    assert np.allclose(pdf.integral('reduced'), 1)


@pytest.mark.pdfs
@pytest.mark.source_pdf
@setting_seed
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

    data = helpers.rndm_gen.uniform(-5, 5, 100000)

    compare_with_numpy(pol0, data, m)

    # Test straight line
    minkit.Chebyshev('pol1', m, p1)

    # Test a parabola
    minkit.Chebyshev('pol2', m, p1, p2)

    # Test a three-degree polynomial
    minkit.Chebyshev('pol2', m, p1, p2, p3)


@pytest.mark.pdfs
@pytest.mark.source_pdf
@setting_seed
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

    data = helpers.rndm_gen.normal(500, 5, 100000)

    # For a very large value of "a", it behaves as a Gaussian
    compare_with_numpy(cb, data, m)

    # The same stands if the tail is flipped
    a.value = - a.value
    compare_with_numpy(cb, data, m)

    # Test the normalization
    assert np.allclose(cb.integral(), 1)
    a.value = +1
    helpers.check_numerical_normalization(cb)
    a.value = -1
    helpers.check_numerical_normalization(cb)


@pytest.mark.pdfs
@pytest.mark.source_pdf
@setting_seed
def test_exponential():
    '''
    Test the "Exponential" PDF
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    k = minkit.Parameter('k', -0.05, bounds=(-0.1, 0))
    e = minkit.Exponential('exponential', m, k)

    data = helpers.rndm_gen.exponential(-1. / k.value, 100000)

    compare_with_numpy(e, data, m)

    helpers.check_numerical_normalization(e)


@pytest.mark.pdfs
@pytest.mark.source_pdf
@setting_seed
def test_exppoly():
    '''
    Test the "ExpPoly" PDF
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    k = minkit.Parameter('k', -0.05, bounds=(-0.1, 0))
    p = minkit.Parameter('p1', 1, bounds=(0, 2))
    pdf = minkit.ExpPoly('exp_poly', m, k, p)

    assert np.allclose(pdf.integral(), 1)
    helpers.check_numerical_normalization(pdf)


@pytest.mark.pdfs
@pytest.mark.source_pdf
@setting_seed
def test_gaussian():
    '''
    Test the "Gaussian" PDF.
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    c = minkit.Parameter('c', 0., bounds=(-2, +2))
    s = minkit.Parameter('s', 1., bounds=(-3, +3))
    g = minkit.Gaussian('gaussian', m, c, s)

    data = helpers.rndm_gen.normal(c.value, s.value, 100000)

    compare_with_numpy(g, data, m)

    helpers.check_numerical_normalization(g)


@pytest.mark.pdfs
@pytest.mark.source_pdf
@setting_seed
def test_landau():
    '''
    Test the "Landau" PDF.
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    c = minkit.Parameter('c', 0., bounds=(-2, +2))
    s = minkit.Parameter('s', 1., bounds=(-3, +3))
    l = minkit.Landau('landau', m, c, s)

    assert np.allclose(l.integral(), 1)
    helpers.check_numerical_normalization(l)


@pytest.mark.pdfs
@pytest.mark.source_pdf
@setting_seed
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

    pol0.numint_config = {'method': 'miser', 'calls': 10000}

    data = helpers.rndm_gen.uniform(-5, 5, 10000)

    compare_with_numpy(pol0, data, m)

    pol0.generate(1000)
    assert np.allclose(pol0.integral(), 1)
    helpers.check_numerical_normalization(pol0)

    # Test straight line
    pol1 = minkit.Polynomial('pol1', m, p1)

    assert np.allclose(pol1.integral(), 1)
    helpers.check_numerical_normalization(pol1)

    # Test a parabola
    pol2 = minkit.Polynomial('pol2', m, p1, p2)

    assert np.allclose(pol2.integral(), 1)
    helpers.check_numerical_normalization(pol2)

    # Test a three-degree polynomial
    pol3 = minkit.Polynomial('pol3', m, p1, p2, p3)

    assert np.allclose(pol3.integral(), 1)
    helpers.check_numerical_normalization(pol3)


@pytest.mark.pdfs
@pytest.mark.source_pdf
@setting_seed
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
    helpers.check_numerical_normalization(pl)
