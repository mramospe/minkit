'''
Test the "minimizers" module.
'''
import numpy as np
import pyfit
import pytest


def pytest_namespace():
    '''
    Variables shared among tests.
    '''
    return {'shared_names': None, 'shared_result': None}


def test_unbinned_minimizer():
    '''
    Test the "unbinned_minimizer" function.
    '''
    m = pyfit.Parameter('m', bounds=(20, 80))
    c = pyfit.Parameter('c', 50, bounds=(30, 70))
    s = pyfit.Parameter('s', 5, bounds=(1, 10))
    g = pyfit.Gaussian('gaussian', m, c, s)

    arr = np.random.normal(50, 5, 10000)

    data = pyfit.DataSet.from_array(arr, m)

    with pyfit.unbinned_minimizer('uml', g, data, minimizer='minuit') as minuit:
        pytest.shared_result = minuit.migrad()

    print(pytest.shared_result)

    assert not pytest.shared_result.fmin.hesse_failed

    pytest.shared_names = [s for s in g.all_args]


def test_simultaneous_minimizer():
    '''
    Test the "simultaneous_minimizer" function.
    '''
    m = pyfit.Parameter('m', bounds=(-5, +5))

    # First Gaussian
    c1 = pyfit.Parameter('c1', 0., bounds=(-2, +2))
    s1 = pyfit.Parameter('s1', 1., bounds=(-3, +3))
    g1 = pyfit.Gaussian('g1', m, c1, s1)

    data1 = g1.generate(size=1000)

    # Second Gaussian
    c2 = pyfit.Parameter('c2', 0., bounds=(-2, +2))
    s2 = pyfit.Parameter('s2', 1., bounds=(-3, +3))
    g2 = pyfit.Gaussian('g2', m, c2, s2)

    data2 = g2.generate(size=10000)

    categories = [pyfit.Category('uml', g1, data1), pyfit.Category('uml', g2, data2)]

    with pyfit.simultaneous_minimizer(categories, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed


def test_migrad_output_to_registry():
    '''
    Test the "migrad_output_to_registry" function.
    '''
    r = pyfit.migrad_output_to_registry(pytest.shared_result)
    assert all(s in r for s in pytest.shared_names)
