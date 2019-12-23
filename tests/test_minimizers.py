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


def test_create_minuit():
    '''
    Test the "create_minuit" function.
    '''
    m = pyfit.Parameter('m', bounds=(20, 80))
    c = pyfit.Parameter('c', 50, bounds=(30, 70))
    s = pyfit.Parameter('s', 5, bounds=(1, 10))
    g = pyfit.Gaussian('gaussian', m, c, s)

    arr = np.random.normal(50, 5, 10000)

    data = pyfit.DataSet.from_array(arr, m)

    with pyfit.create_minuit('uml', g, data) as minuit:
        pytest.shared_result = minuit.migrad()

    pytest.shared_names = [s for s in g.all_args]


def test_migrad_output_to_registry():
    '''
    Test the "migrad_output_to_registry" function.
    '''
    r = pyfit.migrad_output_to_registry(pytest.shared_result)
    assert all(s in r for s in pytest.shared_names)
