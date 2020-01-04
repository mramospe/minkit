'''
Test the "minimizers" module.
'''
import helpers
import numpy as np
import minkit
import pytest

helpers.configure_logging()
minkit.initialize()

# For reproducibility
np.random.seed(98953)


def pytest_namespace():
    '''
    Variables shared among tests.
    '''
    return {'shared_names': None, 'shared_result': None}


@pytest.mark.minimization
def test_unbinned_minimizer():
    '''
    Test the "unbinned_minimizer" function.
    '''
    m = minkit.Parameter('m', bounds=(20, 80))
    c = minkit.Parameter('c', 50, bounds=(30, 70))
    s = minkit.Parameter('s', 5, bounds=(1, 10))
    g = minkit.Gaussian('gaussian', m, c, s)

    arr = np.random.normal(50, 5, 10000)

    data = minkit.DataSet.from_array(arr, m)

    with minkit.unbinned_minimizer('uml', g, data, minimizer='minuit') as minuit:
        pytest.shared_result = minuit.migrad()

    print(pytest.shared_result)

    assert not pytest.shared_result.fmin.hesse_failed

    pytest.shared_names = [s for s in g.all_args]

    # Unweighted fit to uniform distribution fails
    arr = np.random.uniform(*m.bounds, 10000)
    data = minkit.DataSet.from_array(arr, m)

    with minkit.unbinned_minimizer('uml', g, data, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    results = minkit.migrad_output_to_registry(r)

    assert not all(np.allclose(
        p.value, g.all_args[n].value, rtol=0.05) for n, p in results.items())

    # With weights fits correctly
    data.weights = g(data)

    with minkit.unbinned_minimizer('uml', g, data, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    results = minkit.migrad_output_to_registry(r)

    assert all(np.allclose(
        p.value, g.all_args[n].value, rtol=0.05) for n, p in results.items())


@pytest.mark.minimization
def test_simultaneous_minimizer():
    '''
    Test the "simultaneous_minimizer" function.
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))

    # First Gaussian
    c1 = minkit.Parameter('c1', 0., bounds=(-2, +2))
    s1 = minkit.Parameter('s1', 1., bounds=(-3, +3))
    g1 = minkit.Gaussian('g1', m, c1, s1)

    data1 = g1.generate(size=1000)

    # Second Gaussian
    c2 = minkit.Parameter('c2', 0., bounds=(-2, +2))
    s2 = minkit.Parameter('s2', 1., bounds=(-3, +3))
    g2 = minkit.Gaussian('g2', m, c2, s2)

    data2 = g2.generate(size=10000)

    categories = [minkit.Category('uml', g1, data1),
                  minkit.Category('uml', g2, data2)]

    with minkit.simultaneous_minimizer(categories, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed


@pytest.mark.minimization
def test_migrad_output_to_registry():
    '''
    Test the "migrad_output_to_registry" function.
    '''
    r = minkit.migrad_output_to_registry(pytest.shared_result)
    assert all(s in r for s in pytest.shared_names)
