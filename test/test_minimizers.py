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

    initials = g.get_values()

    arr = np.random.normal(c.value, s.value, 10000)

    data = minkit.DataSet.from_array(arr, m)

    with helpers.fit_test(g, rtol=0.05) as test:
        with minkit.unbinned_minimizer('uml', g, data, minimizer='minuit') as minuit:
            test.result = pytest.shared_result = minuit.migrad()

    pytest.shared_names = [p.name for p in g.all_args]

    # Unweighted fit to uniform distribution fails
    arr = np.random.uniform(*m.bounds, 100000)
    data = minkit.DataSet.from_array(arr, m)

    with minkit.unbinned_minimizer('uml', g, data, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    reg = minkit.minuit_to_registry(r)

    assert not np.allclose(reg.get(s.name).value, initials[s.name])

    # With weights fits correctly
    data.weights = minkit.aop.extract_ndarray(g(data))

    with helpers.fit_test(g, rtol=0.05) as test:
        with minkit.unbinned_minimizer('uml', g, data, minimizer='minuit') as minuit:
            test.result = minuit.migrad()


@pytest.mark.minimization
def test_simultaneous_minimizer():
    '''
    Test the "simultaneous_minimizer" function.
    '''
    m = minkit.Parameter('m', bounds=(10, 20))

    # Common mean
    s = minkit.Parameter('s', 1, bounds=(0.1, +3))

    # First Gaussian
    c1 = minkit.Parameter('c1', 15, bounds=(10, 20))
    g1 = minkit.Gaussian('g1', m, c1, s)

    data1 = g1.generate(size=1000)

    # Second Gaussian
    c2 = minkit.Parameter('c2', 15, bounds=(10, 20))
    g2 = minkit.Gaussian('g2', m, c2, s)

    data2 = g2.generate(size=10000)

    categories = [minkit.Category('uml', g1, data1),
                  minkit.Category('uml', g2, data2)]

    with helpers.fit_test(categories, rtol=0.05, simultaneous=True) as test:
        with minkit.simultaneous_minimizer(categories, minimizer='minuit') as minuit:
            test.result = minuit.migrad()


@pytest.mark.minimization
def test_minuit_to_registry():
    '''
    Test the "minuit_to_registry" function.
    '''
    r = minkit.minuit_to_registry(pytest.shared_result)
    assert all(n in r.names for n in pytest.shared_names)
