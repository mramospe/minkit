'''
Test the "minimizers" module.
'''
import helpers
import numpy as np
import minkit
import pytest

helpers.configure_logging()
minkit.initialize()


def pytest_namespace():
    '''
    Variables shared among tests.
    '''
    return {'shared_names': None, 'shared_result': None}


@pytest.mark.minimization
@helpers.setting_numpy_seed
def test_minimizer():
    '''
    Test the "minimizer" function
    '''
    m = minkit.Parameter('m', bounds=(20, 80))
    c = minkit.Parameter('c', 50, bounds=(30, 70))
    s = minkit.Parameter('s', 5, bounds=(1, 10))
    g = minkit.Gaussian('gaussian', m, c, s)

    initials = g.get_values()

    arr = np.random.normal(c.value, s.value, 10000)

    data = minkit.DataSet.from_array(arr, m)

    with helpers.fit_test(g) as test:
        with minkit.minimizer('uml', g, data, minimizer='minuit') as minuit:
            test.result = pytest.shared_result = minuit.migrad()

    pytest.shared_names = [p.name for p in g.all_args]

    # Unweighted fit to uniform distribution fails
    arr = np.random.uniform(*m.bounds, 100000)
    data = minkit.DataSet.from_array(arr, m)

    with minkit.minimizer('uml', g, data, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    reg = minkit.minuit_to_registry(r.params)

    assert not np.allclose(reg.get(s.name).value, initials[s.name])

    # With weights fits correctly
    data.weights = minkit.as_ndarray(g(data))

    with helpers.fit_test(g) as test:
        with minkit.minimizer('uml', g, data, minimizer='minuit') as minuit:
            test.result = minuit.migrad()

    # Test the binned case
    data = data.make_binned(bins=100)

    with helpers.fit_test(g) as test:
        with minkit.minimizer('bml', g, data, minimizer='minuit') as minuit:
            test.result = minuit.migrad()


@pytest.mark.minimization
@helpers.setting_numpy_seed
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

    with helpers.fit_test(categories, simultaneous=True) as test:
        with minkit.simultaneous_minimizer(categories, minimizer='minuit') as minuit:
            test.result = minuit.migrad()


@pytest.mark.minimization
def test_minuit_to_registry():
    '''
    Test the "minuit_to_registry" function.
    '''
    r = minkit.minuit_to_registry(pytest.shared_result.params)
    assert all(n in r.names for n in pytest.shared_names)


@pytest.mark.minimization
def test_scipyminimizer():
    '''
    Test the "SciPyMinimizer" class.
    '''
    m = minkit.Parameter('m', bounds=(10, 20))
    s = minkit.Parameter('s', 1, bounds=(0.5, 2))
    c = minkit.Parameter('c', 15, bounds=(10, 20))
    g = minkit.Gaussian('g', m, c, s)

    # Test the unbinned case
    data = g.generate(10000)

    values = []
    with minkit.minimizer('uml', g, data, minimizer='scipy') as minimizer:
        for m in minkit.minimizers.SCIPY_CHOICES:
            values.append(minimizer.result_to_registry(
                minimizer.minimize(method=m)))

    with minkit.minimizer('uml', g, data, minimizer='minuit') as minimizer:
        reference = minkit.minuit_to_registry(minimizer.migrad().params)

    for reg in values:
        for p, r in zip(reg, reference):
            helpers.check_parameters(p, r, rtol=0.01)

    # Test the binned case
    data = data.make_binned(bins=100)

    values = []
    with minkit.minimizer('bml', g, data, minimizer='scipy') as minimizer:
        for m in minkit.minimizers.SCIPY_CHOICES:
            values.append(minimizer.result_to_registry(
                minimizer.minimize(method=m)))

    with minkit.minimizer('bml', g, data, minimizer='minuit') as minimizer:
        reference = minkit.minuit_to_registry(minimizer.migrad().params)

    for reg in values:
        for p, r in zip(reg, reference):
            helpers.check_parameters(p, r, rtol=0.01)
