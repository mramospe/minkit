########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Test the "minimizers" module.
'''
import helpers
import numpy as np
import minkit
import pytest

helpers.configure_logging()


def pytest_namespace():
    '''
    Variables shared among tests.
    '''
    return {'shared_names': None, 'shared_result': None}


@pytest.mark.minimization
@helpers.setting_seed
def test_minimizer():
    '''
    Test the "minimizer" function
    '''
    m = minkit.Parameter('m', bounds=(30, 70))
    c = minkit.Parameter('c', 50, bounds=(30, 70))
    s = minkit.Parameter('s', 5, bounds=(1, 10))
    g = minkit.Gaussian('gaussian', m, c, s)

    initials = g.get_values()

    arr = helpers.rndm_gen.normal(c.value, s.value, 10000)

    data = minkit.DataSet.from_ndarray(arr, m)

    with helpers.fit_test(g) as test:
        with minkit.minimizer('uml', g, data, minimizer='minuit') as minuit:
            test.result = pytest.shared_result = minuit.migrad()

    pytest.shared_names = [p.name for p in g.all_args]

    # Unweighted fit to uniform distribution fails
    arr = helpers.rndm_gen.uniform(*m.bounds, 100000)
    data = minkit.DataSet.from_ndarray(arr, m)

    with g.restoring_state(), helpers.fit_test(g, fails=True) as test:
        with minkit.minimizer('uml', g, data, minimizer='minuit') as minuit:
            test.result = minuit.migrad()

        reg = g.args.copy()

        assert not np.allclose(reg.get(s.name).value, initials[s.name])

    # With weights fits correctly
    data = minkit.DataSet(data.values, data.data_pars, weights=g(data))

    with helpers.fit_test(g) as test:
        with minkit.minimizer('uml', g, data, minimizer='minuit') as minuit:
            test.result = minuit.migrad()

    # Test the binned case
    data = data.make_binned(bins=100)

    with helpers.fit_test(g) as test:
        with minkit.minimizer('chi2', g, data, minimizer='minuit') as minuit:
            test.result = minuit.migrad()


@pytest.mark.minimization
@helpers.setting_seed
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
def test_scipyminimizer():
    '''
    Test the "SciPyMinimizer" class.
    '''
    m = minkit.Parameter('m', bounds=(10, 20))
    s = minkit.Parameter('s', 2, bounds=(1.5, 2.5))
    c = minkit.Parameter('c', 15, bounds=(13, 17))
    g = minkit.Gaussian('g', m, c, s)

    # Test the unbinned case
    data = g.generate(10000)

    initials = g.get_values()

    for c in (False, True):

        s.constant = c

        values = []
        for m in minkit.minimization.scipy_api.SCIPY_CHOICES:
            with minkit.minimizer('uml', g, data, minimizer=m) as minimizer:
                g.set_values(**initials)
                minimizer.minimize()
                values.append(g.args.copy())

        with minkit.minimizer('uml', g, data, minimizer='minuit') as minimizer:
            g.set_values(**initials)
            minimizer.migrad()
            reference = g.args.copy()

        for reg in values:
            for p, r in zip(reg, reference):
                helpers.check_parameters(p, r, rtol=0.05)

    # Test the binned case
    data = data.make_binned(bins=100)

    for c in (False, True):

        s.constant = c

        values = []
        for m in minkit.minimization.scipy_api.SCIPY_CHOICES:
            with minkit.minimizer('bml', g, data, minimizer=m) as minimizer:
                g.set_values(**initials)
                minimizer.minimize()
                values.append(g.args.copy())

        with minkit.minimizer('bml', g, data, minimizer='minuit') as minimizer:
            g.set_values(**initials)
            minimizer.migrad()
            reference = g.args.copy()

        for reg in values:
            for p, r in zip(reg, reference):
                helpers.check_parameters(p, r, rtol=0.05)


@pytest.mark.minimization
def test_nloptminimizer():
    '''
    Test the "NLoptMinimizer" class.
    '''
    m = minkit.Parameter('m', bounds=(10, 20))
    s = minkit.Parameter('s', 2, bounds=(1.5, 2.5))
    c = minkit.Parameter('c', 15, bounds=(13, 17))
    g = minkit.Gaussian('g', m, c, s)

    k = minkit.Parameter('k', -0.1, bounds=(-1, 0))
    e = minkit.Exponential('e', m, k)

    y = minkit.Parameter('y', 0.5, bounds=(0.4, 1))

    pdf = minkit.AddPDFs.two_components('pdf', g, e, y)

    # Test the unbinned case
    data = pdf.generate(10000)

    initials = pdf.get_values()

    for c in (False, True):

        s.constant = c

        values = []
        for m in minkit.minimization.nlopt_api.NLOPT_CHOICES:
            with minkit.minimizer('uml', pdf, data, minimizer=m) as minimizer:
                pdf.set_values(**initials)
                minimizer.minimize()
                values.append(pdf.args.copy())

        with minkit.minimizer('uml', pdf, data, minimizer='minuit') as minimizer:
            pdf.set_values(**initials)
            minimizer.migrad()
            reference = pdf.args.copy()

        for reg in values:
            for p, r in zip(reg, reference):
                helpers.check_parameters(p, r, rtol=0.05)

    # Test the binned case
    data = data.make_binned(bins=100)

    for c in (False, True):

        s.constant = c

        values = []
        for m in minkit.minimization.nlopt_api.NLOPT_CHOICES:
            with minkit.minimizer('bml', pdf, data, minimizer=m) as minimizer:
                pdf.set_values(**initials)
                minimizer.minimize()
                values.append(pdf.args.copy())

        with minkit.minimizer('bml', pdf, data, minimizer='minuit') as minimizer:
            pdf.set_values(**initials)
            minimizer.migrad()
            reference = pdf.args.copy()

        for reg in values:
            for p, r in zip(reg, reference):
                helpers.check_parameters(p, r, rtol=0.05)


@pytest.mark.minimization
def test_asymmetric_errors():
    '''
    Test the calculation of asymmetric errors.
    '''
    g = helpers.default_gaussian(sigma='s')

    s = g.args.get('s')

    data = g.generate(1000)

    # Test for 1 sigma
    with minkit.minimizer('uml', g, data) as minimizer:
        minimum = minimizer.minimize()
        minimizer.minuit.print_level = 0
        minimizer.asymmetric_errors('s', sigma=1.)
        errors_default = s.asym_errors
        minimizer.minos('s', 1.)
        errors_minos = s.asym_errors

    assert s.asym_errors != ()

    assert np.allclose(errors_default, errors_minos,
                       rtol=1e-4)  # same as MINOS

    # Test for 2 sigma
    with minkit.minimizer('uml', g, data) as minimizer:
        minimizer.minimize()
        minimizer.minuit.print_level = 0
        minimizer.asymmetric_errors('s', sigma=2.)
        errors = s.asym_errors
        minimizer.minos('s', 2.)

    assert s.asym_errors != ()

    assert np.allclose(s.asym_errors, errors, rtol=1e-4)  # compare with MINOS


@pytest.mark.minimization
def test_minimization_names():
    '''
    Check that the minimization names of the different APIs do not cause
    collisions.
    '''
    minuit_api_names = {minkit.minimization.minuit_api.MINUIT}
    scipy_api_names = set(minkit.minimization.scipy_api.SCIPY_CHOICES)
    nlopt_api_names = set(minkit.minimization.nlopt_api.NLOPT_CHOICES)

    assert len(minuit_api_names.intersection(scipy_api_names)) == 0
    assert len(minuit_api_names.intersection(nlopt_api_names)) == 0
    assert len(scipy_api_names.intersection(nlopt_api_names)) == 0
