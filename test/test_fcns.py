########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Test the "fcns" module.
'''
import helpers
import minkit
import numpy as np
import pytest

helpers.configure_logging()


@pytest.mark.minimization
@helpers.setting_seed
def test_binned_maximum_likelihood():
    '''
    Test the "binned_maximum_likelihood" FCN.
    '''
    # Simple fit to a Gaussian
    m = minkit.Parameter('m', bounds=(0, 20))  # bounds to generate data later
    c = minkit.Parameter('c', 10, bounds=(8, 12))
    s = minkit.Parameter('s', 2, bounds=(1, 3))
    g = minkit.Gaussian('gaussian', m, c, s)

    values, edges = np.histogram(
        helpers.rndm_gen.normal(c.value, s.value, 1000), range=m.bounds, bins=100)

    data = minkit.BinnedDataSet.from_ndarray(edges, m, values)

    with helpers.fit_test(g) as test:
        with minkit.minimizer('bml', g, data, minimizer='minuit') as minuit:
            test.result, _ = minuit.migrad()

    # Add constraints
    cc = minkit.Parameter('cc', 10)
    sc = minkit.Parameter('sc', 1)
    gc = minkit.Gaussian('constraint', c, cc, sc)

    with helpers.fit_test(g) as test:
        with minkit.minimizer('bml', g, data, minimizer='minuit', constraints=[gc]) as minuit:
            test.result, _ = minuit.migrad()

    # Test for a composed PDF
    k = minkit.Parameter('k', -0.1, bounds=(-1, 0))
    e = minkit.Exponential('e', m, k)

    y = minkit.Parameter('y', 0.5, bounds=(0, 1))

    pdf = minkit.AddPDFs.two_components('pdf', g, e, y)

    data = pdf.generate(10000)

    values, edges = np.histogram(
        data[m.name].as_ndarray(), range=m.bounds, bins=100)

    data = minkit.BinnedDataSet.from_ndarray(edges, m, values)

    with helpers.fit_test(pdf) as test:
        with minkit.minimizer('bml', pdf, data) as minimizer:
            test.result, _ = minimizer.migrad()

    # Test for a PDF with no "evaluate_binned" function defined
    m = minkit.Parameter('m', bounds=(0, 10))
    a = minkit.Parameter('a', 0)
    theta = minkit.Parameter('theta', 2, bounds=(0, 3))
    alpha = minkit.Parameter('alpha', 0.5)
    beta = minkit.Parameter('beta', 2)
    pdf = minkit.Amoroso('amoroso', m, a, theta, alpha, beta)

    data = pdf.generate(10000)

    values, edges = np.histogram(
        data[m.name].as_ndarray(), range=m.bounds, bins=100)

    data = minkit.BinnedDataSet.from_ndarray(edges, m, values)

    with helpers.fit_test(pdf) as test:
        with minkit.minimizer('bml', pdf, data) as minimizer:
            test.result, _ = minimizer.migrad()


@pytest.mark.minimization
@helpers.setting_seed
def test_binned_extended_maximum_likelihood():
    '''
    Test the "binned_extended_maximum_likelihood" FCN.
    '''
    pdf = helpers.default_add_pdfs(extended=True, yields=('ng', 'ne'))

    ntot = int(pdf.args.get('ng').value + pdf.args.get('ne').value)

    data = pdf.generate(ntot).make_binned(100)

    with helpers.fit_test(pdf) as test:
        with minkit.minimizer('beml', pdf, data) as minimizer:
            test.result, _ = minimizer.migrad()


@pytest.mark.minimization
@helpers.setting_seed
def test_binned_chisquare():
    '''
    Test the "binned_chisquare" FCN.
    '''
    # Single PDF
    m = minkit.Parameter('m', bounds=(0, 20))
    c = minkit.Parameter('c', 10., bounds=(8, 12))
    # all bins must be highly populated
    s = minkit.Parameter('s', 3., bounds=(2, 7))
    g = minkit.Gaussian('gaussian', m, c, s)

    data = g.generate(10000)

    values, edges = np.histogram(
        data[m.name].as_ndarray(), range=m.bounds, bins=100)

    data = minkit.BinnedDataSet.from_ndarray(edges, m, values)

    with helpers.fit_test(g) as test:
        with minkit.minimizer('chi2', g, data) as minimizer:
            test.result, _ = minimizer.migrad()

    # Many PDfs
    k = minkit.Parameter('k', -0.1, bounds=(-1, 0))
    e = minkit.Exponential('exponential', m, k)

    ng = minkit.Parameter('ng', 10000, bounds=(0, 100000))
    ne = minkit.Parameter('ne', 1000, bounds=(0, 100000))

    pdf = minkit.AddPDFs.two_components('pdf', g, e, ng, ne)

    data = pdf.generate(int(ng.value + ne.value))

    values, edges = np.histogram(
        data[m.name].as_ndarray(), range=m.bounds, bins=100)

    data = minkit.BinnedDataSet.from_ndarray(edges, m, values)

    with helpers.fit_test(pdf) as test:
        with minkit.minimizer('chi2', pdf, data) as minimizer:
            test.result, _ = minimizer.migrad()


@pytest.mark.minimization
@helpers.setting_seed
def test_binned_extended_chisquare():
    '''
    Test the "binned_extended_chisquare" FCN.
    '''
    pdf = helpers.default_add_pdfs(extended=True, yields=('ng', 'ne'))

    ntot = int(pdf.args.get('ng').value + pdf.args.get('ne').value)

    data = pdf.generate(ntot).make_binned(100)

    with helpers.fit_test(pdf) as test:
        with minkit.minimizer('echi2', pdf, data) as minimizer:
            test.result, _ = minimizer.migrad()


@pytest.mark.minimization
@helpers.setting_seed
def test_unbinned_extended_maximum_likelihood():
    '''
    Test the "unbinned_extended_maximum_likelihood" FCN.
    '''
    m = minkit.Parameter('m', bounds=(-5, +15))

    # Create an Exponential PDF
    k = minkit.Parameter('k', -0.1, bounds=(-0.2, 0))
    e = minkit.Exponential('exponential', m, k)

    # Create a Gaussian PDF
    c = minkit.Parameter('c', 10., bounds=(8, 12))
    s = minkit.Parameter('s', 1., bounds=(0.5, 2))
    g = minkit.Gaussian('gaussian', m, c, s)

    # Add them together
    ng = minkit.Parameter('ng', 10000, bounds=(0, 100000))
    ne = minkit.Parameter('ne', 1000, bounds=(0, 100000))
    pdf = minkit.AddPDFs.two_components('model', g, e, ng, ne)

    data = pdf.generate(int(ng.value + ne.value))

    with helpers.fit_test(pdf) as test:
        with minkit.minimizer('ueml', pdf, data, minimizer='minuit') as minuit:
            test.result, _ = minuit.migrad()

    # Add constraints
    cc = minkit.Parameter('cc', 10)
    sc = minkit.Parameter('sc', 1)
    gc = minkit.Gaussian('constraint', c, cc, sc)

    with helpers.fit_test(pdf) as test:
        with minkit.minimizer('ueml', pdf, data, minimizer='minuit', constraints=[gc]) as minuit:
            test.result, _ = minuit.migrad()


@pytest.mark.minimization
@helpers.setting_seed
def test_unbinned_maximum_likelihood():
    '''
    Test the "unbinned_maximum_likelihood" FCN.
    '''
    # Simple fit to a Gaussian
    m = minkit.Parameter('m', bounds=(5, 15))
    c = minkit.Parameter('c', 10., bounds=(8, 12))
    s = minkit.Parameter('s', 1., bounds=(0.5, 2))
    g = minkit.Gaussian('gaussian', m, c, s)

    data = g.generate(10000)

    with helpers.fit_test(g) as test:
        with minkit.minimizer('uml', g, data, minimizer='minuit') as minuit:
            test.result, _ = minuit.migrad()

    # Add constraints
    cc = minkit.Parameter('cc', 10)
    sc = minkit.Parameter('sc', 0.1)
    gc = minkit.Gaussian('constraint', c, cc, sc)

    with helpers.fit_test(g) as test:
        with minkit.minimizer('uml', g, data, minimizer='minuit', constraints=[gc]) as minuit:
            test.result, _ = minuit.migrad()
