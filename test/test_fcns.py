'''
Test the "fcns" module.
'''
import helpers
import minkit
import numpy as np
import pytest

helpers.configure_logging()
minkit.initialize()

# For reproducibility
np.random.seed(98953)


@pytest.mark.minimization
def test_binned_maximum_likelihood():
    '''
    Tets the "binned_maximum_likelihood" FCN.
    '''
    # Simple fit to a Gaussian
    m = minkit.Parameter('m', bounds=(5, 15))
    c = minkit.Parameter('c', 10., bounds=(5, 15))
    s = minkit.Parameter('s', 1., bounds=(0.1, 2))
    g = minkit.Gaussian('gaussian', m, c, s)

    values, edges = np.histogram(
        np.random.normal(c.value, s.value, 10000), bins=100)

    data = minkit.BinnedDataSet.from_array(
        0.5 * (edges[1:] + edges[:-1]), m, values)

    with minkit.binned_minimizer('bml', g, data, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed

    results = minkit.migrad_output_to_registry(r)

    for n, p in results.items():
        assert np.allclose(p.value, g.all_args[n].value, rtol=0.05)

    # Add constraints
    cc = minkit.Parameter('cc', 10)
    sc = minkit.Parameter('sc', 0.1)
    gc = minkit.Gaussian('constraint', c, cc, sc)

    with minkit.binned_minimizer('bml', g, data, minimizer='minuit', constraints=[gc]) as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed

    results = minkit.migrad_output_to_registry(r)

    for n, p in results.items():
        assert np.allclose(g.all_args[n].value, p.value, rtol=0.05)


@pytest.mark.minimization
def test_unbinned_extended_maximum_likelihood():
    '''
    Test the "unbinned_extended_maximum_likelihood" FCN.
    '''
    m = minkit.Parameter('m', bounds=(-5, +15))

    # Create an Exponential PDF
    k = minkit.Parameter('k', -0.1, bounds=(-0.2, 0))
    e = minkit.Exponential('exponential', m, k)

    # Create a Gaussian PDF
    c = minkit.Parameter('c', 10., bounds=(5, 15))
    s = minkit.Parameter('s', 1., bounds=(0.1, 2))
    g = minkit.Gaussian('gaussian', m, c, s)

    # Add them together
    ng = minkit.Parameter('ng', 10000, bounds=(0, 100000))
    ne = minkit.Parameter('ne', 1000, bounds=(0, 100000))
    pdf = minkit.AddPDFs.two_components('model', g, e, ng, ne)

    data = pdf.generate(ng.value + ne.value)

    with minkit.unbinned_minimizer('ueml', pdf, data, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed

    results = minkit.migrad_output_to_registry(r)

    for n, p in results.items():
        assert np.allclose(pdf.all_args[n].value, p.value, rtol=0.1)

    # Add constraints
    cc = minkit.Parameter('cc', 10)
    sc = minkit.Parameter('sc', 0.1)
    gc = minkit.Gaussian('constraint', c, cc, sc)

    with minkit.unbinned_minimizer('ueml', pdf, data, minimizer='minuit', constraints=[gc]) as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed

    results = minkit.migrad_output_to_registry(r)

    for n, p in results.items():
        assert np.allclose(pdf.all_args[n].value, p.value, rtol=0.1)


@pytest.mark.minimization
def test_unbinned_maximum_likelihood():
    '''
    Test the "unbinned_maximum_likelihood" FCN.
    '''
    # Simple fit to a Gaussian
    m = minkit.Parameter('m', bounds=(5, 15))
    c = minkit.Parameter('c', 10., bounds=(5, 15))
    s = minkit.Parameter('s', 1., bounds=(0.1, 2))
    g = minkit.Gaussian('gaussian', m, c, s)

    data = g.generate(10000)

    with minkit.unbinned_minimizer('uml', g, data, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed

    results = minkit.migrad_output_to_registry(r)

    for n, p in results.items():
        assert np.allclose(p.value, g.all_args[n].value, rtol=0.05)

    # Add constraints
    cc = minkit.Parameter('cc', 10)
    sc = minkit.Parameter('sc', 0.1)
    gc = minkit.Gaussian('constraint', c, cc, sc)

    with minkit.unbinned_minimizer('uml', g, data, minimizer='minuit', constraints=[gc]) as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed

    results = minkit.migrad_output_to_registry(r)

    for n, p in results.items():
        assert np.allclose(p.value, g.all_args[n].value, rtol=0.05)
