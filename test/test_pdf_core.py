'''
Test the "pdf_core" module.
'''
from helpers import check_parameters, check_pdfs, check_multi_pdfs, fit_test
import json
import helpers
import numpy as np
import os
import minkit
import pytest

helpers.configure_logging()
minkit.initialize()

# For reproducibility
np.random.seed(98953)


@pytest.mark.pdfs
def test_pdf():
    '''
    General tests for the PDF class.
    '''
    # Create a Polynomial PDF
    m = minkit.Parameter('m', bounds=(0, 10))
    p1 = minkit.Parameter('p1', 0.)
    p2 = minkit.Parameter('p2', 0.)
    p = minkit.Polynomial('polynomial', m, p1, p2)

    m.set_range('sides', [(0, 4), (6, 10)])

    # integral
    assert np.allclose(p.integral(integral_range='full', range='full'), 1.)
    assert np.allclose(p.integral(integral_range='sides', range='full'), 0.8)
    assert np.allclose(p.integral(integral_range='sides', range='sides'), 1.)


@pytest.mark.pdfs
@pytest.mark.multipdf
def test_addpdfs(tmpdir):
    '''
    Test the "AddPDFs" class.
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))

    # Create an Exponential PDF
    k = minkit.Parameter('k', -0.05, bounds=(-0.1, 0))
    e = minkit.Exponential('exponential', m, k)

    # Create a Gaussian PDF
    c = minkit.Parameter('c', 0., bounds=(-2, +2))
    s = minkit.Parameter('s', 1., bounds=(-3, +3))
    g = minkit.Gaussian('gaussian', m, c, s)

    # Add them together
    g2e = minkit.Parameter('g2e', 0.5, bounds=(0, 1))
    pdf = minkit.AddPDFs.two_components('model', g, e, g2e)

    assert len(pdf.all_args) == (1 + len(g.args) + len(e.args))

    gdata = np.random.normal(c.value, s.value, 100000)
    edata = np.random.exponential(-1. / k.value, 100000)
    data = np.concatenate([gdata, edata])

    values, edges = np.histogram(data, bins=100, range=m.bounds)

    centers = minkit.DataSet.from_array(0.5 * (edges[1:] + edges[:-1]), m)

    pv = minkit.aop.extract_ndarray(pdf(centers))

    pdf_values = minkit.scale_pdf_values(pv, values, edges)

    assert np.allclose(np.sum(pdf_values), np.sum(values))

    # Test consteness of the PDFs
    k.constant = True
    assert e.constant and not pdf.constant
    g2e.constant = True
    assert not pdf.constant
    for p in pdf.all_args:
        p.constant = True
    assert pdf.constant

    # Test the JSON conversion
    with open(os.path.join(tmpdir, 'pdf.json'), 'wt') as fi:
        json.dump(minkit.pdf_to_json(pdf), fi)

    with open(os.path.join(tmpdir, 'pdf.json'), 'rt') as fi:
        s = minkit.pdf_from_json(json.load(fi))

    check_multi_pdfs(s, pdf)


@pytest.mark.core
def test_category():
    '''
    Test the "Category" class.
    '''
    with pytest.raises(TypeError):
        c = minkit.Category()


@pytest.mark.pdfs
@pytest.mark.multipdf
def test_constpdf(tmpdir):
    '''
    Test a fit with a constant PDF.
    '''
    m = minkit.Parameter('m', bounds=(0, 10))

    # Create an Exponential PDF
    k = minkit.Parameter('k', -0.05)
    e = minkit.Exponential('exponential', m, k)

    # Create a Gaussian PDF
    c = minkit.Parameter('c', 5., bounds=(0, 10))
    s = minkit.Parameter('s', 1., bounds=(0.1, 3))
    g = minkit.Gaussian('gaussian', m, c, s)

    # Add them together
    g2e = minkit.Parameter('g2e', 0.5, bounds=(0, 1))
    pdf = minkit.AddPDFs.two_components('model', g, e, g2e)

    # Check for "get_values" and "set_values"
    p = pdf.norm()
    pdf.set_values(**pdf.get_values())
    assert np.allclose(p, pdf.norm())

    # Test a simple fit
    data = pdf.generate(10000)

    with fit_test(pdf, rtol=0.05) as test:
        with minkit.unbinned_minimizer('uml', pdf, data, minimizer='minuit') as minuit:
            test.result = minuit.migrad()

    # Test the JSON conversion
    with open(os.path.join(tmpdir, 'pdf.json'), 'wt') as fi:
        json.dump(minkit.pdf_to_json(pdf), fi)

    with open(os.path.join(tmpdir, 'pdf.json'), 'rt') as fi:
        s = minkit.pdf_from_json(json.load(fi))

    check_multi_pdfs(s, pdf)


@pytest.mark.pdfs
@pytest.mark.multipdf
def test_convpdfs(tmpdir):
    '''
    Test the "ConvPDFs" class.
    '''
    m = minkit.Parameter('m', bounds=(-20, +20))

    # Create two Gaussians
    c1 = minkit.Parameter('c1', 0, bounds=(-2, +2))
    s1 = minkit.Parameter('s1', 3, bounds=(0.1, +10))
    g1 = minkit.Gaussian('g1', m, c1, s1)

    c2 = minkit.Parameter('c2', 0, bounds=(-2, +2))
    s2 = minkit.Parameter('s2', 4, bounds=(0.1, +10))
    g2 = minkit.Gaussian('g2', m, c2, s2)

    pdf = minkit.ConvPDFs('convolution', g1, g2)

    data = pdf.generate(1000)

    # Check that the output is another Gaussian with bigger standard deviation
    mean = minkit.aop.sum(data[m.name]) / len(data)
    var = minkit.aop.sum((data[m.name] - mean)**2) / len(data)

    assert np.allclose(var, s1.value**2 + s2.value**2, rtol=0.1)

    # Check that the normalization is correct
    with pdf.bind() as proxy:
        assert np.allclose(proxy.integral(), 1.)
        assert np.allclose(proxy.norm(), 1.)
        assert np.allclose(proxy.numerical_normalization(), 1.)

    # Ordinary check for PDFs
    values, edges = np.histogram(minkit.aop.extract_ndarray(
        data[m.name]), bins=100, range=m.bounds)

    centers = minkit.DataSet.from_array(0.5 * (edges[1:] + edges[:-1]), m)

    pv = minkit.aop.extract_ndarray(pdf(centers))

    pdf_values = minkit.scale_pdf_values(pv, values, edges)

    assert np.allclose(np.sum(pdf_values), np.sum(values), rtol=0.01)

    # Test a fit
    with fit_test(pdf, atol=0.1, rtol=0.05) as test:
        with minkit.unbinned_minimizer('uml', pdf, data, minimizer='minuit') as minuit:
            test.result = minuit.migrad()

    # Test the JSON conversion
    with open(os.path.join(tmpdir, 'pdf.json'), 'wt') as fi:
        json.dump(minkit.pdf_to_json(pdf), fi)

    with open(os.path.join(tmpdir, 'pdf.json'), 'rt') as fi:
        s = minkit.pdf_from_json(json.load(fi))

    check_multi_pdfs(s, pdf)


@pytest.mark.pdfs
@pytest.mark.multipdf
def test_prodpdfs(tmpdir):
    '''
    Test the "ProdPDFs" class.
    '''
    # Create two Gaussians
    mx = minkit.Parameter('mx', bounds=(-5, +5))
    cx = minkit.Parameter('cx', 0., bounds=(-2, +2))
    sx = minkit.Parameter('sx', 1., bounds=(-3, +3))
    gx = minkit.Gaussian('gx', mx, cx, sx)

    my = minkit.Parameter('my', bounds=(-5, +5))
    cy = minkit.Parameter('cy', 0., bounds=(-2, +2))
    sy = minkit.Parameter('sy', 1., bounds=(-3, +3))
    gy = minkit.Gaussian('gy', my, cy, sy)

    pdf = minkit.ProdPDFs('pdf', [gx, gy])

    # Test integration
    assert np.allclose(pdf.norm(), pdf.numerical_normalization())

    # Test consteness of the PDFs
    for p in gx.all_args:
        p.constant = True
    assert gx.constant and not pdf.constant
    for p in gy.all_args:
        p.constant = True
    assert pdf.constant

    # Test the JSON conversion
    with open(os.path.join(tmpdir, 'pdf.json'), 'wt') as fi:
        json.dump(minkit.pdf_to_json(pdf), fi)

    with open(os.path.join(tmpdir, 'pdf.json'), 'rt') as fi:
        s = minkit.pdf_from_json(json.load(fi))

    check_multi_pdfs(s, pdf)


@pytest.mark.pdfs
@pytest.mark.source_pdf
def test_sourcepdf(tmpdir):
    '''
    Test the "SourcePDF" class.
    '''
    # Test the construction of a normal PDF
    m = minkit.Parameter('m', bounds=(-5, +5))
    c = minkit.Parameter('c', 0., bounds=(-2, +2))
    s = minkit.Parameter('s', 1., bounds=(-3, +3))
    g = minkit.Gaussian('gaussian', m, c, s)

    # Test the construction of a PDF with variable number of arguments
    m = minkit.Parameter('m', bounds=(-5, +5))
    p1 = minkit.Parameter('p1', 1.)
    p2 = minkit.Parameter('p2', 2.)
    pol0 = minkit.Polynomial('pol0', m)
    pol1 = minkit.Polynomial('pol1', m, p1)
    pol2 = minkit.Polynomial('pol2', m, p1, p2)

    # Test the JSON conversion
    with open(os.path.join(tmpdir, 'pdf.json'), 'wt') as fi:
        json.dump(minkit.pdf_to_json(pol0), fi)

    with open(os.path.join(tmpdir, 'pdf.json'), 'rt') as fi:
        s = minkit.pdf_from_json(json.load(fi))

    check_pdfs(s, pol0)
