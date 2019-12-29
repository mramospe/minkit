'''
Test the "pdf_core" module.
'''
import json
import numpy as np
import os
import pyfit
import pytest

pyfit.initialize()

# For reproducibility
np.random.seed(98953)


def _check_parameters( f, s ):
    '''
    Check that two parameters have the same values for the attributes.
    '''
    for attr in ('name', 'value', 'error', 'constant'):
        assert getattr(f, attr) == getattr(s, attr)
    assert np.all(np.array(f.bounds) == np.array(s.bounds))


def _check_pdfs( f, s ):
    '''
    Check that two PDFs have the same values for the attributes.
    '''
    assert f.name == s.name
    for fa, sa in zip(f.args.values(), s.args.values()):
        _check_parameters(fa, sa)
    for fa, sa in zip(f.data_pars.values(), s.data_pars.values()):
        _check_parameters(fa, sa)


def _check_multi_pdfs( f, s ):
    '''
    Check that two PDFs have the same values for the attributes.
    '''
    assert f.name == s.name
    for fa, sa in zip(f.args.values(), s.args.values()):
        _check_parameters(fa, sa)
    for fa, sa in zip(f.data_pars.values(), s.data_pars.values()):
        _check_parameters(fa, sa)
    for fp, sp in zip(f.pdfs.values(), s.pdfs.values()):
        _check_pdfs(fp, sp)


def test_pdf():
    '''
    General tests for the PDF class.
    '''
    # Create a Polynomial PDF
    m = pyfit.Parameter('m', bounds=(0, 10))
    p1 = pyfit.Parameter('p1', 0.)
    p2 = pyfit.Parameter('p2', 0.)
    p = pyfit.Polynomial('polynomial', m, p1, p2)

    m.set_range('sides', [(0, 4), (6, 10)])

    # integral
    assert np.allclose(p.integral(integral_range='full', range='full'), 1.)
    assert np.allclose(p.integral(integral_range='sides', range='full'), 0.8)
    assert np.allclose(p.integral(integral_range='sides', range='sides'), 1.)

    # We can call the PDF over a set of data providing only some of the values
    data = p.generate(1000, values={'p1': 1.})
    p(data, values={'p1': 1.})


def test_addpdfs( tmp_path ):
    '''
    Test the "AddPDFs" class.
    '''
    m = pyfit.Parameter('m', bounds=(-5, +5))

    # Create an Exponential PDF
    k = pyfit.Parameter('k', -0.05, bounds=(-0.1, 0))
    e = pyfit.Exponential('exponential', m, k)

    # Create a Gaussian PDF
    c = pyfit.Parameter('c', 0., bounds=(-2, +2))
    s = pyfit.Parameter('s', 1., bounds=(-3, +3))
    g = pyfit.Gaussian('gaussian', m, c, s)

    # Add them together
    g2e = pyfit.Parameter('g2e', 0.5, bounds=(0, 1))
    pdf = pyfit.AddPDFs.two_components('model', g, e, g2e)

    assert len(pdf.all_args) == (1 + len(g.args) + len(e.args))

    gdata = np.random.normal(c.value, s.value, 100000)
    edata = np.random.exponential(-1. / k.value, 100000)
    data = np.concatenate([gdata, edata])

    values, edges = np.histogram(data, bins=100, range=m.bounds)

    centers = pyfit.DataSet.from_array(0.5 * (edges[1:] + edges[:-1]), m)

    pv = pyfit.extract_ndarray(pdf(centers))

    pdf_values = pyfit.scale_pdf_values(pv, values, edges)

    assert np.allclose(np.sum(pdf_values), np.sum(values))

    # Test consteness of the PDFs
    k.constant = True
    assert e.constant and not pdf.constant
    g2e.constant = True
    assert not pdf.constant
    for p in pdf.all_args.values():
        p.constant = True
    assert pdf.constant

    # Test the JSON conversion
    with open(os.path.join(tmp_path, 'pdf.json'), 'wt') as fi:
        json.dump(pdf.to_json_object(), fi)

    with open(os.path.join(tmp_path, 'pdf.json'), 'rt') as fi:
        s = pyfit.PDF.from_json_object(json.load(fi))

    _check_multi_pdfs(s, pdf)


def test_category():
    '''
    Test the "Category" class.
    '''
    with pytest.raises(TypeError):
        c = pyfit.Category()


def test_constpdf( tmp_path ):
    '''
    Test a fit with a constant PDF.
    '''
    m = pyfit.Parameter('m', bounds=(0, 10))

    # Create an Exponential PDF
    k = pyfit.Parameter('k', -0.05)
    e = pyfit.Exponential('exponential', m, k)

    # Create a Gaussian PDF
    c = pyfit.Parameter('c', 5., bounds=(0, 10))
    s = pyfit.Parameter('s', 1., bounds=(0.1, 3))
    g = pyfit.Gaussian('gaussian', m, c, s)

    # Add them together
    g2e = pyfit.Parameter('g2e', 0.5, bounds=(0, 1))
    pdf = pyfit.AddPDFs.two_components('model', g, e, g2e)

    data = pdf.generate(10000)

    with pyfit.unbinned_minimizer('uml', pdf, data, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed

    results = pyfit.migrad_output_to_registry(r)

    for n, p in results.items():
        assert np.allclose(pdf.all_args[n].value, p.value, rtol=0.05)

    # Test the JSON conversion
    with open(os.path.join(tmp_path, 'pdf.json'), 'wt') as fi:
        json.dump(pdf.to_json_object(), fi)

    with open(os.path.join(tmp_path, 'pdf.json'), 'rt') as fi:
        s = pyfit.PDF.from_json_object(json.load(fi))

    _check_multi_pdfs(s, pdf)


def test_convpdfs( tmp_path ):
    '''
    Test the "ConvPDFs" class.
    '''
    m = pyfit.Parameter('m', bounds=(-20, +20))

    # Create two Gaussians
    c1 = pyfit.Parameter('c1', 0., bounds=(-2, +2))
    s1 = pyfit.Parameter('s1', 3., bounds=(0.1, +10))
    g1 = pyfit.Gaussian('g1', m, c1, s1)

    c2 = pyfit.Parameter('c2', 0., bounds=(-2, +2))
    s2 = pyfit.Parameter('s2', 4., bounds=(0.1, +10))
    g2 = pyfit.Gaussian('g2', m, c2, s2)

    pdf = pyfit.ConvPDFs('convolution', g1, g2)

    data = pdf.generate(size=10000)

    # Check that the output is another Gaussian with bigger standard deviation
    mean = np.sum(data[m.name]) / len(data)
    var  = np.sum((data[m.name] - mean)**2) / len(data)

    assert np.allclose(var, s1.value**2 + s2.value**2, rtol=0.1)

    # Check that the normalization is correct
    with pdf.bind() as proxy:
        assert np.allclose(proxy.integral(), 1.)
        assert np.allclose(proxy.norm(), 1.)
        assert np.allclose(proxy.numerical_normalization(), 1.)

    # Ordinary check for PDFs
    values, edges = np.histogram(pyfit.extract_ndarray(data[m.name]), bins=100, range=m.bounds)

    centers = pyfit.DataSet.from_array(0.5 * (edges[1:] + edges[:-1]), m)

    pv = pyfit.extract_ndarray(pdf(centers))

    pdf_values = pyfit.scale_pdf_values(pv, values, edges)

    assert np.allclose(np.sum(pdf_values), np.sum(values), rtol=0.01)

    # Test a fit
    with pyfit.unbinned_minimizer('uml', pdf, data, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed

    # Test the JSON conversion
    with open(os.path.join(tmp_path, 'pdf.json'), 'wt') as fi:
        json.dump(pdf.to_json_object(), fi)

    with open(os.path.join(tmp_path, 'pdf.json'), 'rt') as fi:
        s = pyfit.PDF.from_json_object(json.load(fi))

    _check_multi_pdfs(s, pdf)


def test_prodpdfs( tmp_path ):
    '''
    Test the "ProdPDFs" class.
    '''
    # Create two Gaussians
    mx = pyfit.Parameter('mx', bounds=(-5, +5))
    cx = pyfit.Parameter('cx', 0., bounds=(-2, +2))
    sx = pyfit.Parameter('sx', 1., bounds=(-3, +3))
    gx = pyfit.Gaussian('gx', mx, cx, sx)

    my = pyfit.Parameter('my', bounds=(-5, +5))
    cy = pyfit.Parameter('cy', 0., bounds=(-2, +2))
    sy = pyfit.Parameter('sy', 1., bounds=(-3, +3))
    gy = pyfit.Gaussian('gy', my, cy, sy)

    pdf = pyfit.ProdPDFs('pdf', [gx, gy])

    # Test integration
    assert np.allclose(pdf.norm(), pdf.numerical_normalization())

    # Test consteness of the PDFs
    for p in gx.all_args.values():
        p.constant = True
    assert gx.constant and not pdf.constant
    for p in gy.all_args.values():
        p.constant = True
    assert pdf.constant

    # Test the JSON conversion
    with open(os.path.join(tmp_path, 'pdf.json'), 'wt') as fi:
        json.dump(pdf.to_json_object(), fi)

    with open(os.path.join(tmp_path, 'pdf.json'), 'rt') as fi:
        s = pyfit.PDF.from_json_object(json.load(fi))

    _check_multi_pdfs(s, pdf)


def test_sourcepdf( tmp_path ):
    '''
    Test the "SourcePDF" class.
    '''
    # Test the construction of a normal PDF
    m = pyfit.Parameter('m', bounds=(-5, +5))
    c = pyfit.Parameter('c', 0., bounds=(-2, +2))
    s = pyfit.Parameter('s', 1., bounds=(-3, +3))
    g = pyfit.Gaussian('gaussian', m, c, s)

    # Test the construction of a PDF with variable number of arguments
    m = pyfit.Parameter('m', bounds=(-5, +5))
    p1 = pyfit.Parameter('p1', 1.)
    p2 = pyfit.Parameter('p2', 2.)
    pol0 = pyfit.Polynomial('pol0', m)
    pol1 = pyfit.Polynomial('pol1', m, p1)
    pol2 = pyfit.Polynomial('pol2', m, p1, p2)

    # Test the JSON conversion
    with open(os.path.join(tmp_path, 'pdf.json'), 'wt') as fi:
        json.dump(pol0.to_json_object(), fi)

    with open(os.path.join(tmp_path, 'pdf.json'), 'rt') as fi:
        s = pyfit.PDF.from_json_object(json.load(fi))

    _check_pdfs(s, pol0)
