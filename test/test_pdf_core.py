########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Test the "pdf_core" module.
'''
from helpers import check_pdfs, check_multi_pdfs, fit_test, setting_seed
import json
import helpers
import numpy as np
import os
import minkit
import pytest

helpers.configure_logging()

aop = minkit.backends.core.parse_backend()


@pytest.mark.pdfs
@setting_seed
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
@setting_seed
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

    gdata = helpers.rndm_gen.normal(c.value, s.value, 100000)
    edata = helpers.rndm_gen.exponential(-1. / k.value, 100000)
    data = np.concatenate([gdata, edata])

    values, edges = np.histogram(data, bins=100, range=m.bounds)

    centers = minkit.DataSet.from_ndarray(0.5 * (edges[1:] + edges[:-1]), m)

    pdf_values = minkit.utils.core.scaled_pdf_values(
        pdf, centers, values, edges)

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

    # Check copying the PDF
    pdf.copy()


@pytest.mark.core
def test_category():
    '''
    Test the "Category" class.
    '''
    with pytest.raises(TypeError):
        minkit.Category()


@pytest.mark.pdfs
@pytest.mark.multipdf
@setting_seed
def test_constpdf(tmpdir):
    '''
    Test a fit with a constant PDF.
    '''
    pdf = helpers.default_add_pdfs(extended=False)

    # Check for "get_values" and "set_values"
    p = pdf.norm()
    pdf.set_values(**pdf.get_values())
    assert np.allclose(p, pdf.norm())

    # Test a simple fit
    data = pdf.generate(10000)

    with fit_test(pdf) as test:
        with minkit.minimizer('uml', pdf, data, minimizer='minuit') as minuit:
            test.result = minuit.migrad()

    # Test the JSON conversion
    with open(os.path.join(tmpdir, 'pdf.json'), 'wt') as fi:
        json.dump(minkit.pdf_to_json(pdf), fi)

    with open(os.path.join(tmpdir, 'pdf.json'), 'rt') as fi:
        s = minkit.pdf_from_json(json.load(fi))

    check_multi_pdfs(s, pdf)


@pytest.mark.pdfs
@pytest.mark.multipdf
@setting_seed(seed=431582)
def test_convpdfs(tmpdir):
    '''
    Test the "ConvPDFs" class.
    '''
    m = minkit.Parameter('m', bounds=(-20, +20))

    # Create two Gaussians
    c1 = minkit.Parameter('c1', 0, bounds=(-2, +2))
    s1 = minkit.Parameter('s1', 3, bounds=(0.5, +10))
    g1 = minkit.Gaussian('g1', m, c1, s1)

    c2 = minkit.Parameter('c2', 0, bounds=(-2, +2))
    s2 = minkit.Parameter('s2', 4, bounds=(0.5, +10))
    g2 = minkit.Gaussian('g2', m, c2, s2)

    pdf = minkit.ConvPDFs('convolution', g1, g2)

    data = pdf.generate(10000)

    # Check that the output is another Gaussian with bigger standard deviation
    mean = aop.sum(data[m.name]) / len(data)
    var = aop.sum((data[m.name] - mean)**2) / len(data)

    assert np.allclose(var, s1.value**2 + s2.value**2, rtol=0.1)

    # Ordinary check for PDFs
    values, edges = np.histogram(
        data[m.name].as_ndarray(), bins=100, range=m.bounds)

    centers = minkit.DataSet.from_ndarray(0.5 * (edges[1:] + edges[:-1]), m)

    pdf_values = minkit.utils.core.scaled_pdf_values(
        pdf, centers, values, edges)

    assert np.allclose(np.sum(pdf_values), np.sum(values), rtol=0.01)

    # Test a fit
    s2.constant = True  # otherwise the minimization is undefined

    with fit_test(pdf) as test:
        with minkit.minimizer('uml', pdf, data, minimizer='minuit') as minuit:
            test.result = minuit.migrad()

    # Test the JSON conversion
    with open(os.path.join(tmpdir, 'pdf.json'), 'wt') as fi:
        json.dump(minkit.pdf_to_json(pdf), fi)

    with open(os.path.join(tmpdir, 'pdf.json'), 'rt') as fi:
        s = minkit.pdf_from_json(json.load(fi))

    check_multi_pdfs(s, pdf)

    # Check copying the PDF
    pdf.copy()

    # Test for binned data samples
    bdata = data.make_binned(20)

    with fit_test(pdf) as test:
        with minkit.minimizer('bml', pdf, bdata, minimizer='minuit') as minuit:
            test.result = minuit.migrad()


@pytest.mark.pdfs
@pytest.mark.multipdf
@setting_seed
def test_prodpdfs(tmpdir):
    '''
    Test the "ProdPDFs" class.
    '''
    # Create two Gaussians
    mx = minkit.Parameter('mx', bounds=(-5, +5))
    cx = minkit.Parameter('cx', 0., bounds=(-2, +2))
    sx = minkit.Parameter('sx', 1., bounds=(0.1, +3))
    gx = minkit.Gaussian('gx', mx, cx, sx)

    my = minkit.Parameter('my', bounds=(-5, +5))
    cy = minkit.Parameter('cy', 0., bounds=(-2, +2))
    sy = minkit.Parameter('sy', 2., bounds=(0.5, +3))
    gy = minkit.Gaussian('gy', my, cy, sy)

    pdf = minkit.ProdPDFs('pdf', [gx, gy])

    # Test integration
    helpers.check_numerical_normalization(pdf)

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

    # Check copying the PDF
    pdf.copy()

    # Do a simple fit
    for p in pdf.all_real_args:
        p.constant = False

    data = pdf.generate(10000)

    with fit_test(pdf) as test:
        with minkit.minimizer('uml', pdf, data) as minimizer:
            test.result = minimizer.migrad()


@pytest.mark.pdfs
@pytest.mark.source_pdf
@setting_seed
def test_sourcepdf(tmpdir):
    '''
    Test the "SourcePDF" class.
    '''
    # Test the construction of a normal PDF
    m = minkit.Parameter('m', bounds=(-5, +5))
    c = minkit.Parameter('c', 0., bounds=(-2, +2))
    s = minkit.Parameter('s', 1., bounds=(-3, +3))
    minkit.Gaussian('gaussian', m, c, s)

    # Test the construction of a PDF with variable number of arguments
    m = minkit.Parameter('m', bounds=(-5, +5))
    p1 = minkit.Parameter('p1', 1.)
    p2 = minkit.Parameter('p2', 2.)
    pol0 = minkit.Polynomial('pol0', m)
    minkit.Polynomial('pol1', m, p1)
    minkit.Polynomial('pol2', m, p1, p2)

    # Test the JSON conversion
    with open(os.path.join(tmpdir, 'pdf.json'), 'wt') as fi:
        json.dump(minkit.pdf_to_json(pol0), fi)

    with open(os.path.join(tmpdir, 'pdf.json'), 'rt') as fi:
        s = minkit.pdf_from_json(json.load(fi))

    check_pdfs(s, pol0)

    # Check copying the PDF
    pol0.copy()


@pytest.mark.pdfs
@pytest.mark.source_pdf
@setting_seed
def test_evaluation():
    '''
    Test the methods used for evaluation of the PDF.
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    c = minkit.Parameter('c', 0., bounds=(-2, +2))
    s = minkit.Parameter('s', 1., bounds=(-3, +3))
    g = minkit.Gaussian('gaussian', m, c, s)

    m.set_range('reduced', (-3, +3))

    assert not np.allclose(g.function(), g.function('reduced'))

    data = g.generate(1000)

    g(data)  # normal evaluation

    binned_data = data.make_binned(100)

    bv = g.evaluate_binned(binned_data)  # evaluation on a binned data set

    assert np.allclose(bv.sum(), 1.)


@pytest.mark.pdfs
@pytest.mark.source_pdf
def test_formulapdf():
    '''
    Test the "FormulPDF" class.
    '''
    x = minkit.Parameter('x', bounds=(-2. * np.pi, +2 * np.pi))
    a = minkit.Parameter('a', 1., bounds=(0.9, 1.1))
    b = minkit.Parameter('b', 0., bounds=(-1, 1))
    pdf = minkit.FormulaPDF.unidimensional(
        'pdf', 'pow(sin(a * x + b), 2)', x, [a, b])

    norm = pdf.norm()

    data = pdf.generate(10000)

    with helpers.fit_test(pdf) as test:
        with minkit.minimizer('uml', pdf, data) as minimizer:
            test.result = minimizer.migrad()

    # Include the integral
    pdf = minkit.FormulaPDF.unidimensional('pdf', 'pow(sin(a * x + b), 2)', x, [
                                           a, b], primitive='- sin(2 * (a * x + b)) -2 * (a * x + b) / (4 * a)')

    assert np.allclose(norm, pdf.norm())

    with helpers.fit_test(pdf) as test:
        with minkit.minimizer('uml', pdf, data) as minimizer:
            test.result = minimizer.migrad()

    # In two dimensions
    x = minkit.Parameter('x', bounds=(0, 10))
    y = minkit.Parameter('y', bounds=(0, 10))
    ax = minkit.Parameter('ax', -0.01, bounds=(-1, 0))
    ay = minkit.Parameter('ay', -0.01, bounds=(-1, 0))
    pdf = minkit.FormulaPDF(
        'pdf', 'exp(ax * x) * exp(ay * y)', [x, y], [ax, ay])

    data = pdf.generate(10000)

    with helpers.fit_test(pdf) as test:
        with minkit.minimizer('uml', pdf, data) as minimizer:
            test.result = minimizer.migrad()


@pytest.mark.pdfs
@pytest.mark.source_pdf
def test_display_pdfs():
    '''
    Test that the PDFs are displayed correctly as strings.
    '''
    # Define the model
    x = minkit.Parameter('x', bounds=(-5, +5))
    y = minkit.Parameter('y', bounds=(-5, +5))
    c = minkit.Parameter('c', 0, bounds=(-5, +5))

    k = minkit.Parameter('k', -0.1)

    sx = minkit.Parameter('sx', 2, bounds=(1, 3))
    sy = minkit.Parameter('sy', 1, bounds=(0.5, 3))

    gx = minkit.Gaussian('gx', x, c, sx)
    ex = minkit.Exponential('exp', x, k)
    gy = minkit.Gaussian('gy', y, c, sy)

    # Print a single PDF
    print(gx)

    # Print AddPDFs
    y = minkit.Parameter('y', 0.5)
    pdf = minkit.AddPDFs.two_components('pdf', gx, ex, y)
    print(pdf)

    # Print ProdPDFs
    pdf = minkit.ProdPDFs('pdf', [gx, gy])
    print(pdf)

    # Print ConvPDFs
    pdf = minkit.ConvPDFs('pdf', gx, gy)
    print(pdf)


@pytest.mark.pdfs
@pytest.mark.source_pdf
def test_restoring_state():
    '''
    Test that the state of the PDFs is treated correctly.
    '''
    m = minkit.Parameter('m', bounds=(10, 20))
    c = minkit.Parameter('c', 15, bounds=(10, 20))
    s = minkit.Formula('s', '0.1 * {c}', [c])
    g = minkit.Gaussian('g', m, c, s)

    data = g.generate(10000)

    with minkit.minimizer('uml', g, data) as minuit:
        minuit.migrad()
        result = g.args.copy()

    data = g.generate(10000)  # new data set

    with g.restoring_state(), minkit.minimizer('uml', g, data) as minuit:
        minuit.migrad()

    # The values of the PDF must be those of the first minimization
    for f, s in zip(result, g.real_args):
        helpers.check_parameters(f, s)


@pytest.mark.pdfs
@pytest.mark.source_pdf
def test_interppdf(tmpdir):
    '''
    Test the InterpPDF class.
    '''
    m = minkit.Parameter('m', bounds=(-3, +3))
    centers = np.linspace(*m.bounds, 100)
    values = np.exp(-0.5 * centers**2)

    ip = minkit.InterpPDF.from_ndarray('ip', m, centers, values)

    ip.max()  # check that we can calculate the maximum

    # Test the JSON conversion
    with open(os.path.join(tmpdir, 'ip.json'), 'wt') as fi:
        json.dump(minkit.pdf_to_json(ip), fi)

    with open(os.path.join(tmpdir, 'ip.json'), 'rt') as fi:
        p = minkit.pdf_from_json(json.load(fi))

    check_pdfs(p, ip)

    # Check copying the PDF
    ip.copy()

    # Combine the PDF with another
    k = minkit.Parameter('k', -0.1, bounds=(-1, +1))
    e = minkit.Exponential('exp', m, k)

    y = minkit.Parameter('y', 0.5, bounds=(0, 1))

    pdf = minkit.AddPDFs.two_components('pdf', ip, e, y)

    data = pdf.generate(10000)

    with fit_test(pdf) as test:
        with minkit.minimizer('uml', pdf, data, minimizer='minuit') as minimizer:
            test.result = minimizer.migrad()

    bdata = data.make_binned(20)

    with fit_test(pdf) as test:
        with minkit.minimizer('bml', pdf, bdata, minimizer='minuit') as minimizer:
            test.result = minimizer.migrad()

    # Test the construction from a binned data set
    minkit.InterpPDF.from_binned_dataset('pdf', bdata)


@pytest.mark.pdfs
def test_pdf_max():
    '''
    Test the determination of the maximum value of a PDF.
    '''
    pdf = helpers.default_gaussian(sigma='s')

    assert np.allclose(pdf.max(normalized=False), 1.)

    pdf.args.get('s').value = 0.01  # a very narrow peak

    assert np.allclose(pdf.max(normalized=False), 1.)


@pytest.mark.pdfs
def test_numerical_integral():
    '''
    Test the calculation of numerical integrals.
    '''
    pdf = helpers.default_gaussian(data_par='x')

    x = pdf.data_pars.get('x')

    x.set_range('reduced', 0.5 * x.bounds)

    values = {}
    for m in 'qng', 'qag', 'cquad', 'plain', 'miser', 'vegas':
        pdf.numint_config = {'method': m}
        values[m] = pdf.numerical_integral(integral_range='reduced')

    def check_for(methods, rtol):
        vals = np.fromiter((values[m] for m in methods), dtype=np.float64)
        mean = np.mean(vals)
        assert np.allclose(vals, mean, rtol=rtol)

    check_for(values.keys(), rtol=1e-2)  # plain Monte Carlo is less accurate
    check_for(('qng', 'qag', 'cquad', 'miser', 'vegas'), rtol=5e-5)
    check_for(('qng', 'qag', 'cquad', 'vegas'), rtol=1e-6)  # more precise
