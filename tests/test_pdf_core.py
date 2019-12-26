'''
Test the "pdf_core" module.
'''
import numpy as np
import pyfit

pyfit.initialize()

# For reproducibility
np.random.seed(98953)


def test_pdf():
    '''
    General tests for the PDF class.
    '''
    # Create a Polynomial PDF
    m = pyfit.Parameter('m', bounds=(0, 10))
    e = pyfit.Polynomial('polynomial', m)

    m.set_range('sides', [(0, 4), (6, 10)])

    # integral
    assert np.allclose(e.integral(range='full', norm_range='full'), 1.)
    assert np.allclose(e.integral(range='sides', norm_range='full'), 0.8)
    assert np.allclose(e.integral(range='sides', norm_range='sides'), 1.)


def test_addpdfs():
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


def test_prodpdfs():
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


def test_sourcepdf():
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
