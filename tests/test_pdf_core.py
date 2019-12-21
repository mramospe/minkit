'''
Test the "pdf_core" module.
'''
import numpy as np
import pyfit

pyfit.initialize()

# For reproducibility
np.random.seed(98953)


def test_addpdfs():
    '''
    Test the "AddPDFs" class.
    '''
    # Create an Exponential PDF
    m = pyfit.Parameter('m', bounds=(-5, +5))
    k = pyfit.Parameter('k', -0.05, bounds=(-0.1, 0))
    e = pyfit.Exponential('exponential', m, k)

    # Create a Gaussian PDF
    m = pyfit.Parameter('m', bounds=(-5, +5))
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
