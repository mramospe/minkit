'''
Test the "splot" module.
'''
import helpers
import minkit
import numpy as np
import pytest

from minkit.minimization.minuit_api import MinuitMinimizer

helpers.configure_logging()

aop = minkit.backends.core.parse_backend()


@pytest.mark.utils
def test_sweights():
    '''
    Test the "sweights" function.
    '''
    m = minkit.Parameter('m', bounds=(0, +20))

    # Create an Exponential PDF
    k = minkit.Parameter('k', -0.1, bounds=(-0.2, 0))
    e = minkit.Exponential('exponential', m, k)

    # Create a Gaussian PDF
    c = minkit.Parameter('c', 10., bounds=(0, 20))
    s = minkit.Parameter('s', 1., bounds=(0.1, 2))
    g = minkit.Gaussian('gaussian', m, c, s)

    # Add them together
    ng = minkit.Parameter('ng', 10000, bounds=(0, 100000))
    ne = minkit.Parameter('ne', 1000, bounds=(0, 100000))
    pdf = minkit.AddPDFs.two_components('model', g, e, ng, ne)

    data = pdf.generate(int(ng.value + ne.value))

    with minkit.minimizer('ueml', pdf, data, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    # Now we fix the parameters that are not yields, and we re-run the fit
    for p in (e, g):
        for a in p.args:
            a.constant = True

    with minkit.minimizer('ueml', pdf, data, minimizer='minuit') as minuit:
        r = minuit.migrad()
        print(r)

    result = MinuitMinimizer.result_to_registry(r.params)

    # Calculate the s-weights (first comes from the Gaussian, second from the exponential)
    sweights, V = minkit.sweights(pdf.pdfs, result.reduce([
        'ng', 'ne']), data, return_covariance=True)

    # The s-weights are normalized
    assert np.allclose(aop.sum(
        sweights[0]), result.get(ng.name).value)
    assert np.allclose(aop.sum(
        sweights[1]), result.get(ne.name).value)

    # The uncertainty on the yields is reflected in the s-weights
    assert np.allclose(aop.sum(sweights[0]**2), V[0][0])
    assert np.allclose(aop.sum(sweights[1]**2), V[1][1])
