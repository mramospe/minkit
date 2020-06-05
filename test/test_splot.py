########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Test the "splot" module.
'''
import helpers
import minkit
import numpy as np
import pytest

helpers.configure_logging()

aop = minkit.backends.core.parse_backend()


@pytest.mark.utils
def test_sweights():
    '''
    Test the "sweights" function.
    '''
    pdf = helpers.default_add_pdfs(extended=True, yields=['ng', 'ne'])

    ng = pdf.args.get('ng')
    ne = pdf.args.get('ne')

    data = pdf.generate(int(ng.value + ne.value))

    with helpers.fit_test(pdf) as test:
        with minkit.minimizer('ueml', pdf, data, minimizer='minuit') as minuit:
            test.result, _ = minuit.migrad()

    # Now we fix the parameters that are not yields, and we re-run the fit
    for p in pdf.pdfs:
        for a in p.args:
            a.constant = True

    with helpers.fit_test(pdf) as test:
        with minkit.minimizer('ueml', pdf, data, minimizer='minuit') as minuit:
            test.result, _ = minuit.migrad()

    result = pdf.args.copy()

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

    # Check the calculation of the uncertainties of the s-weights
    minkit.sweights_u(data.values.as_ndarray(), sweights[0].as_ndarray())
