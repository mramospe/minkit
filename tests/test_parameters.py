'''
Test the "parameters" module.
'''
import pyfit
import numpy as np


def test_range():
    '''
    Test the "Range" class.
    '''
    # Simple constructor
    v = [(1, 2), (5, 6)]
    r = pyfit.Range(v)

    assert np.allclose(r.bounds, v)

    # Do calculations in a range
    m = pyfit.Parameter('m', bounds=(0, 10))
    k = pyfit.Parameter('k', -0.5, bounds=(-0.8, -0.3))
    e = pyfit.Exponential('exponential', m, k)

    m.set_range('sides', [(0, 4), (6, 10)])

    data = e.generate(10000)

    with pyfit.create_minuit_unbinned('uml', e, data, range='sides') as minuit:
        r = minuit.migrad()
        print(r)