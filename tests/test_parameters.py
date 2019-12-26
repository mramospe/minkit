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

    assert np.allclose(e.norm(range='sides'), e.numerical_normalization(range='sides'))

    data = e.generate(10000)

    with pyfit.unbinned_minimizer('uml', e, data, minimizer='minuit', range='sides') as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed

    results = pyfit.migrad_output_to_registry(r)

    for n, p in results.items():
        assert np.allclose(p.value, e.all_args[n].value, rtol=0.01)
