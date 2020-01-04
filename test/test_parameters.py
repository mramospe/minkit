'''
Test the "parameters" module.
'''
from helpers import check_parameters
import json
import helpers
import minkit
import numpy as np
import os
import pytest

helpers.configure_logging()
minkit.initialize()

# For reproducibility
np.random.seed(98953)


@pytest.mark.core
def test_parameter(tmpdir):
    '''
    Test the "Parameter" class.
    '''
    f = minkit.Parameter('a', 1., (-5, +5),
                         {'sides': [(-5, -2), (+2, +5)]}, 0.1, False)

    with open(os.path.join(tmpdir, 'a.json'), 'wt') as fi:
        json.dump(f.to_json_object(), fi)

    with open(os.path.join(tmpdir, 'a.json'), 'rt') as fi:
        s = minkit.Parameter.from_json_object(json.load(fi))

    check_parameters(f, s)


@pytest.mark.core
def test_range():
    '''
    Test the "Range" class.
    '''
    # Simple constructor
    v = [(1, 2), (5, 6)]
    r = minkit.Range(v)

    assert np.allclose(r.bounds, v)

    # Do calculations in a range
    m = minkit.Parameter('m', bounds=(0, 10))
    k = minkit.Parameter('k', -0.5, bounds=(-0.8, -0.3))
    e = minkit.Exponential('exponential', m, k)

    m.set_range('sides', [(0, 4), (6, 10)])

    assert np.allclose(e.norm(range='sides'),
                       e.numerical_normalization(range='sides'))

    data = e.generate(10000)

    with minkit.unbinned_minimizer('uml', e, data, minimizer='minuit', range='sides') as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed

    results = minkit.migrad_output_to_registry(r)

    for n, p in results.items():
        assert np.allclose(p.value, e.all_args[n].value, rtol=0.05)

    # Test generation of data only in the range
    data = e.generate(10000, range='sides')

    with minkit.unbinned_minimizer('uml', e, data, minimizer='minuit', range='sides') as minuit:
        r = minuit.migrad()
        print(r)

    assert not r.fmin.hesse_failed

    results = minkit.migrad_output_to_registry(r)

    for n, p in results.items():
        assert np.allclose(p.value, e.all_args[n].value, rtol=0.05)


@pytest.mark.core
def test_registry(tmpdir):
    '''
    Test the "Registry" class.
    '''
    a = minkit.Parameter('a', 1., (-5, +5), None, 0.1, False)
    b = minkit.Parameter('b', 0., (-10, +10), None, 2., True)

    f = minkit.Registry.from_list([a, b])

    with open(os.path.join(tmpdir, 'r.json'), 'wt') as fi:
        json.dump(f.to_json_object(), fi)

    with open(os.path.join(tmpdir, 'r.json'), 'rt') as fi:
        s = minkit.Registry.from_json_object(json.load(fi))

    assert f.keys() == s.keys()

    for fv, sv in zip(f.values(), s.values()):
        check_parameters(fv, sv)
