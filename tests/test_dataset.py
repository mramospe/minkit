'''
Tests for the "dataset.py" module.
'''
import pyfit
from pyfit import dataset


def test_evaluation_grid():
    '''
    Test the "evaluation_grid" function.
    '''
    m = pyfit.Parameter('m', bounds=(0, 20))
    reg = pyfit.Registry()
    reg['m'] = m

    n = 100

    # Test single range
    g = dataset.evaluation_grid(reg, m.bounds, n)
    assert len(g) == n

    # Test multi-range
    reg['m'].set_range('sides', [(0, 8), (12, 20)])
    g = dataset.evaluation_grid(reg, [(0, 8), (12, 20)], n)
    assert len(g) == n
