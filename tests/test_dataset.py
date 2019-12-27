'''
Tests for the "dataset.py" module.
'''
import pyfit
from pyfit import dataset
import numpy as np

pyfit.initialize()

# For reproducibility
np.random.seed(98953)


def test_evaluation_grid():
    '''
    Test the "evaluation_grid" function.
    '''
    x = pyfit.Parameter('x', bounds=(0, 20))
    y = pyfit.Parameter('y', bounds=(0, 20))

    n = 100

    # Test single range
    g = dataset.evaluation_grid(pyfit.Registry(x=x), x.bounds, n)
    assert len(g) == n

    # Test multi-range
    g = dataset.evaluation_grid(pyfit.Registry(x=x, y=y), np.concatenate([x.bounds, y.bounds]), n)
    assert len(g) == n**2
