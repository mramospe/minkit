'''
Tests for the "dataset.py" module.
'''
import helpers
import minkit
from minkit import dataset
import numpy as np
import pytest

helpers.configure_logging()
minkit.initialize()

# For reproducibility
np.random.seed(98953)


@pytest.mark.core
def test_evaluation_grid():
    '''
    Test the "evaluation_grid" function.
    '''
    x = minkit.Parameter('x', bounds=(0, 20))
    y = minkit.Parameter('y', bounds=(0, 20))

    n = 100

    # Test single range
    g = dataset.evaluation_grid(minkit.Registry([x]), x.bounds, n)
    assert len(g) == n

    # Test multi-range
    g = dataset.evaluation_grid(minkit.Registry(
        [x, y]), np.concatenate([x.bounds, y.bounds]), n)
    assert len(g) == n**2
