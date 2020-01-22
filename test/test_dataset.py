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


@pytest.mark.core
@helpers.setting_numpy_seed
def test_dataset():
    '''
    Test for the "DataSet" class.
    '''
    numpy_data = np.random.normal(0, 1, 10000)

    m = minkit.Parameter('m', bounds=(-5, +5))
    m.set_range('reduced', (-2, +2))

    data = minkit.DataSet.from_array(numpy_data, m)

    new_data = data.subset(range='reduced')

    assert np.allclose(minkit.core.aop.count_nonzero(
        minkit.core.aop.leq(new_data[m.name], -2.1)), 0)
    assert np.allclose(minkit.core.aop.count_nonzero(
        minkit.core.aop.geq(new_data[m.name], +2.1)), 0)

    binned_data = data.make_binned(bins=100)

    values, _ = np.histogram(numpy_data, range=m.bounds, bins=100)

    assert np.allclose(minkit.as_ndarray(binned_data.values), values)


@pytest.mark.core
@helpers.setting_numpy_seed
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
