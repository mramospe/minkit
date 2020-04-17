'''
Tests for the "dataset.py" module.
'''
import helpers
import minkit
from minkit.pdfs import dataset
import numpy as np
import pytest

helpers.configure_logging()

aop = minkit.backends.core.parse_backend()


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

    new_data = data.subset('reduced')

    assert np.allclose(aop.count_nonzero(
        aop.le(new_data[m.name], -2.1)), 0)

    assert np.allclose(aop.count_nonzero(
        aop.ge(new_data[m.name], +2.1)), 0)

    binned_data = data.make_binned(bins=100)

    values, _ = np.histogram(numpy_data, range=m.bounds, bins=100)

    assert np.allclose(binned_data.values.as_ndarray(), values)

    # Multidimensional case
    x = minkit.Parameter('x', bounds=(-5, +5))
    y = minkit.Parameter('y', bounds=(-5, +5))

    nps = np.empty(10000, dtype=[('x', np.float64), ('y', np.float64)])
    nps['x'] = np.random.normal(0, 0.1, 10000)
    nps['y'] = np.random.normal(0, 0.2, 10000)

    data = minkit.DataSet.from_records(nps, [x, y])

    x.set_range('reduced', (-2, 2))
    y.set_range('reduced', (-3, 3))

    data.subset('reduced')
    data.make_binned(bins=100)
    data.make_binned(bins=(100, 100))


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
    p = minkit.Registry([x])
    b = minkit.base.parameters.bounds_for_range(p, 'full')[0]
    g = dataset.evaluation_grid(aop, p, b, n)
    assert len(g) == n

    # Test multi-range
    p = minkit.Registry([x, y])
    b = minkit.base.parameters.bounds_for_range(p, 'full')[0]
    g = dataset.evaluation_grid(aop, p, b, n)
    assert len(g) == n**2
