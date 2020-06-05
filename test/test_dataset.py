########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
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
@helpers.setting_seed
def test_dataset():
    '''
    Test for the "DataSet" class.
    '''
    numpy_data = helpers.rndm_gen.normal(0, 1, 10000)

    m = minkit.Parameter('m', bounds=(-5, +5))
    m.set_range('reduced', (-2, +2))

    data = minkit.DataSet.from_ndarray(numpy_data, m)

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
    nps['x'] = helpers.rndm_gen.normal(0, 0.1, 10000)
    nps['y'] = helpers.rndm_gen.normal(0, 0.2, 10000)

    data = minkit.DataSet.from_records(nps, [x, y])

    assert len(data) == len(nps)

    x.set_range('reduced', (-2, 2))
    y.set_range('reduced', (-3, 3))

    data.subset('reduced')
    data.make_binned(bins=100)
    data.make_binned(bins=(100, 100))

    r = data.to_records()
    data = minkit.DataSet.from_records(r, data.data_pars)

    assert len(r) == len(data)


@pytest.mark.core
@helpers.setting_seed
def test_evaluation_grid():
    '''
    Test the "evaluation_grid" function.
    '''
    x = minkit.Parameter('x', bounds=(0, 20))
    y = minkit.Parameter('y', bounds=(40, 60))

    n = 100

    # Test single range
    p = minkit.Registry([x])
    b = minkit.base.parameters.bounds_for_range(p, 'full')[0]
    g = dataset.evaluation_grid(aop, p, b, n)
    assert len(g.values.ua) == n

    # Test multi-range
    p = minkit.Registry([x, y])
    b = minkit.base.parameters.bounds_for_range(p, 'full')[0]
    g = dataset.evaluation_grid(aop, p, b, n)
    assert len(g.values.ua) == len(p) * n**2


@pytest.mark.core
@helpers.setting_seed
def test_uniform_sample():
    '''
    Test the "uniform_sample" function.
    '''
    x = minkit.Parameter('x', bounds=(0, 20))
    y = minkit.Parameter('y', bounds=(0, 20))

    n = 100

    # Test single range
    p = minkit.Registry([x])
    b = minkit.base.parameters.bounds_for_range(p, 'full')[0]
    g = dataset.uniform_sample(aop, p, b, n)
    assert len(g.values.ua) == n

    # Test multi-range
    p = minkit.Registry([x, y])
    b = minkit.base.parameters.bounds_for_range(p, 'full')[0]
    g = dataset.uniform_sample(aop, p, b, n)
    assert len(g.values.ua) == len(p) * n
