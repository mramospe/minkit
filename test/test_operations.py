'''
Test operations on arrays.
'''
import helpers
import minkit
import numpy as np
import pytest
from minkit.base import data_types

helpers.configure_logging()

aop = minkit.backends.core.parse_backend()


@pytest.mark.core
@pytest.mark.operations
@helpers.setting_numpy_seed
def test_aop():
    '''
    Test the different operations.
    '''
    n = 1000

    # Ones
    ones = aop.bones(n)
    assert np.allclose(ones.as_ndarray(), np.ones(
        n, dtype=data_types.cpu_bool))

    ones = aop.fones(n)  # Keep as double
    assert np.allclose(ones.as_ndarray(), np.ones(n))

    assert np.allclose((ones * ones).as_ndarray(), np.ones(n))

    # Zeros
    zeros = aop.bzeros(n)
    assert np.allclose(zeros.as_ndarray(), np.zeros(n))

    zeros = aop.fzeros(n)  # Keep as double
    assert np.allclose(zeros.as_ndarray(), np.zeros(n))

    assert np.allclose((zeros * zeros).as_ndarray(), np.zeros(n))

    # count_nonzero
    ones = aop.bones(n)
    assert np.allclose(aop.count_nonzero(ones), len(ones))
    zeros = aop.bzeros(n)
    assert np.allclose(aop.count_nonzero(zeros), 0)

    # Exponential
    zeros = aop.fzeros(n)
    ez = aop.exp(zeros)
    assert np.allclose(ez.as_ndarray(), np.ones(n))

    # Logarithm
    ones = aop.fones(n)
    lo = aop.log(ones)
    assert np.allclose(lo.as_ndarray(), np.zeros(n))

    # Linspace
    ls = aop.linspace(10, 20, n)
    assert np.allclose(ls.as_ndarray(), np.linspace(10, 20, n))

    # amax
    assert np.allclose(aop.max(ls), 20)

    # amin
    assert np.allclose(aop.min(ls), 10)

    # sum
    ls = aop.linspace(1, 100, 100)
    assert np.allclose(aop.sum(ls), 5050)

    # ge
    u = aop.random_uniform(0, 1, 10000)
    s = aop.count_nonzero(aop.ge(u, 0.5)) / len(u)
    assert np.allclose(s, 0.5, rtol=0.05)

    # le
    u = aop.random_uniform(0, 1, 10000)
    s = aop.count_nonzero(aop.le(u, 0.5)) / len(u)
    assert np.allclose(s, 0.5, rtol=0.05)

    # lt
    l = aop.linspace(0, 1, 10000)
    n = aop.random_uniform(0, 1, len(l))

    assert np.allclose(aop.count_nonzero(aop.lt(n, l)),
                       0.5 * len(l), rtol=0.05)

    # concatenate
    l1 = aop.linspace(0, 1, 10000)
    l2 = aop.linspace(1, 2, len(l1))

    l = aop.concatenate((l1, l2))

    assert len(l) == len(l1) + len(l2)
    assert np.allclose(l.as_ndarray()[:len(l1)], l1.as_ndarray())
    assert np.allclose(l.as_ndarray()[len(l1):], l2.as_ndarray())

    # slice_from_boolean
    u = aop.random_uniform(0, 1, 10000)
    c = aop.lt(u, 0.5)
    s = aop.slice_from_boolean(u, c)

    npr = u.as_ndarray()[c.as_ndarray().astype(data_types.cpu_real_bool)]

    assert np.allclose(npr, s.as_ndarray())

    # sum_inside
    l = aop.linspace(0, 10, 101)
    c = 0.5 * (l.take_slice(start=1) + l.take_slice(end=-1))
    e = aop.linspace(0, 10, 11)
    i = data_types.array_int([0, len(e)])
    g = data_types.array_int([1])

    r = aop.sum_inside(i, g, c, e)
    assert np.allclose(r.as_ndarray(), np.full(len(r), 10))

    v = aop.fzeros(len(c))
    r = aop.sum_inside(i, g, c, e, v)
    assert np.allclose(r.as_ndarray(), np.zeros(len(r)))

    ex = aop.linspace(0, 10, 11)
    ey = aop.linspace(0, 10, 11)
    cx = 0.5 * (ex.take_slice(1) + ex.take_slice(end=-1))
    cy = 0.5 * (ey.take_slice(1) + ey.take_slice(end=-1))

    mx, my = aop.meshgrid(cx, cy)

    e = aop.concatenate([ex, ey])
    m = aop.concatenate([mx, my])

    i = data_types.array_int([0, len(e) // 2, len(e)])
    g = data_types.array_int([1, (len(e) - 2) // 2])

    r = aop.sum_inside(i, g, m, e).as_ndarray()

    assert np.allclose(r, np.full(len(r), 1))

    # FFT
    n = 1000

    def gaussian(x, c, s):
        return 1. / (np.sqrt(2. * np.pi) * s) * np.exp(- (x - c)**2 / (2. * s**2))

    x = np.linspace(-20, +20, n, dtype=data_types.cpu_float)
    fa = aop.ndarray_to_farray(gaussian(x, 0, 3))
    fb = aop.ndarray_to_farray(gaussian(x, 0, 4))

    ax = aop.ndarray_to_farray(x)

    fr = aop.fftconvolve(fa, fb, ax)

    assert np.allclose(aop.sum(fr) * (x[1] - x[0]), 1.)
