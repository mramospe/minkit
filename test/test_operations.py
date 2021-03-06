########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Test operations on arrays.
'''
import helpers
import minkit
import numpy as np
import os
import pytest
from minkit.base import data_types
from minkit import darray

helpers.configure_logging()

aop = minkit.backends.core.parse_backend()


@pytest.mark.core
@pytest.mark.operations
@helpers.setting_seed
def test_aop():
    '''
    Test the different operations.
    '''
    n = 1000

    # For GPU implementation
    backend = os.environ.get('MINKIT_BACKEND', None)
    if minkit.backends.core.is_gpu_backend(backend):
        i = minkit.backends.gpu_core.closest_number_divisible(100, 32)
        assert i == 128
        i = minkit.backends.gpu_core.closest_number_divisible(1, 32)
        assert i == 32
        i = minkit.backends.gpu_core.closest_number_divisible(128, 32)
        assert i == 128

    # Ones
    ones = aop.bones(n)
    assert np.allclose(ones.as_ndarray(), np.ones(
        n, dtype=data_types.cpu_bool))

    ones = aop.dones(n)  # Keep as double
    assert np.allclose(ones.as_ndarray(), np.ones(n))

    assert np.allclose((ones * ones).as_ndarray(), np.ones(n))

    # Zeros
    zeros = aop.bzeros(n)
    assert np.allclose(zeros.as_ndarray(), np.zeros(n))

    zeros = aop.dzeros(n)  # Keep as double
    assert np.allclose(zeros.as_ndarray(), np.zeros(n))

    assert np.allclose((zeros * zeros).as_ndarray(), np.zeros(n))

    # count_nonzero
    ones = aop.bones(n)
    assert np.allclose(aop.count_nonzero(ones), len(ones))
    zeros = aop.bzeros(n)
    assert np.allclose(aop.count_nonzero(zeros), 0)

    # Exponential
    zeros = aop.dzeros(n)
    ez = aop.dexp(zeros)
    assert np.allclose(ez.as_ndarray(), np.ones(n))

    # Logarithm
    ones = aop.dones(n)
    lo = aop.log(ones)
    assert np.allclose(lo.as_ndarray(), np.zeros(n))

    # Linspace
    ls = aop.linspace(10, 20, n)
    assert np.allclose(ls.as_ndarray(), np.linspace(10, 20, n))

    # Interpolation
    x = aop.linspace(0, 10, 11)
    y = aop.linspace(10, 20, 11)
    i = aop.linspace(0.5, 9.5, 9)
    e = np.linspace(10.5, 19.5, 9)

    interpolator = aop.make_linear_interpolator(x, y)
    r = interpolator.interpolate(0, i)
    assert np.allclose(r.as_ndarray(), e)

    interpolator = aop.make_spline_interpolator(x, y)
    r = interpolator.interpolate(0, i)
    assert np.allclose(r.as_ndarray(), e)

    # amax
    assert np.allclose(aop.max(ls), 20)

    # amin
    assert np.allclose(aop.min(ls), 10)

    # argmax
    x = aop.linspace(0, 10, 11)

    assert aop.argmax(x) == len(x) - 1

    x = aop.dzeros(7653)  # odd number

    x.ua[9] = 1
    assert aop.argmax(x) == 9

    x.ua[0] = 10
    assert aop.argmax(x) == 0

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

    assert np.allclose(aop.count_nonzero(aop.lt(n, 0.5)),
                       len(l) // 2, rtol=0.05)

    # concatenate
    l1 = aop.linspace(0, 1, 10000)
    l2 = aop.linspace(1, 2, len(l1))

    l = aop.concatenate((l1, l2))

    assert len(l) == len(l1) + len(l2)
    assert np.allclose(l.as_ndarray()[:len(l1)], l1.as_ndarray())
    assert np.allclose(l.as_ndarray()[len(l1):], l2.as_ndarray())

    # meshgrid
    n = 1000
    lb = data_types.array_float([1.],)
    ub = data_types.array_float([2.])
    mm = aop.meshgrid(lb, ub, data_types.array_int([n]))
    nm = np.linspace(lb.item(), ub.item(), n)
    assert np.allclose(mm.as_ndarray(), nm)

    lb = data_types.array_float([0., 1.])
    ub = data_types.array_float([1., 2.])
    mm = aop.meshgrid(lb, ub, data_types.array_int([n, n]))

    ng = np.array([a.flatten() for a in np.meshgrid(
        *tuple(np.linspace(l, u, n) for l, u in zip(lb, ub)))]).T.flatten()

    assert np.allclose(mm.as_ndarray(), ng)

    # random_grid (do not use a symmetric range around 0)
    n = 10000
    lb = data_types.array_float([0, 0])
    ub = data_types.array_float([20, 20])
    u = aop.random_grid(lb, ub, n)

    assert u.length == n
    assert u.size == 2 * n

    x = aop.take_column(u, 0)
    y = aop.take_column(u, 1)

    for i in range(u.ndim):
        c = 0.5 * (lb[i] + ub[i])
        g = aop.count_nonzero(aop.take_column(u, i) >= c) / len(u)
        assert np.allclose(g, 0.5, rtol=0.05)

    # slice_from_boolean
    u = aop.random_uniform(0, 1, 10000)
    c = aop.lt(u, 0.5)
    s = aop.slice_from_boolean(u, c)

    npr = u.as_ndarray()[c.as_ndarray()]

    assert np.allclose(npr, s.as_ndarray())

    # sum_inside
    l = aop.linspace(0, 10, 101)
    c = 0.5 * (l.take_slice(start=1) + l.take_slice(end=len(l) - 1))
    e = aop.linspace(0, 10, 11)
    i = data_types.array_int([0, len(e)])
    g = data_types.array_int([1])

    r = aop.sum_inside(i, g, c, e)
    assert np.allclose(r.as_ndarray(), np.full(len(r), 10))

    v = aop.dzeros(len(c))
    r = aop.sum_inside(i, g, c, e, v)
    assert np.allclose(r.as_ndarray(), np.zeros(len(r)))

    ex = aop.linspace(0, 10, 11)
    ey = aop.linspace(0, 10, 11)
    cx = 0.5 * (ex.take_slice(1) + ex.take_slice(end=len(ex) - 1))
    cy = 0.5 * (ey.take_slice(1) + ey.take_slice(end=len(ey) - 1))

    m = darray.from_ndarray(np.array(list(map(lambda a: a.flatten(),
                                              np.meshgrid(cx.as_ndarray(), cy.as_ndarray())))).T, aop.backend)

    e = aop.concatenate([ex, ey])

    i = data_types.array_int([0, len(e) // 2, len(e)])
    g = data_types.array_int([1, (len(e) - 2) // 2])

    r = aop.sum_inside(i, g, m, e).as_ndarray()

    assert np.allclose(r, np.full(len(r), 1))

    # FFT
    n = 1000

    def gaussian(x, c, s):
        return 1. / (np.sqrt(2. * np.pi) * s) * np.exp(- (x - c)**2 / (2. * s**2))

    x = np.linspace(-20, +20, n, dtype=data_types.cpu_float)
    fa = darray.from_ndarray(gaussian(x, 0, 3), aop.backend)
    fb = darray.from_ndarray(gaussian(x, 0, 4), aop.backend)

    ax = darray.from_ndarray(x, aop.backend)

    fr = aop.fftconvolve(fa, fb, ax)

    assert np.allclose(aop.sum(fr) * (x[1] - x[0]), 1.)

    # Simpson's rule
    with pytest.raises(ValueError):
        aop.simpson_factors(4)

    s = aop.simpson_factors(5).as_ndarray()
    assert np.allclose(s, [1, 4, 2, 4, 1])

    s = aop.simpson_factors(5, 3).as_ndarray()
    r = np.ones(len(s))
    r[:-1] = np.tile([1, 4, 2, 4], 3)
    assert np.allclose(s, r)


@pytest.mark.core
@pytest.mark.operations
def test_display_arrays():
    '''
    Test that the arrays are displayed correctly as strings.
    '''
    print(aop.bzeros(10))  # array of booleans
    print(aop.carange(10))  # array of complex numbers
    print(aop.iarange(10))  # array of integers
    print(aop.dzeros(10))  # array of doubles dim 1
    print(aop.dzeros(10, 2))  # array of doubles dim 2
