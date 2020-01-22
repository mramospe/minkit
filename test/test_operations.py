'''
Test operations on arrays.
'''
import helpers
import minkit
import numpy as np
import pytest
from minkit.operations import types

helpers.configure_logging()
minkit.initialize()


@pytest.mark.core
@pytest.mark.operations
@helpers.setting_numpy_seed
def test_aop():
    '''
    Test the different operations.
    '''
    n = 1000

    # Ones
    ones = minkit.core.aop.ones(n, dtype=types.cpu_bool)
    assert np.allclose(minkit.as_ndarray(ones),
                       np.ones(n, dtype=types.cpu_bool))

    ones = minkit.core.aop.ones(n)  # Keep as double
    assert np.allclose(minkit.as_ndarray(ones), np.ones(n))

    assert np.allclose(minkit.as_ndarray(ones * ones), np.ones(n))

    # Zeros
    zeros = minkit.core.aop.zeros(n, dtype=types.cpu_bool)
    assert np.allclose(minkit.as_ndarray(zeros), np.zeros(n))

    zeros = minkit.core.aop.zeros(n)  # Keep as double
    assert np.allclose(minkit.as_ndarray(zeros), np.zeros(n))

    assert np.allclose(minkit.as_ndarray(zeros * zeros), np.zeros(n))

    # count_nonzero
    ones = minkit.core.aop.ones(n, dtype=types.cpu_bool)
    assert np.allclose(minkit.core.aop.count_nonzero(ones), len(ones))
    zeros = minkit.core.aop.zeros(n, dtype=types.cpu_bool)
    assert np.allclose(minkit.core.aop.count_nonzero(zeros), 0)

    # Exponential
    zeros = minkit.core.aop.zeros(n, dtype=types.cpu_type)
    ez = minkit.core.aop.exp(zeros)
    assert np.allclose(minkit.as_ndarray(ez), np.ones(n))

    # Logarithm
    ones = minkit.core.aop.ones(n, dtype=types.cpu_type)
    lo = minkit.core.aop.log(ones)
    assert np.allclose(minkit.as_ndarray(lo), np.zeros(n))

    # Linspace
    ls = minkit.core.aop.linspace(10, 20, n)
    assert np.allclose(minkit.as_ndarray(ls), np.linspace(10, 20, n))

    # amax
    assert np.allclose(minkit.core.aop.max(ls), 20)

    # amin
    assert np.allclose(minkit.core.aop.min(ls), 10)

    # sum
    ls = minkit.core.aop.linspace(1, 100, 100)
    assert np.allclose(minkit.core.aop.sum(ls), 5050)

    # geq
    u = minkit.core.aop.random_uniform(0, 1, 10000)
    s = minkit.core.aop.count_nonzero(minkit.core.aop.geq(u, 0.5)) / len(u)
    assert np.allclose(s, 0.5, rtol=0.05)

    # leq
    u = minkit.core.aop.random_uniform(0, 1, 10000)
    s = minkit.core.aop.count_nonzero(minkit.core.aop.leq(u, 0.5)) / len(u)
    assert np.allclose(s, 0.5, rtol=0.05)

    # ale
    l = minkit.core.aop.linspace(0, 1, 10000)
    n = minkit.core.aop.random_uniform(0, 1, len(l))

    assert np.allclose(minkit.core.aop.count_nonzero(minkit.core.aop.ale(n, l)),
                       len(l) / 2., rtol=0.05)

    # concatenate
    l1 = minkit.core.aop.linspace(0, 1, 10000)
    l2 = minkit.core.aop.linspace(1, 2, len(l1))

    l = minkit.core.aop.concatenate((l1, l2))

    assert len(l) == len(l1) + len(l2)
    assert np.allclose(minkit.as_ndarray(
        l)[:len(l1)], minkit.as_ndarray(l1))
    assert np.allclose(minkit.as_ndarray(
        l)[len(l1):], minkit.as_ndarray(l2))

    # false_till and true_till
    f = minkit.core.aop.false_till(10, 5)
    fn = np.zeros(len(f))
    fn[5:] = True

    assert np.allclose(minkit.as_ndarray(f), fn)

    t = minkit.core.aop.true_till(10, 5)
    tn = np.ones(len(f))
    tn[5:] = False

    assert np.allclose(minkit.as_ndarray(t), tn)

    # sum_inside
    l = minkit.core.aop.linspace(0, 100, 1001)
    c = (l[1:] + l[:-1]) / 2.
    e = minkit.core.aop.linspace(0, 100, 101)

    r = minkit.core.aop.sum_inside([c], [e])
    assert np.allclose(minkit.as_ndarray(r), np.full(len(r), 10))

    v = minkit.core.aop.zeros(len(c), dtype=types.cpu_type)
    r = minkit.core.aop.sum_inside([c], [e], v)
    assert np.allclose(minkit.as_ndarray(r), np.zeros(len(r)))

    # FFT
    n = 1000

    a = minkit.core.aop.real(
        minkit.core.aop.data_array(np.random.normal(0, 3, n)))
    b = minkit.core.aop.real(
        minkit.core.aop.data_array(np.random.normal(0, 4, n)))

    av = minkit.as_ndarray(minkit.core.aop.real(
        minkit.core.aop.ifft(minkit.core.aop.fft(a))))
    bv = minkit.as_ndarray(minkit.core.aop.real(
        minkit.core.aop.ifft(minkit.core.aop.fft(b))))

    assert np.allclose(av, minkit.as_ndarray(a))
    assert np.allclose(bv, minkit.as_ndarray(b))

    def gaussian(x, c, s):
        return 1. / (np.sqrt(2. * np.pi) * s) * np.exp(- (x - c)**2 / (2. * s**2))

    x = np.linspace(-20, +20, n, dtype=types.cpu_type)
    fa = minkit.core.aop.data_array(gaussian(x, 0, 3))
    fb = minkit.core.aop.data_array(gaussian(x, 0, 4))

    ax = minkit.core.aop.data_array(x)

    fr = minkit.core.aop.real(minkit.core.aop.fftconvolve(fa, fb, ax))

    assert np.allclose(minkit.core.aop.sum(fr) * (x[1] - x[0]), 1.)
