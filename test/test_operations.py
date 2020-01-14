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

# For reproducibility
np.random.seed(98953)


@pytest.mark.core
@pytest.mark.operations
def test_aop():
    '''
    Test the different operations.
    '''
    n = 1000

    # Ones
    ones = minkit.aop.ones(n, dtype=types.cpu_bool)
    assert np.allclose(minkit.aop.extract_ndarray(ones),
                       np.ones(n, dtype=types.cpu_bool))

    ones = minkit.aop.ones(n)  # Keep as double
    assert np.allclose(minkit.aop.extract_ndarray(ones), np.ones(n))

    assert np.allclose(minkit.aop.extract_ndarray(ones * ones), np.ones(n))

    # Zeros
    zeros = minkit.aop.zeros(n, dtype=types.cpu_bool)
    assert np.allclose(minkit.aop.extract_ndarray(zeros), np.zeros(n))

    zeros = minkit.aop.zeros(n)  # Keep as double
    assert np.allclose(minkit.aop.extract_ndarray(zeros), np.zeros(n))

    assert np.allclose(minkit.aop.extract_ndarray(zeros * zeros), np.zeros(n))

    # count_nonzero
    ones = minkit.aop.ones(n, dtype=types.cpu_bool)
    assert np.allclose(minkit.aop.count_nonzero(ones), len(ones))
    zeros = minkit.aop.zeros(n, dtype=types.cpu_bool)
    assert np.allclose(minkit.aop.count_nonzero(zeros), 0)

    # Exponential
    zeros = minkit.aop.zeros(n, dtype=types.cpu_type)
    ez = minkit.aop.exp(zeros)
    assert np.allclose(minkit.aop.extract_ndarray(ez), np.ones(n))

    # Logarithm
    ones = minkit.aop.ones(n, dtype=types.cpu_type)
    lo = minkit.aop.log(ones)
    assert np.allclose(minkit.aop.extract_ndarray(lo), np.zeros(n))

    # Linspace
    ls = minkit.aop.linspace(10, 20, n)
    assert np.allclose(minkit.aop.extract_ndarray(ls), np.linspace(10, 20, n))

    # amax
    assert np.allclose(minkit.aop.max(ls), 20)

    # amin
    assert np.allclose(minkit.aop.min(ls), 10)

    # sum
    ls = minkit.aop.linspace(1, 100, 100)
    assert np.allclose(minkit.aop.sum(ls), 5050)

    # geq
    u = minkit.aop.random_uniform(0, 1, 10000)
    s = minkit.aop.count_nonzero(minkit.aop.geq(u, 0.5)) / len(u)
    assert np.allclose(s, 0.5, rtol=0.05)

    # leq
    u = minkit.aop.random_uniform(0, 1, 10000)
    s = minkit.aop.count_nonzero(minkit.aop.leq(u, 0.5)) / len(u)
    assert np.allclose(s, 0.5, rtol=0.05)

    # ale
    l = minkit.aop.linspace(0, 1, 10000)
    n = minkit.aop.random_uniform(0, 1, len(l))

    assert np.allclose(minkit.aop.count_nonzero(minkit.aop.ale(n, l)),
                       len(l) / 2., rtol=0.05)

    # concatenate
    l1 = minkit.aop.linspace(0, 1, 10000)
    l2 = minkit.aop.linspace(1, 2, len(l1))

    l = minkit.aop.concatenate((l1, l2))

    assert len(l) == len(l1) + len(l2)
    assert np.allclose(minkit.aop.extract_ndarray(
        l)[:len(l1)], minkit.aop.extract_ndarray(l1))
    assert np.allclose(minkit.aop.extract_ndarray(
        l)[len(l1):], minkit.aop.extract_ndarray(l2))

    # FFT
    n = 1000

    a = minkit.aop.real(minkit.aop.array(np.random.normal(0, 3, n)))
    b = minkit.aop.real(minkit.aop.array(np.random.normal(0, 4, n)))

    av = minkit.aop.extract_ndarray(minkit.aop.real(
        minkit.aop.ifft(minkit.aop.fft(a))))
    bv = minkit.aop.extract_ndarray(minkit.aop.real(
        minkit.aop.ifft(minkit.aop.fft(b))))

    assert np.allclose(av, minkit.aop.extract_ndarray(a))
    assert np.allclose(bv, minkit.aop.extract_ndarray(b))

    def gaussian(x, c, s):
        return 1. / (np.sqrt(2. * np.pi) * s) * np.exp(- (x - c)**2 / (2. * s**2))

    x = np.linspace(-20, +20, n, dtype=types.cpu_type)
    fa = minkit.aop.array(gaussian(x, 0, 3))
    fb = minkit.aop.array(gaussian(x, 0, 4))

    ax = minkit.aop.array(x)

    fr = minkit.aop.real(minkit.aop.fftconvolve(fa, fb, ax))

    assert np.allclose(minkit.aop.sum(fr) * (x[1] - x[0]), 1.)
