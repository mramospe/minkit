'''
Test operations on arrays.
'''
import helpers
import minkit
import numpy as np
import pytest

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
    ones = minkit.aop.ones(n, dtype=np.bool)
    assert np.allclose(minkit.aop.extract_ndarray(ones),
                       np.ones(n, dtype=np.bool))

    ones = minkit.aop.ones(n)  # Keep as double
    assert np.allclose(minkit.aop.extract_ndarray(ones), np.ones(n))

    # Zeros
    zeros = minkit.aop.zeros(n, dtype=np.bool)
    assert np.allclose(minkit.aop.extract_ndarray(zeros), np.zeros(n))

    zeros = minkit.aop.zeros(n)  # Keep as double
    assert np.allclose(minkit.aop.extract_ndarray(zeros), np.zeros(n))

    # Exponential
    ez = minkit.aop.exp(zeros)
    assert np.allclose(minkit.aop.extract_ndarray(ez), np.ones(n))

    # Logarithm
    lo = minkit.aop.log(ones)
    assert np.allclose(minkit.aop.extract_ndarray(lo), np.zeros(n))

    # Linspace
    ls = minkit.aop.linspace(10, 20, n)
    assert np.allclose(minkit.aop.extract_ndarray(ls), np.linspace(10, 20, n))

    # FFT
    a = minkit.aop.real(minkit.aop.array(np.random.normal(0, 3, n)))
    b = minkit.aop.real(minkit.aop.array(np.random.normal(0, 4, n)))

    av = minkit.aop.real(minkit.aop.extract_ndarray(
        minkit.aop.ifft(minkit.aop.fft(a))))
    bv = minkit.aop.real(minkit.aop.extract_ndarray(
        minkit.aop.ifft(minkit.aop.fft(b))))

    assert np.allclose(av, minkit.aop.extract_ndarray(a))
    assert np.allclose(bv, minkit.aop.extract_ndarray(b))

    def gaussian(x, c, s):
        return 1. / (np.sqrt(2. * np.pi) * s) * np.exp(- (x - c)**2 / (2. * s**2))

    x = np.linspace(-20, +20, n, dtype=np.float64)
    fa = minkit.aop.array(gaussian(x, 0, 3))
    fb = minkit.aop.array(gaussian(x, 0, 4))

    ax = minkit.aop.array(x)

    fr = minkit.aop.extract_ndarray(
        minkit.aop.real(minkit.aop.fftconvolve(fa, fb, ax)))

    assert np.allclose(fr.sum() * (x[1] - x[0]), 1.)
