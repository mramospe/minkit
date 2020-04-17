'''
Tests for the "backends" module.
'''
import helpers
import minkit
import numpy as np
import os
import pytest


@pytest.mark.core
def test_backend():
    '''
    Test the construction of the backend.
    '''
    with pytest.raises(AttributeError):
        minkit.Backend.DataSet

    bk = minkit.Backend(minkit.backends.core.CPU)

    x = minkit.Parameter('x', bounds=(-1, +1))

    data = np.random.uniform(0, 1, 1000)

    # Test initialization and constructor methods
    bk.DataSet(bk.aop.ndarray_to_farray(data), [x])

    dataset = bk.DataSet.from_array(data, x)

    new_bk = minkit.Backend(minkit.backends.core.CPU)

    # Test the adaption of objects to new backends
    dataset.to_backend(new_bk)


BACKEND = os.environ.get('MINKIT_BACKEND', None)

if BACKEND in (minkit.backends.core.CUDA, minkit.backends.core.OPENCL):

    def test_gpu_backends():
        '''
        Test the change of objects from a CPU to a GPU backend.
        '''
        cpu_backend = minkit.Backend()
        gpu_backend = minkit.Backend(BACKEND)
