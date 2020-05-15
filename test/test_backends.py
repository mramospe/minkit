'''
Tests for the "backends" module.
'''
import helpers
import minkit
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

    data = helpers.rndm_gen.uniform(0, 1, 1000)

    # Test initialization and constructor methods
    bk.DataSet(minkit.darray.from_ndarray(data, bk), [x])

    dataset = bk.DataSet.from_ndarray(data, x)

    new_bk = minkit.Backend(minkit.backends.core.CPU)

    # Test the adaption of objects to new backends
    dataset.to_backend(new_bk)


BACKEND = os.environ.get('MINKIT_BACKEND', None)

if minkit.backends.core.is_gpu_backend(BACKEND):

    def test_gpu_backends():
        '''
        Test the change of objects from a CPU to a GPU backend.
        '''
        minkit.Backend()
        minkit.Backend(BACKEND)
