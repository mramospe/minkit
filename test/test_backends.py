########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
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

    m = bk.Parameter('m')
    c = bk.Parameter('c')
    s = bk.Parameter('s')
    k = bk.Parameter('k')
    y = bk.Parameter('y')

    g = bk.Gaussian('gauss', m, c, s)
    e = bk.Exponential('exponential', m, k)

    bk.AddPDFs.two_components('pdf', g, e, y)

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
