'''
Tests fot the "bindings.py" module.
'''
import numpy as np
import minkit
import pytest

minkit.initialize()

# For reproducibility
np.random.seed(98953)


def test_bind_class_arguments():
    '''
    Test the "bind_class_arguments" function.
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    c = minkit.Parameter('c', 0., bounds=(-2, +2))
    s = minkit.Parameter('s', 1., bounds=(-3, +3))
    g = minkit.Gaussian('gaussian', m, c, s)

    data = g.generate(10000)

    # Single call
    with g.bind() as proxy:
        proxy(data)

    # Call with arguments
    with g.bind(values=None) as proxy:
        proxy(data)

    # Use same arguments as in bind
    with g.bind(values=None) as proxy:
        proxy(data, values=None)

    # Use different arguments as in bind (raises error)
    with g.bind(values=None) as proxy:
        with pytest.raises(ValueError):
            proxy(data, values={'c': 1.})

    # Same tests with positionals
    with g.bind(values=None) as proxy:
        proxy(data, None)

    with g.bind(values=None) as proxy:
        with pytest.raises(ValueError):
            proxy(data, {'c': 1.})
