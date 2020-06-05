########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Tests fot the "bindings.py" module.
'''
import helpers
import minkit
import pytest

helpers.configure_logging()


@pytest.mark.core
@helpers.setting_seed
def test_bind_class_arguments():
    '''
    Test the "bind_class_arguments" function.
    '''
    m = minkit.Parameter('m', bounds=(-5, +5))
    c = minkit.Parameter('c', 0., bounds=(-2, +2))
    s = minkit.Parameter('s', 1., bounds=(-3, +3))
    g = minkit.Gaussian('gaussian', m, c, s)

    m.set_range('sides', ((-5, -2), (+2, +5)))

    data = g.generate(10000)

    # Single call
    with g.bind() as proxy:
        proxy(data)

    # Call with arguments
    with g.bind(range='sides') as proxy:
        proxy(data)

    # Use same arguments as in bind
    with g.bind(range='sides') as proxy:
        proxy(data, range='sides')

    # Use different arguments as in bind (raises error)
    with g.bind(range='sides') as proxy:
        with pytest.raises(ValueError):
            proxy(data, range='full')

    # Same tests with positionals
    with g.bind(range='sides') as proxy:
        proxy(data, 'sides')

    with g.bind(range='sides') as proxy:
        with pytest.raises(ValueError):
            proxy(data, 'full')
