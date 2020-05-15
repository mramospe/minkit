'''
Tests for the "profile.py" module.
'''
import minkit
import numpy as np
import pytest

import helpers


@pytest.mark.utils
def test_fcn_profile():
    '''
    Test the "fcn_profile" function.
    '''
    pdf = helpers.default_gaussian('c', 's')

    args = pdf.args

    c = args.get('c')
    s = args.get('s')

    data = pdf.generate(1000)

    cv = np.linspace(*c.bounds, 10)
    sv = np.linspace(*s.bounds, 10)

    mp = tuple(a.flatten() for a in np.meshgrid(cv, sv))

    minkit.fcn_profile('uml', pdf, data, 'c', cv)
    minkit.fcn_profile('uml', pdf, data, ['c', 's'], mp)


@pytest.mark.utils
def test_simultaneous_fcn_profile():
    '''
    Test the "simultaneous_fcn_profile" function.
    '''
    g1 = helpers.default_gaussian('c1', 's1')
    g2 = helpers.default_gaussian('c2', 's2')

    d1 = g1.generate(1000)
    d2 = g2.generate(1000)

    cats = [minkit.Category('uml', g1, d1), minkit.Category('uml', g2, d2)]

    cv = np.linspace(*g1.args.get('c1').bounds, 10)
    sv = np.linspace(*g2.args.get('s2').bounds, 10)

    mp = tuple(a.flatten() for a in np.meshgrid(cv, sv))

    minkit.simultaneous_fcn_profile(cats, 'c1', cv)
    minkit.simultaneous_fcn_profile(cats, ['c1', 's2'], mp)
