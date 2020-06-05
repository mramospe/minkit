########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Tests for the "profile.py" module.
'''
import minkit
import numpy as np
import pytest

import helpers


@pytest.mark.profiles
def test_fcn_profile():
    '''
    Test the calculation of profiles by the minimizers.
    '''
    pdf = helpers.default_gaussian('g', 'x', 'c', 's')

    args = pdf.args

    c = args.get('c')
    s = args.get('s')

    data = pdf.generate(1000)

    cv = np.linspace(*c.bounds, 10)
    sv = np.linspace(*s.bounds, 10)

    mp = tuple(a.flatten() for a in np.meshgrid(cv, sv))

    with minkit.minimizer('uml', pdf, data) as minimizer:
        minimizer.fcn_profile('c', cv)
        minimizer.fcn_profile(['c', 's'], mp)


@pytest.mark.profiles
def test_minimization_profile():
    '''
    Test the calculation of minimization profiles by the minimizers.
    '''
    pdf = helpers.default_gaussian('g', 'x', 'c', 's')

    args = pdf.args

    c = args.get('c')
    s = args.get('s')

    # Change the bounds to avoid evaluations to zero
    c.bounds = (-2, +2)

    data = pdf.generate(1000)

    cv = np.linspace(*c.bounds, 10)
    sv = np.linspace(*s.bounds, 10)

    mp = tuple(a.flatten() for a in np.meshgrid(cv, sv))

    with minkit.minimizer('uml', pdf, data) as minimizer:
        minimizer.minimize()
        minimizer.minimization_profile('c', cv)
        minimizer.minimization_profile(['c', 's'], mp)


@pytest.mark.profiles
def test_simultaneous_fcn_profile():
    '''
    Test the calculations of profiles in simultaneous minimizers.
    '''
    g1 = helpers.default_gaussian('g1', 'x1', 'c1', 's1')
    g2 = helpers.default_gaussian('g2', 'x2', 'c2', 's2')

    d1 = g1.generate(1000)
    d2 = g2.generate(1000)

    cats = [minkit.Category('uml', g1, d1), minkit.Category('uml', g2, d2)]

    cv = np.linspace(*g1.args.get('c1').bounds, 10)
    sv = np.linspace(*g2.args.get('s2').bounds, 10)

    mp = tuple(a.flatten() for a in np.meshgrid(cv, sv))

    with minkit.simultaneous_minimizer(cats) as minimizer:
        minimizer.fcn_profile('c1', cv)
        minimizer.fcn_profile(['c1', 's2'], mp)


@pytest.mark.profiles
def test_simultaneous_minimization_profile():
    '''
    Test the calculation of minimization profiles by the minimizers.
    '''
    g1 = helpers.default_gaussian('g1', 'x1', 'c1', 's1')
    g2 = helpers.default_gaussian('g2', 'x2', 'c2', 's2')

    c1 = g1.args.get('c1')
    s2 = g2.args.get('s2')

    # Change the bounds to avoid evaluations to zero
    c1.bounds = (-2, +2)
    s2.bounds = (0.5, 1.5)

    d1 = g1.generate(1000)
    d2 = g2.generate(1000)

    cats = [minkit.Category('uml', g1, d1), minkit.Category('uml', g2, d2)]

    cv = np.linspace(*c1.bounds, 10)
    sv = np.linspace(*s2.bounds, 10)

    mp = tuple(a.flatten() for a in np.meshgrid(cv, sv))

    with minkit.simultaneous_minimizer(cats) as minimizer:
        minimizer.minimize()
        minimizer.minimization_profile('c1', cv)
        minimizer.minimization_profile(['c1', 's2'], mp)
