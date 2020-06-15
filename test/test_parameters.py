########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Test the "parameters" module.
'''
from helpers import check_parameters, compare_with_numpy, rndm_gen
import json
import helpers
import minkit
import numpy as np
import os
import pytest

helpers.configure_logging()


@pytest.mark.core
def test_parameter(tmpdir):
    '''
    Test the "Parameter" class.
    '''
    f = minkit.Parameter('a', 1., (-5, +5),
                         {'sides': [(-5, -2), (+2, +5)]}, 0.1, False)

    with open(os.path.join(tmpdir, 'a.json'), 'wt') as fi:
        json.dump(f.to_json_object(), fi)

    with open(os.path.join(tmpdir, 'a.json'), 'rt') as fi:
        s = minkit.Parameter.from_json_object(json.load(fi))

    check_parameters(f, s)

    c = s.copy()

    assert not c is s

    check_parameters(c, s)


@pytest.mark.core
def test_eval_math_expression():
    '''
    Test the "eval_math_expression" function.
    '''
    ev = minkit.base.core.eval_math_expression

    assert ev('1 * 2') == 2  # simple operation

    assert ev('min(1, 2)') == 1  # functions

    assert ev('cos(pi)') == -1  # functions and constants

    with pytest.raises(NameError):
        ev('non_existing_function(1, 2)')  # error if function does not exist

    with pytest.raises(NameError):
        ev('__import__("subprocess")')  # avoid doing nasty things


@pytest.mark.core
@helpers.setting_seed
def test_formula(tmpdir):
    '''
    Test the "Formula" class.
    '''
    a = minkit.Parameter('a', 1)
    b = minkit.Parameter('b', 2)
    c = minkit.Formula('c', '{a} * {b}', [a, b])

    assert np.allclose(c.value, a.value * b.value)

    # Test its use on a PDF
    m = minkit.Parameter('m', bounds=(10, 20))
    c = minkit.Parameter('c', 15, bounds=(10, 20))
    s = minkit.Formula('s', '0.1 + {c} / 10', [c])
    g = minkit.Gaussian('gaussian', m, c, s)

    data = g.generate(10000)

    nd = rndm_gen.normal(c.value, s.value, 10000)

    compare_with_numpy(g, nd, m)

    with helpers.fit_test(g) as test:
        with minkit.minimizer('uml', g, data, minimizer='minuit') as minuit:
            test.result, _ = minuit.migrad()

    # Test the JSON (only for formula)
    with open(os.path.join(tmpdir, 'r.json'), 'wt') as fi:
        json.dump(s.to_json_object(), fi)

    with open(os.path.join(tmpdir, 'r.json'), 'rt') as fi:
        s = minkit.Formula.from_json_object(json.load(fi), g.all_real_args)

    # Test the JSON (whole PDF)
    with open(os.path.join(tmpdir, 'pdf.json'), 'wt') as fi:
        json.dump(minkit.pdf_to_json(g), fi)

    with open(os.path.join(tmpdir, 'pdf.json'), 'rt') as fi:
        minkit.pdf_from_json(json.load(fi))

    # Test the copy of a formula
    new_args = s.args.copy()

    assert all(not o is p for o, p in zip(s.args, s.copy(new_args).args))

    # Test for a formula depending on another formula
    m = minkit.Parameter('m', bounds=(10, 20))
    c = minkit.Parameter('c', 15, bounds=(10, 20))
    d = minkit.Formula('d', '0.1 + {c} / 10', [c])
    s = minkit.Formula('s', '2 * {d}', [d])
    g = minkit.Gaussian('gaussian', m, c, s)

    assert s.value == 3.2

    data = g.generate(10000)

    with helpers.fit_test(g) as test:
        with minkit.minimizer('uml', g, data, minimizer='minuit') as minuit:
            test.result, _ = minuit.migrad()

    # Test the JSON (only for formula)
    with open(os.path.join(tmpdir, 'r.json'), 'wt') as fi:
        json.dump(s.to_json_object(), fi)

    with open(os.path.join(tmpdir, 'r.json'), 'rt') as fi:
        s = minkit.Formula.from_json_object(json.load(fi), g.all_args)

    # Test the copy of a formula depending on another formula
    new_args = s.args.copy()

    assert all(not o is p for o, p in zip(s.args, s.copy(new_args).args))

    # Test the JSON (whole PDF)
    with open(os.path.join(tmpdir, 'pdf.json'), 'wt') as fi:
        json.dump(minkit.pdf_to_json(g), fi)

    with open(os.path.join(tmpdir, 'pdf.json'), 'rt') as fi:
        minkit.pdf_from_json(json.load(fi))


@pytest.mark.core
@helpers.setting_seed
def test_range():
    '''
    Test for disjointed ranges.
    '''
    # Do calculations in a range
    m = minkit.Parameter('m', bounds=(0, 10))
    k = minkit.Parameter('k', -0.5, bounds=(-0.8, -0.3))
    e = minkit.Exponential('exponential', m, k)

    m.set_range('sides', [(0, 4), (6, 10)])

    helpers.check_numerical_normalization(e, range='sides')

    data = e.generate(10000)

    with helpers.fit_test(e) as test:
        with minkit.minimizer('uml', e, data, minimizer='minuit', range='sides') as minuit:
            test.result, _ = minuit.migrad()

    # Test generation of data only in the range
    data = e.generate(10000, range='sides')

    with helpers.fit_test(e) as test:
        with minkit.minimizer('uml', e, data, minimizer='minuit', range='sides') as minuit:
            test.result, _ = minuit.migrad()


@pytest.mark.core
def test_registry(tmpdir):
    '''
    Test the "Registry" class.
    '''
    a = minkit.Parameter('a', 1., (-5, +5), None, 0.1, False)
    b = minkit.Parameter('b', 0., (-10, +10), None, 2., True)

    f = minkit.Registry([a, b])

    with open(os.path.join(tmpdir, 'r.json'), 'wt') as fi:
        json.dump(f.to_json_object(), fi)

    with open(os.path.join(tmpdir, 'r.json'), 'rt') as fi:
        s = minkit.Registry.from_json_object(json.load(fi))

    assert f.names == s.names

    for fv, sv in zip(f, s):
        check_parameters(fv, sv)

    assert all(not o is p for o, p in zip(s, s.copy()))

    # Must raise errors if different objects with the same names are added to the registry
    a2 = minkit.Parameter('a', 1., (-5, +5), None, 0.1, False)

    with pytest.raises(ValueError):
        f.append(a2)

    with pytest.raises(ValueError):
        f.insert(0, a2)

    with pytest.raises(ValueError):
        _ = f + [a2, a2]

    with pytest.raises(ValueError):
        f += [a2, a2]

    # These operations do not raise an error, and the registry is not modified
    pl = len(f)
    f.append(a)
    f.insert(0, a)
    f += [a, a]
    _ = f + [a, b]
    assert len(f) == pl


@pytest.mark.core
@pytest.mark.minimization
@helpers.setting_seed
def test_blinding(tmpdir):
    '''
    Test fits with blinded parameters.
    '''
    iv, ib = 1., (0, 2)

    p = minkit.Parameter('p', value=iv, bounds=ib)

    p.set_blinding_configuration(scale=10, offset=2)

    assert not np.allclose(p.value, iv)  # value is hidden
    assert not np.allclose(p.bounds, ib)  # bounds are hidden

    with p.blind(status=False):
        assert np.allclose(p.value, iv)
        assert np.allclose(p.bounds, ib)

    hv = p.value

    # Test the blinding state in JSON files
    with open(os.path.join(tmpdir, 'p.json'), 'wt') as fi:
        json.dump(p.to_json_object(), fi)

    with open(os.path.join(tmpdir, 'p.json'), 'rt') as fi:
        pn = minkit.Parameter.from_json_object(json.load(fi))

    assert not np.allclose(pn.value, iv)  # value is still hidden
    assert np.allclose(pn.value, hv)  # blinded value is the same as before
    assert not np.allclose(pn.bounds, ib)  # bounds are still hidden

    with pn.blind(status=False):
        assert np.allclose(pn.value, iv)
        assert np.allclose(pn.bounds, ib)

    # Gaussian model with a blinded center
    pdf = helpers.default_gaussian(center='c')

    data = pdf.generate(10000)

    c = pdf.args.get('c')

    iv = c.value  # initial value
    ib = c.bounds  # initial bounds

    initial = pdf.get_values()

    c.set_blinding_configuration(scale=10, offset=2)

    for m in 'minuit', 'L-BFGS-B', 'COBYLA':  # test all the minimizers

        with c.blind(status=False):
            pdf.set_values(**initial)

        helpers.randomize(pdf)
        with minkit.minimizer('uml', pdf, data) as minimizer:
            minimizer.minimize()

        assert not np.allclose(c.value, iv)

        with c.blind(status=False):
            assert np.allclose(c.value, iv, atol=2. * c.error)
            assert np.allclose(c.bounds, ib)

    # Model composed by a background and a signal component with unknown yield
    pdf = helpers.default_add_pdfs(extended=True, yields=('nsig', 'nbkg'))

    nsig = pdf.args.get('nsig')
    nbkg = pdf.args.get('nbkg')

    nsig.value = 1000
    nbkg.value = 10000

    data = pdf.generate(int(nsig.value + nbkg.value))

    nsig.bounds = 0.8 * nsig.value, len(data)
    nbkg.bounds = 0.8 * nbkg.value, len(data)

    iv = nsig.value

    nsig.set_blinding_configuration(scale=10000, offset=100)

    helpers.randomize(pdf)
    with minkit.minimizer('ueml', pdf, data) as minimizer:
        minimizer.minimize()

    assert not np.allclose(nsig.value, iv)

    with nsig.blind(status=False):
        assert np.allclose(nsig.value, iv, atol=2. * nsig.error)

    # Blinding using a formula
    f = minkit.Formula('f', '10 * {nsig}', [nsig])

    pdf.args[pdf.args.index('nsig')] = f

    with nsig.blind(status=False):
        iv = nsig.value
        data = pdf.generate(int(f.value + nbkg.value))

    helpers.randomize(pdf)
    with minkit.minimizer('ueml', pdf, data) as minimizer:
        minimizer.minimize()

    assert not np.allclose(nsig.value, iv)

    with nsig.blind(status=False):
        assert np.allclose(nsig.value, iv, atol=2. * nsig.error)

    # Test the determination of asymmetric errors and profiles
    with minkit.minimizer('ueml', pdf, data) as minimizer:

        minimizer.minuit.print_level = 0

        minimizer.minimize()

        # FCN profile (a linear transformation makes the shapes of the profile
        # be the same)
        v = np.linspace(*nsig.bounds, 20)
        fp = minimizer.fcn_profile('nsig', v)
        mp = minimizer.minimization_profile('nsig', v)
        with nsig.blind(status=False):
            v = np.linspace(*nsig.bounds, 20)
            ufp = minimizer.fcn_profile('nsig', v)
            ump = minimizer.minimization_profile('nsig', v)

        assert np.allclose(fp, ufp)
        assert np.allclose(mp, ump)

        # asymmetric errors
        minimizer.asymmetric_errors('nsig')
        errors = nsig.asym_errors
        with nsig.blind(status=False):
            assert not np.allclose(errors, nsig.asym_errors)

        # minos errors
        minimizer.minos('nsig')
        errors = nsig.asym_errors
        with nsig.blind(status=False):
            assert not np.allclose(errors, nsig.asym_errors)

    # Check that with an offset-based blinding the error of the true value
    # is the same to that of the blinded.
    pdf = helpers.default_gaussian(center='c')

    data = pdf.generate(10000)

    c = pdf.args.get('c')

    c.set_blinding_configuration(offset=2)

    helpers.randomize(pdf)
    with minkit.minimizer('uml', pdf, data) as minimizer:

        minimizer.minimize()

        blinded = c.error
        with c.blind(status=False):
            unblinded = c.error

        assert np.allclose(blinded, unblinded)

    # Check that with an scale-based blinding the relative error of the true
    # value is the same to that of the blinded
    c.set_blinding_configuration(scale=100)

    helpers.randomize(pdf)
    with minkit.minimizer('uml', pdf, data) as minimizer:

        minimizer.minimize()

        blinded = c.error / c.value
        with c.blind(status=False):
            unblinded = c.error / c.value

        assert np.allclose(blinded, unblinded)
