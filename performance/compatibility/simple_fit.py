#!/usr/bin/env python
########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Check that the results of binned and unbinned fits are similar
to those obtained using RooFit.
'''
import argparse
import collections
import minkit
import numpy as np
import ROOT as rt
rt.PyConfig.IgnoreCommandLineOptions = True

RooFitModel = collections.namedtuple(
    'RooFitModel', ['pdf', 'data_par', 'args'])


def gaussian_constraint(backend, var, std=0.1):
    '''
    Create a gaussian constraint for the given backend and variable.
    '''

    if backend == 'minkit':
        cn, sn, gn = f'{var.name}_cc', f'{var.name}_cs', f'{var.name}_constraint'
        c = minkit.Parameter(cn, var.value)
        s = minkit.Parameter(sn, std)
        return minkit.Gaussian(gn, var, c, s)
    elif backend == 'roofit':
        cn, sn, gn = f'{var.GetName()}_cc', f'{var.GetName()}_cs', f'{var.GetName()}_constraint'
        c = rt.RooRealVar(cn, cn, var.getVal())
        s = rt.RooRealVar(sn, sn, std)
        g = rt.RooGaussian(gn, gn, var, c, s)
        return RooFitModel(g, var, [c, s])
    else:
        raise ValueError(f'Unknown backend "{backend}"')


def gaussian_model(backend):
    '''
    Return a gaussian model that can be in the minkit or RooFit backends.
    '''
    if backend == 'minkit':
        m = minkit.Parameter('m', bounds=(30, 50))
        c = minkit.Parameter('c', 40, bounds=(30, 50))
        s = minkit.Parameter('s', 5, bounds=(0.1, 10))
        return minkit.Gaussian('g', m, c, s)
    elif backend == 'roofit':
        m = rt.RooRealVar('m', 'm', 30, 50)
        c = rt.RooRealVar('c', 'c', 40, 30, 50)
        s = rt.RooRealVar('s', 's', 5, 0.1, 10)
        g = rt.RooGaussian('g', 'g', m, c, s)
        return RooFitModel(g, m, [c, s])
    else:
        raise ValueError(f'Unknown backend "{backend}"')


def fit_and_check(fcn, minkit_model, minkit_data, roofit_model, roofit_data, constraints=None):
    '''
    Fit the models of the two backends to a FCN and check that the results are
    the same.
    '''
    if constraints is None:
        minkit_constraint, roofit_constraint = None, rt.RooFit.ExternalConstraints(
            rt.RooArgSet())
    else:
        minkit_constraint, rc = constraints
        roofit_constraint = rt.RooFit.ExternalConstraints(rt.RooArgSet(*rc))

    roofit_model.pdf.fitTo(
        roofit_data, rt.RooFit.Save(), roofit_constraint)
    with minkit.minimizer(fcn, minkit_model, minkit_data, constraints=minkit_constraint) as minimizer:
        minimizer.migrad()

    for p in roofit_model.args:  # check that the value and errors coincide
        mp = minkit_model.args.get(p.GetName())
        assert np.allclose(mp.value, p.getVal())
        assert np.allclose(mp.error, p.getError())


def minkit_data_to_roofit(var, data):
    '''
    Convert a DataSet or BinnedDataSet into a RooDataSet or RooDataHist
    '''
    if data._sample_type == 'binned':
        h = rt.TH1D('', '', len(data), *data.data_pars[0].bounds)
        for i, v in enumerate(data.values.as_ndarray()):
            h.SetBinContent(i + 1, v)
        return rt.RooDataHist('data', 'data', rt.RooArgList(var), h)
    else:
        smp = rt.RooDataSet('data', 'data', rt.RooArgSet(var))
        s = rt.RooArgSet(var)
        for v in data.values.as_ndarray():  # this is slow...
            var.setVal(v)
            smp.add(s)
        return smp


def binned_maximum_likelihood():
    '''
    Check the result of binned maximum likelihood fits.
    '''
    minkit_model = gaussian_model('minkit')
    roofit_model = gaussian_model('roofit')
    minkit_data = minkit_model.generate(10000).make_binned(100)
    roofit_data = minkit_data_to_roofit(roofit_model.data_par, minkit_data)

    minkit_constraints = [gaussian_constraint(
        'minkit', minkit_model.args.get('c'))]

    roofit_constraint_models = [gaussian_constraint(
        'roofit', roofit_model.pdf.getParameters(roofit_data).find('c'))]

    roofit_constraints = [m.pdf for m in roofit_constraint_models]

    fit_and_check('bml', minkit_model, minkit_data, roofit_model, roofit_data)
    fit_and_check('bml', minkit_model, minkit_data, roofit_model, roofit_data,
                  constraints=(minkit_constraints, roofit_constraints))


def unbinned_maximum_likelihood():
    '''
    Check the result of unbinned maximum likelihood fits.
    '''
    minkit_model = gaussian_model('minkit')
    roofit_model = gaussian_model('roofit')
    minkit_data = minkit_model.generate(10000)
    roofit_data = minkit_data_to_roofit(roofit_model.data_par, minkit_data)

    minkit_constraints = [gaussian_constraint(
        'minkit', minkit_model.args.get('c'))]

    roofit_constraint_models = [gaussian_constraint(
        'roofit', roofit_model.pdf.getParameters(roofit_data).find('c'))]

    roofit_constraints = [m.pdf for m in roofit_constraint_models]

    fit_and_check('uml', minkit_model, minkit_data, roofit_model, roofit_data)
    fit_and_check('uml', minkit_model, minkit_data, roofit_model, roofit_data,
                  constraints=(minkit_constraints, roofit_constraints))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(description='Mode to run')

    bml = subparsers.add_parser(
        'bml', description=binned_maximum_likelihood.__doc__)
    bml.set_defaults(function=binned_maximum_likelihood)

    uml = subparsers.add_parser(
        'uml', description=unbinned_maximum_likelihood.__doc__)
    uml.set_defaults(function=unbinned_maximum_likelihood)

    args = parser.parse_args()

    config = vars(args)

    function = config.pop('function')

    function(**config)
