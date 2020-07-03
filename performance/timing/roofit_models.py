########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Some RooFit models to generate and fit.
'''
import ROOT as rt


def basic():
    '''
    Basic Gaussian model.
    '''
    m = rt.RooRealVar('m', 'm', 0, 10, 20)
    c = rt.RooRealVar('c', 'c', 15, 10, 20)
    s = rt.RooRealVar('s', 's', 2, 0.1, 5)
    g = rt.RooGaussian('g', 'g', m, c, s)
    return g, [m, c, s], []


def intermediate():
    '''
    Intermediate model, with a Crystal-ball and an exponential.
    '''
    # Signal
    m = rt.RooRealVar('m', 'm', 0, 5, 25)
    c = rt.RooRealVar('c', 'c', 15, 10, 20)
    s = rt.RooRealVar('s', 's', 1, 0.1, 5)
    a = rt.RooRealVar('a', 'a', 1.5, 0.1, 5)
    n = rt.RooRealVar('n', 'n', 10, 1, 30)
    sig = rt.RooCBShape('cb', 'cb', m, c, s, a, n)

    # Background
    k = rt.RooRealVar('k', 'k', -1e-6, -1e-4, 0)
    bkg = rt.RooExponential('bkg', 'bkg', m, k)

    # Model
    y = rt.RooRealVar('y', 'y', 0.5, 0, 1)
    pdf = rt.RooAddPdf('pdf', 'pdf', sig, bkg, y)

    return pdf, [m, c, s, a, n, k, y], [sig, bkg]


def numeric():
    '''
    Model where numerical integration is needed (Argus).
    '''
    m = rt.RooRealVar('m', 'm', 0, 1)
    mu = rt.RooRealVar('mu', 'mu', 0.9, 0.5, 1)
    c = rt.RooRealVar('c', 'c', 0.2, 0.01, 2)
    p = rt.RooRealVar('p', 'p', 0.7, 0.1, 1)
    pdf = rt.RooArgusBG('pdf', 'pdf', m, mu, c, p)
    return pdf, [m, mu, c, p], []
