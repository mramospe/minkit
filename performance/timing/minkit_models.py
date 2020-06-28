########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Some minkit models to generate and fit.
'''
import minkit


def basic(backend):
    '''
    Basic Gaussian model.
    '''
    m = minkit.Parameter('m', bounds=(10, 20))
    c = minkit.Parameter('c', 15, bounds=(10, 20))
    s = minkit.Parameter('s', 2, bounds=(0.1, 5))
    g = backend.Gaussian('g', m, c, s)
    return g


def intermediate(backend):
    '''
    Intermediate model, with a Crystal-ball and an exponential.
    '''
    # Signal
    m = minkit.Parameter('m', bounds=(5, 25))
    c = minkit.Parameter('c', 15, bounds=(10, 20))
    s = minkit.Parameter('s', 1, bounds=(0.1, 5))
    a = minkit.Parameter('a', 1.5, bounds=(0.1, 5))
    n = minkit.Parameter('n', 10, bounds=(1, 30))
    sig = backend.CrystalBall('sig', m, c, s, a, n)

    # Background
    k = minkit.Parameter('k', -1e-6, bounds=(-1e-4, 0))
    bkg = backend.Exponential('bkg', m, k)

    # Model
    y = minkit.Parameter('y', 0.5, bounds=(0, 1))
    pdf = backend.AddPDFs.two_components('pdf', sig, bkg, y)

    return pdf


def numeric(backend):
    '''
    Model where numerical integration is needed (Argus).
    '''
    m = minkit.Parameter('m', bounds=(0, 1))
    mu = minkit.Parameter('mu', 0.9, bounds=(0.5, 1))
    c = minkit.Parameter('c', 0.2, bounds=(0.01, 2))
    p = minkit.Parameter('p', 0.6, bounds=(0.1, 1))
    pdf = minkit.Argus('argus', m, mu, c, p)
    return pdf
