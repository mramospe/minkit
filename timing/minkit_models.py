'''
Some minkit models to generate and fit.
'''
import minkit


def basic():
    '''
    Basic Gaussian model.
    '''
    m = minkit.Parameter('m', bounds=(10, 20))
    c = minkit.Parameter('c', 15, bounds=(10, 20))
    s = minkit.Parameter('s', 2, bounds=(0.1, 5))
    g = minkit.Gaussian('g', m, c, s)
    return g


def intermediate():
    '''
    Intermediate model, with a Crystal-ball and an exponential.
    '''
    # Signal
    m = minkit.Parameter('m', bounds=(5, 25))
    c = minkit.Parameter('c', 15, bounds=(10, 20))
    s = minkit.Parameter('s', 1, bounds=(0.1, 5))
    a = minkit.Parameter('a', 1.5, bounds=(0.1, 5))
    n = minkit.Parameter('n', 10, bounds=(1, 30))
    sig = minkit.CrystalBall('sig', m, c, s, a, n)

    # Background
    k = minkit.Parameter('k', -1e-6, bounds=(-1e-4, 0))
    bkg = minkit.Exponential('bkg', m, k)

    # Model
    y = minkit.Parameter('y', 0.5, bounds=(0, 1))
    pdf = minkit.AddPDFs.two_components('pdf', sig, bkg, y)

    return pdf
