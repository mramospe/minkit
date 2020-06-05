########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Helper functions to test the minkit package.
'''
import functools
import logging
import minkit
import numpy as np

# Default seed for the generators
DEFAULT_SEED = 4987

rndm_gen = np.random.RandomState(seed=DEFAULT_SEED)


def configure_logging():
    '''
    Configure the logging for the tests.
    '''
    logging.basicConfig(level=logging.INFO)


def check_parameters(f, s, **kwargs):
    '''
    Check that two parameters have the same values for the attributes.
    '''
    assert f.name == s.name
    for attr in ('bounds', 'value', 'error', 'constant', 'asym_errors'):
        fa, sa = getattr(f, attr), getattr(s, attr)
        if fa is not None:
            assert np.allclose(fa, sa, **kwargs)
        else:
            assert sa is None

    for n in f.ranges:
        assert np.allclose(f.get_range(n), s.get_range(n))


def check_pdfs(f, s):
    '''
    Check that two PDFs have the same values for the attributes.
    '''
    assert f.name == s.name
    for fa, sa in zip(f.args, s.args):
        check_parameters(fa, sa)
    for fa, sa in zip(f.data_pars, s.data_pars):
        check_parameters(fa, sa)


def check_multi_pdfs(f, s):
    '''
    Check that two PDFs have the same values for the attributes.
    '''
    assert f.name == s.name
    for fa, sa in zip(f.args, s.args):
        check_parameters(fa, sa)
    for fa, sa in zip(f.data_pars, s.data_pars):
        check_parameters(fa, sa)
    for fp, sp in zip(f.pdfs, s.pdfs):
        check_pdfs(fp, sp)


def check_numerical_normalization(pdf, range=minkit.base.parameters.FULL):
    '''
    Check the numerical normalization of a PDF.
    '''
    assert np.allclose(pdf.numerical_normalization(
        range=range), pdf.norm(range=range), rtol=0.05)


def compare_with_numpy(pdf, numpy_data, data_par, rtol=0.01):
    '''
    Compare the output of the evaluation of a PDF with data taken from
    numpy.
    '''
    # Create the data
    values, edges = np.histogram(numpy_data, bins=100, range=data_par.bounds)

    centers = minkit.DataSet.from_ndarray(
        0.5 * (edges[1:] + edges[:-1]), data_par)

    pdf_values = minkit.utils.core.scaled_pdf_values(
        pdf, centers, values, edges)

    assert np.allclose(np.sum(pdf_values), np.sum(values), rtol=rtol)


def default_add_pdfs(center='c', sigma='s', k='k', extended=False, yields=None):
    '''
    Create a combination of a Gaussian and an exponential.
    '''
    # Simple fit to a Gaussian
    x = minkit.Parameter('x', bounds=(0, 20))  # bounds to generate data later
    c = minkit.Parameter(center, 10, bounds=(8, 12))
    s = minkit.Parameter(sigma, 2, bounds=(1, 3))
    g = minkit.Gaussian('gaussian', x, c, s)

    # Test for a composed PDF
    k = minkit.Parameter(k, -0.1, bounds=(-1, 0))
    e = minkit.Exponential('exponential', x, k)

    if extended:
        ng_name, ne_name = tuple(
            yields if yields is not None else ('ng', 'ne'))
        ng = minkit.Parameter(ng_name, 9000, bounds=(0, 10000))
        ne = minkit.Parameter(ne_name, 1000, bounds=(0, 10000))
        return minkit.AddPDFs.two_components('pdf', g, e, ng, ne)
    else:
        y_name = yields if yields is not None else 'y'
        y = minkit.Parameter(y_name, 0.5, bounds=(0, 1))
        return minkit.AddPDFs.two_components('pdf', g, e, y)


def default_gaussian(pdf_name='g', data_par='x', center='c', sigma='s'):
    '''
    Create a Gaussian function.
    '''
    x = minkit.Parameter(
        data_par, bounds=(-4, +4))
    c = minkit.Parameter(center, 0, bounds=(-4, +4))
    s = minkit.Parameter(sigma, 1, bounds=(0.1, 2.))
    return minkit.Gaussian(pdf_name, x, c, s)


def setting_seed(function=None, seed=DEFAULT_SEED):
    '''
    Decorator to set the NumPy random seed before executing a function.
    '''
    def __wrapper(function):
        @functools.wraps(function)
        def __wrapper(*args, **kwargs):
            global rndm_gen
            rndm_gen = np.random.RandomState(seed=seed)  # set the numpy seed
            minkit.backends.core.parse_backend().set_rndm_seed(seed)  # set the minkit seed
            return function(*args, **kwargs)
        return __wrapper

    if function is not None:
        return __wrapper(function)
    else:
        return __wrapper


class fit_test(object):

    def __init__(self, proxy, nsigma=5, simultaneous=False, fails=False):
        '''
        Save the initial values and do a check afterwards to determine whether
        the fit was successful or not.
        '''
        self.fails = fails

        # Must be set after the minimization ends
        self.result = None

        # Number of standard deviations allowed from the fitted value to the initial
        self.nsigma = nsigma

        # Keep the information on the object to minimize
        self.proxy = proxy
        self.simultaneous = simultaneous

        # Extract the initial values
        if self.simultaneous:
            self.initials = {}
            for c in self.proxy:
                self.initials.update(c.pdf.get_values())
            # Do it in a separate loop to avoid modifying the initial values
            for c in self.proxy:
                self._variate(c.pdf)
        else:
            self.initials = self.proxy.get_values()
            self._variate(self.proxy)

        super(fit_test, self).__init__()

    def __enter__(self):
        '''
        Return itself, so it can be modified.
        '''
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''
        Check the results.
        '''
        if any(map(lambda e: e is not None, (exc_type, exc_value, traceback))):
            return

        if self.result is not None:

            if not self.fails:

                # Print the result
                print(self.result)

                # Check that the fit run fine
                assert not self.result.hesse_failed

                # Check the values of the parameters
                if self.simultaneous:
                    results = minkit.Registry()
                    for c in self.proxy:
                        results += c.pdf.all_real_args
                else:
                    results = self.proxy.all_real_args

                for n, v in self.initials.items():
                    rv = results.get(n)
                    assert np.allclose(
                        v, rv.value, atol=self.nsigma * rv.error)

                # Reset the values of the PDF(s)
                if self.simultaneous:
                    for c in self.proxy:
                        c.pdf.set_values(**self.initials)
                else:
                    self.proxy.set_values(**self.initials)
        else:
            raise RuntimeError(
                'Must set the attribute "result" to the result from the minimization')

    def _variate(self, pdf):
        '''
        Variate the values of a PDF following a normal distribution centered
        in the central value of the parameter, and with standard deviation equal
        to the distance from the bounds divided by the square root of 12.
        If it falls out of bounds, the middle distance between the closest bound
        and the current value is used.
        '''
        for p in filter(lambda a: not a.constant, pdf.all_real_args):

            l, r = p.bounds

            c = 0.5 * (r + l)

            def gen(): return rndm_gen.normal(c, 0.2886751345948129 * (r - l))

            v = gen()
            while v > r or v < l:
                v = gen()

            p.value = v
