'''
Helper functions to test the minkit package.
'''
import contextlib
import functools
import logging
import minkit
import numpy as np


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
    for attr in ('bounds', 'value', 'error', 'constant'):
        fa, sa = getattr(f, attr), getattr(s, attr)
        if fa is not None:
            assert np.allclose(fa, sa, **kwargs)
        else:
            assert sa is None


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


def compare_with_numpy(pdf, numpy_data, data_par, rtol=0.01):
    '''
    Compare the output of the evaluation of a PDF with data taken from
    numpy.
    '''
    # Create the data
    values, edges = np.histogram(numpy_data, bins=100, range=data_par.bounds)

    centers = minkit.DataSet.from_array(
        0.5 * (edges[1:] + edges[:-1]), data_par)

    pdf_values = minkit.plotting.core.scaled_pdf_values(
        pdf, centers, values, edges)

    assert np.allclose(np.sum(pdf_values), np.sum(values), rtol=rtol)


def setting_numpy_seed(function=None, seed=98953):
    '''
    Decorator to set the NumPy random seed before executing a function.
    '''
    def __wrapper(function):
        @functools.wraps(function)
        def __wrapper(*args, **kwargs):
            np.random.seed(seed)
            return function(*args, **kwargs)
        return __wrapper

    if function is not None:
        return __wrapper(function)
    else:
        return __wrapper


class fit_test(object):

    def __init__(self, proxy, nsigma=5, simultaneous=False):
        '''
        Save the initial values and do a check afterwards to determine whether
        the fit was successful or not.
        '''
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

            # Print the result
            print(self.result)

            # Check that the fit run fine
            assert not self.result.fmin.hesse_failed

            # Check the values of the parameters
            results = minkit.minuit_to_registry(self.result.params)
            for n, v in self.initials.items():
                rv = results.get(n)
                assert np.allclose(v, rv.value, atol=self.nsigma * rv.error)

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

            v = np.random.normal(c, 0.2886751345948129 * (r - l))

            if v > r:
                v = 0.5 * (r - c)
            elif v < l:
                v = 0.5 * (c - l)

            p.value = v
