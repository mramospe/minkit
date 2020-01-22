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


def check_parameters(f, s):
    '''
    Check that two parameters have the same values for the attributes.
    '''
    assert f.name == s.name
    for attr in ('bounds', 'value', 'error', 'constant'):
        fa, sa = getattr(f, attr), getattr(s, attr)
        if fa is not None:
            assert np.allclose(fa, sa)
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

    pdf_values = minkit.scale_pdf_values(pdf, centers, values, edges)

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

    def __init__(self, proxy, atol=1e-8, rtol=1e-5, simultaneous=False):
        '''
        Save the initial values and do a check afterwards to determine whether
        the fit was successful or not.
        '''
        # Must be set after the minimization ends
        self.result = None

        # Tolerances for numpy.allclose
        self.atol = atol
        self.rtol = rtol

        # Keep the information on the object to minimize
        self.proxy = proxy
        self.simultaneous = simultaneous

        # Extract the initial values
        if self.simultaneous:
            self.initials = {}
            for c in self.proxy:
                self.initials.update(c.pdf.get_values())
        else:
            self.initials = self.proxy.get_values()

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
            results = minkit.minuit_to_registry(self.result)
            for n, v in self.initials.items():
                assert np.allclose(v, results.get(n).value,
                                   atol=self.atol, rtol=self.rtol)

            # Set the values of the PDF(s)
            r = {p.name: p.value for p in results}
            if self.simultaneous:
                for c in self.proxy:
                    c.pdf.set_values(**r)
            else:
                self.proxy.set_values(**r)
        else:
            raise RuntimeError(
                'Must set the attribute "result" to the result from the minimization')
