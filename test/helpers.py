'''
Helper functions to test the minkit package.
'''
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
    for attr in ('name', 'value', 'error', 'constant'):
        assert getattr(f, attr) == getattr(s, attr)
    assert np.all(np.array(f.bounds) == np.array(s.bounds))


def check_pdfs(f, s):
    '''
    Check that two PDFs have the same values for the attributes.
    '''
    assert f.name == s.name
    for fa, sa in zip(f.args.values(), s.args.values()):
        check_parameters(fa, sa)
    for fa, sa in zip(f.data_pars.values(), s.data_pars.values()):
        check_parameters(fa, sa)


def check_multi_pdfs(f, s):
    '''
    Check that two PDFs have the same values for the attributes.
    '''
    assert f.name == s.name
    for fa, sa in zip(f.args.values(), s.args.values()):
        check_parameters(fa, sa)
    for fa, sa in zip(f.data_pars.values(), s.data_pars.values()):
        check_parameters(fa, sa)
    for fp, sp in zip(f.pdfs.values(), s.pdfs.values()):
        check_pdfs(fp, sp)


def compare_with_numpy(pdf, numpy_data, data_par):
    '''
    Compare the output of the evaluation of a PDF with data taken from
    numpy.
    '''
    # Create the data
    values, edges = np.histogram(numpy_data, bins=100, range=data_par.bounds)

    centers = minkit.DataSet.from_array(
        0.5 * (edges[1:] + edges[:-1]), data_par)

    pv = minkit.aop.extract_ndarray(pdf(centers))

    pdf_values = minkit.scale_pdf_values(pv, values, edges)

    assert np.allclose(np.sum(pdf_values), np.sum(values))
