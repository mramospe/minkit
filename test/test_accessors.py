'''
Tests for the "accessors" module.
'''
import minkit
import numpy as np
import os
import pytest


@pytest.mark.pdfs
@pytest.mark.source_pdf
@pytest.mark.core
def test_add_pdf_src(tmpdir):
    '''
    Test for the "add_pdf_src" function.
    '''
    @minkit.register_pdf
    class NonExistingPDF(minkit.SourcePDF):
        def __init__(self, name, x):
            super(NonExistingPDF, self).__init__(name, [x])

    x = minkit.Parameter('x', bounds=(-10, +10))

    with pytest.raises(RuntimeError):
        NonExistingPDF('non-existing', x)

    with open(os.path.join(tmpdir, 'ExistingPDF.cpp'), 'wt') as fi:
        fi.write('''
        extern "C" {
        double function( double x ) {
        return 1.;
        }
        void evaluate( int len, double *out, double *in ) {
        for ( int i = 0; i < len; ++i )
        out[i] = 1.;
        }
        double normalization( double xmin, double xmax ) {
        return xmax - xmin;
        }
        }
        ''')

    # Add the temporary directory to the places where to look for PDFs
    minkit.add_pdf_src(tmpdir)

    @minkit.register_pdf
    class ExistingPDF(minkit.SourcePDF):
        def __init__(self, name, x):
            super(ExistingPDF, self).__init__(name, [x])

    pdf = ExistingPDF('existing', x)

    assert np.allclose(pdf.integral(), 1)
    assert np.allclose(pdf.numerical_normalization(), pdf.norm())
