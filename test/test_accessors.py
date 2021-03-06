########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Tests for the "accessors" module.
'''
import helpers
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

    x = minkit.Parameter('x', bounds=(0, 10))

    with pytest.raises(FileNotFoundError):
        NonExistingPDF('non-existing', x)

    with open(os.path.join(tmpdir, 'ExistingPDF.xml'), 'wt') as fi:
        fi.write('''
        <PDF data="x" parameters="a">
          <function>
            return a * x;
          </function>
          <integral bounds="xmin xmax">
            return 0.5 * a * (xmax * xmax - xmin * xmin);
          </integral>
        </PDF>
        ''')

    # Add the temporary directory to the places where to look for PDFs
    minkit.add_pdf_src(tmpdir)

    @minkit.register_pdf
    class ExistingPDF(minkit.SourcePDF):
        def __init__(self, name, x, a):
            super(ExistingPDF, self).__init__(name, [x], [a])

    a = minkit.Parameter('a', 1.)
    pdf = ExistingPDF('existing', x, a)

    assert np.allclose(pdf.integral(), 1)
    helpers.check_numerical_normalization(pdf)
