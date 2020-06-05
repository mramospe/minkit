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

    with pytest.raises(RuntimeError):
        NonExistingPDF('non-existing', x)

    with open(os.path.join(tmpdir, 'ExistingPDF.xml'), 'wt') as fi:
        fi.write('''
        <PDF>
          <parameters a="a"/>
          <function>
            <data x="x"/>
            <code>
              return a * x;
            </code>
          </function>
          <integral>
            <bounds xmin="xmin" xmax="xmax"/>
            <code>
              return 0.5 * a * (xmax * xmax - xmin * xmin);
            </code>
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
