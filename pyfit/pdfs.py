'''
Definition of PDFs based in C, OpenCL or CUDA.
'''
from . import accessors
from . import parameters
from . import pdf_core

__all__ = ['Exponential', 'Gaussian', 'Polynomial']


class Exponential(pdf_core.SourcePDF):
    '''
    Definition of an Exponential.
    '''
    def __init__( self, name, x, k ):
        '''
        Create a new PDF with the parameters related to the data and the slope parameter.
        '''
        func, pdf, norm = accessors.access_pdf('Exponential', ndata_pars=1, narg_pars=1)
        super(Exponential, self).__init__(name, func, pdf, norm, [x], [k])


class Gaussian(pdf_core.SourcePDF):
    '''
    Definition of a Gaussian.
    '''
    def __init__( self, name, x, center, sigma ):
        '''
        Create a new PDF with the parameters related to the data, center and standard
        deviation.

        :param x: Parameter
        :type x: running variable
        :param center: center
        :type center: Parameter
        :param sigma: standard deviation
        :type sigma: Parameter
        '''
        func, pdf, norm = accessors.access_pdf('Gaussian', ndata_pars=1, narg_pars=2)
        super(Gaussian, self).__init__(name, func, pdf, norm, [x], [center, sigma])


class Polynomial(pdf_core.SourcePDF):
    '''
    Definition of a polynomial PDF.
    '''
    def  __init__( self, name, x, *coeffs ):
        '''
        Build the class given the name, parameter related to data and the coefficients.
        Coefficients must be sorted from higher to lower order.

        :param name: name of the PDF.
        :type name: str
        :param x: parameter related to the data.
        :type x: Parameter
        :param coeffs: coefficients for the polynomial
        :type coeffs: tuple(Parameter)
        '''
        func, pdf, norm = accessors.access_pdf('Polynomial', ndata_pars=1, nvar_arg_pars=len(coeffs))
        super(Polynomial, self).__init__(name, func, pdf, norm, [x], None, coeffs)

