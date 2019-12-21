'''
Definition of PDFs based in C, OpenCL or CUDA.
'''
from . import accessors
from . import parameters
from . import pdf_core

__all__ = ['Exponential', 'Gaussian']


class Exponential(pdf_core.SourcePDF):
    '''
    Definition of an Exponential.
    '''
    def __init__( self, name, x, k ):
        '''
        Create a new PDF with the parameters related to the data and the slope parameter.
        '''
        pdf, norm = accessors.access_pdf('Exponential', ndata_pars=1, narg_pars=1)

        self.__x = x
        self.__k = k

        super(Exponential, self).__init__(name, pdf, norm, [x], [k])


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
        pdf, norm = accessors.access_pdf('Gaussian', ndata_pars=1, narg_pars=2)

        self.__x = x
        self.__c = center
        self.__s = sigma

        super(Gaussian, self).__init__(name, pdf, norm, [x], [center, sigma])
