'''
Definition of PDFs based in C, OpenCL or CUDA.
'''
from . import accessors
from . import parameters
from . import pdf_core

import logging

__all__ = ['Amoroso', 'Chebyshev', 'CrystalBall',
           'Exponential', 'Gaussian', 'Polynomial', 'PowerLaw']

logger = logging.getLogger(__name__)


@pdf_core.register_pdf
class Amoroso(pdf_core.SourcePDF):
    '''
    Definition of the Amoroso PDF.
    '''

    def __init__(self, name, x, a, theta, alpha, beta):
        '''
        Create a new PDF with the name, parameter related to the data and the argument parameters.

        .. warning: This function is unstable and the evaluation can explode easily for certain combination of parameters, as the normalization is currently done numerically.
        '''
        if alpha.value <= 0:
            logger.warning(
                'Parameter "alpha" for the {self.__class__.__name__} PDF must be greater than zero; check its initial value')

        if alpha.bounds is not None and alpha.bounds[0] <= 0:
            logger.warning(
                'Parameter "alpha" for the {self.__class__.__name__} PDF must be greater than zero; check its bounds')

        super(Amoroso, self).__init__(name, [x], [a, theta, alpha, beta])


@pdf_core.register_pdf
class Chebyshev(pdf_core.SourcePDF):
    '''
    Definition of a Chebyshev polynomial PDF.
    '''

    def __init__(self, name, x, *coeffs):
        '''
        Build the class given the name, parameter related to data and coefficients.
        Coefficients must be sorted from lower to higher order.
        Due to the normalization requirement, the first coefficient corresponds to the
        Chebyshev polynomial of n = 1, thus a straight line.

        :param name: name of the PDF.
        :type name: str
        :param x: parameter related to the data.
        :type x: Parameter
        :param coeffs: coefficients for the polynomial
        :type coeffs: tuple(Parameter)
        '''
        super(Chebyshev, self).__init__(name, [x], None, coeffs)


@pdf_core.register_pdf
class CrystalBall(pdf_core.SourcePDF):
    '''
    Definition of the CrystalBall PDF.
    '''

    def __init__(self, name, x, c, s, a, n):
        '''
        Create a new PDF with the name, parameter related to the data and the argument parameters.
        '''
        super(CrystalBall, self).__init__(name, [x], [c, s, a, n])


@pdf_core.register_pdf
class Exponential(pdf_core.SourcePDF):
    '''
    Definition of an Exponential.
    '''

    def __init__(self, name, x, k):
        '''
        Create a new PDF with the parameters related to the data and the slope parameter.
        '''
        super(Exponential, self).__init__(name, [x], [k])


@pdf_core.register_pdf
class Gaussian(pdf_core.SourcePDF):
    '''
    Definition of a Gaussian.
    '''

    def __init__(self, name, x, center, sigma):
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
        super(Gaussian, self).__init__(name, [x], [center, sigma])


@pdf_core.register_pdf
class Polynomial(pdf_core.SourcePDF):
    '''
    Definition of a polynomial PDF.
    '''

    def __init__(self, name, x, *coeffs):
        '''
        Build the class given the name, parameter related to data and the coefficients.
        Coefficients must be sorted from lower to higher order.

        :param name: name of the PDF.
        :type name: str
        :param x: parameter related to the data.
        :type x: Parameter
        :param coeffs: coefficients for the polynomial
        :type coeffs: tuple(Parameter)
        '''
        super(Polynomial, self).__init__(name, [x], None, coeffs)


@pdf_core.register_pdf
class PowerLaw(pdf_core.SourcePDF):
    '''
    Definition of a power-law PDF.
    '''

    def __init__(self, name, x, c, n):
        '''
        Build the class given the name, parameter related to data and the coefficients.

        :param name: name of the PDF.
        :type name: str
        :param x: parameter related to the data.
        :type x: Parameter
        :param c: asymptote of the power-law.
        :type c: Parameter
        :param n: power of the function.
        :type n: Parameter
        '''
        if c.value > x.bounds[0] and c.value < x.bounds[1]:
            logger.warning(
                'Defining power law with an asymptote that lies in the middle of the integration range')

        if c.bounds is not None:
            if (c.bounds[0] > x.bounds[0] and c.bounds[0] < x.bounds[1]) or (c.bounds[1] > x.bounds[0] and c.bounds[1] < x.bounds[1]):
                logger.warning(
                    'Defining power law with an asymptote that might lie in the middle of the range of interest')

        super(PowerLaw, self).__init__(name, [x], [c, n])
