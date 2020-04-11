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

    def __init__(self, name, x, a, theta, alpha, beta):
        r'''
        Create a new PDF with the name, parameter related to the data and the argument parameters.
        The amoroso distribution is defined as

        .. math:: f\left(x;a,\theta,\alpha,\beta\right) = \left(\frac{x - a}{\theta}\right)^{\alpha\beta - 1} e^{-\left(\frac{x - a}{\theta}\right)^\beta}

        :param name: name of the PDF.
        :type name: str
        :param x: data parameter.
        :type x: Parameter
        :param a: center of the distribution.
        :type a: Parameter
        :param theta: parameter related to the width of the distribution.
        :type theta: Parameter
        :param alpha: power for the distance with respect to the center. Must be greater than zero.
        :type alpha: Parameter
        :param beta: power of the exponential.
        :type beta: Parameter

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

    def __init__(self, name, x, *coeffs):
        r'''
        Build the class given the name, parameter related to data and coefficients.
        Coefficients must be sorted from lower to higher order.
        Due to the normalization requirement, the first coefficient corresponds to the
        Chebyshev polynomial of n = 1, thus a straight line.
        The Chebyshev polynomials are related through

        .. math:: T_0(x) = 1

        .. math:: T_1(x) = x

        .. math:: T_{n + 1}(x) = 2xT_n(x) - T_{n - 1}(x)

        :param name: name of the PDF.
        :type name: str
        :param x: data parameter.
        :type x: Parameter
        :param coeffs: coefficients for the polynomial
        :type coeffs: tuple(Parameter)
        '''
        super(Chebyshev, self).__init__(name, [x], None, coeffs)


@pdf_core.register_pdf
class CrystalBall(pdf_core.SourcePDF):

    def __init__(self, name, x, c, s, a, n):
        r'''
        Create a new PDF with the name, parameter related to the data and the argument parameters.
        This PDF is expressed as

        .. math:: f\left(x;c,\sigma,\alpha,n\right) = e^{- \frac{\left(x - c\right)^2}{2\sigma^2}}

        for values of

        .. math:: \frac{x - c}{\sigma} \geq - |\alpha|

        otherwise, the power law

        .. math:: \frac{A}{\left(B - \frac{x - c}{\sigma}\right)^n}

        is used, where

        .. math:: A = \left(\frac{n}{|\alpha|}\right)^n e^{- \frac{1}{2} |\alpha|^2}

        .. math:: B = \frac{n}{|\alpha|} - |\alpha|

        :param name: name of the PDF.
        :type name: str
        :param x: data parameter.
        :type x: Parameter
        :param c: center of the Gaussian core.
        :type c: Parameter
        :param s: standard deviation of the Gaussian core.
        :type s: Parameter
        :param a: number of standard deviations till the start of the power-law behaviour. A negative value implies the tail is on the right.
        :type a: Parameter
        :param n: power of the power-law.
        :type n: Parameter
        '''
        super(CrystalBall, self).__init__(name, [x], [c, s, a, n])


@pdf_core.register_pdf
class Exponential(pdf_core.SourcePDF):

    def __init__(self, name, x, k):
        r'''
        Create a new PDF with the parameters related to the data and the slope parameter.
        The PDF is defined as.

        .. math:: f\left(x;k\right) = e^{k x}

        :param name: name of the PDF.
        :type name: str
        :param x: data parameter.
        :type x: Parameter
        :param k: parameter of the exponential.
        :type k: Parameter
        '''
        super(Exponential, self).__init__(name, [x], [k])


@pdf_core.register_pdf
class Gaussian(pdf_core.SourcePDF):

    def __init__(self, name, x, center, sigma):
        r'''
        Create a new PDF with the parameters related to the data, center and standard
        deviation.
        The PDF is defined as

        .. math:: f\left(x;c,\sigma\right) = e^{-\frac{\left(x - c\right)^2}{2\sigma^2}}

        :param name: name of the PDF.
        :type name: str
        :param x: Parameter
        :type x: data parameter.
        :param center: center of the Gaussian.
        :type center: Parameter
        :param sigma: standard deviation.
        :type sigma: Parameter
        '''
        super(Gaussian, self).__init__(name, [x], [center, sigma])


@pdf_core.register_pdf
class Polynomial(pdf_core.SourcePDF):

    def __init__(self, name, x, *coeffs):
        r'''
        Build the class given the name, parameter related to data and the coefficients.
        Coefficients must be sorted from lower to higher order.
        Due to the normalization condition, the first parameter is fixed to zero, if no
        coefficients are provided the result is a constant function.
        The PDF is expressed as

        .. math:: f\left(x;a_1,a_2,...\right) = 1 + \sum_{i = 1}^n a_i x^i

        :param name: name of the PDF.
        :type name: str
        :param x: data parameter.
        :type x: Parameter
        :param coeffs: coefficients for the polynomial
        :type coeffs: tuple(Parameter)
        '''
        super(Polynomial, self).__init__(name, [x], None, coeffs)


@pdf_core.register_pdf
class PowerLaw(pdf_core.SourcePDF):

    def __init__(self, name, x, c, n):
        r'''
        Build the class given the name, parameter related to data and the coefficients.
        This PDF is expressed as

        .. math:: f\left(x;c,n\right) = \frac{1}{\left(x - c\right)^n}

        :param name: name of the PDF.
        :type name: str
        :param x: data parameter.
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
