########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Definition of PDFs based in C, OpenCL or CUDA.
'''
from . import pdf_core

import logging

__all__ = ['Amoroso', 'Argus', 'Chebyshev', 'CrystalBall',
           'Exponential', 'ExpPoly', 'Gaussian', 'Landau', 'Polynomial', 'PowerLaw']

logger = logging.getLogger(__name__)


@pdf_core.register_pdf
class Amoroso(pdf_core.SourcePDF):

    def __init__(self, name, x, a, theta, alpha, beta, backend=None):
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

        super().__init__(
            name, [x], [a, theta, alpha, beta], backend=backend)


@pdf_core.register_pdf
class Argus(pdf_core.SourcePDF):

    def __init__(self, name, x, mu, c, p, backend=None):
        r'''
        Create a new PDF with the name, parameter related to the data and the argument parameters.
        If :math:`p = 0.5`, the Amoroso distribution can be simplified as

        .. math:: f\left(x;\mu,c,p\right) = x \sqrt{1 - \frac{x^2}{\mu^2}} e^{- \frac{1}{2}\left(1 - \frac{x^2}{\mu^2}\right)}

        :param name: name of the PDF.
        :type name: str
        :param x: data parameter.
        :type x: Parameter
        :param mu: cut-off of the distribution.
        :type mu: Parameter
        :param c: parameter ruling the curvature of the distribution.
        :type c: Parameter
        :param p: power ruling the peaking strength of the distribution.
        :type p: Parameter
        '''
        super().__init__(name, [x], [mu, c, p], backend=backend)


@pdf_core.register_pdf
class Chebyshev(pdf_core.SourcePDF):

    def __init__(self, name, x, *coeffs, backend=None):
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
        super().__init__(name, [x], None, coeffs, backend)


@pdf_core.register_pdf
class CrystalBall(pdf_core.SourcePDF):

    def __init__(self, name, x, c, s, a, n, backend=None):
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
        super().__init__(name, [x], [c, s, a, n], backend=backend)


@pdf_core.register_pdf
class Exponential(pdf_core.SourcePDF):

    def __init__(self, name, x, k, backend=None):
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
        super().__init__(name, [x], [k], backend=backend)


@pdf_core.register_pdf
class ExpPoly(pdf_core.SourcePDF):

    def __init__(self, name, x, k, *coeffs, backend=None):
        r'''
        Create a new PDF with the parameters related to the data and the slope parameter.
        The PDF is defined as

        .. math:: f\left(x;k\right) = \sum_{i = 0}^n \alpha_i x^i e^{k x},

        where the first parameter :math:`\alpha_0` is set to one due to the
        normalization requirement.

        :param name: name of the PDF.
        :type name: str
        :param x: data parameter.
        :type x: Parameter
        :param k: parameter of the exponential.
        :type k: Parameter
        :param coeffs: coefficients for the polynomial
        :type coeffs: tuple(Parameter)
        '''
        super().__init__(name, [x], [k], coeffs, backend)


@pdf_core.register_pdf
class Gaussian(pdf_core.SourcePDF):

    def __init__(self, name, x, center, sigma, backend=None):
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
        super().__init__(name, [x], [center, sigma], backend=backend)


@pdf_core.register_pdf
class Landau(pdf_core.SourcePDF):

    def __init__(self, name, x, center, sigma, backend=None):
        r'''
        Create a Landau PDF with the parameters related to the data, center and
        scale parameter. The PDF is defined as

        .. math:: f\left(x;c,\sigma\right) = \frac{1}{2\pi i \sigma}\int_{a - i \infty}^{a + i \infty} e^{\lambda s + s \log s} ds,

        where :math:`\lambda = \frac{x - c}{\sigma}` and *a* is an arbitrary
        parameter.

        :param name: name of the PDF.
        :type name: str
        :param x: Parameter
        :type x: data parameter.
        :param center: center of the Landau distribution.
        :type center: Parameter
        :param sigma: scale parameter.
        :type sigma: Parameter
        '''
        super().__init__(name, [x], [center, sigma], backend=backend)


@pdf_core.register_pdf
class Polynomial(pdf_core.SourcePDF):

    def __init__(self, name, x, *coeffs, backend=None):
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
        super().__init__(name, [x], None, coeffs, backend)


@pdf_core.register_pdf
class PowerLaw(pdf_core.SourcePDF):

    def __init__(self, name, x, c, n, backend=None):
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

        super().__init__(name, [x], [c, n], backend=backend)
