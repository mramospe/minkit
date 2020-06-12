########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Definition of classes to evaluate FCNs using PDFs and data sets.
'''
from . import fcns
from ..base import core
from ..base import exceptions
from ..base import parameters
from ..pdfs import pdf_core

import contextlib
import functools
import logging

__all__ = ['Category', 'Evaluator', 'BinnedEvaluator',
           'UnbinnedEvaluator', 'SimultaneousEvaluator']

logger = logging.getLogger(__name__)


def unblind_parameters(method):
    '''
    Wrapper around a method to unblind the parameters. The class owing the
    method must define the property *args*.
    '''
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with contextlib.ExitStack() as stack:
            for p in filter(lambda p: p.blinded is not None, self.args):
                stack.enter_context(p.blind(status=False))
            return method(self, *args, **kwargs)
    return wrapper


class Category(object):

    def __init__(self, fcn, pdf, data):
        '''
        Object serving as a proxy for an FCN to be evaluated using a PDF on a data set.
        The type of data (binned/unbinned) is assumed from the FCN.

        :param fcn: name of the FCN.
        :type fcn: str
        :param pdf: probability density function associated to this category.
        :type pdf: PDF
        :param data: data set associated to this category.
        :type data: DataSet or BinnedDataSet
        '''
        super().__init__()

        self.__data = data
        self.__fcn = fcn
        self.__pdf = pdf

    @property
    def data(self):
        '''
        Data sample.

        :type: DataSet or BinnedDataSet
        '''
        return self.__data

    @property
    def fcn(self):
        '''
        FCN used in the category.

        :type: str
        '''
        return self.__fcn

    @property
    def pdf(self):
        '''
        Probability density function.

        :type: PDF
        '''
        return self.__pdf


# Methods to treat samples with weights
WGTS_RESCALE = 'rescale'
WGTS_NONE = 'none'


class Evaluator(object, metaclass=core.DocMeta):

    def __init__(self):
        '''
        Object to evaluate an FCN on a set of PDFs and data sets.
        '''
        super().__init__()

    def __call__(self, *values):
        '''
        Evaluate the FCN.
        Values must be provided sorted as :meth:`PDF.args`.

        :param values: set of values to evaluate the FCN.
        :type values: tuple(float)
        :returns: Value of the FCN.
        :rtype: float

        .. seealso:: :meth:`Evaluator.fcn`
        '''
        raise exceptions.MethodNotDefinedError(self.__class__, '__call__')

    @property
    def args(self):
        '''
        All the arguments of the evaluator.

        :type: Registry(Parameter)
        '''
        raise exceptions.PropertyNotDefinedError(self.__class__, 'args')

    def fcn(self):
        '''
        Calculate the value of the FCN with the current set of values.

        :returns: Value of the FCN.
        :rtype: float
        '''
        raise exceptions.MethodNotDefinedError(
            self.__class__, 'fcn')

    @contextlib.contextmanager
    def using_caches(self):
        '''
        Create a context where the cache of the PDF is activated. This should be
        done before successive calls to the evaluator.
        '''
        raise exceptions.MethodNotDefinedError(
            self.__class__, 'using_caches')


class BinnedEvaluator(Evaluator):

    def __init__(self, fcn, pdf, data, constraints=None):
        '''
        Proxy class to evaluate an FCN with a PDF on a BinnedDataSet object.

        :param fcn: FCN to be used during minimization.
        :type fcn: str
        :param pdf: PDF to minimize.
        :type pdf: PDF
        :param data: data sample to process.
        :type data: BinnedDataSet
        :param constraints: set of constraints to consider in the minimization.
        :type constraints: list(PDF) or None
        '''
        self.__data = data
        self.__fcn = fcn
        self.__pdf = pdf
        self.__constraints = constraints

        super(BinnedEvaluator, self).__init__()

    def __call__(self, *values):
        for i, p in enumerate(self.args):
            p.value = values[i]
        return self.fcn()

    @property
    def args(self):
        return self.__pdf.all_real_args

    @unblind_parameters
    def fcn(self):
        fcn = self.__fcn(self.__pdf, self.__data)
        return fcn + fcns.evaluate_constraints(self.__constraints)

    @contextlib.contextmanager
    def using_caches(self):
        with self.__pdf.using_cache(pdf_core.CONST):
            yield self


class UnbinnedEvaluator(Evaluator):

    def __init__(self, fcn, pdf, data, range=parameters.FULL, constraints=None, weights_treatment=WGTS_RESCALE):
        r'''
        Proxy class to evaluate an FCN with a PDF.

        :param fcn: FCN to be used during minimization.
        :type fcn: str
        :param pdf: PDF to minimize.
        :type pdf: PDF
        :param data: data sample to process.
        :type data: DataSet
        :param range: range of data to minimize.
        :type range: str
        :param constraints: set of constraints to consider in the minimization.
        :type constraints: list(PDF) or None
        :param weights_treatment: what to do with weighted samples (see below for more information).
        :type weights_treatment: str
        :raises ValueError: If the the way to treat the weights is unknown.

        The treatment of weights when calculating FCNs can lead to unreliable errors
        for the parameters. In general there is no correct way of processing
        the likelihoods. In this package the following methods are supported:

        * *none*: the raw weights are used to calculate the FCN. This will lead to
          completely incorrect uncertainties, since the statistical weight of the
          events in the data sample will not be proportional to the sample weight.
        * *rescale*: in this case the weights are rescaled so
          :math:`\omega^\prime_i = \omega_i \times \frac{\sum_{j = 0}^n \omega_j}{\sum_{j = 0}^n \omega_j^2}`.
          In this case the statistical weight of each event is proportional to the
          sample weight, although the uncertainties will still be incorrect.
        '''
        if weights_treatment == WGTS_RESCALE:
            self.__data = data.subset(range, rescale_weights=True)
        elif weights_treatment == WGTS_NONE:
            self.__data = data.subset(range)
        else:
            raise ValueError(
                f'Unknown weights treatment type "{weights_treatment}"')

        self.__fcn = fcn
        self.__pdf = pdf
        self.__range = range
        self.__constraints = constraints

        super(UnbinnedEvaluator, self).__init__()

    def __call__(self, *values):
        for i, p in enumerate(self.args):
            p.value = values[i]
        return self.fcn()

    @property
    def args(self):
        return self.__pdf.all_real_args

    @property
    def data(self):
        '''
        Data sample.

        :type: DataSet or BinnedDataSet
        '''
        return self.__data

    @property
    def pdf(self):
        '''
        Probability density function.

        :type: PDF
        '''
        return self.__pdf

    @unblind_parameters
    def fcn(self):
        fcn = self.__fcn(self.__pdf, self.__data, self.__range)
        return fcn + fcns.evaluate_constraints(self.__constraints)

    @contextlib.contextmanager
    def using_caches(self):
        with self.__pdf.using_cache(pdf_core.CONST):
            yield self


class SimultaneousEvaluator(Evaluator):

    def __init__(self, evaluators, constraints=None):
        '''
        Build an object to evaluate PDFs on independent data samples.
        This class is not meant to be used by users.

        :param evaluators: list of evaluators to use.
        :type evaluators: list(UnbinnedEvaluator or BinnedEvaluator)
        :param constraints: set of constraints to consider in the minimization.
        :type constraints: list(PDF) or None
        '''
        self.__evaluators = evaluators
        self.__constraints = constraints

        super(SimultaneousEvaluator, self).__init__()

    def __call__(self, *values):

        args = self.args
        sfcn = 0.
        for e in self.__evaluators:
            sfcn += e.__call__(*(values[args.index(a.name)] for a in e.args))

        return sfcn + fcns.evaluate_constraints(self.__constraints)

    @property
    def args(self):

        args = parameters.Registry(self.__evaluators[0].args)
        for e in self.__evaluators[1:]:
            args += e.args

        return args

    @property
    def data(self):
        '''
        Data sample.

        :type: DataSet or BinnedDataSet
        '''
        return self.__data

    @property
    def pdf(self):
        '''
        Probability density function.

        :type: PDF
        '''
        return self.__pdf

    def fcn(self):

        sfcn = 0.
        for e in self.__evaluators:
            sfcn += e.fcn()  # blinded parameters are processed in each call

        return sfcn + fcns.evaluate_constraints(self.__constraints)

    @contextlib.contextmanager
    def using_caches(self):

        with contextlib.ExitStack() as stack:

            for ev in self.__evaluators:
                stack.enter_context(ev.using_caches())

            yield self
