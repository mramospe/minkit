########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
API for minimizers.
'''
from . import evaluators
from . import fcns
from . import minuit_api
from . import scipy_api
from ..base import parameters
from ..pdfs import dataset

import contextlib
import logging

__all__ = ['minimizer', 'simultaneous_minimizer']

logger = logging.getLogger(__name__)

# Names for the minimizers
MINUIT = 'minuit'
SCIPY = 'scipy'


@contextlib.contextmanager
def minimizer(fcn, pdf, data, minimizer=MINUIT, minimizer_config=None, constraints=None, range=parameters.FULL, weights_treatment=evaluators.WGTS_RESCALE):
    r'''
    Create a new minimizer to be used within a context.
    The bounds of the data parameters must not change in the process.
    Changes to the values, errors or the *constant* state of the parameters
    must be done through the :meth:`Minimizer.set_parameter_state` method.

    :param fcn: type of FCN to use for the minimization.
    :type fcn: str
    :param pdf: function to minimize.
    :type pdf: PDF
    :param data: data set to use.
    :type data: UnbinnedDataSet or BinnedDataSet
    :param minimizer: minimizer to use (*minuit* or *scipy*).
    :type minimizer: str
    :param minimizer_config: any extra configuration to be passed to the minimizer. For *minuit*, the argument *forced_parameters* is unavailable.
    :type minimizer_config: dict
    :param constraints: set of constraints to consider in the minimization.
    :type constraints: list(PDF)
    :param range: range of data to minimize (unbinned case only).
    :type range: str
    :param weights_treatment: what to do with weighted samples (see below for more information).
    :type weights_treatment: str
    :returns: Minimizer to call.
    :rtype: Minimizer

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

    .. warning::
       Do not change the bounds of the data parameters, since for unbinned data
       sets a new sample is internally created with values falling in *range*.
       The weights of the sample are also changed according to the
       *weights_treatment* argument.

       It is also very important that any change to the state of a parameter
       within a minimization context is done using the
       :meth:`Minimizer.set_parameter_state` method.
    '''
    mf = fcns.fcn_from_name(fcn)

    if fcns.data_type_for_fcn(fcn) == dataset.BINNED:
        evaluator = evaluators.BinnedEvaluator(mf, pdf, data, constraints)
    else:
        evaluator = evaluators.UnbinnedEvaluator(
            mf, pdf, data, range, constraints, weights_treatment)

    minimizer_config = minimizer_config or {}

    if minimizer == MINUIT:
        yield minuit_api.MinuitMinimizer(evaluator, **minimizer_config)
    elif minimizer == SCIPY:
        yield scipy_api.SciPyMinimizer(evaluator, **minimizer_config)
    else:
        raise ValueError(f'Unknown minimizer "{minimizer}"')


@contextlib.contextmanager
def simultaneous_minimizer(categories, minimizer=MINUIT, minimizer_config=None, constraints=None, range=parameters.FULL, weights_treatment=evaluators.WGTS_RESCALE):
    r'''
    Create a new object to minimizer a :class:`PDF`.
    The bounds of the data parameters must not change in the process.
    Changes to the values, errors or the *constant* state of the parameters
    must be done through the :meth:`Minimizer.set_parameter_state` method.

    :param categories: categories to process.
    :type categories: list(Category)
    :param minimizer: minimizer to use (*minuit* or *scipy*).
    :type minimizer: str
    :param minimizer_config: any extra configuration to be passed to the minimizer. For *minuit*, the argument *forced_parameters* is unavailable.
    :type minimizer_config: dict
    :param constraints: set of constraints to consider in the minimization.
    :type constraints: list(PDF)
    :param range: range of data to minimize.
    :type range: str
    :param weights_treatment: what to do with weighted samples (see below for more information).
    :type weights_treatment: str
    :returns: Minimizer to call.
    :rtype: Minimizer

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

    .. warning::
       Do not change the bounds of the data parameters, since for unbinned data
       sets a new sample is internally created with values falling in *range*.
       The weights of the sample are also changed according to the *weights_treatment*
       argument.

       It is also very important that any change to the state of a parameter
       within a minimization context is done using the
       :meth:`Minimizer.set_parameter_state` method.
    '''
    # Build the simultaneous evaluator
    evals = []
    for cat in categories:

        fcn = fcns.fcn_from_name(cat.fcn)

        if fcns.data_type_for_fcn(cat.fcn) == dataset.BINNED:
            e = evaluators.BinnedEvaluator(
                fcn, cat.pdf, cat.data)
        else:
            e = evaluators.UnbinnedEvaluator(
                fcn, cat.pdf, cat.data, range, weights_treatment=weights_treatment)

        evals.append(e)

    evaluator = evaluators.SimultaneousEvaluator(evals, constraints)

    minimizer_config = minimizer_config or {}

    # Return the minimizer
    if minimizer == MINUIT:
        yield minuit_api.MinuitMinimizer(evaluator, **minimizer_config)
    elif minimizer == SCIPY:
        yield scipy_api.SciPyMinimizer(evaluator, **minimizer_config)
    else:
        raise ValueError(f'Unknown minimizer "{minimizer}"')
