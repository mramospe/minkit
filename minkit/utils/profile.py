'''
Instances to manage profiles.
'''
from ..base import data_types
from ..base import parameters
from ..pdfs import dataset
from ..pdfs import pdf_core
from ..minimization import fcns

import contextlib
import logging
import numpy as np


__all__ = ['fcn_profile', 'simultaneous_fcn_profile']


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def address_constness(args, profile_pars):
    '''
    Create a new context where all the variables in *args* not contained in
    *profile_pars* are marked as constant. The state (constness and value) of
    the parameters is recovered when exiting the context.

    :param args: whole set of arguments.
    :type args: Registry
    :param profile_pars: parameters to calculate a profile.
    :type profile_pars: Registry
    '''
    # Save the state of the parameters (constant, value) and mark the rest of
    # the parameters as constant
    pars_state = {p.name: (p.constant, p.value) for p in args}

    for p in filter(lambda p: p in args, profile_pars):
        p.constant = False  # so we manage the cache correctly

    for p in filter(lambda p: p not in profile_pars, args):
        p.constant = True

    yield  # do whatever operation here

    for n, (c, v) in pars_state.items():
        p = args.get(n)
        p.constant = c
        p.value = v


def process_parameters(args, wa, values):
    '''
    Process an input set of parameters, converting it to a :class:`Registry`,
    if only one parameter is provided.

    :param args: whole set of parameters.
    :type args: Registry
    :param wa: input arguments to process.
    :type wa: str or list(str)
    :param values: values to calculate a profile.
    :type values: numpy.ndarray
    :returns: registry of parameters and values.
    :rtype: Registry, numpy.ndarray
    '''
    values = np.asarray(values)

    if np.asarray(wa).ndim == 0:
        wa = [wa]
        values = np.array([values])

    if len(wa) != len(values):
        raise RuntimeError(
            'Length of the profile values must coincide with profile parameters')

    pars = parameters.Registry(args.get(n) for n in wa)

    return pars, values


def fcn_profile(fcn, pdf, data, wa, values, range=parameters.FULL, constraints=None):
    '''
    Evaluate the profile of an FCN for a set of parameters and values.

    :param fcn: FCN to use.
    :type fcn: str
    :param pdf: PDF to evaluate.
    :type pdf: PDF
    :param data: data set.
    :type data: DataSet or BinnedDataSet
    :param wa: single variable or set of variables.
    :type wa: str or list(str).
    :param values: values for each parameter specified in *wa*.
    :type values: numpy.ndarray
    :param range: normalization range for the unbinned case.
    :type range: str
    :param constraints: set of constraints to consider in the minimization.
    :type constraints: list(PDF)
    :returns: Profile of the FCN for the given values.
    :rtype: numpy.ndarray
    '''
    args = pdf.all_real_args

    profile_pars, values = process_parameters(args, wa, values)

    # Determine the FCN to use and process its options
    if fcns.data_type_for_fcn(fcn) == dataset.UNBINNED:
        opts = dict(range=range)

    fcn_function = fcns.fcn_from_name(fcn)

    fcn_values = data_types.empty_float(len(values[0]))

    with address_constness(args, profile_pars), pdf.using_cache(pdf_core.PDF.CONST):
        for i, vals in enumerate(values.T):
            for p, v in zip(profile_pars, vals):
                p.value = v
            fcn_values[i] = fcn_function(
                pdf, data, **opts) + fcns.evaluate_constraints(constraints)

    return fcn_values


def simultaneous_fcn_profile(categories, wa, values, range=parameters.FULL, constraints=None):
    '''
    Evaluate the profile of a simultaneous FCN for a set of parameters and values.

    :param categories: categories to use.
    :type categories: list(Category)
    :param wa: single variable or set of variables.
    :type wa: str or list(str).
    :param values: values for each parameter specified in *wa*.
    :type values: numpy.ndarray
    :param range: normalization range for the unbinned case.
    :type range: str
    :param constraints: set of constraints to consider in the minimization.
    :type constraints: list(PDF)
    :returns: Profile of the FCN for the given values.
    :rtype: numpy.ndarray
    '''
    # Get all FCNs and all arguments to the PDFs
    all_args = parameters.Registry()
    all_fcns = []
    for cat in categories:

        fcn_function = fcns.fcn_from_name(cat.fcn)

        all_args += cat.pdf.all_real_args

        if fcns.data_type_for_fcn(cat.fcn) == dataset.UNBINNED:
            all_fcns.append(lambda: fcn_function(
                cat.pdf, cat.data, range))
        else:
            all_fcns.append(lambda: fcn_function(
                cat.pdf, cat.data))

    # Compute the profile
    profile_pars, values = process_parameters(all_args, wa, values)

    fcn_values = data_types.empty_float(values.shape[1])
    with contextlib.ExitStack() as stack:

        # Process constness
        stack.enter_context(address_constness(all_args, profile_pars))
        for cat in categories:
            stack.enter_context(cat.pdf.using_cache(pdf_core.PDF.CONST))

        # Evaluate the FCN for different values of the parameters
        for i, vals in enumerate(values):
            for p, v in zip(profile_pars, vals):
                p.value = v

            fcn_values[i] = np.sum(
                data_types.fromiter_float((f() for f in all_fcns))) + fcns.evaluate_constraints(constraints)

    return fcn_values
