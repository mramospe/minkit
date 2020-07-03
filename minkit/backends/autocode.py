########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Define the way to parse XML files defining PDFs.
'''
from . import PACKAGE_PATH
from .core import CPU

from string import Template

import logging
import os
import xml.etree.ElementTree as ET

TEMPLATES_PATH = os.path.join(PACKAGE_PATH, 'src', 'templates')

# Keep the code to compile in several cache strings
with open(os.path.join(TEMPLATES_PATH, 'function.c')) as f:
    FUNCTION_CACHE = Template(f.read())

with open(os.path.join(TEMPLATES_PATH, 'integral.c')) as f:
    INTEGRAL_CACHE = Template(f.read())

with open(os.path.join(TEMPLATES_PATH, 'primitive.c')) as f:
    PRIMITIVE_CACHE = Template(f.read())

with open(os.path.join(TEMPLATES_PATH, 'evaluators.c')) as f:
    EVALUATORS_CACHE = f.read()  # this is a plain string

with open(os.path.join(TEMPLATES_PATH, 'numerical_integral.c')) as f:
    NUMERICAL_INTEGRAL_CACHE = f.read()  # this is a plain string

with open(os.path.join(TEMPLATES_PATH, 'whole.c')) as f:
    WHOLE_CACHE = Template(f.read())


logger = logging.Logger(__name__)


def generate_code(xmlfile, backend, nvar_arg_pars):
    '''
    Generate the source code needed to compile a PDF.

    :param xmlfile: path to an XML file.
    :type xmlfile: str
    :param backend: backend where the code will be compiled.
    :type backend: str
    :returns: code for either CPU or GPU.
    :rtype: str
    '''
    root = ET.parse(xmlfile).getroot()

    format_kwargs = {}

    if backend == CPU:
        double_ptr = 'double *'
        format_kwargs['backend'] = +10  # Must be greater than zero for CPU
    else:
        # Imposed by reikna
        double_ptr = 'GLOBAL_MEM double *'
        format_kwargs['backend'] = -10  # Must be smaller than zero for GPU

    # Parse the parameters
    c = root.get('parameters')
    if c is not None:
        params_arg_names = list(c.split())
        params = list(f'double {v}' for v in params_arg_names)
    else:
        params = []
        params_arg_names = []

    format_kwargs['number_of_parameters'] = len(params)

    c = root.get('variable_parameters')
    if c is not None:
        n, p = tuple(c.split())
        params += [f'int {n}', f'{double_ptr}{p}']
        params_arg_names += [n, p]
        format_kwargs['has_variable_parameters'] = 'true'
        format_kwargs['nvar_arg_pars'] = nvar_arg_pars
    else:
        format_kwargs['has_variable_parameters'] = 'false'
        format_kwargs['nvar_arg_pars'] = ''

    params_args = ', '.join(params)

    # Determine whether a preamble is needed
    p = root.find('preamble')
    if p is not None:
        format_kwargs['preamble_code'] = p.text or ''
    else:
        format_kwargs['preamble_code'] = ''

    # Process the function
    p = root.find('function')

    data_arg_names = root.get('data').split()

    format_kwargs['ndimensions'] = len(data_arg_names)

    data_args = ', '.join(f'double {v}' for v in data_arg_names)

    format_kwargs['function'] = FUNCTION_CACHE.substitute(function_code=p.text,
                                                          function_arguments=', '.join([data_args, params_args]))

    # Check if the "integral" field has been filled
    pi = root.find('integral')
    pp = root.find('primitive')

    if pi is not None or pp is not None:

        if pi is not None and pp is not None:  # both have been defined
            logger.info(
                'Specified integral and primitive for the same PDF, using the former')

        if pi is not None:  # use the integral

            xml_bounds = pi.get('bounds').split()

            bounds = ', '.join(f'double {v}' for v in xml_bounds)

            format_kwargs['integral'] = INTEGRAL_CACHE.substitute(integral_code=pi.text,
                                                                  integral_arguments=', '.join((bounds, params_args)))

        else:  # use the primitive

            bounds = ', '.join(f'double {p}_{bd}' for bd in (
                'min', 'max') for p in data_arg_names)

            famin = ', '.join(
                s for s in [f'{p}_min' for p in data_arg_names] + params_arg_names)
            famax = ', '.join(
                s for s in [f'{p}_max' for p in data_arg_names] + params_arg_names)

            format_kwargs['integral'] = PRIMITIVE_CACHE.substitute(primitive_arguments=', '.join((data_args, params_args)),
                                                                   primitive_code=pp.text,
                                                                   integral_arguments=', '.join(
                                                                       (bounds, params_args)),
                                                                   primitive_fwd_args_min=famin,
                                                                   primitive_fwd_args_max=famax)
    else:
        format_kwargs['integral'] = ''

    format_kwargs['evaluators'] = EVALUATORS_CACHE

    format_kwargs['numerical_integral'] = NUMERICAL_INTEGRAL_CACHE

    # Prepare the template
    whole_template = WHOLE_CACHE.substitute(**format_kwargs)

    return whole_template


def xml_from_formula(formula, data_pars, arg_pars, primitive=None):
    '''
    Generate XML code using the given formula as a function.

    :param formula: formula associated to the PDF.
    :type formula: str
    :param data_pars: data parameters.
    :type data_pars: Registry(Parameter)
    :param arg_pars: argument parameters.
    :type arg_pars: Registry(Parameter)
    :param primitive: if provided, use this formula in order to calculate
       the primitive.
    :type primitive: str or None
    '''
    data = ' '.join(p.name for p in data_pars)
    parameters = ' '.join(p.name for p in arg_pars)

    top = ET.Element('PDF', {'data': data, 'parameters': parameters})

    if os.linesep in formula:
        raise ValueError('Line breaks are not allowed inside formulas')

    function = ET.SubElement(top, 'function')
    function.text = f'return {formula};'

    if primitive is not None:

        if os.linesep in primitive:
            raise ValueError('Line breaks are not allowed inside formulas')

        primitive_element = ET.SubElement(top, 'primitive')
        primitive_element.text = f'return {primitive};'

    return ET.tostring(top, encoding='unicode')
