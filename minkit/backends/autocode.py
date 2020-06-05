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

import os
import xml.etree.ElementTree as ET

TEMPLATES_PATH = os.path.join(PACKAGE_PATH, 'src', 'templates')

# Keep the code to compile in several cache strings
with open(os.path.join(TEMPLATES_PATH, 'function.c')) as f:
    FUNCTION_CACHE = Template(f.read())

with open(os.path.join(TEMPLATES_PATH, 'integral.c')) as f:
    INTEGRAL_CACHE = Template(f.read())

with open(os.path.join(TEMPLATES_PATH, 'evaluators.c')) as f:
    EVALUATORS_CACHE = f.read()  # this is a plain string

with open(os.path.join(TEMPLATES_PATH, 'numerical_integral.c')) as f:
    NUMERICAL_INTEGRAL_CACHE = f.read()  # this is a plain string

with open(os.path.join(TEMPLATES_PATH, 'whole.c')) as f:
    WHOLE_CACHE = Template(f.read())


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

    tags = [c.tag for c in root.getchildren()]

    if 'function' not in tags:
        raise RuntimeError('Expected field "function"')

    # Parse the parameters
    c = root.find('parameters')
    if c is not None:
        params = list(f'double {v}' for _, v in c.items())
    else:
        params = []

    format_kwargs['number_of_parameters'] = len(params)

    c = root.find('variable_parameters')
    if c is not None:
        n, p = tuple(v for _, v in c.items())
        params += [f'int {n}', f'{double_ptr}{p}']
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

    d = p.find('data')

    format_kwargs['ndimensions'] = len(d.items())

    data_args = ', '.join(f'double {v}' for _, v in d.items())

    format_kwargs['function'] = FUNCTION_CACHE.substitute(function_code=p.find('code').text,
                                                          function_arguments=', '.join([data_args, params_args]))

    # Check if the "integral" field has been filled
    p = root.find('integral')

    if p is not None:

        xml_bounds = p.find('bounds')

        bounds = ', '.join(f'double {v}' for _, v in xml_bounds.items())

        format_kwargs['integral'] = INTEGRAL_CACHE.substitute(integral_code=p.find('code').text,
                                                              integral_arguments=', '.join((bounds, params_args)))
    else:
        format_kwargs['integral'] = ''

    format_kwargs['evaluators'] = EVALUATORS_CACHE

    format_kwargs['numerical_integral'] = NUMERICAL_INTEGRAL_CACHE

    # Prepare the template
    whole_template = WHOLE_CACHE.substitute(**format_kwargs)

    return whole_template
