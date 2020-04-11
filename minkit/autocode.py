'''
Define the way to parse XML files defining PDFs.
'''
import xml.etree.ElementTree as ET
from . import core
from . import PACKAGE_PATH

import os

__all__ = []


def generate_code(xmlfile, backend):
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

    if backend == core.CPU:
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
    else:
        format_kwargs['has_variable_parameters'] = 'false'

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

    with open(os.path.join(PACKAGE_PATH, 'templates/function.c')) as f:
        format_kwargs['function'] = f.read().format(function_code=p.find('code').text,
                                                    function_arguments=', '.join([data_args, params_args]))

    # Check if the "integral" field has been filled
    p = root.find('integral')

    if p is not None:

        xml_bounds = p.find('bounds')

        bounds = ', '.join(f'double {v}' for _, v in xml_bounds.items())

        with open(os.path.join(PACKAGE_PATH, 'templates/integral.c')) as f:
            format_kwargs['integral'] = f.read().format(integral_code=p.find('code').text,
                                                        integral_arguments=', '.join((bounds, params_args)))
    else:
        format_kwargs['integral'] = ''

    with open(os.path.join(PACKAGE_PATH, 'templates/evaluators.c')) as f:
        format_kwargs['evaluators'] = f.read()

    # Prepare the template
    with open(os.path.join(PACKAGE_PATH, 'templates/whole.c')) as f:
        whole_template = f.read().format(**format_kwargs)

    return whole_template
