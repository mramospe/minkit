########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Definition of the backend where to store the data and run the jobs.
'''
import contextlib
import importlib
import inspect
import logging
import math
import os
import pkgutil
import time

__all__ = ['timer']


logger = logging.getLogger(__name__)


class DocMeta(type):

    def __init__(cls, name, bases, namespace):
        '''
        Metaclass that allows to inherit the docstring from the first base
        with an available docstring for that method.
        '''
        super().__init__(name, bases, namespace)

        # Iterate over the instances in the class
        for attr, obj in namespace.items():

            if obj.__doc__:
                continue  # skip existing methods with docstrings

            for base in filter(lambda b: hasattr(b, attr), bases):

                f = getattr(base, attr)

                if f.__doc__:
                    obj.__doc__ = f.__doc__
                    break  # we set that from the first we find


def get_exposed_package_objects(path):
    '''
    Process a given path, taking all the exposed objects in it and returning
    a dictionary with their names and respective pointers.

    :param path: path to the package.
    :type path: str
    :returns: Names and objects that are exposed.
    :rtype: dict(str, object)
    '''
    pkg = os.path.normpath(path[path.rfind('minkit'):]).replace(os.sep, '.')

    dct = {}

    for loader, module_name, ispkg in pkgutil.walk_packages([path]):

        if module_name.endswith('setup') or module_name.endswith('__'):
            continue

        # Import all classes and functions
        mod = importlib.import_module('.' + module_name, package=pkg)

        for n, c in inspect.getmembers(mod):
            if n in mod.__all__:
                dct[n] = c

    return dct


# Allowed mathematical objects
MATH_OBJECTS = {n: f for n, f in inspect.getmembers(math, inspect.isbuiltin)}
MATH_OBJECTS['pi'] = math.pi
MATH_OBJECTS['max'] = max
MATH_OBJECTS['min'] = min


def eval_math_expression(expression):
    '''
    Evaluate a mathematical expression on the safest manner possible. Only
    functions defined in the "math" module are allowed.

    :param expression: input expression.
    :type expression: str
    :returns: evaluation of the expression.
    '''
    code = compile(expression, '', 'eval')

    for name in code.co_names:
        if name not in MATH_OBJECTS:
            raise NameError(
                f'Use of {name} not allowed in a mathematical expression; functions and constants allowed: {sorted(MATH_OBJECTS.keys())}')

    return eval(code, {'__builtins__': {}}, MATH_OBJECTS)


@contextlib.contextmanager
def timer():
    '''
    Create an object that, on exit, displays the time elapsed.
    '''
    start = time.time()
    yield
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)

    logger.info(f'Time elapsed: {hours}h {minutes}m {seconds}s')
