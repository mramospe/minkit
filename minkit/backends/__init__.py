########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
In this module the exposed objects are defined manually to avoid loading
GPU-specific modules.
'''
import importlib
import inspect
import os

PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))

exposed_modules = ('core', 'aop', 'arrays')

pkg = os.path.normpath(
    PACKAGE_PATH[PACKAGE_PATH.rfind('minkit'):]).replace(os.sep, '.')

__all__ = []
for m in exposed_modules:

    mod = importlib.import_module('.' + m, package=pkg)

    for n, c in inspect.getmembers(mod):
        if n in mod.__all__:
            globals()[n] = c

    __all__ += mod.__all__


__all__ = list(sorted(__all__))
