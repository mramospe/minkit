# Python
import importlib
import inspect
import os
import pkgutil

# Keep the path to this package
PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))


__all__ = []
for loader, module_name, ispkg in pkgutil.walk_packages(__path__):

    if module_name.endswith('setup') or module_name.endswith('__') or ispkg:
        continue

    if 'operations' in module_name:
        # Skip, since we might end up loading PyCUDA
        continue

    # Import all classes and functions
    mod = importlib.import_module('.' + module_name, package='minkit')

    __all__ += mod.__all__

    for n, c in inspect.getmembers(mod):
        if n in mod.__all__:
            globals()[n] = c


__all__ = list(sorted(__all__))
