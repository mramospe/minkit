########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
from .core import get_exposed_package_objects

import os

PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))

dct = get_exposed_package_objects(PACKAGE_PATH)

globals().update(dct)

__all__ = list(sorted(dct.keys()))
