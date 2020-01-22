'''
Hold the version of the package.
'''

__all__ = ['__version__']

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution('minkit').version
except Exception:
    __version__ = 'unknown'
