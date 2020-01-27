#!/ usr / bin / env python
'''
Setup script for the "minkit" package
'''

import os
import subprocess
import sys
from setuptools import Command, setup, find_packages


class CheckFormatCommand(Command):
    '''
    Check the format of the files in the given directory. This script takes only
    one argument, the directory to process. A recursive look-up will be done to
    look for python files in the sub-directories and determine whether the files
    have the correct format.
    '''
    description = 'check the format of the files of a certain type in a given directory'

    user_options = [
        ('directory=', 'd', 'directory to process'),
        ('file-type=', 't', 'file type (python|all)'),
    ]

    def initialize_options(self):
        '''
        Running at the begining of the configuration.
        '''
        self.directory = None
        self.file_type = None

    def finalize_options(self):
        '''
        Running at the end of the configuration.
        '''
        if self.directory is None:
            raise Exception('Parameter --directory is missing')
        if not os.path.isdir(self.directory):
            raise Exception('Not a directory {}'.format(self.directory))
        if self.file_type is None:
            raise Exception('Parameter --file-type is missing')
        if self.file_type not in ('python', 'all'):
            raise Exception('File type must be either "python" or "all"')

    def run(self):
        '''
        Execution of the command action.
        '''
        matched_files = []
        for root, _, files in os.walk(self.directory):
            for f in files:
                if self.file_type == 'python' and not f.endswith('.py'):
                    continue
                matched_files.append(os.path.join(root, f))

        process = subprocess.Popen(['autopep8', '--diff'] + matched_files,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()

        if process.returncode < 0:
            raise RuntimeError('Call to autopep8 exited with error {}\nMessage:\n{}'.format(
                abs(returncode), stderr))

        if len(stdout):
            raise RuntimeError(
                'Found differences for files in directory "{}" with file type "{}"'.format(self.directory, self.file_type))


# Setup function
setup(

    name='minkit',

    description='Package to perform fits in both CPUs and GPUs',

    cmdclass={'check_format': CheckFormatCommand},

    # Read the long description from the README
    long_description=open('README.rst').read(),

    # Keywords to search for the package
    keywords='hep high energy physics fit pdf probability',

    # Find all the packages in this directory
    packages=find_packages(),

    # Install data
    package_dir={'minkit': 'minkit'},
    package_data={'minkit': ['cpu/*.cpp', 'gpu/*.c', 'operations/src/*.c']},

    # Test requirements
    setup_requires=['numpy', 'iminuit', 'numdifftools',
                    'scipy', 'uncertainties', 'pytest-runner'],

    tests_require=['pytest'],
)
