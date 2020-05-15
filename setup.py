'''
Setup script for the "minkit" package
'''

import os
import subprocess
from setuptools import Command, setup, find_packages

PWD = os.path.abspath(os.path.dirname(__file__))


class DirectoryChecker(Command):

    user_options = [
        ('directory=', 'd', 'directory to process'),
    ]

    def _check_process(self, name, process, compare=None):
        '''
        Check the return code of a process. If "compare" is given, it is used to
        compare the output with it. By default it just checks that there is no output.
        '''
        stdout, stderr = process.communicate()

        if process.returncode < 0:
            raise RuntimeError(
                f'Call to {name} exited with error {abs(process.returncode)}; the message is:\n{stderr}')

        if compare is None:
            if len(stdout):
                raise RuntimeError(
                    f'Found problems for files in directory "{self.directory}"')
        else:
            if stdout.decode('ascii') != compare:
                raise RuntimeError(
                    f'Found problems for files in directory "{self.directory}"')

    def initialize_options(self):
        '''
        Running at the begining of the configuration.
        '''
        self.directory = None

    def finalize_options(self):
        '''
        Running at the end of the configuration.
        '''
        if self.directory is None:
            raise Exception('Parameter --directory is missing')
        if not os.path.isdir(self.directory):
            raise Exception('Not a directory {}'.format(self.directory))


class CheckFormatCommand(DirectoryChecker):

    description = 'check the format of the files of a certain type in a given directory'

    def run(self):
        '''
        Execution of the command action.
        '''
        python_files, c_files = [], []
        for root, _, files in os.walk(self.directory):
            for f in filter(lambda f: f.endswith('.py'), files):
                python_files.append(os.path.join(root, f))
            for f in filter(lambda f: any(map(f.endswith, ('.h', '.c', '.xml'))), files):
                c_files.append(os.path.join(root, f))

        # Check python files
        process = subprocess.Popen(['autopep8', '--diff'] + python_files,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        self._check_process('autopep8', process)

        # Check the C files
        for fl in c_files:
            process = subprocess.Popen(['clang-format', fl],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            with open(fl) as f:
                self._check_process('clang-format', process, compare=f.read())


class CheckPyFlakesCommand(DirectoryChecker):

    description = 'run pyflakes in order to detect unused objects and errors in the code'

    def run(self):
        '''
        Execution of the command action.
        '''
        python_files = []
        for root, _, files in os.walk(self.directory):
            for f in filter(lambda f: f.endswith('.py'), files):
                python_files.append(os.path.join(root, f))

        process = subprocess.Popen(['pyflakes'] + python_files,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        self._check_process('pyflakes', process)


# Determine the source files
src_path = os.path.join(PWD, 'minkit', 'backends', 'src')
rel_path = os.path.join('backends', 'src')

data_files = [os.path.join(rel_path, d, f) for d in ('gpu', 'templates', 'xml') for f in os.listdir(
    os.path.join(src_path, d))]

# Setup function
setup(

    name='minkit',

    description='Package to perform fits in both CPUs and GPUs',

    cmdclass={'check_format': CheckFormatCommand,
              'check_pyflakes': CheckPyFlakesCommand},

    # Read the long description from the README
    long_description=open('README.rst').read(),

    # Keywords to search for the package
    keywords='hep high energy physics fit pdf probability',

    # Find all the packages in this directory
    packages=find_packages(),

    # Install data
    package_dir={'minkit': 'minkit'},
    package_data={'minkit': data_files},

    # Install requirements
    install_requires=['iminuit', 'numpy', 'numdifftools',
                      'scipy', 'uncertainties', 'pytest-runner'],

    tests_require=['pytest'],
)
