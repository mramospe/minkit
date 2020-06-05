'''
Setup script for the "minkit" package
'''

import os
import re
import subprocess
import warnings

from setuptools import Command, setup, find_packages

PWD = os.path.abspath(os.path.dirname(__file__))


def check_format_process(name, directory, process, compare=None):
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
                f'Found problems for files in directory "{directory}"')
    else:
        if stdout.decode('ascii') != compare:
            raise RuntimeError(
                f'Found problems for files in directory "{directory}"')


def files_with_extension(ext, where):
    '''
    Return all the files with the given extension in the package.
    '''
    return [os.path.join(root, f) for root, _, files in os.walk(where) for f in filter(lambda s: s.endswith(ext), files)]


def license_for_language(language):
    '''
    Create the license string for the given language.
    '''
    with open(os.path.join(PWD, 'LICENSE.txt')) as f:
        lines = f.read().split('\n')[:3] # take the first paragraphs

    if language == 'python':

        ml = max(map(len, lines)) + 2 # for the extra # and the whitespace

        text = '\n'.join(f'# {l}' if l else '#' for l in lines)

        return ml * '#' + f'\n{text}\n' + ml * '#' + '\n'

    elif language == 'c':

        ml = max(map(len, lines)) + 1

        text = '\n'.join(f' * {l}' if l else ' *' for l in lines)

        return '/*' + ml * '*' + f'\n{text}\n ' + ml * '*' + '*/\n'

    elif language == 'xml':

        text = '\n'.join(f' {l}' if l else '' for l in lines)

        return f'<!--\n{text}\n-->\n'
    else:
        raise ValueError(f'Unknown programming language "{language}"')


class DirectoryWorker(Command):

    user_options = [
        ('regex=', 'r', 'regular expression defining the directories to process'),
    ]

    def initialize_options(self):
        '''
        Running at the begining of the configuration.
        '''
        self.regex = None
        self.directories = None

    def finalize_options(self):
        '''
        Running at the end of the configuration.
        '''
        if self.regex is None:
            raise Exception('Parameter --regex is missing')

        m = re.compile(self.regex)

        self.directories = list(os.path.join(PWD, d) for d in filter(
            lambda s: os.path.isdir(s) and (m.match(s) is not None), os.listdir(PWD)))

        if len(self.directories) == 0:
            warnings.warn('Empty list of directories', RuntimeWarning)


class ApplyFormatCommand(DirectoryWorker):

    description = 'apply the format to the files in the given directories'

    def run(self):
        '''
        Execution of the command action.
        '''
        for directory in self.directories:

            # Format the python files
            python_files = files_with_extension('.py', directory)
            python_proc = None if not python_files else subprocess.Popen(['autopep8', '-i'] + python_files)

            # Format C files
            c_files = files_with_extension('.c', directory) + files_with_extension('.h', directory)
            c_proc = None if not c_files else subprocess.Popen(['clang-format', '-i'] + c_files)

            # Format the XML files
            xml_files = files_with_extension('.xml', directory)
            xml_proc = None if not xml_files else subprocess.Popen(['clang-format', '-i'] + xml_files)

            killall = lambda: python_proc.kill() and c_proc.kill() and xml_proc.kill()

            # Wait for the processes to finish
            if python_proc is not None and python_proc.wait() != 0:
                killall()
                raise RuntimeError('Problems found while formatting python files')

            if c_proc is not None and c_proc.wait() != 0:
                killall()
                raise RuntimeError('Problems found while formatting C files')

            if xml_proc is not None and xml_proc.wait() != 0:
                killall()
                raise RuntimeError('Problems found while formatting XML files')


class ApplyCopyrightCommand(DirectoryWorker):

    description = 'add the copyright to the files in the directories'

    def _check_add_copyright(self, path, preamble):
        '''
        Check that the given file has the provided copyright and add it in case
        it is not present.
        '''
        with open(path) as f:
            data = f.read()

        if not preamble in data:

            if data.startswith('#!'): # is a script
                lines = data.splitlines(keepends=True)
                text = f'{lines[0]}{preamble}{"".join(lines[1:])}'
            else:
                text = f'{preamble}{data}'

            with open(path, 'wt') as f:
                f.write(text)

    def run(self):
        '''
        Execution of the command action.
        '''
        for directory in self.directories:
            python_files = files_with_extension('.py', directory)
            c_files = files_with_extension('.c', directory) + files_with_extension('.h', directory)
            xml_files = files_with_extension('.xml', directory)

            lic = license_for_language('python')
            for pf in python_files:
                self._check_add_copyright(pf, lic)

            lic = license_for_language('c')
            for cf in c_files:
                self._check_add_copyright(cf, lic)

            lic = license_for_language('xml')
            for xf in xml_files:
                self._check_add_copyright(xf, lic)


class CheckFormatCommand(DirectoryWorker):

    description = 'check the format of the files of a certain type in a given directory'

    def run(self):
        '''
        Execution of the command action.
        '''
        for directory in self.directories:
            python_files = files_with_extension('.py', directory)
            c_files = files_with_extension('.c', directory) + files_with_extension('.h', directory) + files_with_extension('.xml', directory)

            # Check python files
            process = subprocess.Popen(['autopep8', '--diff'] + python_files,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

            check_format_process('autopep8', directory, process)

            # Check the C files
            for fl in c_files:
                process = subprocess.Popen(['clang-format', fl],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
                with open(fl) as f:
                    check_format_process(
                        'clang-format', directory, process, compare=f.read())


class CheckPyFlakesCommand(DirectoryWorker):

    description = 'run pyflakes in order to detect unused objects and errors in the code'

    def run(self):
        '''
        Execution of the command action.
        '''
        for directory in self.directories:
            python_files = files_with_extension('.py', directory)

            process = subprocess.Popen(['pyflakes'] + python_files,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

            check_format_process('pyflakes', directory, process)


class CheckCopyrightCommand(DirectoryWorker):

    description = 'check that the copyright is present in the directory files'

    def _check_copyright(self, path, preamble):
        '''
        Check that the given file has the provided copyright.
        '''
        with open(path) as f:
            text = f.read()
            if not text.startswith(preamble):
                # it might be an executable (skip the first line)
                t = ''.join(text.splitlines(keepends=True)[1:])
                if not t.startswith(preamble):
                    raise RuntimeError(f'Copyright not present in file "{path}"')

    def run(self):
        '''
        Execution of the command action.
        '''
        for directory in self.directories:

            python_files = files_with_extension('.py', directory)
            c_files = files_with_extension('.c', directory) + files_with_extension('.h', directory)
            xml_files = files_with_extension('.xml', directory)

            lic = license_for_language('python')
            for pf in python_files:
                self._check_copyright(pf, lic)

            lic = license_for_language('c')
            for cf in c_files:
                self._check_copyright(cf, lic)

            lic = license_for_language('xml')
            for xf in xml_files:
                self._check_copyright(xf, lic)


class RemoveCopyrightCommand(DirectoryWorker):

    description = 'remove the copyright from the files in the given directories'

    def _remove_copyright(self, path, preamble):
        '''
        Remove the copyright if present in the given file.
        '''
        with open(path) as f:
            text = f.read()

        if preamble in text:
            with open(path, 'wt') as f:
                f.write(text.replace(preamble, '', 1))

    def run(self):
        '''
        Execution of the command action.
        '''
        for directory in self.directories:

            python_files = files_with_extension('.py', directory)
            c_files = files_with_extension('.c', directory) + files_with_extension('.h', directory)
            xml_files = files_with_extension('.xml', directory)

            lic = license_for_language('python')
            for pf in python_files:
                self._remove_copyright(pf, lic)

            lic = license_for_language('c')
            for cf in c_files:
                self._remove_copyright(cf, lic)

            lic = license_for_language('xml')
            for xf in xml_files:
                self._remove_copyright(xf, lic)


# Determine the source files
src_path = os.path.join(PWD, 'minkit', 'backends', 'src')
rel_path = os.path.join('backends', 'src')

data_files = [os.path.join(rel_path, d, f) for d in ('gpu', 'templates', 'xml') for f in os.listdir(
    os.path.join(src_path, d))]

# Setup function
setup(

    name='minkit',

    description='Package to perform fits in both CPUs and GPUs',

    cmdclass={'apply_copyright': ApplyCopyrightCommand,
              'apply_format': ApplyFormatCommand,
              'check_copyright': CheckCopyrightCommand,
              'check_format': CheckFormatCommand,
              'check_pyflakes': CheckPyFlakesCommand,
              'remove_copyright': RemoveCopyrightCommand},

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
    install_requires=['iminuit>=1.3', 'numpy>=1.17', 'numdifftools>=0.9.39',
                      'scipy>=1.3.2', 'uncertainties>=3.1.2'],

    tests_require=['pytest', 'pytest-runner'],
)
