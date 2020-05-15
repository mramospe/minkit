#!/usr/bin/env python
'''
Apply the format to the files in the package.
'''
import argparse
import os
import subprocess

PWD = os.path.abspath(os.path.dirname(__file__))


def files_with_extension(ext, where='minkit'):
    '''
    Return all the files with the given extension in the package.
    '''
    return [os.path.join(root, f) for root, _, files in os.walk(where) for f in filter(lambda s: s.endswith(ext), files)]


def main():

    # Format the python files
    python_files = files_with_extension('.py') + files_with_extension(
        '.py', where='test') + files_with_extension('.py', where='docs')
    python_proc = subprocess.Popen(['autopep8', '-i'] + python_files)

    # Format C files
    c_files = files_with_extension('.c') + files_with_extension('.h')

    c_proc = subprocess.Popen(['clang-format', '-i'] + c_files)

    # Format the XML files
    xml_files = files_with_extension('.xml')
    xml_proc = subprocess.Popen(['clang-format', '-i'] + xml_files)

    # Wait for the processes to finish
    if python_proc.wait() != 0:
        raise RuntimeError('Problems found while formatting C files')

    if c_proc.wait() != 0:
        raise RuntimeError('Problems found while formatting python files')

    if xml_proc.wait() != 0:
        raise RuntimeError('Problems found while formatting C files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main(**vars(args))
