#!/usr/bin/env python
'''
Apply the format to the files in the package.
'''
import argparse
import os
import shutil
import string
import subprocess
import tempfile

PWD = os.path.abspath(os.path.dirname(__file__))

C_FILES_WITH_FIELDS = ('whole.c', 'function.c', 'integral.c')


def files_with_extension(ext, where='minkit'):
    '''
    Return all the files with the given extension in the package.
    '''
    return [os.path.join(root, f) for root, _, files in os.walk(where) for f in filter(lambda s: s.endswith(ext), files)]


def main():

    with tempfile.TemporaryDirectory() as tmpdir:

        # Format the python files
        python_files = files_with_extension('.py') + files_with_extension('.py', where='test') + files_with_extension('.py', where='docs')
        python_proc = subprocess.Popen(['autopep8', '-i'] + python_files)

        # Format C files
        c_files = files_with_extension('.c') + files_with_extension('.h')

        new_c_files = [os.path.join(tmpdir, os.path.basename(f))
                       for f in c_files]

        files_fields = {}
        for of, nf in zip(c_files, new_c_files):

            shutil.copy(of, nf)

            with open(nf, 'rt') as fi:

                text = fi.read()

                if any(map(lambda s: nf.endswith(s), C_FILES_WITH_FIELDS)):
                    fields = [f for _, f, _,
                              _ in string.Formatter().parse(text) if f]
                else:
                    fields = None

            if fields:
                for f in fields:
                    text = text.replace(f'{{{f}}}', f'___{f}___;')

                text = text.replace('{{', '{').replace('}}', '}')

                with open(nf, 'wt') as of:
                    of.write(text)

                files_fields[nf] = fields

        c_proc = subprocess.Popen(['clang-format', '-i'] + new_c_files)

        for nf, fields in files_fields.items():

            with open(nf, 'rt') as f:
                text = f.read()

            text = text.replace('{', '{{').replace('}', '}}')

            for f in fields:
                text = text.replace(f'___{f}___;', f'{{{f}}}')

            with open(nf, 'wt') as f:
                f.write(text)

        for fi, fo in zip(new_c_files, c_files):
            shutil.copy(fi, fo)

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
