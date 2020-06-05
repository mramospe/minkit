#!/usr/bin/env python
########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Generate the plots about the performance.
'''
import argparse
import cycler
import json
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

number_of_events = [1000, 10000, 100000, 1000000, 10000000]

logger = logging.getLogger(__name__)

# Format for the plots
matplotlib.rcParams['figure.figsize'] = (10, 8)
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['legend.fontsize'] = 15
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['lines.markersize'] = 8
matplotlib.rcParams['patch.linewidth'] = 2
for t in ('xtick', 'ytick'):
    matplotlib.rcParams[f'{t}.major.width'] = 1.5
    matplotlib.rcParams[f'{t}.minor.width'] = 1.0
    matplotlib.rcParams[f'{t}.major.size'] = 10
    matplotlib.rcParams[f'{t}.minor.size'] = 5


def run(jobtype, model, repetitions, directory, backends, extra_args):
    '''
    Main function to execute.
    '''
    # Run with minkit
    for bk in {'cpu', 'opencl', 'cuda'}.intersection(backends):

        logger.info(f'Processing for backend "{bk}" and model "{model}"')

        ofile = os.path.join(directory, f'{bk}.txt')
        with open(ofile, 'wt'):
            pass

        for nevts in number_of_events:

            logger.info(f'- {nevts}')

            cflags = 'CFLAGS="-shared -fPIC -std=c++0x"'
            p = subprocess.Popen(f'{cflags} python minkit_script.py {jobtype} {model} {nevts} {repetitions} {ofile} --backend {bk}',
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            _, stderr = p.communicate()
            if p.poll() != 0:
                stderr = stderr.decode()
                raise RuntimeError(f'Job failed with errors:\n{stderr}')

    # Run with RooFit
    for bk in {'roofit'}.intersection(backends):

        cfg = extra_args.get('roofit', {})

        logger.info(f'Processing for backend "{bk}" and model "{model}"')

        for ncpu in cfg.get('ncpu', [1]):

            logger.info(f'Running on {ncpu} cores')

            ofile = os.path.join(directory, f'{bk}_ncpu_{ncpu}.txt')
            with open(ofile, 'wt'):
                pass

            for nevts in number_of_events:

                logger.info(f'- {nevts}')

                p = subprocess.Popen(f'python roofit_script.py {jobtype} {model} {nevts} {repetitions} {ofile}',
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                stdout, stderr = p.communicate()
                if p.poll() != 0:
                    raise RuntimeError(f'Job failed with errors:\n{stderr}')


def plot(files, output, show):
    '''
    Generate output plots.
    '''
    # Get the data and plot the results
    fig, (ax, lax) = plt.subplots(1, 2, figsize=(12, 6))

    values = np.empty((len(files), len(number_of_events)), dtype=np.float64)
    errors = np.empty((len(files), len(number_of_events)), dtype=np.float64)

    for i, f in enumerate(files):
        values[i], errors[i] = np.loadtxt(f).T

    def _process_backend(f):
        name = os.path.basename(f).replace('.txt', '')
        if 'roofit' in name:
            ncpu = name.split('_')[-1]
            return f'roofit (ncpu={ncpu})'
        else:
            return name

    # Normalize to maximum
    tmax = values.max()

    backends = list(map(_process_backend, files))

    raw_cyc = cycler.cycler(ls=('-', '--', ':', '-.',
                                (0, (5, 10)),  # loosely dashed
                                (0, (3, 10, 1, 10)),  # loosely dashdotted
                                (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
                                (0, (3, 1, 1, 1, 1, 1))),  # densely dashdotdotted
                            marker=('*', '^', 's', 'x', 'v', 'P', 'D', 'o'))

    cyc = (len(backends) // len(raw_cyc) + 1) * raw_cyc

    for i, (b, c) in enumerate(zip(backends, cyc)):

        v = values[i] / tmax
        e = errors[i] / tmax
        for a in ax, lax:
            a.errorbar(number_of_events, v, yerr=e, label=b, **c)

    lax.set_yscale('log', nonposy='clip')
    for a in ax, lax:
        a.set_xlabel('Number of events')
        a.set_ylabel('Time scaled to maximum')
        a.set_xscale('log', nonposx='clip')
        a.legend(loc='upper left')

    fig.tight_layout()

    fig.savefig(output)

    if show:
        plt.show()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)

    subparsers = parser.add_subparsers(description='Mode to run')

    all_backends = ('cpu', 'cuda', 'opencl', 'roofit')

    parser_run = subparsers.add_parser('run', help=run.__doc__)
    parser_run.set_defaults(function=run)
    parser_run.add_argument('model', type=str, choices=('basic', 'intermediate'),
                            default='basic',
                            help='Model to use')
    parser_run.add_argument('jobtype', type=str, choices=('generate', 'fit'),
                            default='fit',
                            help='Type of job to execute')
    parser_run.add_argument('repetitions', type=int, default=100,
                            help='Number of repetitions per process')
    parser_run.add_argument('--directory', type=str, default='./',
                            help='Where to put the output files')
    parser_run.add_argument('--backends', nargs='*', default=all_backends,
                            help='Backends to process')
    parser_run.add_argument('--extra-args', type=json.loads,
                            help='Arguments to be forwarded to the backend runner. '
                            'It must be provided as a JSON dictionary. '
                            'Only the "roofit" backend accepts configuration with the "ncpu" key. '
                            'You can set it via --extra-args \'{"roofit": {"ncpu": [1, 4]}}\'')

    parser_plot = subparsers.add_parser('plot', help=plot.__doc__)
    parser_plot.set_defaults(function=plot)
    parser_plot.add_argument('files', nargs='*', help='Files to process')
    parser_plot.add_argument('--output', type=str, default='result.pdf',
                             help='Name of the output file')
    parser_plot.add_argument('--show', action='store_true',
                             help='Whether to show the results')

    args = parser.parse_args()

    opts = vars(args)

    func = opts.pop('function')

    func(**opts)
