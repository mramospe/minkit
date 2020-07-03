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
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess

number_of_events = [1000, 10000, 100000, 1000000, 10000000]

logger = logging.getLogger(__name__)

# Format for the plots
matplotlib.rcParams['figure.figsize'] = (12, 5)
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


def run(jobtype, model, repetitions, directory, backends):
    '''
    Main function to execute.
    '''
    # Run with minkit
    for bk in sorted({'cpu', 'opencl', 'cuda'}.intersection(backends)):

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
    for bk in sorted({'roofit'}.intersection(backends)):

        logger.info(f'Processing for backend "{bk}" and model "{model}"')

        ofile = os.path.join(directory, f'{bk}.txt')
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
    fig, (ax, lax) = plt.subplots(1, 2)

    values = np.empty((len(files), len(number_of_events)), dtype=np.float64)
    errors = np.empty((len(files), len(number_of_events)), dtype=np.float64)

    for i, f in enumerate(files):
        values[i], errors[i] = np.loadtxt(f).T

    backends = list(
        map(lambda s: os.path.splitext(os.path.basename(s))[0], files))

    # Normalize to maximum
    tmax = values.max()

    raw_cyc = cycler.cycler(ls=('-', '--', ':', '-.',
                                (0, (5, 10)),  # loosely dashed
                                (0, (3, 10, 1, 10)),  # loosely dashdotted
                                (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
                                (0, (3, 1, 1, 1, 1, 1))),  # densely dashdotdotted
                            marker=('*', '^', 's', 'x', 'v', 'P', 'D', 'o'))

    cyc = (len(backends) // len(raw_cyc) + 1) * raw_cyc

    lax.set_yscale('log', nonposy='clip')
    for a in ax, lax:

        t = a.twinx()

        def convert_time_to_relative(ax):
            y1, y2 = ax.get_ylim()
            t.set_ylim(y1 / tmax, y2 / tmax)
            t.figure.canvas.draw()

        a.callbacks.connect('ylim_changed', convert_time_to_relative)

        for i, (b, c) in enumerate(zip(backends, cyc)):
            a.errorbar(number_of_events,
                       values[i], yerr=errors[i], label=b, **c)

        a.set_xlabel('Number of events')
        a.set_ylabel('Time (s)')
        t.set_ylabel('Time scaled to maximum')
        a.set_xscale('log', nonposx='clip')
        a.set_xticks(number_of_events)

        if a.get_yscale() == 'log':
            a.legend(loc='best')
            t.set_yscale('log', nonposy='clip')
        else:
            a.legend(loc='best')

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
    parser_run.add_argument('model', type=str, choices=('basic', 'intermediate', 'numeric'),
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
