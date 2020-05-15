#!/usr/bin/env python
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
import tempfile

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


def run(jobtype, model, repetitions, directory, backends):
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

        ofile = os.path.join(directory, f'{bk}.txt')
        with open(ofile, 'wt'):
            pass

        logger.info(f'Processing for backend "{bk}" and model "{model}"')

        for nevts in number_of_events:

            logger.info(f'- {nevts}')

            p = subprocess.Popen(f'python roofit_script.py {jobtype} {model} {nevts} {repetitions} {ofile}',
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            stdout, stderr = p.communicate()
            if p.poll() != 0:
                raise RuntimeError(f'Job failed with errors:\n{stderr}')


def plot(directory, output, show, backends):
    '''
    Generate output plots.
    '''
    # Get the data and plot the results
    fig, (ax, lax) = plt.subplots(1, 2, figsize=(12, 6))

    values = np.empty(len(number_of_events), dtype=[
                      (b, np.float64) for b in backends])
    errors = np.empty(len(number_of_events), dtype=[
                      (b, np.float64) for b in backends])

    for m in backends:
        values[m], errors[m] = np.loadtxt(
            os.path.join(directory, f'{m}.txt')).T

    # Normalize to maximum
    tmax = values.view((np.float64, len(values.dtype.names))).max()

    cyc = cycler.cycler(label=backends,
                        ls=('-', '--', ':', '-.'),
                        marker=('*', '^', 's', 'x'))

    for c in cyc:

        m = c['label']

        v = values[m] / tmax
        e = errors[m] / tmax
        for a in ax, lax:
            a.errorbar(number_of_events, v, yerr=e, **c)

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

    parser_plot = subparsers.add_parser('plot', help=plot.__doc__)
    parser_plot.set_defaults(function=plot)
    parser_plot.add_argument('directory', type=str, default='./',
                             help='Where to get the files')
    parser_plot.add_argument('--output', type=str, default='result.pdf',
                             help='Name of the output file')
    parser_plot.add_argument('--show', action='store_true',
                             help='Whether to show the results')
    parser_plot.add_argument('--backends', nargs='*', default=all_backends,
                             help='Backends to process')

    args = parser.parse_args()

    opts = vars(args)

    func = opts.pop('function')

    func(**opts)