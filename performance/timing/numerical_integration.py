#!/usr/bin/env python
########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Study the performance of numerical integration methods.
'''
import argparse
import cycler
import logging
import matplotlib.pyplot as plt
import minkit
import numpy as np
import time

logger = logging.getLogger(__name__)

RAW_CYC = cycler.cycler(ls=('-', '--', ':', '-.',
                            (0, (5, 10)),  # loosely dashed
                            (0, (3, 10, 1, 10)),  # loosely dashdotted
                            (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
                            (0, (3, 1, 1, 1, 1, 1))),  # densely dashdotdotted
                        marker=('*', '^', 's', 'x', 'v', 'P', 'D', 'o'))


def basic():
    '''
    Basic model.
    '''
    x = minkit.Parameter('x', bounds=(-5, +5))
    c = minkit.Parameter('c', 0, bounds=(-5, +5))
    s = minkit.Parameter('s', 1, bounds=(0.1, 5))
    g = minkit.Gaussian('g', x, c, s)
    return g


def intermediate():
    '''
    Model composed by a single narrow Gaussian function.
    '''
    x = minkit.Parameter('x', bounds=(-5, +5))
    c = minkit.Parameter('c', 0, bounds=(-5, +5))
    s = minkit.Parameter('s', 0.5, bounds=(1e-3, 5))
    g = minkit.Gaussian('g', x, c, s)
    return g


def hard():
    '''
    Model composed by three narrow Gaussian functions.
    '''
    x1 = minkit.Parameter('x1', bounds=(-10, +10))
    c1 = minkit.Parameter('c1', -5, bounds=(-7, -3))
    s1 = minkit.Parameter('s1', 0.01, bounds=(1e-4, 0.1))
    g1 = minkit.Gaussian('g1', x1, c1, s1)

    x2 = minkit.Parameter('x2', bounds=(-10, +10))
    c2 = minkit.Parameter('c2', 0, bounds=(-2, +2))
    s2 = minkit.Parameter('s2', 0.01, bounds=(1e-4, 0.1))
    g2 = minkit.Gaussian('g2', x2, c2, s2)

    x3 = minkit.Parameter('x3', bounds=(-10, +10))
    c3 = minkit.Parameter('c3', +5, bounds=(+2, +7))
    s3 = minkit.Parameter('s3', 0.01, bounds=(1e-4, 0.1))
    g3 = minkit.Gaussian('g3', x3, c3, s3)

    return minkit.ProdPDFs('pdf', [g1, g2, g3])


def monte_carlo(model, calls, repetitions, output, show):

    methods = ('plain', 'miser', 'vegas')

    if model == 'basic':
        pdf = basic()
    else:
        pdf = hard()

    logger.info(f'Processing for model "{model}"')

    true_value = pdf.norm()  # provided it is computed analitically by default

    values, times = {}, {}

    for m in methods:

        logger.info(f'Evaluating for method {m} (printing number of calls)')

        values[m] = []
        times[m] = []

        for n in calls:

            logger.info(f'- {n}')

            # deactivate the warning messages changing the relative tolerance
            pdf.numint_config = {'method': m, 'calls': n, 'rtol': 1}

            method_values = np.empty(repetitions, dtype=np.float64)
            method_times = np.empty(repetitions, dtype=np.float64)

            for i in range(repetitions):
                start = time.time()
                I = pdf.numerical_normalization()
                end = time.time()
                method_values[i] = I
                method_times[i] = end - start

            values[m].append(np.mean(method_values))
            times[m].append(np.mean(method_times))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))

    cyc = (len(methods) // len(RAW_CYC) + 1) * RAW_CYC

    for m, c in zip(methods, cyc):
        ax0.plot(calls, np.abs(np.asarray(
            values[m]) - true_value), label=m, **c)
        ax1.plot(calls, np.asarray(times[m]) * 1e-3, label=m, **c)

    ax0.set_ylabel('distance to true value')

    ax1.set_ylabel('time (ms)')

    for ax in ax0, ax1:
        ax.set_xscale('log', nonposx='clip')
        ax.set_yscale('log', nonposy='clip')
        ax.set_xlabel('number of calls')
        ax.legend(loc='best')

    fig.tight_layout()
    fig.savefig(output)
    if show:
        plt.show()


def performance(model, rtol, repetitions, output, show):

    if model == 'basic':
        pdf = basic()
    else:
        pdf = hard()

    logger.info(f'Processing for model "{model}"')

    true_value = pdf.norm()  # provided it is computed analitically by default

    values, times = {}, {}

    for rt in rtol:

        logger.info(f'Evaluating for tolerance {rt:.2e}')

        if rt >= 1e-2:
            methods = ('qng', 'qag', 'cquad', 'plain', 'miser', 'vegas')
        elif rt >= 1e-4:  # remove plain Monte Carlo
            methods = ('qng', 'qag', 'cquad', 'miser', 'vegas')
        elif rt >= 1e-5:  # remove MISER
            methods = ('qng', 'qag', 'cquad', 'vegas')
        else:  # remove VEGAS
            methods = ('qng', 'qag', 'cquad')

        for m in methods:

            if m not in values:
                values[m] = []
                times[m] = []

            pdf.numint_config = {'method': m, 'rtol': rt}

            method_values = np.empty(repetitions, dtype=np.float64)
            method_times = np.empty(repetitions, dtype=np.float64)

            for i in range(repetitions):
                start = time.time()
                I = pdf.numerical_normalization()
                end = time.time()
                method_values[i] = I
                method_times[i] = end - start

            values[m].append(np.mean(method_values))
            times[m].append(np.mean(method_times))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))

    cyc = (len(values) // len(RAW_CYC) + 1) * RAW_CYC

    for (m, v), c in zip(values.items(), cyc):
        rt = rtol[:len(v)]
        ax0.plot(rt, np.abs(np.asarray(v) - true_value), label=m, **c)
        ax1.plot(rt, np.asarray(times[m]) * 1e-3,
                 label=m, **c)  # to milliseconds

    ax0.set_ylabel('distance to true value')

    ax1.set_ylabel('time (ms)')

    for ax in ax0, ax1:
        ax.set_xscale('log', nonposx='clip')
        ax.set_yscale('log', nonposy='clip')
        ax.set_xlabel('relative error')
        ax.legend(loc='best')
        l, r = ax.get_xlim()
        ax.set_xlim(r, l)

    fig.tight_layout()
    fig.savefig(output)
    if show:
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)

    subparsers = parser.add_subparsers(help='Mode to run')

    pp = subparsers.add_parser('performance', help=performance.__doc__)
    pp.set_defaults(function=performance)
    pp.add_argument('--rtol', nargs='*', default=(1e-1, 1e-2, 1e-3, 1e-4,
                                                  1e-5, 1e-6, 1e-7), help='Values for the relative error to consider')

    mp = subparsers.add_parser('monte-carlo', help=monte_carlo.__doc__)
    mp.set_defaults(function=monte_carlo)
    mp.add_argument('--calls', nargs='*',
                    default=(100, 1000, 10000, 100000, 1000000))

    for p in pp, mp:
        p.add_argument('--model', '-m', default='basic', choices=('basic', 'intermediate', 'hard'),
                       help='Model to use. The "basic" model is composed by a single well-behaved Gaussian function. '
                       'The "intermediate" model is composed by a narrow Gaussian function. '
                       'The "hard" model is composed by three narrow Gaussian functions.')
        p.add_argument('--repetitions', '-r', type=int, default=100,
                       help='Number of times to calculate the integral with each method')
        p.add_argument('--output', '-o', type=str, default='numerical_integration.png',
                       help='Name of the output file for the plot')
        p.add_argument('--show', '-s', action='store_true',
                       help='Whether to show the output plots')

    args = parser.parse_args()

    config = dict(vars(args))

    function = config.pop('function')

    function(**config)
