#!/usr/bin/env python
########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Generate a sample using the minkit package.
'''
import argparse
import logging
import minkit
import minkit_models
import numpy as np
import time

logging.basicConfig(level=logging.INFO)


def fit(pdf, nevts, repetitions):
    '''
    Generate data following the given model and fit it.
    '''
    times = np.empty(repetitions, dtype=np.float64)
    initials = {p.name: np.random.uniform(
        *p.bounds) for p in pdf.all_real_args}
    for i in range(len(times)):
        data = pdf.generate(nevts)
        start = time.time()
        with minkit.minimizer('uml', pdf, data) as minimizer:
            minimizer.migrad()
        end = time.time()
        times[i] = end - start
        pdf.set_values(**initials)
    return times


def generate(pdf, nevts, repetitions):
    '''
    Generate data following the given model.
    '''
    times = np.empty(repetitions, dtype=np.float64)
    for i in range(len(times)):
        start = time.time()
        pdf.generate(nevts)
        end = time.time()
        times[i] = end - start
    return times


def main(jobtype, model, nevts, repetitions, outputfile, backend):

    bk = minkit.Backend(backend)

    pdf = getattr(minkit_models, model)(bk)

    if jobtype == 'generate':
        times = generate(pdf, nevts, repetitions)
    elif jobtype == 'fit':
        times = fit(pdf, nevts, repetitions)
    else:
        raise ValueError(f'Unknown job type {jobtype}')

    with open(outputfile, 'at') as f:
        f.write(f'{times.mean()} {times.std()}\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobtype', type=str, choices=('generate', 'fit'),
                        default='fit',
                        help='Type of job to execute')
    parser.add_argument('model', type=str, choices=('basic', 'intermediate', 'numeric'),
                        default='basic',
                        help='Model to use')
    parser.add_argument('nevts', type=int, default=100000,
                        help='Number of events to generate')
    parser.add_argument('repetitions', type=int, default=10,
                        help='Number of repetitions')
    parser.add_argument('outputfile', type=str,
                        help='Where to save the result')
    parser.add_argument('--backend', type=str, default='cpu',
                        choices=('cpu', 'cuda', 'opencl'),
                        help='Backend to use')
    args = parser.parse_args()
    main(**vars(args))
