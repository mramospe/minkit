#!/usr/bin/env python
########################################
# MIT License
#
# Copyright (c) 2020 Miguel Ramos Pernas
########################################
'''
Generate a sample using the RooFit package.
'''
import time
import argparse
import roofit_models
import multiprocessing
import numpy as np
import ROOT as rt
rt.PyConfig.IgnoreCommandLineOptions = True


def fit(pdf, nevts, repetitions, m, pars, ncpu):
    '''
    Generate data following the given model and fit it.
    '''
    times = np.empty(repetitions, dtype=np.float64)
    initials = {p.GetName(): np.random.uniform(p.getMin(), p.getMax())
                for p in pars}
    for i in range(len(times)):
        data = pdf.generate(rt.RooArgSet(m), nevts, rt.RooFit.NumCPU(ncpu))
        start = time.time()
        pdf.fitTo(data, rt.RooFit.Save(), rt.RooFit.NumCPU(ncpu))
        end = time.time()
        times[i] = end - start
        for p in pars:
            p.setVal(initials[p.GetName()])
    return times


def generate(pdf, nevts, repetitions, m, ncpu):
    '''
    Generate data following the given model.
    '''
    times = np.empty(repetitions, dtype=np.float64)
    for i in range(len(times)):
        start = time.time()
        pdf.generate(rt.RooArgSet(m), nevts, rt.RooFit.NumCPU(ncpu))
        end = time.time()
        times[i] = end - start
    return times


def main(jobtype, model, nevts, repetitions, outputfile, ncpu):

    pdf, pars, extra = getattr(roofit_models, model)()

    m = pars[0]

    if jobtype == 'generate':
        times = generate(pdf, nevts, repetitions, m, ncpu)
    elif jobtype == 'fit':
        times = fit(pdf, nevts, repetitions, m, pars[1:], ncpu)
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
    parser.add_argument('--ncpu', type=int, default=1, choices=tuple(range(multiprocessing.cpu_count())),
                        help='Number of threads to run')
    args = parser.parse_args()
    main(**vars(args))
