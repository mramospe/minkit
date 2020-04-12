#!/usr/bin/env python
'''
Generate a sample using the RooFit package.
'''
import time
import argparse
import roofit_models
import numpy as np
import ROOT as rt
rt.PyConfig.IgnoreCommandLineOptions = True


def fit(pdf, nevts, repetitions, m, pars):
    '''
    Generate data following the given model and fit it.
    '''
    times = np.empty(repetitions, dtype=np.float64)
    initials = {p.GetName(): np.random.uniform(p.getMin(), p.GetMax())
                for p in pars}
    for i in range(len(times)):
        data = pdf.generate(rt.RooArgSet(m), nevts)
        start = time.time()
        r = pdf.fitTo(data, rt.RooFit.Save())
        end = time.time()
        times[i] = end - start
        for p in pars:
            p.setVal(initials[p.GetName()])
    return times


def generate(pdf, nevts, repetitions, m):
    '''
    Generate data following the given model.
    '''
    times = np.empty(repetitions, dtype=np.float64)
    for i in range(len(times)):
        start = time.time()
        data = pdf.generate(rt.RooArgSet(m), nevts)
        end = time.time()
        times[i] = end - start
    return times


def main(jobtype, model, nevts, repetitions, outputfile, backend):

    pdf, pars, extra = getattr(roofit_models, model)()

    m = pars[0]

    if jobtype == 'generate':
        times = generate(pdf, nevts, repetitions, m)
    elif jobtype == 'fit':
        times = fit(pdf, nevts, repetitions, m, pars[1:])
    else:
        raise ValueError(f'Unknown job type {jobtype}')

    with open(outputfile, 'at') as f:
        f.write(f'{times.mean()} {times.std()}\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobtype', type=str, choices=('generate', 'fit'),
                        default='fit',
                        help='Type of job to execute')
    parser.add_argument('model', type=str, choices=('basic', 'intermediate'),
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
