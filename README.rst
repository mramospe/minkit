======
MinKit
======

.. image:: https://travis-ci.org/mramospe/minkit.svg?branch=master
   :target: https://travis-ci.org/mramospe/minkit

.. image:: https://img.shields.io/badge/documentation-link-blue.svg
   :target: https://minkit.readthedocs.io/en/latest

.. image:: https://codecov.io/gh/mramospe/minkit/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/mramospe/minkit

MinKit is a python package meant to minimize probability density functions (PDFs) in chi-squared and maximum likelihood fits.
It also provides tools to calculate and handle parameter errors.
The package offers a common interface to use `iminuit <https://iminuit.readthedocs.io/en/latest>`__, `SciPy <https://docs.scipy.org/doc>`__ or `NLopt <https://nlopt.readthedocs.io/en/latest>`__ minimizers.
In order to do numerical integrations, the `GSL <https://www.gnu.org/software/gsl/doc/html>`__ libraries are used.
The utilization of both CPU and GPU devices is supported in MinKit.
For GPU backends, it is relied in the `Reikna <http://reikna.publicfields.net/en/latest>`__ package, that is a common interface for `PyCUDA <https://documen.tician.de/pycuda>`__ and `PyOpenCL <https://documen.tician.de/pyopencl>`__.
It is not necessary to have the previous packages installed if working only with CPU backends.
