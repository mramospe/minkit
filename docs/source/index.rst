.. minkit documentation master file, created by
   sphinx-quickstart on Fri Dec  8 18:24:26 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

This package provides tools to fit probability density functions (PDFs) to both unbinned and binned data, using different minimizers.
MinKit offers a common interface to use `iminuit <https://iminuit.readthedocs.io/en/latest>`__, `SciPy <https://docs.scipy.org/doc>`__ or `NLopt <https://nlopt.readthedocs.io/en/latest>`__ minimizers.
In order to do numerical integrations, the `GSL <https://www.gnu.org/software/gsl/doc/html>`__ libraries are used.
The utilization of both CPU and GPU devices is supported.
For GPU backends, it is relied in the `Reikna <http://reikna.publicfields.net/en/latest>`__ package, that is a common interface for `PyCUDA <https://documen.tician.de/pycuda>`__ and `PyOpenCL <https://documen.tician.de/pyopencl>`__.
It is not necessary to have the previous packages installed if working only with CPU backends.

Classes meant for the user are imported directly from the main module

.. code-block:: python

   import minkit

   x = minkit.Parameter('x', bounds=(-10, +10))
   c = minkit.Parameter('c', 0.)
   s = minkit.Parameter('s', 1.)
   g = minkit.Gaussian('Gaussian', x, c, s)

   data = g.generate(10000)

These lines define the parameters used by a Gaussian function, and a data set is generated
following this distribution.
The sample can be easily fitted calling:

.. code-block:: python

   with minkit.minimizer('uml', g, data) as minimizer:
      minimizer.minimize()

After this process, the parameters of the PDF take values corresponding to the minimization point.

.. toctree::
   :maxdepth: 2

   installation
   user
   gpu
   reference
   performance
