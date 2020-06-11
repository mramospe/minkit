.. minkit documentation master file, created by
   sphinx-quickstart on Fri Dec  8 18:24:26 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

This package provides tools to fit probability density functions (PDFs) to both unbinned and binned data, using different minimizers (like `Minuit <https://iminuit.readthedocs.io/en/latest/reference.html>`__).
The MinKit package appears as an alternative to existing minimization packages, like `RooFit <https://root.cern.ch/roofit>`__.
The idea is to provide a friendly pure python API to do minimization and calculations with PDFs.
It has support for both CPU and GPU backends, being very easy for the user to change from one to the other.
PDFs are implemented in C++, OpenCL and CUDA, allowing a fast evaluation of the functions.

The package is built on top of the `numpy <https://numpy.org>`__ and `iminuit <https://iminuit.readthedocs.io/en/latest>`__ packages.
The interface with CUDA and OpenCL is handled using `Reikna <http://reikna.publicfields.net>`__, which is itself an API for `PyCUDA <https://documen.tician.de/pycuda>`__ and `PyOpenCL <https://documen.tician.de/pyopencl>`__.

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
