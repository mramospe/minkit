======
MinKit
======

.. image:: https://img.shields.io/travis/mramospe/minkit.svg
   :target: https://travis-ci.org/mramospe/minkit.svg?branch=master

.. image:: https://img.shields.io/badge/documentation-link-blue.svg
   :target: https://mramospe.github.io/minkit/

.. image:: https://codecov.io/gh/mramospe/minkit/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/mramospe/minkit?branch=master

.. inclusion-marker-do-not-remove

This package provides tools to fit probability density functions (PDFs) to both unbinned and binned data, using different minimizers (like `Minuit <https://iminuit.readthedocs.io/en/latest/reference.html>`__).
The MinKit package appears as an alternative to existing minimization packages, like `RooFit <https://root.cern.ch/roofit>`__.
The idea is to provide a friendly pure python API to do minimization and calculations with PDFs.
It has support for both CPU and GPU backends, being very easy for the user to change from one to the other.
PDFs are implemented in C++, OpenCL and CUDA, allowing a fast evaluation of the functions.

The package is built on top of the `numpy <https://numpy.org>`__ and `iminuit <https://iminuit.readthedocs.io/en/latest>`__ packages.
The interface with CUDA and OpenCL is handled using `reikna <http://reikna.publicfields.net>`__, which is itself an API for `PyCUDA <https://documen.tician.de/pycuda>`__ and `PyOpenCL <https://documen.tician.de/pyopencl>`__.

Basic example
=============

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

   with minkit.unbinned_minimizer('uml', g, data) as minimizer:
      r = minimizer.migrad()

In this case *minimizer* is a `Minuit <https://iminuit.readthedocs.io/en/latest/reference.html#minuit>`__ instance, since by default `Minuit <https://iminuit.readthedocs.io/en/latest/reference.html#minuit>`__ is used to do the minimization.
The string *uml* specifies the type of figure to minimize (FCN), unbinned-maximum likelihood, in this case.

The compilation of C++ sources is completely system dependent (Linux, MacOS, Windows), and it also depends on the way python
has been installed.
The PDFs in this package need the C++ standard from 2011.
Depending on the system, functions might need to be compiled with extra flags that are not used by default in `distutils <https://docs.python.org/3/library/distutils.html>`__.
If you get errors of the type:

.. code-block:: bash

   relocation R_X86_64_PC32 against undefined symbol

suggesting to use *-fPIC* option (when the system is using *gcc* to compile C code) or

.. code-block:: bash

   error: ‘erf’ is not a member of ‘std’

more likely it is needed to specify the flags to use.
In order to do so, simply execute your script setting the value of the environmental variable *CFLAGS* accordingly:

.. code-block:: bash

   CFLAGS="-fPIC -std=c++11" python script.py


Fast installation:
==================

This package is available on `PyPi <https://pypi.org/>`__, so simply type

.. code-block:: bash

   pip install minkit

to install the package in your current python environment.
To use the **latest development version**, clone the repository and install with *pip*:

.. code-block:: bash

   git clone https://github.com/mramospe/minkit.git
   pip install minkit

In order to profit from certain features of the package, like numerical integration, it is necessary
that the system has the *GSL* libraries visible to the compiler.
To install them on Linux, you can simply run

.. code-block:: bash

   sudo apt-get install libgsl-dev

Depending on the system, you might need to set also the necessary environment variables
specifying the path to the include and libraries directory, like

.. code-block:: bash

   export CFLAGS="$CFLAGS -I/usr/include -L/usr/lib/x86_64-linux-gnu"
