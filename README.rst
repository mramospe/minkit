======
minkit
======

.. image:: https://img.shields.io/travis/mramospe/minkit.svg
   :target: https://travis-ci.org/mramospe/minkit

.. image:: https://img.shields.io/badge/documentation-link-blue.svg
   :target: https://mramospe.github.io/minkit/

.. image:: https://codecov.io/gh/mramospe/minkit/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/mramospe/minkit

.. inclusion-marker-do-not-remove

This package provides tools to fit probability density functions (PDFs) to both unbinned and binned data, using different minimizers (like `Minuit <https://iminuit.readthedocs.io/en/latest/reference.html>`__).
It has support for both CPU and GPU backends, providing an easy API in order to manage both backends, being transparent for the user.
PDFs are implemented in C++ and CUDA, allowing a fast evaluation of the functions.

The package is built on top of the `numpy <https://numpy.org/>`__ and `iminuit <https://iminuit.readthedocs.io/en/latest/>`__ packages.
The interface with CUDA is handled using `PyCUDA <https://documen.tician.de/pycuda>`__.

Basic example
=============

Classes meant for the user are imported directly from the main module

.. code-block:: python

   import minkit
   minkit.initialize()

The second line allows to define the backend where the execution will take place (CPU/GPU).
By default (like in this case) it is set to CPU.
Let us now define a very simple normal PDF:

.. code-block:: python

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

In this case "minimizer" is a `Minuit <https://iminuit.readthedocs.io/en/latest/reference.html#minuit>`__ instance, since by default `Minuit <https://iminuit.readthedocs.io/en/latest/reference.html#minuit>`__ is used to do the minimization.
The string "uml" specifies the type of figure to minimize (FCN), unbinned-maximum likelihood, in this case.

The compilation of C++ sources is completely system dependent (Linux, MacOS, Windows), and it also depends on the way python
has been installed.
The PDFs in this package need the C++ standard from 2011.
In some old systems, functions need to be compiled with extra flags, that are not used by default in `distutils <https://docs.python.org/3/library/distutils.html>`__.
If you get errors of the type:

.. code-block:: bash

   relocation R_X86_64_PC32 against undefined symbol

suggesting to use "-fPIC" option (when the system is using gcc to compile C code) or

.. code-block:: bash

   error: ‘erf’ is not a member of ‘std’

more likely it is needed to specify the standard and flags to use.
In order to do so, simply execute your script setting the value of the environmental variable "CFLAGS" accordingly:

.. code-block:: bash

   CFLAGS="-fPIC -std=c++11" python script.py


Installation:
=============

This package is available on `PyPi <https://pypi.org/>`__, so simply type

.. code-block:: bash

   pip install minkit

to install the package in your current python environment.
To use the **latest development version**, clone the repository and install with `pip`:

.. code-block:: bash

   git clone https://github.com/mramospe/minkit.git
   pip install minkit

Remember that you can also install the package in-place, something very useful for developers, by calling

.. code-block:: bash

   git clone https://github.com/mramospe/minkit.git
   pip install -e minkit
