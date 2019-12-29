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

This package provides tools to fit probability density functions (PDFs) to both unbinned and binned data, using different minimizers (like `Minuit <https://iminuit.readthedocs.io/en/latest/reference.html>`_).
It has support for both CPU and GPU backends, providing an easy API in order to manage both backends, being transparent for the user.
PDFs are implemented in C++ and CUDA, allowing a fast evaluation of the functions.

The package is built on top of the `numpy <https://numpy.org/>`_ and `iminuit <https://iminuit.readthedocs.io/en/latest/>`_ packages.
The interface with CUDA is handled using `PyCUDA <https://documen.tician.de/pycuda>`_.

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

In this case "minimizer" is a `Minuit <https://iminuit.readthedocs.io/en/latest/reference.html#minuit`_ instance, since by default `Minuit <https://iminuit.readthedocs.io/en/latest/reference.html#minuit>`_ is used to do the minimization.
The string "uml" specifies the type of figure to minimize (FCN), unbinned-maximum likelihood, in this case.

Installation:
=============

This package is available on `PyPi <https://pypi.org/>`_, so simply type

.. code-block:: bash

   pip install minkit

to install the package in your current python environment.
To use the **latest development version**, clone the repository and install with `pip`:

.. code-block:: bash

   git clone https://github.com/mramospe/hep_spt.git
   pip install hep_spt

Remember that you can also install the package in-place, something very useful for developers, by calling

.. code-block:: bash

   git clone https://github.com/mramospe/hep_spt.git
   pip install -e hep_spt
