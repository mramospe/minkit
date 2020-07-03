Installation
============

The MinKit package allows to minimize probability density functions (PDFs) using
a python API, offering the opportunity to use iminuit or scipy minimizers.
The PDFs are compiled at runtime, allowing any user do define custom PDFs.

Installation with PIP
---------------------

This package is available on `PyPi <https://pypi.org/>`__, so to install it simply type

.. code-block:: bash

   pip install minkit

To use the **latest development version**, clone the repository and install with *pip*:

.. code-block:: bash

   git clone https://github.com/mramospe/minkit.git
   pip install minkit

Remember that you can also install the package in-place, something very useful for developers, by calling

.. code-block:: bash

   git clone https://github.com/mramospe/minkit.git
   pip install -e minkit

This package uses the GNU scientific library (GSL), which needs to be accessible
in order to compile the source code generated at runtime. In addition, the C++
standard used is C++11 or greater. Be sure to have the necessary environment
variables set before running any script. In Ubuntu, this can be done executing:

.. code-block:: bash

   sudo apt-get install libgsl-dev
   export CFLAGS="$CFLAGS -fPIC -std=c++11 -I/usr/include"
   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu"

Installation with CONDA
-----------------------

This package is also available in *conda*.
However, the dependencies are located on a different channel to that of this package.
The dependencies live in the *conda-forge* channel.
In order to properly handle the different channels, it is recommended that you run

.. code-block:: bash

   conda config --add channels conda-forge

which is equivalent to


.. code-block:: bash

   conda config --prepend channels conda-forge

Alternatively, you can also execute

.. code-block:: bash

   conda config --append channels conda-forge

to give the new channel the lowest priority.
Afterwards you can simply run

.. code-block:: bash

   conda install -c mramospe minkit

in order to install MinKit.

.. _notes-for-gpu-compatibility:

Notes for GPU compatibility
---------------------------

This package is capable to work in CPU and GPU backends, and has been designed
to work in both CUDA and OpenCL. The GPU operations are done using the
`reikna <http://reikna.publicfields.net>`__ package. In order to make MinKit
run in GPUs, it becomes necessary to have installed
`reikna <http://reikna.publicfields.net>`__,
and `pycuda <https://documen.tician.de/pycuda/>`__ or
`pyopencl <https://documen.tician.de/pyopencl/>`__ depending if we have CUDA or
OpenCL installed in our system.
The dependencies are not automatically handled by *pip* or *conda*, so you will
need to run

.. code-block:: bash

   pip install reikna pycuda pyopencl

or

.. code-block:: bash

   conda install -c conda-forge reikna pycuda pyopencl

in order to install them.
