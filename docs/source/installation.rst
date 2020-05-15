Installation
============

The Minkit package allows to minimize probability density functions (PDFs) using
a python API, offering the opportunity to use iminuit or scipy minimizers.
The PDFs are compiled at runtime, allowing any user do define custom PDFs.

This package uses the GNU scientific library (GSL), which needs to be accessible
in order to compile the source code generated at runtime. In addition, the C++
standard used is C++11 or greater. Be sure to have the necessary environment
variables set before running any script. In Ubuntu, this can be done executing:

.. code-block:: bash

   sudo apt-get install libgsl-dev
   export CFLAGS="-fPIC -std=c++11 -I/usr/include"
   export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

This package is capable to work in CPU and GPU backends, and has been designed
to work in both CUDA and OpenCL. The GPU operations are done using the
`reikna <http://reikna.publicfields.net>`__ package. In order to make Minkit
run in GPUs, it becomes necessary to have installed
`reikna <http://reikna.publicfields.net>`__,
and `pycuda <https://documen.tician.de/pycuda/>`__ or
`pyopencl <https://documen.tician.de/pyopencl/>`__ depending if we have CUDA or
OpenCL installed in our system.