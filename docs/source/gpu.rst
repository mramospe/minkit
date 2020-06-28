Working on GPUs
===============

By default, the MinKit package only works in CPU backends.
However, the usage of GPUs is supported through the
`Reikna <http://reikna.publicfields.net>`__, which is itself an API for
`PyCUDA <https://documen.tician.de/pycuda>`__ and
`PyOpenCL <https://documen.tician.de/pyopencl>`__.
In order to enable MinKit to work on GPUs, it is necessary to install at least
two of the previous packages (or both).
Please visit :ref:`notes-for-gpu-compatibility` to know about the dependencies
of the package and the way to install them.

There are two (compatible) ways of specifying the backend in MinKit.
The environmental variable *MINKIT_BACKEND* controls the default backend
that is used in the package.
In case a script is created with no explicit declaration of a backend, the
following lines will execute it in CPU, CUDA and OpenCL backends, sequentially:

.. code-block:: bash

   MINKIT_BACKEND=CPU python script.py
   MINKIT_BACKEND=CUDA python script.py
   MINKIT_BACKEND=OPENCL python script.py

The name of the backend is not case sensitive, so it is also possible to
specify them as *cpu*, *cuda* and *opencl*, respectively.

In a single system it is possible to control several GPUs.
The environmental variable that controls the device number is *MINKIT_DEVICE*,
so

.. code-block:: bash

   MINKIT_BACKEND=CUDA MINKIT_DEVICE=0 python script.py
   MINKIT_BACKEND=CUDA MINKIT_DEVICE=1 python script.py

will execute the script in the first device and then in the second.

The backend can also be specified in python scripts, where different backends
can coexist.
It is possible to configure the script so MinKit asks the user what device to
use:

.. code-block:: python

   import minkit
   bk1 = minkit.Backend('cuda', device=0)
   bk2 = minkit.Backend('cuda', interactive=True)
   bk3 = minkit.Backend('cuda', device=2, interactive=True)

In the previous example, the first backend will work on the first device, whilst
in the second it will depend on the input from the user.
In the third case, the device is set initially to *2*, however, if this device
does not exist, MinKit will ask the user to introduce a new backend.
For the last case, if *interactive* is not provided the first device will be
used and a warning will be displayed.

The only objects that depend on backends are PDFs and data sets, and they can
be directly built using the backend instance.

.. code-block:: python

   import minkit

   bk = minkit.Backend('cuda', device=0)

   m = minkit.Parameter('m', bounds=(10, 20))
   c = minkit.Parameter('c', 15)
   s = minkit.Parameter('s', 2)

   ga = bk.Gaussian('g', m, c, s)
   gb = minkit.Gaussian('g', m, c, s, backend=bk)

In the last example, the two last lines are equivalent.
