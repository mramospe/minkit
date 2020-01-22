Performance
===========

The performance of the package has been measured taking the implementation of the `RooFit <https://root.cern.ch/roofit>`__ as a reference in order to evaluate the timing.
The tests have been run using all the different backends available with `minkit <https://mramospe.github.io/minkit>`__, and compared together.
It has been tested both the generation of random numbers and the minimization procedure for two different models with different complexity.
The first model is composed by a simple Gaussian function, whilst the second is the addition of a `Crystal-ball <https://en.wikipedia.org/wiki/Crystal_Ball_function>`__ and an exponential PDFs.
The GPU execution has been done using an `Nvidia GeForce GTX 1080 Ti <https://www.nvidia.com/es-es/geforce/products/10series/geforce-gtx-1080-ti>`__, and the CPU execution using an `Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz <https://ark.intel.com/content/www/es/es/ark/products/91754/intel-xeon-processor-e5-2680-v4-35m-cache-2-40-ghz.html>`__.
The results can be seen in the following figures:

.. figure:: figs/basic_generate.png

   Generate data for a model composed by a Gaussian PDF.

.. figure:: figs/intermediate_generate.png

   Generate data for a model composed by a Crystal-ball and an exponential PDFs.

.. figure:: figs/basic_fit.png

   Generate and fit to a Gaussian PDF.

.. figure:: figs/intermediate_fit.png

   Generate and fit to a model composed by a Crystal-ball and an exponential PDFs.

If the model is simple, generating random numbers is faster for `RooFit <https://root.cern.ch/roofit>`__ than for MinKit, for all the backends, but it soon becomes similar once the model starts to be a bit more complicated.
For a small number of entries, the timing for fitting remains around the same order of magnitude for the CPU implementation and `RooFit <https://root.cern.ch/roofit>`__, being the GPU backends a bit slower.
Howerver, once the number of entries is greater than 10000, the GPU starts to be drastically faster with respect to any CPU implementation.
Once the model starts to be slightly complicated, one can find two orders of magnitude of difference in timing between CPU and GPU.
Note that the CPU implementation of MinKit has an speed close to that of `RooFit <https://root.cern.ch/roofit>`__, despite the latter is purely implemented in C++.
