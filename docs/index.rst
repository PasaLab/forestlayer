.. ForestLayer documentation master file, created by
   sphinx-quickstart on Wed Jan 10 16:41:17 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ForestLayer: A Scalable and Fast Distributed Deep Forest Library
==================================================================

ForestLayer is a scalable, fast deep forest learning library based on Scikit-learn and Ray.
It provides rich data processing, model training, and serving modules to help researchers and engineers build practical deep forest learning workflows.
It internally embedded task parallelization mechanism using Ray, which is a popular flexible, high-performance distributed execution framework proposed by U.C.Berkeley.
ForestLayer aims to enable faster experimentation as possible and reduce the delay from idea to result.
Hope is that ForestLayer can bring you good researches and good products.

You can refer to `Deep Forest Paper <https://arxiv.org/abs/1702.08835>`__, `Ray Project <https://github.com/ray-project/ray>`__ to find more details.

.. toctree::
   :maxdepth: 1
   :caption: Installation

   source/Installation.rst

.. toctree::
   :maxdepth: 1
   :caption: ForestLayer

   source/introduction.rst
   source/deepforest.rst
   source/GettingStarted.rst

.. toctree::
   :maxdepth: 2
   :caption: Examples

   source/Examples.rst

.. toctree::
   :maxdepth: 3
   :caption: API

   source/forestlayer.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
