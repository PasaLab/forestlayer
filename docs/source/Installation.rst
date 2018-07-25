Installation On Linux Platform
==============================


ForestLayer has install prerequisites including scikit-learn, keras, numpy, ray and joblib.

For GPU support, CUDA and cuDNN are required, but now we have not support GPU yet.

The simplest way to install ForestLayer in your python program is:

::

    [for master version] pip install git+https://github.com/whatbeg/forestlayer.git
    [for stable version] pip install forestlayer

Build from Source
-----------------

Alternatively, you can install ForestLayer from the github source:

Dependencies
~~~~~~~~~~~~

::
    $ git clone https://github.com/whatbeg/forestlayer.git
    $ cd forestlayer
    $ pip install -r requirement.txt

Install ForestLayer
~~~~~~~~~~~~~~~~~~~

::

    $ python setup.py install


.. note::

   Now ForestLayer does not support Python 3.x and Windows Platform and MacOS Platform.
