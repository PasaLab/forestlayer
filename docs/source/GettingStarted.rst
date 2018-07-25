Getting Started
===============

Getting Started: 30 seconds to ForestLayer
------------------------------------------

The core data structure of ForestLayer is layers and graph. Layers are
basic modules to implement different data processing, and the graph is
like a model that organize layers, the basic type of graph is a stacking
of layers, and now we only support this type of graph.

Take MNIST classification task as an example.

First, we use the Keras API to load mnist data and do some
pre-processing.

.. code:: python

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # preprocessing

Next, we construct multi-grain scan windows and estimators every window
and then initialize a ``MultiGrainScanLayer``. The Window class is lies
in ``forestlayer.layers.window`` package and the estimators are
represented as ``EstimatorArgument``\ s, which will be used later in
layers to create actual estimator object.

.. code:: python

    from forestlayer.layers.layer import MultiGrainScanLayer
    from forestlayer.estimators.arguments import CompletelyRandomForest, RandomForest
    from forestlayer.layers.window import Window

    rf1 = CompletelyRandomForest(min_samples_leaf=10)
    rf2 = RandomForest(min_samples_leaf=10)

    windows = [Window(win_x=7, win_y=7, stride_x=2, stride_y=2, pad_x=0, pad_y=0),
               Window(11, 11, 2, 2)]

    est_for_windows = [[rf1, rf2],
                       [rf1, rf2]]

    mgs = MultiGrainScanLayer(windows=windows,
                              est_for_windows=est_for_windows,
                              n_class=10)

After multi-grain scan, we consider that building a pooling layer to
reduce the dimension of generated feature vectors, so that reduce the
computation and storage complexity and risk of overfiting.

.. code:: python

    from forestlayer.layers.layer import PoolingLayer
    from forestlayer.layers.factory import MaxPooling

    pools = [[MaxPooling(2, 2), MaxPooling(2, 2)],
             [MaxPooling(2, 2), MaxPooling(2, 2)]]

    pool = PoolingLayer(pools=pools)

And then we add a concat layer to concatenate the output of estimators
of the same window.

.. code:: python

    from forestlayer.layers.layer import ConcatLayer
    concatlayer = ConcatLayer()

Then, we construct the cascade part of the model, we use an auto-growing
cascade layer to build our deep forest model.

.. code:: python

    est_configs = [
        CompletelyRandomForest(),
        CompletelyRandomForest(),
        RandomForest(),
        RandomForest()
    ]

    data_save_dir = osp.join(get_data_save_base(), 'mnist')
    model_save_dir = osp.join(get_model_save_base(), 'mnist')

    auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs,
                                           early_stopping_rounds=4,
                                           stop_by_test=True,
                                           n_classes=10,
                                           data_save_dir=data_save_dir,
                                           model_save_dir=model_save_dir)

Last, we construct a graph to stack these layers to make them as a
complete model.

.. code:: python

    model = Graph()
    model.add(mgs)
    model.add(pool)
    model.add(concatlayer)
    model.add(auto_cascade)

You also can call ``model.summary()`` like Keras to see the appearance
of the model.

After building the model, you can fit the model, and then evaluate or
predict using the fit model.

.. code:: python

    model.fit(x_train, y_train)
    model.evaluate(x_test, y_test)
    result = model.predict(x_in)

For more examples and tutorials, you can refer to `examples <https://github.com/whatbeg/forestlayer/tree/master/examples>`__ to find more details.

Enable Distributed Training
------------------------------------------

Standalone
~~~~~~~~~~

Fill this in.

Cluster
~~~~~~~

Fill this in.


