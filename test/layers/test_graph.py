# -*- coding:utf-8 -*-
"""
Test Suites of layers.graph.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import numpy as np
import os.path as osp
import unittest
from forestlayer.layers.window import Window, Pooling
from forestlayer.layers.layer import MultiGrainScanLayer, PoolingLayer, ConcatLayer, AutoGrowingCascadeLayer
from forestlayer.layers.graph import Graph
from forestlayer.estimators.estimator_configs import RandomForestConfig, ExtraRandomForestConfig
from forestlayer.utils.storage_utils import get_data_save_base
from keras.datasets import mnist


class TestGraph(unittest.TestCase):
    def setUp(self):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        X = np.reshape(x_train, (60000, -1, 28, 28))
        x_train = X[:120, :, :, :]
        y_train = y_train[:120]
        x_test = np.reshape(x_test[:60], (60, -1, 28, 28))
        y_test = y_test[:60]
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        print('X_train: ', x_train.shape, 'y: ', y_train.shape)
        print(' X_test: ', x_test.shape, 'y: ', y_test.shape)

    def _init(self):
        self.est_configs = [
            ExtraRandomForestConfig(n_estimators=40),
            ExtraRandomForestConfig(n_estimators=40),
            RandomForestConfig(n_estimators=40),
            RandomForestConfig(n_estimators=40)
        ]

        windows = [Window(7, 7, 2, 2, 0, 0),
                   Window(11, 11, 2, 2, 0, 0)]

        rf1 = ExtraRandomForestConfig(min_samples_leaf=10)
        rf2 = RandomForestConfig(min_samples_leaf=10)

        est_for_windows = [[rf1, rf2],
                           [rf1, rf2]]

        mgs = MultiGrainScanLayer(dtype=np.float32,
                                  windows=windows,
                                  est_for_windows=est_for_windows,
                                  n_class=10)

        pools = [[Pooling(2, 2, "max"), Pooling(2, 2, "max")],
                 [Pooling(2, 2, "max"), Pooling(2, 2, "max")]]

        poolayer = PoolingLayer(pools=pools)

        concat_layer = ConcatLayer()

        auto_cascade = AutoGrowingCascadeLayer(est_configs=self.est_configs,
                                               early_stopping_rounds=2,
                                               stop_by_test=False,
                                               data_save_rounds=4,
                                               n_classes=10,
                                               data_save_dir=osp.join(get_data_save_base(),
                                                                      'test_graph', 'auto_cascade'))
        return mgs, poolayer, concat_layer, auto_cascade

    def test_fit(self):
        print('test fit')
        mgs, poolayer, concat_layer, auto_cascade = self._init()
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit(self.x_train, self.y_train)

    def test_fit_transform(self):
        print("test fit_transform")
        mgs, poolayer, concat_layer, auto_cascade = self._init()
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        print(self.x_train.shape, self.x_test.shape, self.y_train.shape)
        model.fit_transform(self.x_train, self.y_train, self.x_test)

    def test_fit_transform_full(self):
        print("test fit_transform_full")
        mgs, poolayer, concat_layer, auto_cascade = self._init()
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit_transform(self.x_train, self.y_train, self.x_test, self.y_test)

    def test_transform(self):
        print("test transform")
        mgs, poolayer, concat_layer, auto_cascade = self._init()
        mgs.keep_in_mem = True
        auto_cascade.keep_in_mem = True
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit(self.x_train, self.y_train)
        model.transform(self.x_test)

    def test_fit_predict(self):
        print("test fit and predict")
        mgs, poolayer, concat_layer, auto_cascade = self._init()
        mgs.keep_in_mem = True
        auto_cascade.keep_in_mem = True
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit(self.x_train, self.y_train)
        model.predict(self.x_test)

    def test_fit_evaluate(self):
        print("test fit and evaluate")
        mgs, poolayer, concat_layer, auto_cascade = self._init()
        mgs.keep_in_mem = True
        auto_cascade.keep_in_mem = True
        model = Graph()
        model.add(mgs)
        model.add(poolayer)
        model.add(concat_layer)
        model.add(auto_cascade)
        model.fit(self.x_train, self.y_train)
        model.evaluate(self.x_test, self.y_test)


if __name__ == '__main__':
    unittest.main()

