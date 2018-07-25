# -*- coding:utf-8 -*-
"""
Test Suites of layers.layer.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import numpy as np
import os.path as osp
import unittest
from forestlayer.layers.window import Window
from forestlayer.layers.factory import MaxPooling
from forestlayer.layers.layer import (MultiGrainScanLayer, PoolingLayer,
                                      ConcatLayer, CascadeLayer, AutoGrowingCascadeLayer)
from forestlayer.estimators.estimator_configs import RandomForestConfig, ExtraRandomForestConfig
from forestlayer.layers.graph import Graph
from forestlayer.utils.storage_utils import get_data_save_base
import ray
from keras.datasets import mnist
from forestlayer.datasets import uci_adult

ray.init()


class TestLayerForMNIST(unittest.TestCase):
    def setUp(self):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.reshape(x_train, (60000, -1, 28, 28))
        x_test = np.reshape(x_test, (10000, -1, 28, 28))
        self.x_train = x_train[:160, :, :, :]
        self.y_train = y_train[:160]
        self.x_test = x_test[:80, :, :, :]
        self.y_test = y_test[:80]
        print("================ MNIST ===================")
        print('X_train: ', self.x_train.shape, 'y: ', self.y_train.shape)
        print(' X_test: ', self.x_test.shape, 'y: ', self.y_test.shape)

    def _init(self, distribute=False):
        windows = [Window(7, 7, 2, 2, 0, 0),
                   Window(11, 11, 2, 2, 0, 0)]

        rf1 = ExtraRandomForestConfig(n_estimators=40, min_samples_leaf=10)
        rf2 = RandomForestConfig(n_estimators=40, min_samples_leaf=10)

        est_for_windows = [[rf1, rf2],
                           [rf1, rf2]]

        mgs = MultiGrainScanLayer(windows=windows,
                                  est_for_windows=est_for_windows,
                                  n_class=10,
                                  distribute=distribute)

        pools = [[MaxPooling(), MaxPooling()],
                 [MaxPooling(), MaxPooling()]]

        poolayer = PoolingLayer(pools=pools)

        concat_layer = ConcatLayer()

        self.est_configs = [
            ExtraRandomForestConfig(n_estimators=40),
            ExtraRandomForestConfig(n_estimators=40),
            RandomForestConfig(n_estimators=40),
            RandomForestConfig(n_estimators=40)
        ]

        cascade = CascadeLayer(est_configs=self.est_configs,
                               n_classes=10,
                               keep_in_mem=True,
                               data_save_dir=osp.join(get_data_save_base(), 'test_layer', 'cascade'))

        auto_cascade = AutoGrowingCascadeLayer(est_configs=self.est_configs,
                                               early_stopping_rounds=3,
                                               data_save_rounds=4,
                                               stop_by_test=True,
                                               n_classes=10,
                                               data_save_dir=osp.join(get_data_save_base(),
                                                                      'test_layer', 'auto_cascade'))

        return mgs, poolayer, concat_layer, cascade, auto_cascade

    def test_fit_transform(self):
        print('test fit_transform')

        mgs, poolayer, concat_layer, cascade, auto_cascade = self._init()

        res_train, res_test = mgs.fit_transform(self.x_train, self.y_train, self.x_test, self.y_test)

        res_train, res_test = poolayer.fit_transform(res_train, None, res_test, None)

        res_train, res_test = concat_layer.fit_transform(res_train, None, res_test)

        res_train, res_test = auto_cascade.fit_transform(res_train, self.y_train, res_test)

        print('res train / test shape: ', res_train.shape, res_test.shape)

    def test_fit(self):
        print('test fit')

        mgs, poolayer, concat_layer, cascade, auto_cascade = self._init()
        res_train = mgs.fit(self.x_train, self.y_train)

        res_train = poolayer.fit(res_train, self.y_train)

        res_train = concat_layer.fit(res_train, None)

        res_train = auto_cascade.fit(res_train, self.y_train)

    def test_predict(self):
        print('test predict')

        mgs, poolayer, concat_layer, cascade, auto_cascade = self._init()
        mgs.keep_in_mem = True
        res_train = mgs.fit(self.x_train, self.y_train)
        predicted = mgs.predict(self.x_test)

        res_train = poolayer.fit(res_train, self.y_train)
        predicted = poolayer.predict(predicted)

        res_train = concat_layer.fit(res_train, None)
        predicted = concat_layer.predict(predicted)
        auto_cascade.keep_in_mem = True
        res_train = auto_cascade.fit(res_train, self.y_train)
        auto_cascade.evaluate(predicted, self.y_test)

    def test_distribute_mgs_fit(self):
        mgs, _, _, _, _ = self._init(distribute=True)
        res_trains = mgs.fit(self.x_train, self.y_train)

    def test_non_dis_mgs_fit(self):
        mgs, _, _, _, _ = self._init(distribute=False)
        res_trains = mgs.fit(self.x_train, self.y_train)

    def test_speed_of_distribution_of_mgs_fit(self):
        import time
        start = time.time()
        self.test_distribute_mgs_fit()
        print('distributed mgs fit cost {} s'.format(time.time() - start))
        start = time.time()
        self.test_non_dis_mgs_fit()
        print('Non-distributed mgs fit cost {} s'.format(time.time() - start))

    def test_distribute_mgs_fit_transform(self):
        mgs, _, _, _, _ = self._init(distribute=True)
        res_trains = mgs.fit_transform(self.x_train, self.y_train, self.x_test, self.y_test)

    def test_non_dis_mgs_fit_transform(self):
        mgs, _, _, _, _ = self._init(distribute=False)
        res_trains = mgs.fit_transform(self.x_train, self.y_train, self.x_test, self.y_test)

    def test_speed_of_distribution_of_mgs_fit_transform(self):
        import time
        start = time.time()
        self.test_distribute_mgs_fit()
        print('distributed mgs fit cost {} s'.format(time.time() - start))
        start = time.time()
        self.test_non_dis_mgs_fit()
        print('Non-distributed mgs fit cost {} s'.format(time.time() - start))

    # def test_pool_layer_idempotency(self):
    #     mgs, poolayer, concat_layer, cascade, auto_cascade = self._init()
    #     x_train, y_train = self.x_train, self.y_train
    #     res_train = mgs.fit(x_train, y_train)
    #     before_str = list2str(res_train, 2)
    #     poolayer.fit(res_train, None)
    #     after_str = list2str(res_train, 2)
    #     assert before_str == after_str, '{} is not equal to {}'.format(before_str, after_str)
    #
    # def test_concat_layer_idempotency(self):
    #     mgs, poolayer, concat_layer, cascade, auto_cascade = self._init()
    #     x_train, y_train = self.x_train, self.y_train
    #     res_train = mgs.fit(x_train, y_train)
    #     before_str = list2str(res_train, 2)
    #     concat_layer.fit(res_train, None)
    #     after_str = list2str(res_train, 2)
    #     assert before_str == after_str, '{} is not equal to {}'.format(before_str, after_str)


class TestLayerForUCIADULT(unittest.TestCase):
    def setUp(self):
        (self.x_train, self.y_train, self.x_test, self.y_test) = uci_adult.load_data()
        print("=============== UCI ADULT ===============")
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')
        print(self.x_train.shape[1], 'features')

    def _init(self, distribute=False):
        self.est_configs = [
            ExtraRandomForestConfig(n_estimators=20),
            ExtraRandomForestConfig(n_estimators=20),
            RandomForestConfig(n_estimators=20),
            RandomForestConfig(n_estimators=20)
        ]

        gc = CascadeLayer(est_configs=self.est_configs,
                          n_classes=2,
                          data_save_dir=osp.join(get_data_save_base(), 'test_layer', 'cascade'),
                          distribute=distribute)

        agc = AutoGrowingCascadeLayer(est_configs=self.est_configs,
                                      early_stopping_rounds=2,
                                      stop_by_test=False,
                                      data_save_rounds=4,
                                      n_classes=2,
                                      data_save_dir=osp.join(get_data_save_base(),
                                                             'test_layer', 'auto_cascade'),
                                      distribute=distribute)
        return gc, agc

    def test_uci_graph(self):
        print('test uci_graph')
        gc, agc = self._init()
        model = Graph()
        model.add(agc)
        model.fit_transform(self.x_train, self.y_train, self.x_test, self.y_test)

    def test_fit_predict(self):
        print('test fit and predict')
        gc, agc = self._init()
        agc.keep_in_mem = True
        agc.fit(self.x_train, self.y_train)
        agc.evaluate(self.x_test, self.y_test)

    def test_graph_fit_evaluate(self):
        print('test fit and evaluate')
        gc, agc = self._init()
        agc.keep_in_mem = True
        model = Graph()
        model.add(agc)
        model.fit(self.x_train, self.y_train)
        model.evaluate(self.x_test, self.y_test)

    def test_graph_transform(self):
        print('test graph transform')
        gc, agc = self._init()
        agc.keep_in_mem = True
        model = Graph()
        model.add(agc)
        model.fit(self.x_train, self.y_train)
        model.transform(self.x_test)


if __name__ == '__main__':
    unittest.main()


