# -*- coding:utf-8 -*-
"""
Test Suites of layers.layer: module layer.summary_info.
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
from forestlayer.utils.log_utils import list2str
import ray
from keras.datasets import mnist
from forestlayer.datasets import uci_adult


class TestModelSummary(unittest.TestCase):
    def setUp(self):
        windows = [Window(7, 7, 2, 2, 0, 0),
                   Window(11, 11, 2, 2, 0, 0)]

        rf1 = ExtraRandomForestConfig(n_estimators=40, min_samples_leaf=10)
        rf2 = RandomForestConfig(n_estimators=40, min_samples_leaf=10)

        est_for_windows = [[rf1, rf2],
                           [rf1, rf2]]

        self.mgs = MultiGrainScanLayer(windows=windows,
                                  est_for_windows=est_for_windows,
                                  n_class=10)
        pools = [[MaxPooling(), MaxPooling()],
                 [MaxPooling(), MaxPooling()]]
        self.poolayer = PoolingLayer(pools=pools)
        self.concat_layer = ConcatLayer()
        self.est_configs = [
            ExtraRandomForestConfig(n_estimators=40),
            ExtraRandomForestConfig(n_estimators=40),
            RandomForestConfig(n_estimators=40),
            RandomForestConfig(n_estimators=40)
        ]

        self.cascade = CascadeLayer(est_configs=self.est_configs,
                               n_classes=10,
                               keep_in_mem=True,
                               data_save_dir=osp.join(get_data_save_base(), 'test_layer', 'cascade'))
        self.auto_cascade = AutoGrowingCascadeLayer(est_configs=self.est_configs,
                                               early_stopping_rounds=3,
                                               data_save_rounds=4,
                                               stop_by_test=True,
                                               n_classes=10,
                                               data_save_dir=osp.join(get_data_save_base(),
                                                                      'test_layer', 'auto_cascade'))

    def test_mgs_layer_summary(self):
        print(self.mgs.summary_info)

    def test_pool_layer_summary(self):
        print(self.poolayer.summary_info)

    def test_concat_layer_summary(self):
        print(self.concat_layer.summary_info)

    def test_cascade_layer_summary(self):
        print(self.cascade.summary_info)

    def test_auto_cascade_layer_summary(self):
        print(self.auto_cascade.summary_info)

    def test_graph_summary(self):
        model = Graph()
        model.add(self.mgs, self.poolayer, self.concat_layer, self.auto_cascade)
        model.summary()


if __name__ == '__main__':
    unittest.main()
