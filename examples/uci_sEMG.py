# -*- coding:utf-8 -*-
"""
UCI_sEMG Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import uci_sEMG
from forestlayer.layers.factory import MGSWindow, MeanPooling
from forestlayer.layers.layer import MultiGrainScanLayer, AutoGrowingCascadeLayer, PoolingLayer, ConcatLayer
from forestlayer.layers.graph import Graph
from forestlayer.estimators.estimator_configs import ExtraRandomForestConfig, RandomForestConfig, Basic4x2

x_train, y_train, x_test, y_test = uci_sEMG.load_data()

print("x_train.shape = {}, y_train.shape = {}".format(x_train.shape, y_train.shape))
print("x_test.shape = {}, y_test.shape = {}".format(x_test.shape, y_test.shape))

windows = [MGSWindow((1, 157)),
           MGSWindow((1, 375)),
           MGSWindow((1, 750))]

rf1 = ExtraRandomForestConfig(n_folds=3, min_samples_leaf=10, max_features='auto')
rf2 = RandomForestConfig(n_folds=3, min_samples_leaf=10)

est_for_windows = [[rf1, rf2],
                   [rf1, rf2],
                   [rf1, rf2]]

mgs = MultiGrainScanLayer(windows=windows,
                          est_for_windows=est_for_windows,
                          n_class=6)

pools = [
    [MeanPooling(), MeanPooling()],
    [MeanPooling(), MeanPooling()],
    [MeanPooling(), MeanPooling()]
]

pool_layer = PoolingLayer(pools=pools)

concat_layer = ConcatLayer()

est_configs = Basic4x2()

auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs,
                                       early_stopping_rounds=4,
                                       n_classes=6,
                                       look_index_cycle=[[0, 1], [2, 3], [4, 5]])

model = Graph()
model.add(mgs)
model.add(pool_layer)
# model.add(concat_layer)
model.add(auto_cascade)
model.fit_transform(x_train, y_train, x_test, y_test)



