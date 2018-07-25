# -*- coding:utf-8 -*-
"""
UCI_YEAST Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import uci_iris
from forestlayer.estimators.estimator_configs import ExtraRandomForestConfig, RandomForestConfig
from forestlayer.layers.layer import AutoGrowingCascadeLayer
from forestlayer.layers.graph import Graph
from forestlayer.utils.storage_utils import get_data_save_base, get_model_save_base
import os.path as osp
import time

(x_train, y_train, x_test, y_test) = uci_iris.load_data()

print('x_train shape: {}'.format(x_train.shape))
print('x_test.shape: {}'.format(x_test.shape))

start_time = time.time()

est_configs = [
    ExtraRandomForestConfig(),
    ExtraRandomForestConfig(),
    ExtraRandomForestConfig(),
    ExtraRandomForestConfig(),
    RandomForestConfig(),
    RandomForestConfig(),
    RandomForestConfig(),
    RandomForestConfig()
]

data_save_dir = osp.join(get_data_save_base(), 'uci_iris')
model_save_dir = osp.join(get_model_save_base(), 'uci_iris')

auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs,
                                       early_stopping_rounds=4,
                                       n_classes=3,
                                       stop_by_test=False,
                                       data_save_dir=data_save_dir,
                                       model_save_dir=model_save_dir,
                                       distribute=False,
                                       dis_level=0,
                                       verbose_dis=False,
                                       seed=0)

model = Graph()
model.add(auto_cascade)
model.fit_transform(x_train, y_train, x_test, y_test)

print("time cost: {}".format(time.time() - start_time))
