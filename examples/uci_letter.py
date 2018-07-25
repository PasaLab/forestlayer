# -*- coding:utf-8 -*-
"""
UCI_LETTER Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import uci_letter
from forestlayer.layers import Graph, AutoGrowingCascadeLayer
from forestlayer.utils.storage_utils import get_data_save_base
from forestlayer.estimators.estimator_configs import ExtraRandomForestConfig, RandomForestConfig
import time
import os.path as osp

start_time = time.time()

(X_train, y_train, X_test, y_test) = uci_letter.load_data()


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

agc = AutoGrowingCascadeLayer(est_configs=est_configs,
                              early_stopping_rounds=4,
                              stop_by_test=True,
                              n_classes=26,
                              data_save_dir=osp.join(get_data_save_base(), 'uci_letter', 'auto_cascade'),
                              keep_in_mem=False,
                              distribute=False,
                              dis_level=0,
                              verbose_dis=False,
                              seed=0)

model = Graph()
model.add(agc)
model.fit_transform(X_train, y_train, X_test, y_test)

end_time = time.time()

print("Time cost: {}".format(end_time-start_time))


