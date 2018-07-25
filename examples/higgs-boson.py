# -*- coding:utf-8 -*-
"""
Higgs Boson Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import higgs_boson
from forestlayer.layers import Graph, AutoGrowingCascadeLayer
from forestlayer.utils.storage_utils import get_data_save_base
from forestlayer.estimators.estimator_configs import RandomForestConfig, ExtraRandomForestConfig
import time
import numpy as np
import ray
import sys
import os.path as osp

# ray.init()

if len(sys.argv) > 1:
    sz = sys.argv[1]
else:
    sz = "1K"

start_time = time.time()
(x_train, y_train, x_test, y_test) = higgs_boson.load_data(sz)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train.shape[1], 'features')

est_configs = [
    RandomForestConfig(),
    RandomForestConfig(),
    RandomForestConfig(),
    RandomForestConfig(),
    ExtraRandomForestConfig(),
    ExtraRandomForestConfig(),
    ExtraRandomForestConfig(),
    ExtraRandomForestConfig()
]

data_save_dir = osp.join(get_data_save_base(), 'higgs', 'higgs-{}'.format(sz))

agc = AutoGrowingCascadeLayer(est_configs=est_configs,
                              early_stopping_rounds=4,
                              max_layers=0,
                              stop_by_test=True,
                              n_classes=2,
                              data_save_rounds=0,
                              data_save_dir=data_save_dir,
                              keep_in_mem=False,
                              distribute=False,
                              dis_level=2,
                              verbose_dis=True,
                              seed=0)

model = Graph()
model.add(agc)
model.fit_transform(x_train, y_train, x_test, y_test)

end_time = time.time()
print('time cost: {}'.format(end_time - start_time))

