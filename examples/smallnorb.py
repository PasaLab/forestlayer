# -*- coding:utf-8 -*-
"""
Small NORB Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import smallnorb
from forestlayer.datasets.dataset import get_dataset_dir, set_dataset_dir
from forestlayer.layers import Graph, MultiGrainScanLayer, PoolingLayer, ConcatLayer, AutoGrowingCascadeLayer
from forestlayer.estimators.estimator_configs import ExtraRandomForestConfig, RandomForestConfig
from forestlayer.layers.window import Window
from forestlayer.layers.factory import MeanPooling
from forestlayer.utils.log_utils import set_logging_level
from forestlayer.utils.storage_utils import get_data_save_base, get_model_save_base
import time
import os.path as osp

start_time = time.time()

set_logging_level('DEBUG')
# set_dataset_dir(osp.join(osp.expanduser('~'), 'forestlayer', 'data'))

(x_train, y_train, x_train_plus), (x_test, y_test, x_test_plus) = smallnorb.load_data(osp.join(get_dataset_dir(), "NORB"))

x_train = x_train[:100]
y_train = y_train[:100]
x_train_plus = x_train_plus[:100]
x_test = x_test[:50]
y_test = y_test[:50]
x_test_plus = x_test_plus[:50]

print('train shape and plus shape', x_train.shape, x_train_plus.shape)
print('test shape and plus shape', x_test.shape, x_test_plus.shape)

rf1 = ExtraRandomForestConfig(n_folds=3, n_jobs=-1, min_samples_leaf=10, max_features='auto')
rf2 = RandomForestConfig(n_folds=3, n_jobs=-1, min_samples_leaf=10)

windows = [Window(win_x=24, win_y=24, stride_x=2, stride_y=2, pad_x=0, pad_y=0),
           Window(34, 34, 2, 2),
           Window(48, 48, 2, 2)]

est_for_windows = [[rf1, rf2],
                   [rf1, rf2],
                   [rf1, rf2]]

data_save_dir = osp.join(get_data_save_base(), 'small_norb')
model_save_dir = osp.join(get_model_save_base(), 'small_norb')

mgs = MultiGrainScanLayer(windows=windows,
                          est_for_windows=est_for_windows,
                          n_class=10,
                          distribute=False,
                          keep_in_mem=False,
                          data_save_dir=data_save_dir,
                          cache_in_disk=True,
                          seed=0)

model = Graph()
model.add(mgs)
# model.add(pool)
# model.add(concatlayer)
# model.add(auto_cascade)
model.fit_transform(x_train, y_train, x_test, y_test)

print('time cost: {}'.format(time.time() - start_time))

