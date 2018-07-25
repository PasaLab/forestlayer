# -*- coding:utf-8 -*-
"""
CIFAR10 Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.layers import Graph, MultiGrainScanLayer, PoolingLayer, ConcatLayer, AutoGrowingCascadeLayer
from forestlayer.estimators.estimator_configs import ExtraRandomForestConfig, RandomForestConfig
from forestlayer.layers.window import Window
from forestlayer.layers.factory import MeanPooling
from forestlayer.utils.storage_utils import get_data_save_base, get_model_save_base
from keras.datasets import cifar10
import os.path as osp
import time

start_time = time.time()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.transpose((0, 3, 1, 2))
x_test = x_test.transpose((0, 3, 1, 2))
y_train = y_train.reshape((y_train.shape[0]))
y_test = y_test.reshape((y_test.shape[0]))

x_train = x_train.reshape(50000, -1, 32, 32)
x_test = x_test.reshape(10000, -1, 32, 32)
x_train = x_train[:200, :, :, :]
x_test = x_test[:100, :, :, :]
y_train = y_train[:200]
y_test = y_test[:100]

print(x_train.shape, y_train.shape, 'train')
print(x_test.shape, y_test.shape, 'test')

rf1 = ExtraRandomForestConfig(n_folds=3, n_jobs=-1, min_samples_leaf=10, max_features='auto')
rf2 = RandomForestConfig(n_folds=3, n_jobs=-1, min_samples_leaf=10)

windows = [Window(win_x=8, win_y=8, stride_x=2, stride_y=2, pad_x=0, pad_y=0),
           Window(11, 11, 2, 2),
           Window(16, 16, 2, 2)]

est_for_windows = [[rf1, rf2],
                   [rf1, rf2],
                   [rf1, rf2]]

data_save_dir = osp.join(get_data_save_base(), 'cifar10')
model_save_dir = osp.join(get_model_save_base(), 'cifar10')

mgs = MultiGrainScanLayer(windows=windows,
                          est_for_windows=est_for_windows,
                          n_class=10,
                          distribute=False,
                          dis_level=0,
                          keep_in_mem=False,
                          data_save_dir=data_save_dir,
                          cache_in_disk=False,
                          seed=0)

pools = [[MeanPooling(2, 2), MeanPooling(2, 2)],
         [MeanPooling(2, 2), MeanPooling(2, 2)],
         [MeanPooling(2, 2), MeanPooling(2, 2)]]

pool = PoolingLayer(pools=pools,
                    cache_in_disk=False,
                    data_save_dir=data_save_dir)

concatlayer = ConcatLayer(cache_in_disk=False,
                          data_save_dir=data_save_dir)

est_configs = [
    ExtraRandomForestConfig(n_estimators=1000),
    ExtraRandomForestConfig(n_estimators=1000),
    ExtraRandomForestConfig(n_estimators=1000),
    ExtraRandomForestConfig(n_estimators=1000),
    RandomForestConfig(n_estimators=1000),
    RandomForestConfig(n_estimators=1000),
    RandomForestConfig(n_estimators=1000),
    RandomForestConfig(n_estimators=1000)
]

auto_cascade = AutoGrowingCascadeLayer(name='auto-cascade',
                                       est_configs=est_configs,
                                       early_stopping_rounds=8,
                                       look_index_cycle=[
                                           [0],
                                           [1],
                                           [2],
                                           [0, 1, 2]
                                       ],
                                       stop_by_test=True,
                                       n_classes=10,
                                       data_save_dir=data_save_dir,
                                       data_save_rounds=5,
                                       model_save_dir=model_save_dir,
                                       distribute=False,
                                       seed=0)

model = Graph()
model.add(mgs)
model.add(pool)
model.add(concatlayer)
model.add(auto_cascade)
model.fit_transform(x_train, y_train, x_test, y_test)

print('time cost: {}'.format(time.time() - start_time))
