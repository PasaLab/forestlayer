# -*- coding:utf-8 -*-
"""
Distributed MNIST Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.layers import Graph, MultiGrainScanLayer, PoolingLayer, ConcatLayer, AutoGrowingCascadeLayer
from forestlayer.estimators.estimator_configs import ExtraRandomForestConfig, RandomForestConfig
from forestlayer.layers.window import Window
from forestlayer.layers.factory import MeanPooling
from forestlayer.utils.storage_utils import get_data_save_base, get_model_save_base, getmbof
from keras.datasets import mnist
import os.path as osp
import ray
import time

start_time = time.time()

"""Stand alone mode"""
ray.init()

"""Cluster mode"""
# ray.init(redis_address="192.168.x.x:6379")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, -1, 28, 28)
x_test = x_test.reshape(10000, -1, 28, 28)

# small data for example.
x_train = x_train[:600, :, :, :]
x_test = x_test[:300, :, :, :]
y_train = y_train[:600]
y_test = y_test[:300]

print(x_train.shape, 'train', x_train.dtype, getmbof(x_train))
print(x_test.shape, 'test', x_test.dtype, getmbof(x_test))

rf1 = ExtraRandomForestConfig(n_jobs=-1, min_samples_leaf=10, max_features="auto")
rf2 = RandomForestConfig(n_jobs=-1, min_samples_leaf=10)

windows = [Window(win_x=7, win_y=7, stride_x=2, stride_y=2, pad_x=0, pad_y=0),
           Window(10, 10, 2, 2),
           Window(13, 13, 2, 2)]

est_for_windows = [[rf1, rf2],
                   [rf1, rf2],
                   [rf1, rf2]]

mgs = MultiGrainScanLayer(windows=windows,
                          est_for_windows=est_for_windows,
                          n_class=10,
                          distribute=True,
                          dis_level=2,
                          keep_in_mem=False)

pools = [[MeanPooling(2, 2), MeanPooling(2, 2)],
         [MeanPooling(2, 2), MeanPooling(2, 2)],
         [MeanPooling(2, 2), MeanPooling(2, 2)]]

pool = PoolingLayer(pools=pools)

concatlayer = ConcatLayer()

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

data_save_dir = osp.join(get_data_save_base(), 'mnist')
model_save_dir = osp.join(get_model_save_base(), 'mnist')

auto_cascade = AutoGrowingCascadeLayer(name='auto-cascade',
                                       est_configs=est_configs,
                                       early_stopping_rounds=4,
                                       stop_by_test=True,
                                       n_classes=10,
                                       data_save_dir=data_save_dir,
                                       model_save_dir=model_save_dir,
                                       distribute=True,
                                       dis_level=2)

model = Graph()
model.add(mgs)
model.add(pool)
model.add(concatlayer)
model.add(auto_cascade)
model.fit_transform(x_train, y_train, x_test, y_test)

print('time cost: {}'.format(time.time() - start_time))
