# -*- coding:utf-8 -*-
"""
IMDB Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import imdb
from forestlayer.layers.layer import AutoGrowingCascadeLayer
from forestlayer.estimators.estimator_configs import ExtraRandomForestConfig, RandomForestConfig
from forestlayer.utils.storage_utils import get_data_save_base, get_model_save_base
import os.path as osp
import time
import ray

"""Stand alone mode"""
ray.init()

"""Cluster mode"""
# ray.init(redis_address="192.168.x.x:6379")

(x_train, y_train, x_test, y_test) = imdb.load_data('tfidf')

start_time = time.time()

print('x_train.shape', x_train.shape, x_train.dtype)
print('x_test.shape', x_test.shape, x_test.dtype)

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

data_save_dir = osp.join(get_data_save_base(), 'imdb')
model_save_dir = osp.join(get_model_save_base(), 'imdb')

cascade = AutoGrowingCascadeLayer(est_configs=est_configs,
                                  early_stopping_rounds=4,
                                  stop_by_test=True,
                                  n_classes=2,
                                  data_save_dir=data_save_dir,
                                  model_save_dir=model_save_dir,
                                  distribute=True,
                                  dis_level=0,
                                  seed=0)

cascade.fit_transform(x_train, y_train, x_test, y_test)

print("Time cost: {}".format(time.time()-start_time))
