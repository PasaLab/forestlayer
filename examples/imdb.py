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

(x_train, y_train, x_test, y_test) = imdb.load_data('tfidf')

print('x_train.shape', x_train.shape)
print('x_test.shape', x_test.shape)

# x_train = x_train[:10]
# y_train = y_train[:10]
# x_test = x_test[:5]
# y_test = y_test[:5]

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

data_save_dir = osp.join(get_data_save_base(), 'fashion_mnist')
model_save_dir = osp.join(get_model_save_base(), 'fashion_mnist')

cascade = AutoGrowingCascadeLayer(est_configs=est_configs,
                                  early_stopping_rounds=4,
                                  stop_by_test=True,
                                  n_classes=2,
                                  data_save_dir=data_save_dir,
                                  model_save_dir=model_save_dir)

cascade.fit_transform(x_train, y_train, x_test, y_test)
