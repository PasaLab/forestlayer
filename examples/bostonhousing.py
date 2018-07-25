# -*- coding:utf-8 -*-
"""
Boston Housing Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from keras.datasets import boston_housing
from forestlayer.estimators.estimator_configs import ExtraRandomForestConfig, RandomForestConfig, GBDTConfig
from forestlayer.layers.layer import AutoGrowingCascadeLayer

(x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.25)

print("x_train: {}".format(x_train.shape))
print("x_test: {}".format(x_test.shape))

est_configs = [
    RandomForestConfig(),
    ExtraRandomForestConfig(),
    GBDTConfig()
]

cascade = AutoGrowingCascadeLayer(task='regression',
                                  est_configs=est_configs,
                                  early_stopping_rounds=3,
                                  keep_in_mem=False)

cascade.fit_transform(x_train, y_train, x_test, y_test)


