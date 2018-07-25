# -*- coding:utf-8 -*-
"""
Distributed UCI_ADULT Example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets import uci_adult
from forestlayer.estimators.estimator_configs import RandomForestConfig, ExtraRandomForestConfig
from forestlayer.layers.layer import AutoGrowingCascadeLayer
from forestlayer.layers.graph import Graph
import ray
import time


"""Stand alone mode"""
ray.init()

"""Cluster mode"""
# ray.init(redis_address="192.168.x.x:6379")

(x_train, y_train, x_test, y_test) = uci_adult.load_data()

start_time = time.time()

est_configs = [
    ExtraRandomForestConfig(n_jobs=-1),
    ExtraRandomForestConfig(n_jobs=-1),
    ExtraRandomForestConfig(n_jobs=-1),
    ExtraRandomForestConfig(n_jobs=-1),
    RandomForestConfig(n_jobs=-1),
    RandomForestConfig(n_jobs=-1),
    RandomForestConfig(n_jobs=-1),
    RandomForestConfig(n_jobs=-1)
]

auto_cascade = AutoGrowingCascadeLayer(est_configs=est_configs,
                                       early_stopping_rounds=4,
                                       n_classes=2,
                                       distribute=True,
                                       seed=0)

model = Graph()
model.add(auto_cascade)
model.summary()
model.fit_transform(x_train, y_train, x_test, y_test)

end_time = time.time()
print('time cost: {}'.format(end_time - start_time))

