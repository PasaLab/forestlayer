# -*- coding:utf-8 -*-
"""
Regression task example.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from forestlayer.estimators.estimator_configs import RandomForestConfig, ExtraRandomForestConfig
from forestlayer.layers.layer import AutoGrowingCascadeLayer

rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

est_configs = [
    RandomForestConfig(),
    ExtraRandomForestConfig()
]

cascade = AutoGrowingCascadeLayer(task='regression', est_configs=est_configs, early_stopping_rounds=3, keep_in_mem=True)
cascade.fit(X, y)
y1 = cascade.predict(X)
y1 = y1.reshape(-1)

abr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                        n_estimators=300, random_state=rng)
abr.fit(X, y)
y2 = abr.predict(X)

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y2, c="g", label="n_estimators=300", linewidth=1.5)
plt.plot(X, y1, c="r", label="n_estimators=500", linewidth=1.5)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Regression")
plt.legend()
plt.show()
