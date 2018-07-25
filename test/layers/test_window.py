# -*- coding:utf-8 -*-
"""
Test Suites of layers.window.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import numpy as np
from forestlayer.layers.window import Window, Pooling
from keras.datasets import mnist


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = np.reshape(x_train, (60000, -1, 28, 28))

print('x_train: ', X.shape)

win = Window(7, 7, 2, 2, 0, 0)
x = win.fit_transform(X)

# print('x_win: ', x.shape)
assert x.shape == (60000, 11, 11, 49)

pool = Pooling(2, 2, "mean")
y = pool.fit_transform(x.transpose(0, 3, 1, 2))

# print('pooled shape: ', y.shape)
assert y.shape == (60000, 49, 6, 6)
