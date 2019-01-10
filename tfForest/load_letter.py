# -*- coding:utf-8 -*-
"""
UCI_LETTER dataset loading.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from keras.utils.data_utils import get_file
import numpy as np


def load_letter():
    """
    Load UCI LETTER data, if not exists in data_base_dir, download it and put into data_base_dir.

    :return: x_train, y_train, x_test, y_test
    """
    data_path = "../DeepForestTF_Data/letter-recognition.data"
    with open(data_path) as f:
        rows = [row.strip().split(',') for row in f.readlines()]
    n_datas = len(rows)
    X = np.zeros((n_datas, 16), dtype=np.float32)
    y = np.zeros(n_datas, dtype=np.int32)
    for i, row in enumerate(rows):
        X[i, :] = list(map(float, row[1:]))
        y[i] = ord(row[0]) - ord('A')
    x_train, y_train = X[:16000], y[:16000]
    x_test, y_test = X[16000:], y[16000:]
    return x_train, x_test, y_train, y_test

