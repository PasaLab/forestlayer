# -*- coding:utf-8 -*-
"""
UCI Iris dataset loading.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import os.path as osp
import pandas as pd
from sklearn.model_selection import train_test_split
from .dataset import get_dataset_dir
from keras.utils.data_utils import get_file
from ..utils.log_utils import get_logger, get_logging_level


def load_data():
    """
    Load UCI Iris data. If not exists in data/, download it and put into data_base_dir.

    :return: X_train, y_train, X_test, y_test
    """
    data_path = 'iris.data'
    data_path = get_file(data_path,
                         origin='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                         cache_subdir='uci_iris',
                         cache_dir=get_dataset_dir())
    df = pd.read_csv(data_path, header=None)
    class2num = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    df[4] = df[4].apply(lambda x: class2num[x])
    X = df.values[:, :-1]
    y = df[4].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    load_data()
