# -*- coding:utf-8 -*-
"""
Higgs-boson dataset loading.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.datasets.dataset import get_dataset_dir
from sklearn.model_selection import train_test_split
from forestlayer.utils.log_utils import get_logger, get_logging_level
import numpy as np
import os.path as osp


def load_data(size='1K'):
    """
    Load HIGGS BOSON data.

    :param size: data size
                 '1K', '10K', '100K', '1M', default is '1K'.
    :return: x_train, y_train, x_test, y_test
    """
    data_path = osp.join("higgs", "HIGGS-{}.csv".format(size))
    # data_path = get_file(data_path,
    #                      origin='http://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz',
    #                      cache_subdir='higgs-boson',
    #                      cache_dir=get_dataset_dir())
    data_path = osp.join(get_dataset_dir(), data_path)
    data = np.loadtxt(data_path, delimiter=',')
    y = data[:, 0]
    X = data[:, 1:]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return x_train, y_train, x_test, y_test




