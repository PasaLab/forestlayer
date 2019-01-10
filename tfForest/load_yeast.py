# -*- coding:utf-8 -*-
"""
UCI_sEMG dataset loading.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import numpy as np
import re
from sklearn.model_selection import train_test_split


def load_yeast():
    """
    Load YEAST data.

    :return:
    """
    data_path = "../DeepForestTF_Data/yeast.data"
    label_path = "../DeepForestTF_Data/yeast.label"
    id2label = {}
    label2id = {}
    # id to label mapping and label to id mapping.
    with open(label_path) as f:
        for row in f:
            cols = row.strip().split(" ")
            id2label[int(cols[0])] = cols[1]
            label2id[cols[1]] = int(cols[0])
    with open(data_path) as f:
        rows = f.readlines()
    n_datas = len(rows)
    X = np.zeros((n_datas, 8), dtype=np.float32)
    y = np.zeros(n_datas, dtype=np.int32)

    for i, row in enumerate(rows):
        cols = re.split(" +", row.strip())
        X[i, :] = list(map(float, cols[1:1+8]))
        y[i] = label2id[cols[-1]]
    train_idx, test_idx = train_test_split(range(n_datas), random_state=0, train_size=0.7, stratify=y)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


