# -*- coding:utf-8 -*-
"""
UCI_sEMG dataset loading.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import numpy as np
import scipy.io as sio
import os.path as osp
from .dataset import get_dataset_dir
from keras.utils.data_utils import get_file
from sklearn.model_selection import train_test_split

move2label = dict()
move2label['spher_ch1'] = 0
move2label['spher_ch2'] = 0
move2label['tip_ch1'] = 1
move2label['tip_ch2'] = 1
move2label['palm_ch1'] = 2
move2label['palm_ch2'] = 2
move2label['lat_ch1'] = 3
move2label['lat_ch2'] = 3
move2label['cyl_ch1'] = 4
move2label['cyl_ch2'] = 4
move2label['hook_ch1'] = 5
move2label['hook_ch2'] = 5


def load_data():
    data_dir_path = 'sEMG_Basic_Hand_movements_upatras.zip'
    get_file(data_dir_path,
             origin='http://archive.ics.uci.edu/ml/machine-learning-databases/'
                    '00313/sEMG_Basic_Hand_movements_upatras.zip',
             cache_subdir='uci_sEMG',
             cache_dir=get_dataset_dir(),
             extract=True)
    data_base_1 = osp.join(get_dataset_dir(), 'uci_sEMG', 'Database 1')
    x, y = None, None
    for mat_name in ('female_1.mat', 'female_2.mat', 'female_3.mat', 'male_1.mat', 'male_2.mat'):
        x_cur, y_cur = load_mat(osp.join(data_base_1, mat_name))
        if x is None:
            x, y = x_cur, y_cur
        else:
            x = np.vstack((x, x_cur))
            y = np.concatenate((y, y_cur))
    n_data = x.shape[0]
    train_index, test_index = train_test_split(range(n_data), random_state=0, train_size=0.7, test_size=0.3, stratify=y)
    return (x[train_index][:, np.newaxis, :, np.newaxis], y[train_index],
            x[test_index][:, np.newaxis, :, np.newaxis], y[test_index])


def load_mat(mat_path):
    x, y = None, None
    data = sio.loadmat(mat_path)
    for k in sorted(move2label.keys()):
        x_cur = data[k]
        y_cur = np.full(x_cur.shape[0], move2label[k], dtype=np.int32)
        if x is None:
            x, y = x_cur, y_cur
        else:
            x = np.vstack((x, x_cur))
            y = np.concatenate((y, y_cur))
    return x, y

