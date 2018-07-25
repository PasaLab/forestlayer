# -*- coding:utf-8 -*-
"""
Storage Utilities, include cache utilities.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import os.path as osp
import os
import numpy as np
from ..backend.backend import get_base_dir
from collections import Iterable
import sys
try:
    import cPickle as pickle
except:
    import pickle


_DATA_SAVE_BASE = osp.join(get_base_dir(), 'run_data')

_MODEL_SAVE_BASE = osp.join(get_base_dir(), 'run_model')


def name2path(name):
    """
    Replace '/' in name by '-'
    """
    return name.replace("/", "-")


def is_path_exists(path):
    """
    Judge if path exists.

    :param path:
    :return:
    """
    return path is not None and osp.exists(path)


def check_dir(path):
    """
    Check directory existence, if not, create the directory.

    :param path:
    :return:
    """
    if path is None:
        return
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        try:
            os.makedirs(d)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise e
            print("Attempted to create '{}', but the directory already "
                  "exists.".format(d))
        # Change the directory permissions so others can use it. This is
        # important when multiple people are using the same machine. But now we don't need it.
        # os.chmod(d, 0o0777)


def numpy_to_disk_path(cache_dir, phase, data_name):
    data_path = osp.join(cache_dir, phase, name2path(data_name) + '.npy')
    return data_path


def output_disk_path(cache_dir, layer, phase, data_name, file_type='pkl'):
    assert file_type in ['pkl', 'txt'], 'Unknown file type.'
    data_path = osp.join(cache_dir, layer, phase, name2path(data_name) + '.{}'.format(file_type))
    return data_path


def load_disk_cache(data_path, file_type='pkl'):
    if file_type == 'pkl':
        with open(data_path, 'rb') as f:
            res = pickle.load(f)
    elif file_type == 'txt':
        res = np.loadtxt(data_path)
    else:
        raise ValueError('Unknown file type.')
    return res


def save_disk_cache(save_path, x2save, file_type='pkl'):
    if file_type == 'pkl':
        with open(save_path, "wb") as f:
            pickle.dump(x2save, f, pickle.HIGHEST_PROTOCOL)
    elif file_type == 'txt':
        np.savetxt(save_path, x2save, fmt='%s')
    else:
        raise ValueError('Unknown file type.')


def get_data_save_base():
    """
    Get data save base dir.

    :return:
    """
    global _DATA_SAVE_BASE
    _DATA_SAVE_BASE = osp.join(get_base_dir(), 'run_data')
    return _DATA_SAVE_BASE


def set_data_save_base(dir_path):
    """
    Set data save base dir.

    :param dir_path:
    :return:
    """
    global _DATA_SAVE_BASE
    _DATA_SAVE_BASE = dir_path
    check_dir(_DATA_SAVE_BASE)


def get_model_save_base():
    """
    Get model save base dir.

    :return:
    """
    global _MODEL_SAVE_BASE
    _MODEL_SAVE_BASE = osp.join(get_base_dir(), 'run_model')
    return _MODEL_SAVE_BASE


def set_model_save_base(dir_path):
    """
    Set model save base dir.
    """
    global _MODEL_SAVE_BASE
    _MODEL_SAVE_BASE = dir_path
    check_dir(_MODEL_SAVE_BASE)


def getmbof(x):
    if isinstance(x, np.ndarray):
        return "{:.2f}MB".format(x.itemsize * x.size / 1048576.0)
    return "{:.2f}MB".format(sys.getsizeof(x) / 1048576.0)


def getkbof(x):
    if isinstance(x, np.ndarray):
        return "{:.2f}KB".format(x.itemsize * x.size / 1024.0)
    return "{:.2f}KB".format(sys.getsizeof(x) / 1024.0)


def save_model(model, file_path, overrite=True):
    pass


def savetxt(filename, obj):
    if isinstance(obj, Iterable):
        with open(filename, 'w') as f:
            for x in obj:
                f.write("{}\n".format(x))
    else:
        raise NotImplementedError("obj type of {} is not supported!")
