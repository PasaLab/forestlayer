# -*- coding:utf-8 -*-
"""
Base dataset.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import os.path as osp
import os
from ..backend.backend import get_base_dir

_DATASET_DIR = osp.join(get_base_dir(), 'data')
_DATA_CACHE_DIR = osp.join(get_base_dir(), 'data-cache')


def get_dataset_dir():
    """
    Get data base dir, data_base_dir is used to store input data used to learning.

    :return: data_base_dir
    """
    global _DATASET_DIR
    # _DATASET_DIR = osp.join(get_base_dir(), 'data')
    if not osp.exists(_DATASET_DIR):
        os.makedirs(_DATASET_DIR)
    return _DATASET_DIR


def set_dataset_dir(dir_path):
    """
    Set data base dir, data_base_dir is used to store input data used to learning.

    :param dir_path: set data_base_dir to dir_path
    :return:
    """
    global _DATASET_DIR
    _DATASET_DIR = dir_path
    if not osp.exists(_DATASET_DIR):
        os.makedirs(_DATASET_DIR)


def get_data_cache_base():
    """
    Get data_cache_base_dir, which is used to store intermediate data during learning.

    :return: data_cache_base_dir
    """
    global _DATA_CACHE_DIR
    _DATA_CACHE_DIR = osp.join(get_base_dir(), 'data-cache')
    if not osp.exists(_DATA_CACHE_DIR):
        os.makedirs(_DATA_CACHE_DIR)
    return _DATA_CACHE_DIR


def set_data_cache_base(dir_path):
    """
    Set data_cache_base_dir, which is used to store intermediate data during learning.

    :param dir_path: set data_cache_base_dir to dir_path
    :return:
    """
    global _DATA_CACHE_DIR
    _DATA_CACHE_DIR = dir_path
    if not osp.exists(_DATA_CACHE_DIR):
        os.makedirs(_DATA_CACHE_DIR)
