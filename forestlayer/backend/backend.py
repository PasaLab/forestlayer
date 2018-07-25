# -*- coding:utf-8 -*-
"""
Scikit-learn and Ray as backend.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import ray
import os.path as osp
import os
import numpy as np
import time

_BASE_DIR = osp.expanduser(osp.join('~', '.forestlayer'))

if not osp.exists(_BASE_DIR):
    os.makedirs(_BASE_DIR)

if not os.access(_BASE_DIR, os.W_OK):
    _BASE_DIR = osp.join('/tmp', '.forestlayer')
    if not osp.exists(_BASE_DIR):
        os.makedirs(_BASE_DIR)


def set_base_dir(dir_path):
    """
    Set base_dir, which is data_base_dir, data_cache_base_dir, model_save_dir, data_save_dir base on.

    :param dir_path: set the base_dir to dir_path
    :return:
    """
    global _BASE_DIR
    _BASE_DIR = dir_path
    if not osp.exists(_BASE_DIR):
        os.makedirs(_BASE_DIR)
    if not os.access(_BASE_DIR, os.W_OK):
        _BASE_DIR = osp.join('/tmp', '.forestlayer')
        if not osp.exists(_BASE_DIR):
            os.makedirs(_BASE_DIR)


def get_base_dir():
    """
    Get base_dir, which is data_base_dir, data_cache_base_dir, model_save_dir, data_save_dir base on.

    :return:
    """
    global _BASE_DIR
    return _BASE_DIR


def pb2pred(y_proba):
    """
    Probability to prediction, just one numpy.argmax

    :param y_proba: probability
    :return:
    """
    y_pred = np.argmax(y_proba.reshape((-1, y_proba.shape[-1])), 1)
    return y_pred


@ray.remote
def stat_nodes(sleep_time=0.001):
    time.sleep(sleep_time)
    return ray.services.get_node_ip_address()


def get_num_nodes(sleep_time=0.001, num_task=1000):
    # Get a list of the IP addresses of the nodes that have joined the cluster.
    nodes = set(ray.get([stat_nodes.remote(sleep_time) for _ in range(num_task)]))
    return list(nodes), len(nodes)
