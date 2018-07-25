# -*- coding:utf-8 -*-
"""
Test Suites of utils.log_utils.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.utils.log_utils import get_logging_base, get_logger, set_logging_base, set_logging_level,\
    get_logging_level, set_logging_dir, get_logging_dir
import os.path as osp
import logging

print('log_base = {}'.format(get_logging_base()))

set_logging_base(osp.expanduser(osp.join('~', 'forestlayer', 'log')))

print('after set, log_base = {}'.format(get_logging_base()))

LOGGER = get_logger()

print('log level = {}'.format(get_logging_level()))

set_logging_level(logging.WARN)

print('after update, log level = {}'.format(get_logging_level()))

print('log dir = {}'.format(get_logging_dir()))

set_logging_dir(osp.expanduser(osp.join('~', 'forestlayer', 'log2')))

print('after update, log dir = {}'.format(get_logging_dir()))

