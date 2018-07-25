# -*- coding:utf-8 -*-
"""
Log Utilities.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import os
import os.path as osp
import time
import logging
from ..backend.backend import get_base_dir
from .storage_utils import check_dir

_LOG_BASE = osp.join(get_base_dir(), 'log')


def get_logging_base():
    """
    Get logging base dir, which is used to store log data.
    Default logging base is ~/.forestlayer/log
    logging base contains one or more logging dir.

    :return:
    """
    global _LOG_BASE
    return _LOG_BASE


def set_logging_base(dir_path):
    """
    Set logging base dir, which is used to store log data.
    Default logging base is ~/.forestlayer/log
    logging base contains one or more logging dir.

    :param dir_path:
    :return:
    """
    global _LOG_BASE
    _LOG_BASE = dir_path
    check_dir(_LOG_BASE)


logging.basicConfig(format="[ %(asctime)s][%(module)s.%(funcName)s] %(message)s")

DEFAULT_LEVEL = logging.INFO
OLD_LOGGING_DIR = osp.join(get_logging_base(), "forestlayer")
DEFAULT_LOGGING_DIR = osp.join(get_logging_base(), "forestlayer")
fh = None
str2level = {'DEBUG': logging.DEBUG,
             'WARN': logging.WARN,
             'INFO': logging.INFO,
             'ERROR': logging.ERROR
             }


def strftime(t=None):
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t or time.time()))


def init_fh():
    """
    Initialize log file handler.

    :return:
    """
    global fh, DEFAULT_LOGGING_DIR, OLD_LOGGING_DIR
    if fh is not None and DEFAULT_LOGGING_DIR == OLD_LOGGING_DIR:
        return
    if DEFAULT_LOGGING_DIR is None:
        return
    OLD_LOGGING_DIR = DEFAULT_LOGGING_DIR
    if not osp.exists(DEFAULT_LOGGING_DIR):
        os.makedirs(DEFAULT_LOGGING_DIR)
    logging_path = osp.join(DEFAULT_LOGGING_DIR, strftime() + ".log")
    fh = logging.FileHandler(logging_path)
    fh.setFormatter(logging.Formatter("[ %(asctime)s][%(module)s.%(funcName)s] %(message)s"))


def set_logging_level(default_level):
    """
    Set logging level.
    Accept parameter: logging.WARN, logging.DEBUG, logging.ERROR, logging.INFO(default)
     or their string representation. 'WARN', 'DEBUG', 'ERROR', 'INFO'.

    :param default_level:
    :return:
    """
    global DEFAULT_LEVEL
    if isinstance(default_level, str):
        default_level = str2level[default_level]
    DEFAULT_LEVEL = default_level


def get_logging_level():
    """
    Get logging level.

    :return:
    """
    global DEFAULT_LEVEL
    return DEFAULT_LEVEL


def set_logging_dir(default_logging_dir):
    """
    Set logging dir.
    Default logging dir is ~/.forestlayer/log/forestlayer
    NOTE: Now we recommend not to modify logging dir, because it might cause unknown in-identity.

    :param default_logging_dir: ~/.forestlayer/log/forestlayer
    :return:
    """
    global DEFAULT_LOGGING_DIR
    DEFAULT_LOGGING_DIR = default_logging_dir


def get_logging_dir():
    """
    Get logging dir.
    Default logging dir is ~/.forestlayer/log/forestlayer

    :return:
    """
    global DEFAULT_LOGGING_DIR
    return DEFAULT_LOGGING_DIR


def get_logger(name="forestlayer", level=None):
    """
    Get a logger.

    :param name:
    :param level:
    :return:
    """
    level = level or DEFAULT_LEVEL
    logger = logging.getLogger(name)
    logger.setLevel(level)
    init_fh()
    if fh is not None:
        logger.addHandler(fh)
    return logger


def list2str(lis, dim):
    """
    List to String.

    :param lis:
    :param dim:
    :return:
    """
    result = "["
    for l in lis:
        if dim == 1:
            result += '{},'.format(l.shape)
        elif dim == 2:
            result += '['
            for j in l:
                result += '{} '.format(j.shape)
            result += '], '
        elif dim == 3:
            result += '['
            for j in l:
                result += '['
                for k in j:
                    result += '{} '.format(k.shape)
                result += '], '
            result += '], '
        else:
            raise NotImplementedError
    result += ']'
    return result


def list_type2str(lis, dim):
    """
    List to String.

    :param lis:
    :param dim:
    :return:
    """
    result = "["
    for l in lis:
        if dim == 1:
            result += '{},'.format(l)
        elif dim == 2:
            result += '['
            for j in l:
                result += '{} '.format(j)
            result += '], '
        elif dim == 3:
            result += '['
            for j in l:
                result += '['
                for k in j:
                    result += '{} '.format(k)
                result += '], '
            result += '], '
        else:
            raise NotImplementedError
    result += ']'
    return result

