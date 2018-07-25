# -*- coding:utf-8 -*-
"""
Factory methods to Layers.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from .layer import PoolingLayer
from ..estimators.estimator_configs import ExtraRandomForestConfig, RandomForestConfig
from .window import Window, Pooling


def MGSWindow(wins=(7, 7), strides=(1, 1), pads=(0, 0)):
    """
    Multi-grain Scan window.

    :param wins:
    :param strides:
    :param pads:
    :return:
    """
    assert len(wins) == len(strides) == len(pads), 'wins({}), strides({}), pads({}) SHAPE in-conform'.format(
        len(wins), len(strides), len(pads))
    assert len(wins) >= 2, 'len(wins) = {}, should >= 2'.format(len(wins))
    return Window(win_x=wins[0], win_y=wins[1], stride_x=strides[0], stride_y=strides[1], pad_x=pads[0], pad_y=pads[1])


def MaxPooling(win_x=2, win_y=2):
    """
    Max Pooling.

    :param win_x:
    :param win_y:
    :return:
    """
    assert win_x is not None and win_x >= 1, 'win_x = {}, invalid!'.format(win_x)
    assert win_y is not None and win_y >= 1, 'win_y = {}, invalid!'.format(win_y)
    return Pooling(win_x=win_x, win_y=win_y, pool_strategy="max")


def MeanPooling(win_x=2, win_y=2):
    """
    Mean Pooling.

    :param win_x:
    :param win_y:
    :return:
    """
    assert win_x is not None and win_x >= 1, 'win_x = {}, invalid!'.format(win_x)
    assert win_y is not None and win_y >= 1, 'win_y = {}, invalid!'.format(win_y)
    return Pooling(win_x=win_x, win_y=win_y, pool_strategy="mean")


def MaxPooling2x2Layer(win_x=2, win_y=2):
    """
    MaxPooling Layer with 2x2 pools, each pooling is MaxPooling(win_x, win_y).

    :param win_x:
    :param win_y:
    :return:
    """
    pools = [[MaxPooling(win_x, win_y), MaxPooling(win_x, win_y)],
             [MaxPooling(win_x, win_y), MaxPooling(win_x, win_y)]]
    return PoolingLayer(pools=pools)


def MeanPooling2x2Layer(win_x=2, win_y=2):
    """
    MeanPooling Layer with 2x2 pools, each pooling is MeanPooling(win_x, win_y).

    :param win_x:
    :param win_y:
    :return:
    """
    pools = [[MeanPooling(win_x, win_y), MeanPooling(win_x, win_y)],
             [MeanPooling(win_x, win_y), MeanPooling(win_x, win_y)]]
    return PoolingLayer(pools=pools)


def EstForWin2x2(**kwargs):
    """
    Estimator For Windows with 2x2 estimators, one CompletelyRandomForest, one RandomForest.

    :param kwargs:
    :return:
    """
    rf1 = ExtraRandomForestConfig(**kwargs)
    rf2 = RandomForestConfig(**kwargs)
    est_for_windows = [[rf1, rf2], [rf1, rf2]]
    return est_for_windows








