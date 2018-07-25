# -*- coding:utf-8 -*-
"""
Multi-grain scan windows.
This code is partly borrowed from Ji.Feng.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import numpy as np
from joblib import Parallel, delayed
from ..utils.log_utils import get_logger, get_logging_level

LOGGER = get_logger('multi-grain scan window')


def get_windows_channel(x, x_win, des_id, nw, nh, win_x, win_y, stride_x, stride_y):
    """
    X: N x C x H x W
    X_win: N x nc x nh x nw
    (k, di, dj) in range(X.channel, win_y, win_x)
    Note: This code is borrowed from Ji. Feng.
    """
    # des_id = (k * win_y + di) * win_x + dj
    dj = des_id % win_x
    di = des_id // win_x % win_y
    k = des_id // win_x // win_y
    src = x[:, k, di:di+nh*stride_y:stride_y, dj:dj+nw*stride_x:stride_x].ravel()
    des = x_win[des_id, :]
    np.copyto(des, src)


def get_windows(x, win_x, win_y, stride_x=1, stride_y=1, pad_x=0, pad_y=0):
    """
    Parallelling get_windows

    :param x: numpy.ndarray. n x c x h x w
    :param win_x:
    :param win_y:
    :param stride_x:
    :param stride_y:
    :param pad_x:
    :param pad_y:
    :return: numpy.ndarray. n x nh x nw x nc,
                            nc = win_y * win_x * c,
                            nh = (h - win_y) / stride_y + 1,
                            nw = (w - win_x) / stride_x + 1
    """
    assert len(x.shape) == 4, 'len(X.shape) should be 4, but {}'.format(x.shape)
    n, c, h, w = x.shape
    if pad_y > 0:
        x = np.concatenate((x, np.zeros((n, c, pad_y, w), dtype=x.dtype)), axis=2)
        x = np.concatenate((np.zeros((n, c, pad_y, w), dtype=x.dtype), x), axis=2)
    n, c, h, w = x.shape
    if pad_x > 0:
        x = np.concatenate((x, np.zeros((n, c, h, pad_x), dtype=x.dtype)), axis=3)
        x = np.concatenate((np.zeros((n, c, h, pad_x), dtype=x.dtype), x), axis=3)
    n, c, h, w = x.shape
    nc = win_y * win_x * c
    nh = (h - win_y) / stride_y + 1
    nw = (w - win_x) / stride_x + 1
    x_win = np.empty((nc, n * nh * nw), dtype=x.dtype)
    # LOGGER.debug("get_windows_start: X.shape={}, X_win.shape={}, nw={}, nh={}, channel={},"
    #              " win=({}x{}), stride=({}x{})".format(
    #               x.shape, x_win.shape, nw, nh, c, win_x, win_y, stride_x, stride_y))
    Parallel(n_jobs=-1, backend="threading", verbose=0)(
            delayed(get_windows_channel)(x, x_win, des_id, nw, nh, win_x, win_y, stride_x, stride_y)
            for des_id in range(c * win_x * win_y))
    x_win = x_win.transpose((1, 0))
    x_win = x_win.reshape((n, nh, nw, nc))
    LOGGER.info("got_windows: X.shape={}, X_win.shape={}".format(x.shape, x_win.shape))
    return x_win


class Window(object):
    """
    A window is rectangular that includes input pixels and move along the axis to extract features of images.
    A window in deep forest is not same as in traditional deep convolution neural networks, a window here has not
     weight, it just see the pixels located in the window, and extract and flatten it to input features of next layer.
    Now we only consider 2D window.
    """
    def __init__(self, win_x=None, win_y=None, stride_x=1, stride_y=1, pad_x=0, pad_y=0, name=None):
        """
        A 2D window has several key parameters.

        :param win_x: window length at X-axis
        :param win_y: window length at Y-axis
        :param stride_x: stride at X-axis to move every time
        :param stride_y: stride at Y-axis to move every time
        :param pad_x: if padding is not None, padding pad_x to the X-axis of images.
        :param pad_y: if padding is not None, padding pad_y to the Y-axis of images.
        :param name: window name
        """
        LOGGER.setLevel(get_logging_level())
        assert win_x is not None and win_y is not None, "win_x, win_y should not be None!"
        self.win_x = win_x
        self.win_y = win_y
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.pad_x = pad_x
        self.pad_y = pad_y
        if name:
            self.name = name
        else:
            self.name = "win/" + "{}x{}".format(win_x, win_y)

    def fit_transform(self, X):
        """
        Fit and transform the input X.

        :param X:
        :return:
        """
        # LOGGER.info("Multi-grain Scan window [{}] is fitting...".format(self.name))
        return get_windows(X, self.win_x, self.win_y, self.stride_x, self.stride_y, self.pad_x, self.pad_y)

    def __str__(self):
        return "win/{}x{}".format(self.win_x, self.win_y)


class Pooling(object):
    """
    A pooling to reduce the dimension of generated feature vectors, so that reduce the computation
     and storage complexity and risk of overfitting.
    """
    def __init__(self, win_x=None, win_y=None, pool_strategy=None, name=None):
        """
        Pooling has several key parameters: win_x, win_y, pool_strategy.

        :param win_x: pooling window length at X-axis
        :param win_y: pooling window length at Y-axis
        :param pool_strategy: pooling strategy, [max or mean]
        :param name: pooling name
        """
        LOGGER.setLevel(get_logging_level())
        assert win_x is not None and win_y is not None, "win_x, win_y should not be None!"
        self.win_x = win_x
        self.win_y = win_y
        self.pool_strategy = pool_strategy if pool_strategy else "mean"
        if name:
            self.name = name
        else:
            self.name = "{}pool/".format(pool_strategy) + "{}x{}".format(win_x, win_y)

    def fit_transform(self, x):
        """
        Fit transform the input x.

        :param x:
        :return:
        """
        # LOGGER.info("Multi-grain Scan pooling [{}] is running..., x.dtype={}".format(self.name, x.dtype))
        return self._transform(x)

    def _transform(self, x):
        """
        Transform inner method.

        :param x:
        :return:
        """
        if x is None or x is []:
            return x
        n, c, h, w = x.shape
        nh = (h - 1) / self.win_x + 1
        nw = (w - 1) / self.win_y + 1
        x_pool = np.empty((n, c, nh, nw), dtype=x.dtype)
        for k in range(c):
            for di in range(nh):
                for dj in range(nw):
                    si = di * self.win_x
                    sj = dj * self.win_y
                    src = x[:, k, si:si+self.win_x, sj:sj+self.win_y]
                    src = src.reshape((x.shape[0], -1))
                    if self.pool_strategy == 'max':
                        x_pool[:, k, di, dj] = np.max(src, axis=1)
                    elif self.pool_strategy == 'mean':
                        x_pool[:, k, di, dj] = np.mean(src, axis=1)
                    else:
                        raise ValueError('Unknown pool strategy!')
        return x_pool

    def transform(self, x):
        """
        Transform the input X.

        :param x:
        :return:
        """
        if x is None or x is []:
            return x
        return self._transform(x)

    def __str__(self):
        return "{}pool/{}x{}".format(self.pool_strategy, self.win_x, self.win_y)

