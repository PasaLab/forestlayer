# -*- coding:utf-8 -*-
"""
Metrics.
"""

import numpy as np
from sklearn import metrics


class Metrics(object):
    """
    Metrics are used to evaluate predictions and ground-truth value.
    """
    def __init__(self, name=''):
        self.name = name

    def __call__(self, y_true, y_pred, prefix='', logger=None):
        """
        Call method of metrics.

        :param y_true:
        :param y_pred:
        :param prefix:
        :param logger:
        :return:
        """
        if y_true is None or y_pred is None:
            return
        if not isinstance(y_pred, type(np.array)):
            y_pred = np.asarray(y_pred)
        if y_pred.shape[1] > 1:
            return self.calc_proba(y_true, y_pred, prefix=prefix, logger=logger)
        elif y_pred.shape[1] == 1:
            return self.calc(y_true, y_pred, prefix=prefix, logger=logger)
        else:
            raise ValueError('y_pred.shape={} does not confirm the restriction!'.format(y_pred.shape))

    def calc(self, y_true, y_pred, prefix='', logger=None):
        """
        Calc metric from y_true and y_prediction.

        :param y_true:
        :param y_pred:
        :param prefix:
        :param logger:
        :return:
        """
        raise NotImplementedError

    def calc_proba(self, y_true, y_proba, prefix='', logger=None):
        """
        Calc metric from y_true and y_probability.

        :param y_true:
        :param y_proba:
        :param prefix:
        :param logger:
        :return:
        """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class Accuracy(Metrics):
    """
    Accuracy metric.
    """
    def __init__(self, name=''):
        super(Accuracy, self).__init__(name)

    def calc(self, y_true, y_pred, prefix='', logger=None):
        """
        Calc Accuracy metric from y_true and y_prediction.

        :param y_true:
        :param y_pred:
        :param prefix:
        :param logger:
        :return:
        """
        if y_true is None or y_pred is None:
            return
        acc = 100. * np.sum(np.asarray(y_true) == y_pred) / len(y_true)
        if logger is not None:
            logger.info('{} Accuracy({}) = {:.4f}%'.format(prefix, self.name, acc))
        return acc

    def calc_proba(self, y_true, y_proba, prefix='', logger=None):
        """
        Calc Accuracy metric from y_true and y_probability.

        :param y_true:
        :param y_proba:
        :param prefix:
        :param logger:
        :return:
        """
        y_true = y_true.reshape(-1)
        y_pred = np.argmax(y_proba.reshape((-1, y_proba.shape[-1])), 1)
        acc = 100. * np.sum(y_true == y_pred) / len(y_true)
        if logger is not None:
            logger.info('{} Accuracy({}) = {:.4f}%'.format(prefix, self.name, acc))
        return acc


class AUC(Metrics):
    def __init__(self, name=''):
        super(AUC, self).__init__(name)

    def calc(self, y_true, y_pred, prefix='', logger=None):
        """
        Calc AUC metric from y_true and y_prediction.

        :param y_true:
        :param y_pred:
        :param prefix:
        :param logger:
        :return:
        """
        assert y_pred.shape[1] == 2, 'auc metric is restricted to the binary classification task!'
        return self.calc_proba(y_true, y_pred, prefix, logger)

    def calc_proba(self, y_true, y_proba, prefix='', logger=None):
        """
        Calc AUC metric from y_true and y_probability.

        :param y_true:
        :param y_proba:
        :param prefix:
        :param logger:
        :return:
        """
        assert y_proba.shape[1] == 2, 'auc metric is restricted to the binary classification task!'
        y_true = y_true.reshape(-1)
        auc_result = auc(y_true, y_proba)
        if logger is not None:
            logger.info('{} AUC({}) = {:.4f}'.format(prefix, self.name, auc_result))
        return auc_result


class MSE(Metrics):
    """
    MSE metric.
    """
    def __init__(self, name=''):
        super(MSE, self).__init__(name)

    def __call__(self, y_true, y_pred, prefix='', logger=None):
        return self.calc(y_true, y_pred, prefix, logger)

    def calc(self, y_true, y_pred, prefix='', logger=None):
        """
        Calc MSE metric from y_true and y_prediction.

        :param y_true:
        :param y_pred:
        :param prefix:
        :param logger:
        :return:
        """
        mse_result = metrics.mean_squared_error(y_true, y_pred)
        if logger is not None:
            logger.info('{} MSE({}) = {:.4f}'.format(prefix, self.name, mse_result))
        return mse_result

    def calc_proba(self, y_true, y_proba, prefix='', logger=None):
        """
        Calc MSE metric from y_true and y_prediction.
        MSE is used to regression task, so y_proba is the y_prediction.

        :param y_true:
        :param y_proba:
        :param prefix:
        :param logger:
        :return:
        """
        if len(y_true.shape) == 1:
            y_proba = y_proba.reshape(-1)
        elif len(y_true.shape) == 2:
            y_proba = y_proba.reshape((y_true.shape[0], -1))
        else:
            raise ValueError('y_true.shape should not exceed 2-dim, but {}'.format(y_true.shape))
        return self.calc(y_true, y_proba, prefix, logger)


class RMSE(MSE):
    def __init__(self, name=''):
        super(RMSE, self).__init__(name)

    def calc(self, y_true, y_pred, prefix='', logger=None):
        res = super(RMSE, self).calc(y_true, y_pred, prefix)
        rmse = np.sqrt(res)
        if logger is not None:
            logger.info('{} RMSE({}) = {:.4f}'.format(prefix, self.name, rmse))
        return rmse

    def calc_proba(self, y_true, y_proba, prefix='', logger=None):
        res = super(RMSE, self).calc_proba(y_true, y_proba, prefix, logger)
        return res


def accuracy(y_true, y_pred):
    return 1.0 * np.sum(np.asarray(y_true) == y_pred) / len(y_true)


def accuracy_pb(y_true, y_proba):
    y_true = y_true.reshape(-1)
    y_pred = np.argmax(y_proba.reshape((-1, y_proba.shape[-1])), 1)
    return 1.0 * np.sum(y_true == y_pred) / len(y_true)


def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y))], dtype=np.float)
    g = g[np.lexsort((g[:, 2], -1 * g[:, 1]))]
    gs = g[:, 0].cumsum().sum() / g[:, 0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)


def gini_nor(y_true, y_proba):
    y_proba = [item[1] for item in y_proba]
    y_proba = np.array(y_proba)
    return gini(y_true, y_proba) / gini(y_true, y_true)


def auc(y_true, y_proba):
    y_proba = [item[1] for item in y_proba]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba, pos_label=1)
    return metrics.auc(fpr, tpr)


def mse(y_true, y_pred):
    mse_result = metrics.mean_squared_error(y_true, y_pred)
    return mse_result

