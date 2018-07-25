# -*- coding:utf-8 -*-
"""
Base Estimators Wrapper Definition.
"""

import numpy as np
import os.path as osp
import sys
from ..utils.storage_utils import is_path_exists, check_dir, name2path, getmbof
from ..utils.log_utils import get_logger, get_logging_level

# self.LOGGER = get_logger('estimators.base_estimator')


class BaseEstimator(object):
    """
    Base Estimators Wrapper Definition.
    """
    def __init__(self, task=None, est_class=None, name=None, est_args=None):
        """
        Initialize BaseEstimators.

        :param task: what task does this estimator to execute ('classification' or 'regression')
        :param est_class: estimator class
        :param name: estimator name
        :param est_args: dict, estimator argument
        """
        self.LOGGER = get_logger('estimators.base_estimator')
        self.LOGGER.setLevel(get_logging_level())
        self.name = name
        self.task = task
        self.est_class = est_class
        self.est_args = est_args if est_args is not None else {}
        self.cache_suffix = '.pkl'
        self.est = None

    def _init_estimators(self):
        """
        Initialize an estimator.

        :return:
        """
        return self.est_class(**self.est_args)

    def fit(self, X, y, cache_dir=None):
        """
        Fit estimator.
        If cache_path exists, we do not need to re-fit.
        If cache_path is not None, save model to disk.

        :param X:
        :param y:
        :param cache_dir:
        :return:
        """
        self.LOGGER.debug("X_train.shape={}, y_train.shape={}".format(X.shape, y.shape))
        cache_path = self._cache_path(cache_dir=cache_dir)
        # cache it
        if is_path_exists(cache_path):
            self.LOGGER.info('Found estimator from {}, skip fit'.format(cache_path))
            return
        est = self._init_estimators()
        self._fit(est, X, y)
        if cache_path is not None:
            self.LOGGER.info("Save estimator to {} ...".format(cache_path))
            check_dir(cache_path)
            self._save_model_to_disk(self.est, cache_path)
            # keep out memory
            self.est = None
        else:
            # keep in memory
            self.est = est

    def predict(self, X, cache_dir=None, batch_size=None):
        """
        Predict results.
        if cache_path is not None, we can load model from disk, else we take from memory.
        and then we decide whether use batch predict.

        :param X:
        :param cache_dir:
        :param batch_size:
        :return:
        """
        self.LOGGER.debug("X.shape={}".format(X.shape))
        cache_path = self._cache_path(cache_dir)
        # cache
        if cache_path is not None:
            self.LOGGER.info("Load estimator from {} ...".format(cache_path))
            est = self._load_model_from_disk(cache_path)
            self.LOGGER.info("done ...")
        else:
            est = self.est
        batch_size = batch_size or self._default_predict_batch_size(est=est, X=X, task=self.task)
        if batch_size > 0:
            y_pred = self._batch_predict(est, X, batch_size)
        else:
            y_pred = self._predict(est, X)
        self.LOGGER.debug("y_proba.shape={}".format(y_pred.shape))
        return y_pred

    def predict_proba(self, X, cache_dir=None, batch_size=None):
        """
        Predict probability.
        if cache_path is not None, we can load model from disk, else we take from memory.
        and then we decide whether use batch predict.

        :param X:
        :param cache_dir:
        :param batch_size:
        :return:
        """
        if self.task == 'regression':
            return self.predict(X, cache_dir=cache_dir, batch_size=batch_size)
        self.LOGGER.debug("X.shape={}, size = {}".format(X.shape, getmbof(X)))
        cache_path = self._cache_path(cache_dir)
        # cache
        if cache_path is not None:
            self.LOGGER.info("Load estimator from {} ...".format(cache_path))
            est = self._load_model_from_disk(cache_path)
            self.LOGGER.info("done ...")
        else:
            est = self.est
        batch_size = batch_size or self._default_predict_batch_size(est, X, self.task)
        if batch_size > 0:
            y_proba = self._batch_predict_proba(est, X, batch_size)
        else:
            y_proba = self._predict_proba(est, X)
        self.LOGGER.debug("y_proba.shape={}, size = {}, dtype = {}".format(y_proba.shape, getmbof(y_proba), y_proba.dtype))
        return y_proba

    def _batch_predict_proba(self, est, X, batch_size):
        """
        Predict probability in batch.

        :param est:
        :param X:
        :param batch_size:
        :return:
        """
        self.LOGGER.debug("X.shape={}, batch_size={}".format(X.shape, batch_size))
        verbose_backup = 0
        # clear verbose
        if hasattr(est, "verbose"):
            verbose_backup = est.verbose
            est.verbose = 0
        n_datas = X.shape[0]
        y_pred_proba = None
        for j in range(0, n_datas, batch_size):
            self.LOGGER.info("[batch_predict_proba][batch_size={}] ({}/{})".format(batch_size, j, n_datas))
            cur_x = X[j:j+batch_size]
            y_cur = self._predict_proba(est, cur_x)
            self.LOGGER.debug('[cur_x.dtype = {}][y_cur.dtype = {}]'.format(cur_x.dtype, y_cur.dtype))
            self.LOGGER.debug('[cur_x.shape = {}][y_cur.shape = {}]'.format(cur_x.shape, y_cur.shape))
            self.LOGGER.debug("[cur_x.size = {}][y_cur.size = {}]".format(getmbof(cur_x), getmbof(y_cur)))
            if j == 0:
                n_classes = y_cur.shape[1]
                y_pred_proba = np.empty((n_datas, n_classes), dtype=np.float64)
            y_pred_proba[j:j+batch_size, :] = y_cur
        self.LOGGER.debug('[y_pred_proba size = {}, dtype = {}]'.format(getmbof(y_pred_proba), y_pred_proba.dtype))
        # restore verbose
        if hasattr(est, "verbose"):
            est.verbose = verbose_backup
        return y_pred_proba

    def _batch_predict(self, est, X, batch_size):
        """
        Predict result in batch.

        :param est:
        :param X:
        :param batch_size:
        :return:
        """
        self.LOGGER.debug("X.shape={}, batch_size={}".format(X.shape, batch_size))
        verbose_backup = 0
        # clear verbose
        if hasattr(est, "verbose"):
            verbose_backup = est.verbose
            est.verbose = 0
        n_datas = X.shape[0]
        y_pred = None
        for j in range(0, n_datas, batch_size):
            self.LOGGER.info("[batch_predict_proba][batch_size={}] ({}/{})".format(batch_size, j, n_datas))
            y_cur = self._predict(est, X[j:j + batch_size])
            if j == 0:
                y_pred = np.empty((n_datas,), dtype=np.float64)
            y_pred[j:j + batch_size] = y_cur
        # restore verbose
        if hasattr(est, "verbose"):
            est.verbose = verbose_backup
        return y_pred

    def _cache_path(self, cache_dir):
        """
        Get cache_path (model)

        :param cache_dir:
        :return:
        """
        if cache_dir is None:
            return None
        return osp.join(cache_dir, name2path(self.name) + self.cache_suffix)

    def _load_model_from_disk(self, cache_path):
        """
        Load model from disk.

        :param cache_path:
        :return:
        """
        raise NotImplementedError()

    def _save_model_to_disk(self, est, cache_path):
        """
        Save model to disk.

        :param est:
        :param cache_path:
        :return:
        """
        raise NotImplementedError()

    def _default_predict_batch_size(self, est, X, task):
        """
        You can re-implement this function when inherent this class

        Return
        ------
        predict_batch_size (int): default=0
            if = 0,  predict_proba without batches
            if > 0, then predict_proba with batches
            sklearn predict_proba is not so inefficient, has to do this
        """
        return 0

    def _fit(self, est, X, y):
        """
        Fit est on (X, y)

        :param est:
        :param X:
        :param y:
        :return:
        """
        est.fit(X, y)

    def _predict_proba(self, est, X):
        """
        Predict probability inner method.

        :param est:
        :param X:
        :return:
        """
        return est.predict_proba(X)

    def _predict(self, est, X):
        """
        Predict result inner method.

        :param est:
        :param X:
        :return:
        """
        return est.predict(X)

    def copy(self):
        """
        copy

        :return:
        """
        return BaseEstimator(est_class=self.est_class, **self.est_args)

    @property
    def is_classification(self):
        """
        True if the task is classification.

        :return:
        """
        return self.task == 'classification'

