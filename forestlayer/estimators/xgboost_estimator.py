# -*- coding:utf-8 -*-
"""
XGBoost Estimator, K-fold wrapper version.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from base_estimator import BaseEstimator
from ..utils.storage_utils import is_path_exists, check_dir
from ..utils.log_utils import get_logger
import xgboost as xgb
import numpy as np
import joblib

# self.LOGGER = get_logger('estimators.xgboost')


class XGBoostClassifier(BaseEstimator):
    """
    XGBoost Classifier extends from BaseEstimator.
    """
    def __init__(self, name, est_args):
        """

        :param name:
        :param est_args:
        """
        super(XGBoostClassifier, self).__init__(task='classification',
                                                est_class=XGBoostClassifier,
                                                name=name,
                                                est_args=est_args)
        self.cache_suffix = '.pkl'
        self.est = None
        self.n_class = est_args.get('num_class')
        if self.n_class is None:
            self.n_class = 2
        self.num_boost_round = est_args.get('num_boost_round', 10)
        self.obj = est_args.get('objective', None)
        self.maximize = est_args.get('maximize', False)
        self.early_stopping_rounds = est_args.get('early_stopping_rounds', None)
        self.verbose_eval = est_args.get('verbose_eval', True)
        self.learning_rates = est_args.get('learning_rates', None)

    def fit(self, X, y, cache_dir=None):
        """

        :param X:
        :param y:
        :param cache_dir:
        :return:
        """
        cache_path = self._cache_path(cache_dir=cache_dir)
        # cache it
        if is_path_exists(cache_path):
            self.LOGGER.info('Found estimator from {}, skip fit'.format(cache_path))
            return
        if not isinstance(X, xgb.DMatrix):
            X = xgb.DMatrix(X, label=y)
        watch_list = [(X, 'train'), ]
        est = xgb.train(self.est_args,
                        dtrain=X,
                        num_boost_round=self.num_boost_round,
                        maximize=self.maximize,
                        early_stopping_rounds=self.early_stopping_rounds,
                        evals=watch_list,
                        verbose_eval=self.verbose_eval,
                        learning_rates=self.learning_rates)
        if cache_path is not None:
            self.LOGGER.info("Save estimator to {} ...".format(cache_path))
            check_dir(cache_path)
            self._save_model_to_disk(self.est, cache_path)
            # keep out memory
            self.est = None
        else:
            # keep in memory
            self.est = est

    def _fit(self, est, X, y):
        pass

    def _predict_proba(self, est, X):
        """

        :param est:
        :param X:
        :return:
        """
        if type(X) == list or not isinstance(X, xgb.DMatrix):
            xg_test = xgb.DMatrix(X)
        else:
            xg_test = X
        y_pred = est.predict(xg_test)
        if self.n_class == 2:
            y_proba = []
            for item in y_pred:
                tmp = list()
                tmp.append(1 - item)
                tmp.append(item)
                y_proba.append(tmp)
            y_proba = np.array(y_proba)
        else:
            y_proba = y_pred.reshape((-1, self.n_class))
        return y_proba

    def _load_model_from_disk(self, cache_path):
        return joblib.load(cache_path)

    def _save_model_to_disk(self, est, cache_path):
        joblib.dump(est, cache_path)


class XGBoostRegressor(BaseEstimator):
    """
    XGBoost Classifier extends from BaseEstimator.
    """
    def __init__(self, name, est_args):
        """
        Initialize XGBoostRegressor.

        :param name:
        :param est_args:
        """
        super(XGBoostRegressor, self).__init__(task='regression',
                                               est_class=XGBoostRegressor,
                                               name=name,
                                               est_args=est_args)
        self.cache_suffix = '.pkl'
        self.est = None
        self.num_boost_round = est_args.get('num_boost_round', 10)
        self.obj = est_args.get('objective', None)
        self.maximize = est_args.get('maximize', False)
        self.early_stopping_rounds = est_args.get('early_stopping_rounds', None)
        self.verbose_eval = est_args.get('verbose_eval', True)
        self.learning_rates = est_args.get('learning_rates', None)

    def fit(self, X, y, cache_dir=None):
        """
        Fit.

        :param X:
        :param y:
        :param cache_dir:
        :return:
        """
        cache_path = self._cache_path(cache_dir=cache_dir)
        # cache it
        if is_path_exists(cache_path):
            self.LOGGER.info('Found estimator from {}, skip fit'.format(cache_path))
            return
        if not isinstance(X, xgb.DMatrix):
            X = xgb.DMatrix(X, label=y)
        watch_list = [(X, 'train'), ]
        est = xgb.train(self.est_args,
                        dtrain=X,
                        num_boost_round=self.num_boost_round,
                        maximize=self.maximize,
                        early_stopping_rounds=self.early_stopping_rounds,
                        evals=watch_list,
                        verbose_eval=self.verbose_eval,
                        learning_rates=self.learning_rates)
        if cache_path is not None:
            self.LOGGER.info("Save estimator to {} ...".format(cache_path))
            check_dir(cache_path)
            self._save_model_to_disk(self.est, cache_path)
            # keep out memory
            self.est = None
        else:
            # keep in memory
            self.est = est

    def _fit(self, est, X, y):
        pass

    def _predict(self, est, X):
        """
        Predict inner method.

        :param est:
        :param X:
        :return:
        """
        return self._predict_proba(est, X)

    def _predict_proba(self, est, X):
        """
        Predict probability inner method.

        :param est:
        :param X:
        :return:
        """
        if type(X) == list or not isinstance(X, xgb.DMatrix):
            xg_test = xgb.DMatrix(X)
        else:
            xg_test = X
        y_pred = est.predict(xg_test)
        return y_pred

    def _load_model_from_disk(self, cache_path):
        """
        Load model from disk inner method.

        :param cache_path:
        :return:
        """
        return joblib.load(cache_path)

    def _save_model_to_disk(self, est, cache_path):
        """
        Save model to disk inner method.

        :param est:
        :param cache_path:
        :return:
        """
        joblib.dump(est, cache_path)
