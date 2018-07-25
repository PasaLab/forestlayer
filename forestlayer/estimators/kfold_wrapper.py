# -*- coding:utf-8 -*-
"""
K-fold wrapper definition.
This page of code was partly borrowed from Ji. Feng.
"""

from __future__ import print_function
import os.path as osp
import numpy as np
import ray
import copy
import time
from time import time as get_time
import redis
import forestlayer as fl
try:
    import cPickle as pickle
except ImportError:
    import pickle
from sklearn.model_selection import KFold, StratifiedKFold
from .sklearn_estimator import *
from .estimator_configs import EstimatorConfig
from ..utils.log_utils import get_logger
from ..utils.storage_utils import name2path, getmbof
from ..utils.metrics import Accuracy, MSE
from ..backend.common import add_fit_time, add_kfold_time
from collections import defaultdict
import heapq
import math
from psutil import virtual_memory
MAX_RAND_SEED = np.iinfo(np.int32).max
str2est_class = {
    'classification': {
        'FLCRF': FLCRFClassifier,
        'FLRF': FLRFClassifier,
        'FLGBDT': FLGBDTClassifier,
        'FLXGB': FLXGBoostClassifier,
        'FLLGBM': FLLGBMClassifier,
    },
    'regression': {
        'FLCRF': FLCRFRegressor,
        'FLRF': FLRFRegressor,
        'FLGBDT': FLGBDTRegressor,
        'FLXGB': FLXGBoostRegressor,
        'FLLGBM': FLLGBMRegressor,
    }
}


class KFoldWrapper(object):
    def __init__(self, name, n_folds, task, est_type, seed=None, dtype=np.float32,
                 eval_metrics=None, cache_dir=None, keep_in_mem=None, est_args=None, cv_seed=None):
        """
        Initialize a KFoldWrapper.

        :param name:
        :param n_folds:
        :param task:
        :param est_type:
        :param seed:
        :param eval_metrics:
        :param cache_dir:
        :param keep_in_mem:
        :param est_args:
        """
        self.LOGGER = get_logger('estimators.kfold_wrapper')
        self.name = name
        self.n_folds = n_folds
        self.task = task
        self.est_type = est_type
        self.est_args = est_args if est_args is not None else {}
        self.seed = seed
        self.dtype = dtype
        self.cv_seed = cv_seed
        # assign a cv_seed
        if self.cv_seed is None:
            if isinstance(self.seed, np.random.RandomState):
                self.cv_seed = copy.deepcopy(self.seed)
            else:
                self.cv_seed = self.seed
        self.eval_metrics = eval_metrics if eval_metrics is not None else []
        if cache_dir is not None:
            self.cache_dir = osp.join(cache_dir, name2path(self.name))
        else:
            self.cache_dir = None
        self.keep_in_mem = keep_in_mem
        self.fit_estimators = [None for _ in range(n_folds)]
        self.n_dims = None

    def _init_estimator(self, k):
        """
        Initialize k-th estimator in K-fold CV.

        :param k: the order number of k-th estimator.
        :return: initialed estimator
        """
        est_args = self.est_args.copy()
        est_name = '{}/{}'.format(self.name, k)
        # TODO: consider if add a random_state, actually random_state of each estimator can be set in est_configs in
        # main program by users, so we need not to set random_state there.
        # More importantly, if some estimators have no random_state parameter, this assignment can throw problems.
        if self.est_type in ['FLXGB']:
            if est_args.get('seed', None) is None:
                est_args['seed'] = copy.deepcopy(self.seed)
        elif self.est_type in ['FLCRF', 'FLRF', 'FLGBDT', 'FLLGBM']:
            if est_args.get('random_state', None) is None:
                est_args['random_state'] = copy.deepcopy(self.seed)
        est_class = est_class_from_type(self.task, self.est_type)
        return est_class(est_name, est_args)

    def fit_transform(self, X, y, y_stratify=None, test_sets=None):
        """
        Fit and transform.

        :param X: (ndarray) n x k or n1 x n2 x k
                            to support windows_layer, X could have dim >2
        :param y: (ndarray) y (ndarray):
                            n or n1 x n2
        :param y_stratify: (list) used for StratifiedKFold or None means no stratify
        :param test_sets: (list) optional.
                   A list of (prefix, X_test, y_test) pairs.
                   predict_proba for X_test will be returned
                   use with keep_model_in_mem=False to save mem usage
                   y_test could be None, otherwise use eval_metrics for debugging
        :return:
        """
        if self.keep_in_mem is None:
            self.keep_in_mem = False
        assert 2 <= len(X.shape) <= 3, "X.shape should be n x k or n x n2 x k"
        assert len(X.shape) == len(y.shape) + 1
        if y_stratify is not None:
            assert X.shape[0] == len(y_stratify)
        test_sets = test_sets if test_sets is not None else []
        # K-Fold split
        n_stratify = X.shape[0]
        if self.n_folds == 1:
            cv = [(range(len(X)), range(len(X)))]
        else:
            if y_stratify is None:
                skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.cv_seed)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify))]
            else:
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.cv_seed)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify), y_stratify)]
        y_proba_train = None
        y_probas_test = []
        self.n_dims = X.shape[-1]
        inverse = False
        for k in range(self.n_folds):
            est = self._init_estimator(k)
            if not inverse:
                train_idx, val_idx = cv[k]
            else:
                val_idx, train_idx = cv[k]
            # fit on k-fold train
            est.fit(X[train_idx].reshape((-1, self.n_dims)), y[train_idx].reshape(-1), cache_dir=self.cache_dir)
            # predict on k-fold validation, this y_proba.dtype is float64
            y_proba = est.predict_proba(X[val_idx].reshape((-1, self.n_dims)),
                                        cache_dir=self.cache_dir)
            if not est.is_classification:
                y_proba = y_proba[:, np.newaxis]  # add one dimension
            if len(X.shape) == 3:
                y_proba = y_proba.reshape((len(val_idx), -1, y_proba.shape[-1]))
            self.log_metrics(self.name, y[val_idx], y_proba, "train_{}".format(k))

            # merging result
            if k == 0:
                if len(X.shape) == 2:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1]), dtype=self.dtype)
                else:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1], y_proba.shape[2]), dtype=self.dtype)
                y_proba_train = y_proba_cv
            y_proba_train[val_idx, :] += y_proba

            if self.keep_in_mem:
                self.fit_estimators[k] = est

            # test
            for vi, (prefix, X_test, _) in enumerate(test_sets):
                # keep float32 data type, save half of memory and communication.
                y_proba = est.predict_proba(X_test.reshape((-1, self.n_dims)),
                                            cache_dir=self.cache_dir)
                if not est.is_classification:
                    y_proba = y_proba[:, np.newaxis]
                if len(X.shape) == 3:
                    y_proba = y_proba.reshape((X_test.shape[0], X_test.shape[1], y_proba.shape[-1]))
                if k == 0:
                    y_probas_test.append(y_proba)
                else:
                    y_probas_test[vi] += y_proba
        if inverse and self.n_folds > 1:
            y_proba_train /= (self.n_folds - 1)
        for y_proba in y_probas_test:
            y_proba /= self.n_folds

        # log train average
        self.log_metrics(self.name, y, y_proba_train, "train_avg")
        # y_test can be None
        for vi, (test_name, X_test, y_test) in enumerate(test_sets):
            if y_test is not None:
                self.log_metrics(self.name, y_test, y_probas_test[vi], test_name)
        return y_proba_train, y_probas_test

    def transform(self, x_tests):
        """
        Transform data.

        :param x_tests:
        :return:
        """
        # TODO: using model loaded from disk
        if x_tests is None or x_tests == []:
            return []
        if isinstance(x_tests, (list, tuple)):
            self.LOGGER.warn('transform(x_tests) only support single ndarray instead of list of ndarrays')
            x_tests = x_tests[0]
        proba_result = None
        for k, est in enumerate(self.fit_estimators):
            y_proba = est.predict_proba(x_tests.reshape((-1, self.n_dims)), cache_dir=self.cache_dir)
            if not est.is_classification:
                y_proba = y_proba[:, np.newaxis]  # add one dimension
            if len(x_tests.shape) == 3:
                y_proba = y_proba.reshape((x_tests.shape[0], x_tests.shape[1], y_proba.shape[-1]))
            if k == 0:
                proba_result = y_proba
            else:
                proba_result += y_proba
            proba_result /= self.n_folds
        return proba_result

    def log_metrics(self, est_name, y_true, y_proba, y_name):
        """
        y_true (ndarray): n or n1 x n2
        y_proba (ndarray): n x n_classes or n1 x n2 x n_classes
        """
        if self.eval_metrics is None:
            return
        for metric in self.eval_metrics:
            acc = metric.calc_proba(y_true, y_proba)
            self.LOGGER.info("{}({} - {}) = {:.4f}{}".format(
                metric.__class__.__name__, est_name, y_name, acc, '%' if isinstance(metric, Accuracy) else ''))

    def _predict_proba(self, est, X):
        """
        Predict probability inner method.

        :param est:
        :param X:
        :return:
        """
        return est.predict_proba(X)

    def copy(self):
        """
        copy.
        NOTE: This copy does not confirm object consistency now, be careful to use it.
        TODO: check object consistency.

        :return:
        """
        return KFoldWrapper(name=self.name,
                            n_folds=self.n_folds,
                            task=self.task,
                            est_type=self.est_type,
                            seed=self.seed,
                            eval_metrics=self.eval_metrics,
                            cache_dir=self.cache_dir,
                            keep_in_mem=self.keep_in_mem,
                            est_args=self.est_args,
                            cv_seed=self.cv_seed)


@ray.remote
class DistributedKFoldWrapper(object):
    def __init__(self, name=None, n_folds=3, task='classification', est_type=None, seed=None, dtype=np.float32,
                 splitting=False, eval_metrics=None, cache_dir=None, keep_in_mem=None, est_args=None, cv_seed=None,
                 win_shape=None, pool=None, train_start_ends=None, x_train_group_or_id=None, x_test_group_or_id=None):
        """
        Initialize a KFoldWrapper.

        :param name:
        :param n_folds:
        :param task:
        :param est_type:
        :param seed:
        :param dtype:
        :param splitting: whether is in splitting, if true, we do not transfer proba results to self.dtype
                           (may float32, default proba results of forests is float64) to keep Floating-point precision,
                           after merge, we transfer it to self.dtype.
                           if false, we directly transfer it to self.dtype (may be float32) to save memory.
        :param eval_metrics:
        :param cache_dir:
        :param keep_in_mem:
        :param est_args:
        """
        # log_info is used to store logging strings, which will be return to master node after fit/fit_transform
        # every log info in logs consists of several parts: (level, base_string, data)
        # for example, ('INFO', 'Accuracy(win - 0 - estimator - 0 - 3folds - train_0) = {:.4f}%', 26.2546)
        self.logs = []
        self.name = name
        self.n_folds = n_folds
        self.task = task
        self.est_type = est_type
        self.est_args = est_args if est_args is not None else {}
        self.seed = seed
        if isinstance(seed, basestring):
            self.seed = pickle.loads(seed)
        self.dtype = dtype
        self.splitting = splitting
        self.cv_seed = cv_seed
        if self.cv_seed is None:
            self.cv_seed = self.seed
        self.eval_metrics = eval_metrics if eval_metrics is not None else []
        if cache_dir is not None:
            self.cache_dir = osp.join(cache_dir, name2path(self.name))
        else:
            self.cache_dir = None
        self.keep_in_mem = keep_in_mem
        self.fit_estimators = [None for _ in range(n_folds)]
        self.n_dims = None
        # win_shape and pool only belong to MGS layer.
        self.win_shape = win_shape
        self.pool = pool
        # Lazy assemble
        self.x_train_group_or_id = x_train_group_or_id
        self.x_test_group_or_id = x_test_group_or_id
        self.train_start_ends = train_start_ends

    def pool_shape(self, pool, win_shape):
        h, w = win_shape
        nh = (h - 1) / pool.win_x + 1
        nw = (w - 1) / pool.win_y + 1
        return nh, nw

    def get_fit_estimators(self):
        return self.fit_estimators

    def _init_estimator(self, k):
        """
        Initialize k-th estimator in K-fold CV.

        :param k: the order number of k-th estimator.
        :return: initialed estimator
        """
        est_args = self.est_args.copy()
        est_name = '{}/{}'.format(self.name, k)
        if self.est_type in ['FLXGB']:
            if est_args.get('seed', None) is None:
                est_args['seed'] = copy.deepcopy(self.seed)
        elif self.est_type in ['FLCRF', 'FLRF', 'FLGBDT', 'FLLGBM']:
            if est_args.get('random_state', None) is None:
                est_args['random_state'] = copy.deepcopy(self.seed)
        est_class = est_class_from_type(self.task, self.est_type)
        return est_class(est_name, est_args)

    def query(self, x_train, x_test, win, wi, redis_addr):
        """
        [WIP] Query redis server to keep only one copy for every window scan in single machine.

        :param x_train:
        :param x_test:
        :param win:
        :param wi:
        :param redis_addr:
        :return:
        """
        local_ip = ray.services.get_node_ip_address()
        rhost, rport = redis_addr.split(':')[:2]
        redis_client = redis.StrictRedis(host=rhost, port=int(rport))
        key_train_seal = local_ip + ",train{},seal".format(wi)
        key_train = local_ip + ",train{}".format(wi)
        time.sleep(np.random.random())  # randomly sleep [0, 1) seconds to reduce collision
        query_train_seal = redis_client.get(key_train_seal)
        if query_train_seal is None:
            # become a publisher
            redis_client.setex(key_train_seal, 30, '1')
            start_time = get_time()
            x_wins_train = win.fit_transform(x_train)
            # self.logs.append("GENERATE x_wins_train={} in {} for {}".format(getmbof(x_wins_train), local_ip, wi))
            x_wins_train_id = ray.put(x_wins_train)
            now_time = get_time()
            if now_time - start_time < 5:  # wait for all other process to subscribe this channel
                time.sleep(5 - now_time + start_time)
            redis_client.publish(key_train, x_wins_train_id.id())
        else:
            try:
                # become a subscriber
                p = redis_client.pubsub()
                p.subscribe(key_train)
                start_time = time.time()
                for message in p.listen():
                    if message['type'] == 'message':
                        data = message['data']
                        x_wins_train = ray.get(ray.local_scheduler.ObjectID(data))
                        # self.logs.append("Bingo! we get off-the-shelf x_wins_train {}!"
                        #                  " in {} for {}".format(getmbof(x_wins_train), local_ip, wi))
                        break
                    time.sleep(0.1)
                    if time.time() - start_time > 5:
                        x_wins_train = win.fit_transform(x_train)
                        p.unsubscribe(key_train)
                        break
            finally:
                p.unsubscribe(key_train)
        key_test_seal = local_ip + ",test{},seal".format(wi)
        key_test = local_ip + ",test{}".format(wi)
        time.sleep(np.random.random())  # randomly sleep [0, 1) seconds to reduce collision
        query_test_seal = redis_client.get(key_test_seal)
        if query_test_seal is None:
            # become a publisher
            redis_client.setex(key_test_seal, 30, '1')
            start_time = get_time()
            x_wins_test = win.fit_transform(x_test)
            # self.logs.append("GENERATE x_wins_test={} in {} for {}".format(getmbof(x_wins_test), local_ip, wi))
            x_wins_test_id = ray.put(x_wins_test)
            now_time = get_time()
            if now_time - start_time < 5:  # wait for all other process to subscribe this channel
                time.sleep(5 - now_time + start_time)
            redis_client.publish(key_test, x_wins_test_id.id())
        else:
            try:
                # become a subscriber
                p = redis_client.pubsub()
                p.subscribe(key_test)
                start_time = time.time()
                for message in p.listen():
                    if message['type'] == 'message':
                        data = message['data']
                        x_wins_test = ray.get(ray.local_scheduler.ObjectID(data))
                        # self.logs.append("Bingo! we get off-the-shelf x_wins_test {}!"
                        #                  " in {} for {}".format(getmbof(x_wins_test), local_ip, wi))
                        break
                    time.sleep(0.1)
                    if time.time() - start_time > 5:
                        x_wins_test = win.fit_transform(x_test)
                        p.unsubscribe(key_test)
                        break
            finally:
                p.unsubscribe(key_test)
        return x_wins_train, x_wins_test

    def assemble(self, x, n_xs, group_or_id):
        if group_or_id is None:
            return x
        x_cur = np.zeros((n_xs, 0), dtype=self.dtype)
        for (start, end) in self.train_start_ends:
            x_cur = np.hstack((x_cur, group_or_id[:, start:end]))
        x_cur = np.hstack((x_cur, x))
        return x_cur

    def fit_transform_lazyscan(self, x_train, y_train, x_test, y_test, win, wi, redis_addr):
        x_wins_train, x_wins_test = self.query(x_train, x_test, win, wi, redis_addr)
        x_wins_train = x_wins_train.reshape((x_wins_train.shape[0], -1, x_wins_train.shape[-1]))
        x_wins_test = x_wins_test.reshape((x_wins_test.shape[0], -1, x_wins_test.shape[-1]))
        y_win = y_train[:, np.newaxis].repeat(x_wins_train.shape[1], axis=1)
        y_stratify = y_win[:, 0]
        y_win_test = None if y_test is None else y_test[:, np.newaxis].repeat(x_wins_test.shape[1], axis=1)
        test_sets = [('testOfWin{}'.format(wi), x_wins_test, y_win_test)]
        return self.fit_transform(x_wins_train, y_win, y_stratify, test_sets)

    def fit_transform(self, X, y, y_stratify=None, test_sets=None):
        """
        Fit and transform.

        :param X: (ndarray) n x k or n1 x n2 x k
                            to support windows_layer, X could have dim >2
        :param y: (ndarray) y (ndarray):
                            n or n1 x n2
        :param y_stratify: (list) used for StratifiedKFold or None means no stratify
        :param test_sets: (list) optional.
                   A list of (prefix, X_test, y_test) pairs.
                   predict_proba for X_test will be returned
                   use with keep_model_in_mem=False to save mem usage
                   y_test could be None, otherwise use eval_metrics for debugging
        :return:
        """
        self.logs.append("{}:{} Running on {}".format(self.name, self.est_args.get('n_estimators', -1),
                                                      ray.services.get_node_ip_address()))
        self.logs.append("{} start {}".format(ray.services.get_node_ip_address(), time.time()))
        if self.keep_in_mem is None:
            self.keep_in_mem = False
        assert 2 <= len(X.shape) <= 3, "X.shape should be n x k or n x n2 x k"
        assert len(X.shape) == len(y.shape) + 1
        if y_stratify is not None:
            assert X.shape[0] == len(y_stratify)
        test_sets = test_sets if test_sets is not None else []
        # K-Fold split
        n_stratify = X.shape[0]
        if self.x_train_group_or_id is not None:
            X = self.assemble(X, n_stratify, self.x_train_group_or_id)
        if self.x_test_group_or_id is not None:
            new_test_sets = []
            for tup in test_sets:
                new_test_sets.append((tup[0], self.assemble(tup[1], tup[1].shape[0], self.x_test_group_or_id), tup[2]))
            test_sets = new_test_sets
            del new_test_sets
        if self.n_folds == 1:
            cv = [(range(len(X)), range(len(X)))]
        else:
            if y_stratify is None:
                skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.cv_seed)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify))]
            else:
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.cv_seed)
                cv = [(t, v) for (t, v) in skf.split(range(n_stratify), y_stratify)]
        # K-fold fit
        y_proba_train = None
        y_probas_test = []
        self.n_dims = X.shape[-1]
        inverse = False
        for k in range(self.n_folds):
            # fuse mechanism. Keep memory using safety.
            fuse()
            est = self._init_estimator(k)
            if not inverse:
                train_idx, val_idx = cv[k]
            else:
                val_idx, train_idx = cv[k]
            # fit on k-fold train
            est.fit(X[train_idx].reshape((-1, self.n_dims)), y[train_idx].reshape(-1), cache_dir=self.cache_dir)
            # predict on k-fold validation
            y_proba = est.predict_proba(X[val_idx].reshape((-1, self.n_dims)), cache_dir=self.cache_dir)
            if not est.is_classification:
                y_proba = y_proba[:, np.newaxis]  # add one dimension
            if len(X.shape) == 3:
                y_proba = y_proba.reshape((len(val_idx), -1, y_proba.shape[-1]))
            self.log_metrics(self.name, y[val_idx], y_proba, "train_{}".format(k))

            # merging result
            if k == 0:
                if len(X.shape) == 2:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1]))
                else:
                    y_proba_cv = np.zeros((n_stratify, y_proba.shape[1], y_proba.shape[2]))
                y_proba_train = y_proba_cv
            y_proba_train[val_idx, :] += y_proba

            if self.keep_in_mem:
                self.fit_estimators[k] = est

            # test
            for vi, (prefix, X_test, y_test) in enumerate(test_sets):
                y_proba = est.predict_proba(X_test.reshape((-1, self.n_dims)), cache_dir=self.cache_dir)
                if not est.is_classification:
                    y_proba = y_proba[:, np.newaxis]
                if len(X.shape) == 3:
                    y_proba = y_proba.reshape((X_test.shape[0], X_test.shape[1], y_proba.shape[-1]))
                if k == 0:
                    y_probas_test.append(y_proba)
                else:
                    y_probas_test[vi] += y_proba
        if inverse and self.n_folds > 1:
            y_proba_train /= (self.n_folds - 1)
        for y_proba in y_probas_test:
            y_proba /= self.n_folds

        # log train average
        self.log_metrics(self.name, y, y_proba_train, "train_avg")
        # y_test can be None
        for vi, (test_name, X_test, y_test) in enumerate(test_sets):
            if y_test is not None:
                self.log_metrics(self.name, y_test, y_probas_test[vi], test_name)
        # Advance pooling
        if self.pool is not None:
            # self.logs.append("Advance pooling in shape: {}".format(y_proba_train.shape))
            nh, nw = self.win_shape
            n_class = y_proba_train.shape[-1]
            y_proba_train = y_proba_train.reshape((-1, nh, nw, n_class)).transpose((0, 3, 1, 2))
            # self.logs.append("y_proba_train.shape = {}".format(y_proba_train.shape))
            y_proba_train = self.pool.fit_transform(y_proba_train)
            # self.logs.append("y_proba_train.shape 2 = {}".format(y_proba_train.shape))
            pool_nh, pool_nw = self.pool_shape(self.pool, self.win_shape)
            # LOGGER.info("pool_nh, pool_nw = {}, remember = {}".format((pool_nh, pool_nw), remember_middle))
            if len(X.shape) == 3:
                y_proba_train = (y_proba_train.reshape((-1, n_class, pool_nh, pool_nw))
                                 .transpose((0, 2, 3, 1))
                                 .reshape((-1, pool_nh*pool_nw, n_class)))
            else:
                y_proba_train = (y_proba_train.reshape((-1, n_class, pool_nh, pool_nw))
                                 .transpose((0, 2, 3, 1))
                                 .reshape((-1, n_class)))
            for yi, y_proba_test in enumerate(y_probas_test):
                y_proba_test = y_proba_test.reshape((-1, nh, nw, n_class)).transpose((0, 3, 1, 2))
                y_probas_test[yi] = self.pool.fit_transform(y_proba_test)
                pool_nh, pool_nw = self.pool_shape(self.pool, self.win_shape)
                if len(X.shape) == 3:
                    y_probas_test[yi] = (y_probas_test[yi].reshape((-1, n_class, pool_nh, pool_nw))
                                         .transpose((0, 2, 3, 1))
                                         .reshape((-1, pool_nh*pool_nw, n_class)))
                else:
                    y_probas_test[yi] = (y_probas_test[yi].reshape((-1, n_class, pool_nh, pool_nw))
                                         .transpose((0, 2, 3, 1))
                                         .reshape((-1, n_class)))
            # self.logs.append("Advance pooling out shape: {}".format(y_proba_train.shape))
        # if not splitting, we directly let it be self.dtype to potentially save memory
        if not self.splitting and y_proba_train.dtype != self.dtype:
            y_proba_train = y_proba_train.astype(self.dtype)
        self.logs.append("{} end {}".format(ray.services.get_node_ip_address(), time.time()))
        return y_proba_train, y_probas_test, self.logs

    def transform(self, x_tests):
        """
        Transform data.

        :param x_tests:
        :return:
        """
        # TODO: using model loaded from disk
        if x_tests is None or x_tests == []:
            return []
        if isinstance(x_tests, (list, tuple)):
            self.logs.append(('WARN', 'transform(x_tests) only support single ndarray instead of list of ndarrays', 0))
            x_tests = x_tests[0]
        proba_result = None
        for k, est in enumerate(self.fit_estimators):
            y_proba = est.predict_proba(x_tests.reshape((-1, self.n_dims)), cache_dir=self.cache_dir)
            if not est.is_classification:
                y_proba = y_proba[:, np.newaxis]  # add one dimension
            if len(x_tests.shape) == 3:
                y_proba = y_proba.reshape((x_tests.shape[0], x_tests.shape[1], y_proba.shape[-1]))
            if k == 0:
                proba_result = y_proba
            else:
                proba_result += y_proba
            proba_result /= self.n_folds
        return proba_result

    def log_metrics(self, est_name, y_true, y_proba, y_name):
        """
        Logging evaluation metrics.

        :param est_name: estimator name.
        :param y_true: (ndarray) n or n1 x n2
        :param y_proba: (ndarray) n x n_classes or n1 x n2 x n_classes
        :param y_name: 'train_{no.}' or 'train' or 'test', identify a name for this info.
        :return:
        """
        if self.eval_metrics is None:
            return
        for metric in self.eval_metrics:
            acc = metric.calc_proba(y_true, y_proba)
            self.logs.append(('INFO', "{a}({b} - {c}) = {d}{e}".format(
                a=metric.__class__.__name__, b=est_name, c=y_name, d="{:.4f}",
                e='%' if isinstance(metric, Accuracy) else ''), acc))

    @staticmethod
    def _predict_proba(est, X):
        """
        Predict probability inner method.

        :param est:
        :param X:
        :return:
        """
        return est.predict_proba(X)

    def copy(self):
        """
        copy.
        NOTE: This copy does not confirm object consistency now, be careful to use it.
        TODO: check object consistency.

        :return:
        """
        return DistributedKFoldWrapper.remote(name=self.name,
                                              n_folds=self.n_folds,
                                              task=self.task,
                                              est_type=self.est_type,
                                              seed=self.seed,
                                              dtype=self.dtype,
                                              splitting=self.splitting,
                                              eval_metrics=self.eval_metrics,
                                              cache_dir=self.cache_dir,
                                              keep_in_mem=self.keep_in_mem,
                                              est_args=self.est_args,
                                              cv_seed=self.cv_seed)


class SplittingKFoldWrapper(object):
    """
    Wrapper for splitting forests to smaller forests.
    TODO: support intelligent load-aware splitting method.
    """
    def __init__(self, dis_level=0, estimators=None, ei2wi=None, num_workers=None, seed=None,
                 windows=None, pools=None, task='classification',
                 eval_metrics=None, keep_in_mem=False, cv_seed=None, dtype=np.float32):
        """
        Initialize SplittingKFoldWrapper.

        :param dis_level: distributed level, or parallelization level, 0 / 1 / 2
                           0 means lowest parallelization level, parallelization is len(self.est_configs).
                           1 means we will split the forests in some condition to making more full use of
                            cluster resources, so the parallelization may be larger than len(self.est_configs).
                           2 means that anyway we must split forests.
                           Now 2 is the HIGHEST_DISLEVEL
        :param estimators: base estimators.
        :param ei2wi: estimator to window it belongs to.
        :param num_workers: number of workers in the cluster.
        :param seed: random state.
        :param task: regression or classification.
        :param eval_metrics: evaluation metrics.
        :param keep_in_mem: boolean, if keep the model in mem, now we do not support model
                             saving of splitting_kfold_wrapper.
        :param cv_seed: cross validation random state.
        :param dtype: data type.
        """
        self.LOGGER = get_logger('estimators.splitting_kfold_wrapper')
        self.dis_level = dis_level
        self.estimators = estimators
        self.ei2wi = ei2wi
        self.num_workers = num_workers
        self.seed = seed
        self.windows = windows
        self.pools = pools
        self.task = task
        self.dtype = dtype
        self.eval_metrics = eval_metrics
        self.keep_in_mem = keep_in_mem
        self.cv_seed = cv_seed
        for ei, est in enumerate(self.estimators):
            # convert estimators from EstimatorConfig to dictionary.
            if isinstance(est, EstimatorConfig):
                self.estimators[ei] = est.get_est_args().copy()

    def scan_shape(self, window, x_shape):
        n, c, h, w = x_shape
        nh = (h - window.win_y) / window.stride_y + 1
        nw = (w - window.win_x) / window.stride_x + 1
        return nh, nw

    def splitting(self, ests, in_win_shapes=None):
        """
        Splitting method.
        Judge if we should to split and how we split.

        :param ests:
        :return:
        """
        assert isinstance(ests, list), 'estimators should be a list, but {}'.format(type(ests))
        should_split, split_scheme = determine_split(dis_level=self.dis_level,
                                                     num_workers=self.num_workers, ests=ests)
        split_ests = []
        split_ests_ratio = []
        split_group = []
        self.LOGGER.info('dis_level = {}, num_workers = {}, num_estimators = {}, should_split? {}'.format(
            self.dis_level, self.num_workers, len(ests), should_split))
        # TODO: what if self.seed is an object of RandomState?
        if self.cv_seed is None:
            self.cv_seed = copy.deepcopy(self.seed)
        if should_split:
            i = 0
            new_ei2wi = dict()
            for ei, est in enumerate(ests):
                wi, wei = self.ei2wi[ei]
                num_trees = est.get('n_estimators', 500)
                est_name = 'win-{}-estimator-{}-{}folds'.format(wi, wei, est.get('n_folds', 3))
                split_ei = split_scheme[ei]
                if split_ei[0] == -1:
                    gen_est = self._init_estimators(est.copy(), wi, wei, self.seed, self.cv_seed,
                                                    splitting=False)
                    split_ests.append(gen_est)
                    split_ests_ratio.append([1.0, ])
                    split_group.append([i, ])
                    i += 1
                    continue
                trees_sum = sum(split_ei)
                total_split = len(split_ei)
                cum_sum_split = split_ei[:]
                for ci in range(1, total_split):
                    cum_sum_split[ci] += cum_sum_split[ci - 1]
                if self.seed is not None:
                    # TODO: what if self.seed is an object of RandomState?
                    common_seed = (self.seed + hash("[estimator] {}".format(est_name))) % 1000000007
                    seeds = [np.random.RandomState(common_seed) for _ in split_ei]
                    for si in range(1, total_split):
                        seeds[si].randint(MAX_RAND_SEED, size=cum_sum_split[si - 1])
                else:
                    seeds = [np.random.mtrand._rand for _ in split_ei]
                    for si in range(1, total_split):
                        seeds[si].randint(MAX_RAND_SEED, size=cum_sum_split[si - 1])
                self.LOGGER.debug('{} trees split to {}'.format(num_trees, split_ei))
                args = est.copy()
                ratio_i = []
                if self.pools is not None:
                    win_shape = in_win_shapes[wi]
                    pool = self.pools[wi][wei]
                else:
                    win_shape = None
                    pool = None
                for sei in range(total_split):
                    args['n_estimators'] = split_ei[sei]
                    sub_est = self._init_estimators(args, wi, wei, seeds[sei], self.cv_seed, splitting=True,
                                                    win_shape=win_shape, pool=pool)
                    split_ests.append(sub_est)
                    ratio_i.append(np.float64(split_ei[sei] / float(trees_sum)))
                    new_ei2wi[i + sei] = (wi, wei)
                split_ests_ratio.append(ratio_i)
                split_group.append([li for li in range(i, i + total_split)])
                i += total_split
            self.ei2wi = new_ei2wi
        else:
            for ei, est in enumerate(ests):
                wi, wei = self.ei2wi[ei]
                if self.pools is not None:
                    win_shape = in_win_shapes[wi]
                    pool = self.pools[wi][wei]
                else:
                    win_shape = None
                    pool = None
                gen_est = self._init_estimators(est.copy(),
                                                wi, wei, self.seed, self.cv_seed, splitting=False,
                                                win_shape=win_shape, pool=pool)
                split_ests.append(gen_est)
                split_ests_ratio.append([1.0, ])
            split_group = [[i, ] for i in range(len(ests))]
        return split_ests, split_ests_ratio, split_group

    def _init_estimators(self, args, wi, ei, seed, cv_seed, splitting=False, win_shape=None, pool=None):
        """
        Initialize distributed kfold wrapper. dumps the seed if seed is a np.random.RandomState.

        :param args:
        :param wi:
        :param ei:
        :param seed:
        :param cv_seed:
        :return:
        """
        est_args = args.copy()
        est_name = 'win-{}-estimator-{}-{}folds'.format(wi, ei, est_args['n_folds'])
        n_folds = int(est_args['n_folds'])
        est_args.pop('n_folds')
        est_type = est_args['est_type']
        est_args.pop('est_type')
        # seed, if seed is not None and is integer, we add it with estimator name.
        # if seed is already a RandomState, just pickle it in order to pass to every worker.
        if seed is not None and not isinstance(seed, np.random.RandomState):
            seed = (seed + hash("[estimator] {}".format(est_name))) % 1000000007
        if isinstance(seed, np.random.RandomState):
            seed = pickle.dumps(seed, pickle.HIGHEST_PROTOCOL)
        # we must keep the cross validation seed same, but keep the seed not the same
        # so that no duplicate forest are generated, but exactly same cross validation datasets are generated.
        if cv_seed is not None and not isinstance(cv_seed, np.random.RandomState):
            cv_seed = (cv_seed + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            cv_seed = (0 + hash("[estimator] {}".format(est_name))) % 1000000007
        return get_dist_estimator_kfold(name=est_name,
                                        n_folds=n_folds,
                                        task=self.task,
                                        est_type=est_type,
                                        eval_metrics=self.eval_metrics,
                                        seed=seed,
                                        dtype=self.dtype,
                                        splitting=splitting,
                                        keep_in_mem=self.keep_in_mem,
                                        est_args=est_args,
                                        cv_seed=cv_seed,
                                        win_shape=win_shape,
                                        pool=pool)

    def fit(self, x_wins_train, y_win):
        """
        Fit. This method do splitting fit and collect/merge results of distributed forests.

        :param x_wins_train:
        :param y_win:
        :return:
        """
        split_ests, split_ests_ratio, split_group = self.splitting(self.estimators)
        self.LOGGER.debug('split_group = {}'.format(split_group))
        self.LOGGER.debug('split_ests_ratio = {}'.format(split_ests_ratio))
        self.LOGGER.debug('ei2wi = {}'.format(self.ei2wi))
        x_wins_train_obj_ids = [ray.put(x_wins_train[wi]) for wi in range(len(x_wins_train))]
        y_win_obj_ids = [ray.put(y_win[wi]) for wi in range(len(y_win))]
        y_stratify = [ray.put(y_win[wi][:, 0]) for wi in range(len(y_win))]
        # the base kfold_wrapper of SplittingKFoldWrapper must be DistributedKFoldWrapper,
        # so with the y_proba_train, y_proba_tests, there is a log info list will be return.
        # so, ests_output is like (y_proba_train, y_proba_tests, logs)
        ests_output = [est.fit_transform.remote(x_wins_train_obj_ids[self.ei2wi[ei][0]],
                                                y_win_obj_ids[self.ei2wi[ei][0]],
                       y_stratify[self.ei2wi[ei][0]]) for ei, est in enumerate(split_ests)]
        est_group = merge_group(split_group, split_ests_ratio, ests_output, self.dtype)
        est_group_result = ray.get(est_group)
        return est_group_result

    def fit_transform(self, x_train, y_train, x_test, y_test):
        x_wins_train = [None for _ in range(len(self.windows))]
        x_wins_test = [None for _ in range(len(self.windows))]
        nhs, nws = [None for _ in range(len(self.windows))], [None for _ in range(len(self.windows))]
        y_win = [None for _ in range(len(self.windows))]
        y_win_test = [None for _ in range(len(self.windows))]
        test_sets = [None for _ in range(len(self.windows))]
        for wi, win in enumerate(self.windows):
            x_wins_train[wi] = win.fit_transform(x_train)
            x_wins_test[wi] = win.fit_transform(x_test)
            _, nhs[wi], nws[wi], _ = x_wins_train[wi].shape
            x_wins_train[wi] = x_wins_train[wi].reshape((x_wins_train[wi].shape[0], -1, x_wins_train[wi].shape[-1]))
            x_wins_test[wi] = x_wins_test[wi].reshape((x_wins_test[wi].shape[0], -1, x_wins_test[wi].shape[-1]))
            y_win[wi] = y_train[:, np.newaxis].repeat(x_wins_train[wi].shape[1], axis=1)
            y_win_test[wi] = None if y_test is None else y_test[:, np.newaxis].repeat(x_wins_test[wi].shape[1], axis=1)
            test_sets[wi] = [('testOfWin{}'.format(wi), x_wins_test[wi], y_win_test[wi])]
            self.LOGGER.debug(
                'x_wins_train[{}] size={}, dtype={}'.format(wi, getmbof(x_wins_train[wi]), x_wins_train[wi].dtype))
            self.LOGGER.debug('y_win[{}] size={}, dtype={}'.format(wi, getmbof(y_win[wi]), y_win[wi].dtype))
            self.LOGGER.debug(
                'x_wins_train[{}] size={}, dtype={}'.format(wi, getmbof(x_wins_train[wi]), x_wins_train[wi].dtype))
            self.LOGGER.debug('y_win[{}] size={}, dtype={}'.format(wi, getmbof(y_win[wi]), y_win[wi].dtype))
        split_ests, split_ests_ratio, split_group = self.splitting(self.estimators,
                                                                   in_win_shapes=[(nhs[k], nws[k]) for k in range(len(nhs))])
        self.LOGGER.debug('split_group = {}'.format(split_group))
        self.LOGGER.debug('split_ests_ratio = {}'.format(["{:.18f}".format(r) for r in split_ests_ratio[0]]))
        self.LOGGER.debug('new ei2wi = {}'.format(self.ei2wi))
        x_wins_train_obj_ids = [ray.put(x_wins_train[wi]) for wi in range(len(x_wins_train))]
        y_win_obj_ids = [ray.put(y_win[wi]) for wi in range(len(y_win))]
        y_stratify = [ray.put(y_win[wi][:, 0]) for wi in range(len(y_win))]
        test_sets_obj_ids = [ray.put(test_sets[wi]) for wi in range(len(test_sets))]
        self.LOGGER.info("[NO Lazy Scan] Put all input down!")
        ests_output = [est.fit_transform.remote(x_wins_train_obj_ids[self.ei2wi[ei][0]],
                                                y_win_obj_ids[self.ei2wi[ei][0]], y_stratify[self.ei2wi[ei][0]],
                                                test_sets=test_sets_obj_ids[self.ei2wi[ei][0]])
                       for ei, est in enumerate(split_ests)]
        est_group = merge_group(split_group, split_ests_ratio, ests_output, self.dtype)
        est_group_result = ray.get(est_group)
        return est_group_result

    def fit_transform_lazyscan(self, x_train, y_train, x_test, y_test):
        """
        Fit and transform. This method do splitting fit and collect/merge results of distributed forests.

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        win_shapes = []
        for win in self.windows:
            nh, nw = self.scan_shape(win, x_train.shape)
            win_shapes.append((nh, nw))
        split_ests, split_ests_ratio, split_group = self.splitting(self.estimators, in_win_shapes=win_shapes)
        self.LOGGER.debug('split_group = {}'.format(split_group))
        self.LOGGER.debug('split_ests_ratio = {}'.format(split_ests_ratio))
        self.LOGGER.debug('new ei2wi = {}'.format(self.ei2wi))
        x_train_id = ray.put(x_train)
        y_train_id = ray.put(y_train)
        x_test_id = ray.put(x_test)
        y_test_id = ray.put(y_test)
        self.LOGGER.info("[Lazy Scan] Put all input down!")
        # the base kfold_wrapper of SplittingKFoldWrapper must be DistributedKFoldWrapper,
        # so with the y_proba_train, y_proba_tests, there is a log info list will be return.
        # so, ests_output is like (y_proba_train, y_proba_tests, logs)
        ests_output = [est.fit_transform_lazyscan.remote(x_train_id, y_train_id, x_test_id, y_test_id,
                                                         self.windows[self.ei2wi[ei][0]], self.ei2wi[ei][0],
                                                         fl.get_redis_address())
                       for ei, est in enumerate(split_ests)]
        est_group = merge_group(split_group, split_ests_ratio, ests_output, self.dtype)
        est_group_result = ray.get(est_group)
        return est_group_result


class CascadeSplittingKFoldWrapper(object):
    """
    Wrapper for splitting forests to smaller forests.
    TODO: support intelligent load-aware splitting method.
    """
    def __init__(self, dis_level=0, estimators=None, num_workers=None, seed=None, task='classification',
                 eval_metrics=None, keep_in_mem=False, cv_seed=None, dtype=np.float32,
                 layer_id=None, train_start_ends=None, x_train_group_or_id=None, x_test_group_or_id=None):
        """
        Initialize CascadeSplittingKFoldWrapper.

        :param dis_level: distributed level, or parallelization level, 0 / 1 / 2 / 3
                           0 means lowest parallelization level, parallelization is len(self.est_configs).
                           1 means triple-split.
                           2 means bin-split.
                           3 means avg split
        :param estimators: base estimators.
        :param num_workers: number of workers in the cluster.
        :param seed: random state.
        :param task: regression or classification.
        :param eval_metrics: evaluation metrics.
        :param keep_in_mem: boolean, if keep the model in mem, now we do not support model
                             saving of cascade_splitting_kfold_wrapper.
        :param cv_seed: cross validation random state.
        :param dtype: data type.
        """
        self.LOGGER = get_logger('estimators.cascade_splitting_kfold_wrapper')
        self.dis_level = dis_level
        self.estimators = estimators
        self.num_workers = num_workers
        self.seed = seed
        self.task = task
        self.dtype = dtype
        self.eval_metrics = eval_metrics
        self.keep_in_mem = keep_in_mem
        self.cv_seed = cv_seed
        # cascade
        self.layer_id = layer_id
        for ei, est in enumerate(self.estimators):
            # convert estimators from EstimatorConfig to dictionary.
            if isinstance(est, EstimatorConfig):
                self.estimators[ei] = est.get_est_args().copy()
        self.x_train_group_or_id = x_train_group_or_id
        self.x_test_group_or_id = x_test_group_or_id
        self.train_start_ends = train_start_ends

    def splitting(self, ests):
        """
        Splitting method.
        Judge if we should to split and how we split.

        :param ests:
        :return:
        """
        assert isinstance(ests, list), 'estimators should be a list, but {}'.format(type(ests))
        should_split, split_scheme = determine_split(dis_level=self.dis_level,
                                                     num_workers=self.num_workers, ests=ests)
        split_ests = []
        split_ests_ratio = []
        split_group = []
        # self.LOGGER.info('dis_level = {}, num_workers = {}, num_estimators = {}, should_split? {}'.format(
        #     self.dis_level, self.num_workers, len(ests), should_split))
        if self.cv_seed is None:
            self.cv_seed = copy.deepcopy(self.seed)
        if should_split:
            i = 0
            for ei, est in enumerate(ests):
                num_trees = est.get('n_estimators', 500)
                est_name = 'layer-{}-estimator-{}-{}folds'.format(self.layer_id, ei,
                                                                  est.get('n_folds', 3))
                split_ei = split_scheme[ei]
                if split_ei[0] == -1:
                    gen_est = self._init_estimators(est.copy(), self.layer_id, ei, self.seed, self.cv_seed,
                                                    splitting=False)
                    split_ests.append(gen_est)
                    split_ests_ratio.append([1.0, ])
                    split_group.append([i, ])
                    i += 1
                    continue
                trees_sum = sum(split_ei)
                total_split = len(split_ei)
                cum_sum_split = split_ei[:]
                for ci in range(1, total_split):
                    cum_sum_split[ci] += cum_sum_split[ci - 1]
                if self.seed is not None:
                    common_seed = (self.seed + hash("[estimator] {}".format(est_name))) % 1000000007
                    seeds = [np.random.RandomState(common_seed) for _ in split_ei]
                    for si in range(1, total_split):
                        seeds[si].randint(MAX_RAND_SEED, size=cum_sum_split[si - 1])
                else:
                    seeds = [np.random.mtrand._rand for _ in split_ei]
                    for si in range(1, total_split):
                        seeds[si].randint(MAX_RAND_SEED, size=cum_sum_split[si - 1])
                self.LOGGER.debug('{} trees split to {}'.format(num_trees, split_ei))
                # self.LOGGER.debug('seeds = {}'.format([seed.get_state()[-3] for seed in seeds]))
                args = est.copy()
                ratio_i = []
                for sei in range(total_split):
                    args['n_estimators'] = split_ei[sei]
                    sub_est = self._init_estimators(args, self.layer_id, ei, seeds[sei], self.cv_seed, splitting=True)
                    split_ests.append(sub_est)
                    ratio_i.append(np.float64(split_ei[sei]/float(trees_sum)))
                split_ests_ratio.append(ratio_i)
                split_group.append([li for li in range(i, i + total_split)])
                i += total_split
        else:
            for ei, est in enumerate(ests):
                gen_est = self._init_estimators(est.copy(), self.layer_id, ei, self.seed, self.cv_seed, splitting=False)
                split_ests.append(gen_est)
                split_ests_ratio.append([1.0, ])
            split_group = [[i, ] for i in range(len(ests))]
        return split_ests, split_ests_ratio, split_group

    def _init_estimators(self, args, layer_id, ei, seed, cv_seed, splitting=False):
        """
        Initialize distributed kfold wrapper. dumps the seed if seed is a np.random.RandomState.

        :param args:
        :param layer_id:
        :param ei:
        :param seed:
        :param cv_seed:
        :return:
        """
        est_args = args.copy()
        est_name = 'layer-{}-estimator-{}-{}folds'.format(layer_id, ei, est_args['n_folds'])
        n_folds = int(est_args['n_folds'])
        est_args.pop('n_folds')
        est_type = est_args['est_type']
        est_args.pop('est_type')
        # seed, if seed is not None and is integer, we add it with estimator name.
        # if seed is already a RandomState, just pickle it in order to pass to every worker.
        if seed is not None and not isinstance(seed, np.random.RandomState):
            seed = (seed + hash("[estimator] {}".format(est_name))) % 1000000007
        if isinstance(seed, np.random.RandomState):
            seed = pickle.dumps(seed, pickle.HIGHEST_PROTOCOL)
        # we must keep the cross validation seed same, but keep the seed not the same
        # so that no duplicate forest are generated, but exactly same cross validation datasets are generated.
        if cv_seed is not None and not isinstance(cv_seed, np.random.RandomState):
            cv_seed = (cv_seed + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            cv_seed = (0 + hash("[estimator] {}".format(est_name))) % 1000000007
        return get_dist_estimator_kfold(name=est_name,
                                        n_folds=n_folds,
                                        task=self.task,
                                        est_type=est_type,
                                        eval_metrics=self.eval_metrics,
                                        seed=seed,
                                        dtype=self.dtype,
                                        splitting=splitting,
                                        keep_in_mem=self.keep_in_mem,
                                        est_args=est_args,
                                        cv_seed=cv_seed,
                                        train_start_ends=self.train_start_ends,
                                        x_train_group_or_id=self.x_train_group_or_id,
                                        x_test_group_or_id=self.x_test_group_or_id)

    def fit(self, x_train, y_train, y_stratify):
        # TODO: support lazy assemble later
        split_ests, split_ests_ratio, split_group = self.splitting(self.estimators)
        self.LOGGER.debug('split_group = {}'.format(split_group))
        self.LOGGER.debug('split_ests_ratio = {}'.format(split_ests_ratio))
        x_train_obj_id = ray.put(x_train)
        y_train_obj_id = ray.put(y_train)
        y_stratify_obj_id = ray.put(y_stratify)
        # the base kfold_wrapper of SplittingKFoldWrapper must be DistributedKFoldWrapper,
        # so with the y_proba_train, y_proba_tests, there is a log info list will be return.
        # so, ests_output is like (y_proba_train, y_proba_tests, logs)
        ests_output = [est.fit_transform.remote(x_train_obj_id, y_train_obj_id, y_stratify_obj_id,
                                                test_sets=None)
                       for est in split_ests]
        est_group = merge_group(split_group, split_ests_ratio, ests_output, self.dtype)
        est_group_result = ray.get(est_group)
        return est_group_result, split_ests, split_group

    def fit_transform(self, x_train, y_train, y_stratify, test_sets=None):
        split_ests, split_ests_ratio, split_group = self.splitting(self.estimators)
        self.LOGGER.debug('split_group = {}'.format(split_group))
        self.LOGGER.debug('split_ests_ratio = {}'.format(split_ests_ratio))
        x_train_obj_id = ray.put(x_train)
        y_train_obj_id = ray.put(y_train)
        y_stratify_obj_id = ray.put(y_stratify)
        test_sets_obj_id = ray.put(test_sets)
        # the base kfold_wrapper of SplittingKFoldWrapper must be DistributedKFoldWrapper,
        # so with the y_proba_train, y_proba_tests, there is a log info list will be return.
        # so, ests_output is like (y_proba_train, y_proba_tests, logs)
        ests_output = [est.fit_transform.remote(x_train_obj_id, y_train_obj_id, y_stratify_obj_id,
                                                test_sets=test_sets_obj_id)
                       for est in split_ests]
        # k = ray.get(ests_output)
        # try:
        #     est_group_result = local_merge_group(split_group, split_ests_ratio, k, self.dtype)
        # except Exception, e:
        #     print(e)
        est_group = merge_group(split_group, split_ests_ratio, ests_output, self.dtype)
        try:
            est_group_result = ray.get(est_group)
        except Exception, e:
            print(e)
            raise Exception(e)
        return est_group_result, split_ests, split_group


@ray.remote
def merge(tup_1, ratio1, tup_2, ratio2, dtype=np.float32):
    """
    Merge 2 tuple of (y_proba_train, y_proba_tests, logs).
    NOTE: Now in splitting mode, the logs will be approximate log, because we should calculate metrics after collect
     y_proba, but now we calculate metrics on small forests, respectively, so the average of metrics of two forests
     must be inaccurate, you should mind this and do not worry about it. After merge, the average y_proba is the true
     proba, so the final results is absolutely right!

    :param tup_1: tuple like (y_proba_train, y_proba_tests, logs)
    :param ratio1: ratio occupied by tuple 1
    :param tup_2: tuple like (y_proba_train, y_proba_tests, logs)
    :param ratio2: ratio occupied by tuple 2
    :param dtype: result data type. when we invoke merge, we must in splitting mode, in this mode, we will keep
                   origin float-point precision (may be float64), and when we combine result of small forests, we
                   should convert the data type to self.dtype (may be float32), which can reduce memory and
                   communication overhead.
    :return: tuple of results, (y_proba_train: numpy.ndarray,
              y_proba_tests: numpy.ndarray, may be None, list of y_proba_test,
              logs: list of tuple which contains log level, log info)
    """
    tests = []
    for i in range(len(tup_1[1])):
        tests.append((tup_1[1][i] * ratio1 + tup_2[1][i] * ratio2).astype(dtype))
    mean_dict = defaultdict(float)
    logs = []
    for t1 in tup_1[2]:
        if t1[0] == 'INFO':
            mean_dict[','.join(t1[:2])] += t1[2] * ratio1
        else:
            logs.append(t1)
    for t2 in tup_2[2]:
        if t2[0] == 'INFO':
            mean_dict[','.join(t2[:2])] += t2[2] * ratio2
        else:
            logs.append(t2)
    for key in mean_dict.keys():
        # mean_dict[key] = mean_dict[key]
        key_split = key.split(',')
        logs.append((key_split[0], key_split[1], mean_dict[key]))
    logs.sort()
    return (tup_1[0] * ratio1 + tup_2[0] * ratio2).astype(dtype), tests, logs


def local_merge(tup_1, ratio1, tup_2, ratio2, dtype=np.float32):
    """
    Merge 2 tuple of (y_proba_train, y_proba_tests, logs).
    NOTE: Now in splitting mode, the logs will be approximate log, because we should calculate metrics after collect
     y_proba, but now we calculate metrics on small forests, respectively, so the average of metrics of two forests
     must be inaccurate, you should mind this and do not worry about it. After merge, the average y_proba is the true
     proba, so the final results is absolutely right!

    :param tup_1: tuple like (y_proba_train, y_proba_tests, logs)
    :param ratio1: ratio occupied by tuple 1
    :param tup_2: tuple like (y_proba_train, y_proba_tests, logs)
    :param ratio2: ratio occupied by tuple 2
    :param dtype: result data type. when we invoke merge, we must in splitting mode, in this mode, we will keep
                   origin float-point precision (may be float64), and when we combine result of small forests, we
                   should convert the data type to self.dtype (may be float32), which can reduce memory and
                   communication overhead.
    :return: tuple of results, (y_proba_train: numpy.ndarray,
              y_proba_tests: numpy.ndarray, may be None, list of y_proba_test,
              logs: list of tuple which contains log level, log info)
    """
    tests = []
    for i in range(len(tup_1[1])):
        tests.append((tup_1[1][i] * ratio1 + tup_2[1][i] * ratio2).astype(dtype))
    mean_dict = defaultdict(float)
    logs = []
    for t1 in tup_1[2]:
        if t1[0] == 'INFO':
            mean_dict[','.join(t1[:2])] += t1[2] * ratio1
        else:
            logs.append(t1)
    for t2 in tup_2[2]:
        if t2[0] == 'INFO':
            mean_dict[','.join(t2[:2])] += t2[2] * ratio2
        else:
            logs.append(t2)
    for key in mean_dict.keys():
        # mean_dict[key] = mean_dict[key]
        key_split = key.split(',')
        logs.append((key_split[0], key_split[1], mean_dict[key]))
    logs.sort()
    return (tup_1[0] * ratio1 + tup_2[0] * ratio2).astype(dtype), tests, logs


def est_class_from_type(task, est_type):
    """
    Get estimator class from task ('classification' or 'regression') and estimator type (a string).

    :param task: what task does this estimator to execute ('classification' or 'regression')
    :param est_type: estimator type (a string)
    :return: a concrete estimator instance
    """
    return str2est_class[task][est_type]


def get_estimator(name, task, est_type, est_args):
    """
    Get an estimator.

    :param name: estimator name
    :param task: what task does this estimator to execute ('classification' or 'regression')
    :param est_type: estimator type (a string)
    :param est_args: estimator arguments
    :return: a concrete estimator instance
    """
    est_class = est_class_from_type(task, est_type)
    return est_class(name, est_args)


def get_estimator_kfold(name, n_folds=3, task='classification', est_type='FLRF', eval_metrics=None, seed=None,
                        dtype=np.float32, cache_dir=None, keep_in_mem=True, est_args=None, cv_seed=None,
                        win_shape=None, pool=None):
    """
    A factory method to get a k-fold estimator.

    :param name: estimator name
    :param n_folds: how many folds to execute in cross validation
    :param task: what task does this estimator to execute ('classification' or 'regression')
    :param est_type: estimator type (a string)
    :param eval_metrics: evaluation metrics. [Default: Accuracy (classification), MSE (regression)]
    :param seed: random seed
    :param dtype: data type
    :param cache_dir: data cache dir to cache intermediate data
    :param keep_in_mem: whether keep the model in memory
    :param est_args: estimator arguments
    :param cv_seed: random seed for cross validation
    :param win_shape: None
    :param pool: None
    :return: a KFoldWrapper instance of concrete estimator
    """
    # est_class = est_class_from_type(task, est_type)
    if eval_metrics is None:
        if task == 'classification':
            eval_metrics = [Accuracy('accuracy')]
        else:
            eval_metrics = [MSE('MSE')]
    return KFoldWrapper(name,
                        n_folds,
                        task,
                        est_type,
                        seed=seed,
                        dtype=dtype,
                        eval_metrics=eval_metrics,
                        cache_dir=cache_dir,
                        keep_in_mem=keep_in_mem,
                        est_args=est_args,
                        cv_seed=cv_seed)


def get_dist_estimator_kfold(name, n_folds=3, task='classification', est_type='FLRF', eval_metrics=None,
                             seed=None, dtype=np.float32, splitting=False, cache_dir=None,
                             keep_in_mem=True, est_args=None, cv_seed=None, win_shape=None, pool=None,
                             train_start_ends=None, x_train_group_or_id=None, x_test_group_or_id=None):
    """
    A factory method to get a distributed k-fold estimator.

    :param name: estimator name
    :param n_folds: how many folds to execute in cross validation
    :param task: what task does this estimator to execute ('classification' or 'regression')
    :param est_type: estimator type (a string)
    :param eval_metrics: evaluation metrics. [Default: Accuracy (classification), MSE (regression)]
    :param seed: random seed
    :param dtype: data type
    :param splitting: whether in splitting mode
    :param cache_dir: data cache dir to cache intermediate data
    :param keep_in_mem: whether keep the model in memory
    :param est_args: estimator arguments
    :param cv_seed: random seed for cross validation
    :param win_shape: None
    :param pool: None
    :param train_start_ends: None
    :param x_train_group_or_id: None
    :param x_test_group_or_id: None
    :return: a KFoldWrapper instance of concrete estimator
    """
    # est_class = est_class_from_type(task, est_type)
    # print("Now I am ", est_class.__class__)
    if eval_metrics is None:
        if task == 'classification':
            eval_metrics = [Accuracy('accuracy')]
        else:
            eval_metrics = [MSE('MSE')]
    return DistributedKFoldWrapper.remote(name=name,
                                          n_folds=n_folds,
                                          task=task,
                                          est_type=est_type,
                                          seed=seed,
                                          dtype=dtype,
                                          splitting=splitting,
                                          eval_metrics=eval_metrics,
                                          cache_dir=cache_dir,
                                          keep_in_mem=keep_in_mem,
                                          est_args=est_args,
                                          cv_seed=cv_seed,
                                          win_shape=win_shape,
                                          pool=pool,
                                          train_start_ends=train_start_ends,
                                          x_train_group_or_id=x_train_group_or_id,
                                          x_test_group_or_id=x_test_group_or_id)


def determine_split(dis_level, num_workers, ests):
    """
    Greedy split finding algorithm and bin-Split and non-Split.

    :param dis_level: 0, 2, 3
    :param num_workers: the number of workers
    :param ests: estimators, each with some trees
    :return: should_split(True or False), split_scheme
    """
    if dis_level == 0:
        return False, []
    if dis_level == 3:
        splits = []
        for i, est in enumerate(ests):
            num_trees = est.get('n_estimators', 500)
            if est.get('est_type') in ['FLRF', 'FLCRF']:
                splits.append([num_trees / 3,
                               (num_trees-num_trees/3)/2,
                               (num_trees-num_trees/3)-(num_trees-num_trees/3)/2])
            else:
                splits.append([-1])
        return True, splits
    if dis_level == 2:
        splits = []
        for i, est in enumerate(ests):
            num_trees = est.get('n_estimators', 500)
            if est.get('est_type') in ['FLRF', 'FLCRF']:
                splits.append([num_trees / 2, num_trees - num_trees / 2])
            else:
                splits.append([-1])
        return True, splits
    if dis_level == 1:
        forest_ests = [est for est in ests if est.get('est_type') in ['FLRF', 'FLCRF']]
        non_forest_estimators = len(ests) - len(forest_ests)
        num_trees = map(lambda x: x.get('n_estimators', 500), forest_ests)
        tree_sum = sum(num_trees)
        if tree_sum <= 0:
            return False, []
        # As every node generally capable of handling 2 forests simultaneously, we regard one node as two nodes.
        # But when input data are large, one node may not be able to handle the concurrent training of two forests.
        # So this parameter should be considered later.
        # TODO: Consider when one node can handle two tasks.
        denom = max(1, num_workers * 2 - non_forest_estimators)
        avg_trees = int(math.ceil(tree_sum / float(denom)))
        splits = []
        should_split = False
        forest_idx_tuples = []
        for i, est in enumerate(ests):
            split_i = []
            if est.get('est_type') not in ['FLRF', 'FLCRF']:
                split_i.append(-1)
                splits.append(split_i)
                continue
            if not should_split and num_trees[i] > avg_trees:
                should_split = True
            while num_trees[i] > avg_trees:
                split_i.append(avg_trees)
                forest_idx_tuples.append((avg_trees, i))
                num_trees[i] -= avg_trees
            if num_trees[i] > 0:
                split_i.append(num_trees[i])
                forest_idx_tuples.append((num_trees[i], i))
            splits.append(split_i)
        make_span_should_split, splits = greedy_makespan_split(splits, avg_trees, denom, forest_idx_tuples)
        return should_split or make_span_should_split, splits
    if 500 >= dis_level >= 4:
        splits = []
        for i, est in enumerate(ests):
            num_trees = est.get('n_estimators', 500)
            dis_level = min(dis_level, num_trees)
            if est.get('est_type') in ['FLRF', 'FLCRF']:
                tmp = []
                div = num_trees / dis_level
                mod = num_trees % dis_level
                for v in range(dis_level):
                    if mod > 0:
                        tmp.append(div + 1)
                        mod -= 1
                    else:
                        tmp.append(div)
                splits.append(tmp)
            else:
                splits.append([-1])
        return True, splits
    return False, None


def merge_group(split_group, split_ests_ratio, ests_output, self_dtype):
    """
    Merge split estimators output.

    :param split_group: [[0, 1, 2], [3, 4, 5]]
    :param split_ests_ratio: [[0.332, 0.334, 0.334], [0.332, 0.334, 0.334]]
    :param ests_output: [out0, out1, out2, out3, out4, out5]
    :param self_dtype: np.float32 or np.float64
    :return:
    """
    est_group = []
    for gi, grp in enumerate(split_group):
        ests_ratio = split_ests_ratio[gi]
        assert equaleps(sum(ests_ratio), 1.0), "The sum of est_ratio is not equal to 1, but {}!".format(sum(ests_ratio))
        group = [ests_output[i] for i in grp[:]]
        if len(grp) > 2:
            while len(group) > 1:
                if len(group) == 2:
                    dtype = self_dtype
                else:
                    dtype = np.float64
                group = group[2:] + [merge.remote(group[0], ests_ratio[0],
                                                  group[1], ests_ratio[1], dtype=dtype)]
                ests_ratio = ests_ratio[2:] + [np.float64(1.0)]
            est_group.append(group[0])
        elif len(grp) == 2:
            # tree reduce
            est_group.append(merge.remote(group[0], ests_ratio[0],
                                          group[1], ests_ratio[1], dtype=self_dtype))
        else:
            est_group.append(group[0])
    return est_group


def equaleps(a, b):
    eps = 1e-10
    if b + eps >= a >= b - eps:
        return True
    return False


def local_merge_group(split_group, split_ests_ratio, ests_output, self_dtype):
    """
    Merge split estimators output.

    :param split_group: [[0, 1, 2], [3, 4, 5]]
    :param split_ests_ratio: [[0.332, 0.334, 0.334], [0.332, 0.334, 0.334]]
    :param ests_output: [out0, out1, out2, out3, out4, out5]
    :param self_dtype: np.float32 or np.float64
    :return:
    """
    est_group = []
    for gi, grp in enumerate(split_group):
        ests_ratio = split_ests_ratio[gi]
        assert equaleps(sum(ests_ratio), 1.0), "The sum of est_ratio is not equal to 1, but {}!".format(sum(ests_ratio))
        group = [ests_output[i] for i in grp[:]]
        if len(grp) > 2:
            while len(group) > 1:
                if len(group) == 2:
                    dtype = self_dtype
                else:
                    dtype = np.float64
                group = group[2:] + [local_merge(group[0], ests_ratio[0],
                                                 group[1], ests_ratio[1], dtype=dtype)]
                ests_ratio = ests_ratio[2:] + [np.float64(1.0)]
            est_group.append(group[0])
        elif len(grp) == 2:
            # tree reduce
            est_group.append(local_merge(group[0], ests_ratio[0],
                                         group[1], ests_ratio[1], dtype=self_dtype))
        else:
            est_group.append(group[0])
    return est_group


def find_first_can_put_entirely(lis, start, x, capacity):
    for i in range(start, len(lis)):
        if lis[i] + x <= capacity:
            return i
    return -1


def greedy_makespan_split(splits, avg, num_workers, forest_idx_tuples):
    after_tuples = []
    heap = []
    should_split = False
    for tup in forest_idx_tuples:
        heapq.heappush(heap, (-tup[0], tup[1]))
    num_tasks = len(forest_idx_tuples)
    fill_in = [0 for _ in range(num_workers)]
    first_unfilled = -1
    for i in range(num_workers):
        top = heapq.heappop(heap)
        # print(top)
        fill_in[i] += -top[0]
        after_tuples.append((-top[0], top[1]))
        if fill_in[i] < avg and first_unfilled == -1:
            first_unfilled = i
    for i in range(num_workers, num_tasks):
        if first_unfilled >= len(fill_in):
            print("Index out")
            break
        # for task i
        if not heap:
            print("Heap empty")
            break
        top = heapq.heappop(heap)
        # print(top)
        can_put_idx = find_first_can_put_entirely(fill_in, first_unfilled, -top[0], avg)
        tag = 0
        if can_put_idx == -1:
            can_put_idx = first_unfilled
            tag = 1
        if fill_in[can_put_idx] + -top[0] >= avg:
            if (-top[0] - avg + fill_in[can_put_idx]) > 0:
                heapq.heappush(heap, (-(-top[0] - avg + fill_in[can_put_idx]), top[1]))
                should_split = True
            after_tuples.append((avg - fill_in[can_put_idx], top[1]))
            fill_in[can_put_idx] = avg
            can_put_idx += 1
            first_unfilled += tag
        else:
            fill_in[can_put_idx] += -top[0]
            after_tuples.append((-top[0], top[1]))
    while heap:
        top = heapq.heappop(heap)
        after_tuples.append((-top[0], top[1]))
    new_splits = [[] for _ in range(len(splits))]
    for i in range(len(new_splits)):
        if splits[i][0] == -1:
            new_splits[i].append(-1)
    for tup in after_tuples:
        num_forest = tup[0]
        idx = tup[1]
        new_splits[idx].append(num_forest)
    return should_split, new_splits


def fuse():
    if virtual_memory().used > virtual_memory().total * 0.95:
        # raise EnvironmentError("Too heavy node! Used {:.1f}% memory in {} Killed!".format(
        #     virtual_memory().used / virtual_memory().total, ray.services.get_node_ip_address()))
        import os
        os.system('ray stop')

