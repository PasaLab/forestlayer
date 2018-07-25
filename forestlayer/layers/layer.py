# -*- coding:utf-8 -*-
"""
Base layers definition.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import numpy as np
import datetime
import os.path as osp
try:
    import cPickle as pickle
except ImportError:
    import pickle
import ray
import forestlayer as fl
from collections import defaultdict
from ..utils.log_utils import get_logger, list2str
from ..utils.layer_utils import check_list_depth
from ..utils.storage_utils import check_dir, getmbof, output_disk_path, load_disk_cache, save_disk_cache
from ..utils.metrics import Metrics, Accuracy, AUC, MSE, RMSE
from ..estimators import get_estimator_kfold, get_dist_estimator_kfold, EstimatorConfig
from ..estimators.kfold_wrapper import SplittingKFoldWrapper, CascadeSplittingKFoldWrapper
from forestlayer.backend.backend import get_num_nodes
str2dtype = {'float16': np.float16, 'float32': np.float32, 'float64': np.float64}


class Layer(object):
    """Abstract base layer class.

    # Properties
        name: String
        input_shape: Shape tuple.
        output_shape: Shape tuple.
        input, output: Input / output tensors.
        num_estimators: Number of estimators in this layer.
        estimators: Estimators in this layer.
        summary_info: Summary information of this layer.

    # Methods
        call(x): Where the layer logic lives.
        __call__(x): Wrapper around the layer logic (`call`).

    # Class Methods
        from_config(config)
    """
    def __init__(self, batch_size=None, dtype=np.float32, name=None):
        """
        Initialize a layer.

        :param batch_size:
        :param dtype:
        :param name:
        """
        self.LOGGER = get_logger('layer')
        self.batch_size = batch_size
        if not name:
            prefix = self.__class__.__name__
            name = _to_snake_case(prefix) + "_" + str(id(self))
        self.name = name
        # Set dtype.
        if dtype is None:
            dtype = np.float32
        elif isinstance(dtype, basestring):
            dtype = str2dtype[dtype]
        self.dtype = dtype
        # num of workers, the basis of the split, default is None, which means un-set.
        self._num_workers = None
        # distribute, identify whether use distributed training. default is False.
        self.distribute = False

    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, n):
        # TODO: layer fit/fit_transform num_workers calculation
        assert isinstance(n, int), 'num_workers should be int, but {}'.format(type(n))
        assert n > 0, 'num_workers should be set to a positive number. but {}'.format(n)
        self._num_workers = n

    def init_num_workers(self):
        if self._num_workers is not None:
            return
        nodes, num_nodes = get_num_nodes()
        self._num_workers = num_nodes
        self.LOGGER.info('Get number of workers: {}, total {}'.format(nodes, num_nodes))

    def call(self, x_trains):
        raise NotImplementedError

    def __call__(self, x_trains):
        raise NotImplementedError

    def fit(self, x_trains, y_trains):
        """
        Fit datasets, return a list or single ndarray: train_outputs.
        NOTE: may change x_trains, y_trains. For efficiency, we do not keep the idempotency of a single layer, but
         we keep the idempotency when you use `Graph` to fit. That's to say, if you create a `Layer` instance called
         `layer`, and do `layer.fit(x_trains, y_trains)`, the input x_trains might be changed! But if you use a
         Graph() to wrapper it, the input will never be changed! So we recommend to use
         ```
         >>> graph = Graph()
         >>> graph.add(layer)
         >>> graph.fit(x_trains, y_trains)
         ```

        :param x_trains: train data
        :param y_trains: train labels
        :return: train_outputs
        """
        raise NotImplementedError

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
        """
        Fit and Transform datasets, return two lists or two single ndarrays: train_outputs, test_outputs.

        :param x_trains: train datasets
        :param y_trains: train labels
        :param x_tests: test datasets
        :param y_tests: test labels
        :return: train_outputs, test_outputs
        """
        raise NotImplementedError

    def transform(self, inputs):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError

    def predict_proba(self, inputs):
        raise NotImplementedError

    def evaluate(self, inputs, labels):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    @property
    def summary_info(self):
        return self.__str__()


class DataCachingMixin(object):
    """
    WIP DataCachingMixin.
    """
    def __init__(self, cache_in_disk=False, data_save_dir=None):
        self.cache_in_disk = cache_in_disk
        self.data_save_dir = data_save_dir

    def _check_disk_cache(self, x, phase):
        if not self.cache_in_disk or not self.data_save_dir:
            return False
        data_path = self._get_disk_path(x, phase)
        if osp.exists(data_path):
            return data_path
        return False

    def _get_disk_path(self, x, phase):
        raise NotImplementedError


class MultiGrainScanLayer(Layer):
    """
    Multi-grain Scan Layer
    """
    def __init__(self, batch_size=None, dtype=np.float32, name=None, task='classification',
                 windows=None, est_for_windows=None, n_class=None, keep_in_mem=False,
                 cache_in_disk=False, data_save_dir=None, eval_metrics=None, seed=None,
                 distribute=False, dis_level=1, verbose_dis=True, num_workers=None, lazyscan=True,
                 pre_pools=None):
        """
        Initialize a multi-grain scan layer.

        :param batch_size:
        :param dtype:
        :param name:
        :param task:
        :param windows:
        :param est_for_windows:
        :param n_class:
        :param keep_in_mem:
        :param eval_metrics:
        :param seed:
        :param distribute: whether use distributed training. If use, you should `import ray`
                           and write `ray.init(<redis-address>)` at the beginning of the main program.
        :param dis_level: distributed level, or parallelization level, 0 / 1 / 2 / 3
                           0 means lowest parallelization level, parallelization is len(self.est_configs).
                           1 means triple-split.
                           2 means bin-split.
                           3 means avg split
        :param lazyscan: if open lazyscan, default is True.
        :param pre_pools: if we does pre pooling, we fill this variable.
        """
        if not name:
            prefix = 'multi_grain_scan'
            name = prefix + '_' + str(id(self))
        super(MultiGrainScanLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        self.windows = windows  # [Win, Win, Win, ...]
        self.est_for_windows = est_for_windows  # [[est1, est2], [est1, est2], [est1, est2], ...]
        # TODO: check windows and est_for_windows
        assert task in ['regression', 'classification'], 'task unknown! task = {}'.format(task)
        self.task = task
        if self.task == 'regression':
            self.n_class = 1
        else:
            assert n_class is not None, 'n_class should not be None!'
            self.n_class = n_class
        self.seed = seed
        assert isinstance(distribute, bool), 'distribute variable should be Boolean, but {}'.format(type(distribute))
        self.distribute = distribute
        self.dis_level = dis_level
        self.verbose_dis = verbose_dis
        # initialize num_workers if not provided
        if distribute is True:
            if num_workers is None:
                self.init_num_workers()
            else:
                self.num_workers = num_workers
        self.keep_in_mem = keep_in_mem
        self.cache_in_disk = cache_in_disk
        self.data_save_dir = data_save_dir
        self.eval_metrics = eval_metrics
        self.lazy_scan = lazyscan
        if fl.get_redis_address() is None or fl.get_redis_address().split(':')[0] == "127.0.0.1":
            self.LOGGER.warn("In standalone mode, it's unnecessary to enable lazyscan, we close it!")
            self.lazy_scan = False
        self.pre_pools = pre_pools  # [[pool1, pool2], [pool1, pool2], [pool1, pool2], ...]

    def call(self, x_train, **kwargs):
        pass

    def __call__(self, x_train, **kwargs):
        pass

    def scan(self, window, x):
        """
        Multi-grain scan.

        :param window:
        :param x:
        :return:
        """
        return window.fit_transform(x)

    def scan_shape(self, window, x_shape):
        n, c, h, w = x_shape
        nh = (h - window.win_y) / window.stride_y + 1
        nw = (w - window.win_x) / window.stride_x + 1
        return nh, nw

    def pool_shape(self, pool, win_shape):
        h, w = win_shape
        nh = (h - 1) / pool.win_x + 1
        nw = (w - 1) / pool.win_y + 1
        return nh, nw

    def _init_estimator(self, est_arguments, wi, ei, win_shape=None, pool=None):
        """
        Initialize an estimator.

        :param est_arguments:
        :param wi:
        :param ei:
        :return:
        """
        est_args = est_arguments.get_est_args()
        est_name = 'win-{}-estimator-{}-{}folds'.format(wi, ei, est_args['n_folds'])
        n_folds = int(est_args['n_folds'])
        est_args.pop('n_folds')
        est_type = est_args['est_type']
        est_args.pop('est_type')
        # seed
        if self.seed is not None:
            seed = (self.seed + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            seed = None
        if self.distribute:
            get_est_func = get_dist_estimator_kfold
        else:
            get_est_func = get_estimator_kfold
            # single machine mode, no need to pre pool
            win_shape = None
            pool = None
        # print('init_estimator seed = {}, cv_seed = {}'.format(seed, cv_seed))
        return get_est_func(name=est_name,
                            n_folds=n_folds,
                            task=self.task,
                            est_type=est_type,
                            eval_metrics=self.eval_metrics,
                            seed=seed,
                            dtype=self.dtype,
                            keep_in_mem=self.keep_in_mem,
                            est_args=est_args,
                            cv_seed=seed,
                            win_shape=win_shape,
                            pool=pool)

    def fit(self, x_train, y_train):
        """
        Fit.

        :param x_train:
        :param y_train:
        :return:
        """
        x_train, y_train = self._check_input(x_train, y_train)
        # check if output of fit is exists in disk, if yes, we do not to re-train the model, just load the cached data.
        train_path = self._check_disk_cache(x_train, 'train')
        if train_path is not False:
            self.LOGGER.info("Cache hit! Loading data from {}, skip fit!".format(train_path))
            return load_disk_cache(data_path=train_path)
        if self.distribute and self.dis_level >= 1:
            return self._dis_fit(x_train, y_train)
        x_wins_train = []
        for win in self.windows:
            x_wins_train.append(self.scan(win, x_train))
        self.LOGGER.info('X_wins of train: {}'.format([win.shape for win in x_wins_train]))
        x_win_est_train = []
        for wi, ests_for_win in enumerate(self.est_for_windows):
            if not isinstance(ests_for_win, (list, tuple)):
                ests_for_win = [ests_for_win]
            win_est_train = []
            # X_wins[wi] = (60000, 11, 11, 49)
            _, nh, nw, _ = x_wins_train[wi].shape
            # (60000, 121, 49)
            x_wins_train[wi] = x_wins_train[wi].reshape((x_wins_train[wi].shape[0], -1, x_wins_train[wi].shape[-1]))
            y_win = y_train[:, np.newaxis].repeat(x_wins_train[wi].shape[1], axis=1)
            for ei, est in enumerate(ests_for_win):
                if isinstance(est, EstimatorConfig):
                    est = self._init_estimator(est, wi, ei)
                ests_for_win[ei] = est
            if self.distribute:
                x_wins_train_obj_id = ray.put(x_wins_train[wi])
                y_win_obj_id = ray.put(y_win)
                y_stratify = ray.put(y_win[:, 0])
                y_proba_trains = ray.get([est.fit_transform.remote(x_wins_train_obj_id, y_win_obj_id, y_stratify)
                                          for est in ests_for_win])
            else:
                y_proba_trains = [est.fit_transform(x_wins_train[wi], y_win, y_win[:, 0]) for est in ests_for_win]
            for y_proba_train_tup in y_proba_trains:
                y_proba_train = y_proba_train_tup[0]
                y_proba_train = y_proba_train.reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))
                if len(y_proba_train_tup) == 3 and self.verbose_dis:
                    for log in y_proba_train_tup[2]:
                        if log[0] == 'INFO':
                            self.LOGGER.info("{}".format(log[1].format(log[2])))
                        elif log[0] == 'WARN':
                            self.LOGGER.warn("{}".format(log))
                        else:
                            self.LOGGER.info(str(log))
                win_est_train.append(y_proba_train)
            if self.keep_in_mem:
                self.est_for_windows[wi] = ests_for_win
            x_win_est_train.append(win_est_train)
        # if there are no est_for_windows, x_win_est_train is empty.
        # we let it to be scan result.
        if len(x_win_est_train) == 0:
            return x_wins_train
        self.LOGGER.info('x_win_est_train.shape: {}'.format(list2str(x_win_est_train, 2)))
        if self.cache_in_disk and self.data_save_dir:
            data_path = self._get_disk_path(x_train, 'train')
            check_dir(data_path)
            save_disk_cache(data_path, x_win_est_train)
            self.LOGGER.info("Saving data x_win_est_train to {}".format(data_path))
        return x_win_est_train

    def _dis_fit(self, x_train, y_train):
        # TODO: Add lazy scan for _dis_fit
        # TODO: Add advance pooling for _dis_fit
        """
        Fit.

        :param x_train: training data
        :param y_train: training labels
        :return:
        """
        x_train, y_train = self._check_input(x_train, y_train)
        x_wins_train = []
        for win in self.windows:
            x_wins_train.append(self.scan(win, x_train))
        self.LOGGER.info('X_wins of train: {}'.format([win.shape for win in x_wins_train]))
        self.LOGGER.debug('Scaned dtype for X_wins of train: {}'.format([win.dtype for win in x_wins_train]))
        x_win_est_train = []
        ests = []
        est_offsets = [0, ]
        ei2wi = dict()
        nhs, nws = [None for _ in range(len(self.windows))], [None for _ in range(len(self.windows))]
        y_win = [None for _ in range(len(self.windows))]
        for wi, ests_for_win in enumerate(self.est_for_windows):
            if not isinstance(ests_for_win, (list, tuple)):
                ests_for_win = [ests_for_win]
            for ei, est in enumerate(ests_for_win):
                ests.append(est)
                ei2wi[est_offsets[wi] + ei] = (wi, ei)
            est_offsets.append(est_offsets[-1] + len(ests_for_win))
            _, nhs[wi], nws[wi], _ = x_wins_train[wi].shape
            x_wins_train[wi] = x_wins_train[wi].reshape((x_wins_train[wi].shape[0], -1, x_wins_train[wi].shape[-1]))
            y_win[wi] = y_train[:, np.newaxis].repeat(x_wins_train[wi].shape[1], axis=1)
            self.LOGGER.debug('x_wins_train[{}] size={}, dtype={}'.format(wi, getmbof(x_wins_train[wi]), x_wins_train[wi].dtype))
            self.LOGGER.debug('y_win[{}] size={}, dtype={}'.format(wi, getmbof(y_win[wi]), y_win[wi].dtype))
        splitting = SplittingKFoldWrapper(dis_level=self.dis_level, estimators=ests, ei2wi=ei2wi,
                                          num_workers=self.num_workers, seed=self.seed,
                                          task=self.task, eval_metrics=self.eval_metrics,
                                          keep_in_mem=self.keep_in_mem, cv_seed=self.seed, dtype=self.dtype)
        ests_output = splitting.fit(x_wins_train, y_win)
        for wi, ests_for_win in enumerate(self.est_for_windows):
            win_est_train = []
            # X_wins[wi] = (60000, 11, 11, 49)
            nh, nw = nhs[wi], nws[wi]
            # (60000, 121, 49)
            y_proba_trains = ests_output[est_offsets[wi]:est_offsets[wi+1]]
            for y_proba_train_tup in y_proba_trains:
                y_proba_train = y_proba_train_tup[0]
                y_proba_train = y_proba_train.reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))
                win_est_train.append(y_proba_train)
                if len(y_proba_train_tup) == 3 and self.verbose_dis:
                    for log in y_proba_train_tup[2]:
                        if log[0] == 'INFO':
                            self.LOGGER.info("{}".format(log[1].format(log[2])))
                        elif log[0] == 'WARN':
                            self.LOGGER.warn("{}".format(log))
                        else:
                            self.LOGGER.info(str(log))

            if self.keep_in_mem:
                self.est_for_windows[wi] = ests_for_win
            else:
                self.est_for_windows[wi] = None
            x_win_est_train.append(win_est_train)
        # if there are no est_for_windows, x_win_est_train is empty.
        # we let it to be scan result.
        if len(x_win_est_train) == 0:
            x_win_est_train = x_wins_train
        self.LOGGER.info('x_win_est_train.shape: {}'.format(list2str(x_win_est_train, 2)))
        if self.cache_in_disk and self.data_save_dir:
            data_path = self._get_disk_path(x_train, 'train')
            check_dir(data_path)
            save_disk_cache(data_path, x_win_est_train)
            self.LOGGER.info("Saving data x_win_est_train to {}".format(data_path))
        return x_win_est_train

    def _dis_fit_transform(self,  x_train, y_train, x_test=None, y_test=None):
        """
        Fit and transform.

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        x_train, y_train = self._check_input(x_train, y_train)
        x_test, y_test = self._check_input(x_test, y_test)
        self.LOGGER.debug('x_train size={}, dtype={}'.format(getmbof(x_train), x_train.dtype))
        self.LOGGER.debug(' x_test size={}, dtype={}'.format(getmbof(x_test), x_test.dtype))
        x_win_est_train = []
        x_win_est_test = []
        ests = []
        est_offsets = [0, ]
        ei2wi = dict()
        nhs, nws = [None for _ in range(len(self.windows))], [None for _ in range(len(self.windows))]
        # usually the size of x_test is smaller than x_train, use x_test here to save time
        for wi, win in enumerate(self.windows):
            nhs[wi], nws[wi] = self.scan_shape(win, x_test.shape)
        for wi, ests_for_win in enumerate(self.est_for_windows):
            if not isinstance(ests_for_win, (list, tuple)):
                ests_for_win = [ests_for_win]
            for ei, est in enumerate(ests_for_win):
                ests.append(est)
                ei2wi[est_offsets[wi] + ei] = (wi, ei)
            est_offsets.append(est_offsets[-1] + len(ests_for_win))
        self.LOGGER.debug('est_offsets = {}'.format(est_offsets))
        self.LOGGER.debug('ei2wi = {}'.format(ei2wi))
        splitting = SplittingKFoldWrapper(dis_level=self.dis_level, estimators=ests, ei2wi=ei2wi,
                                          num_workers=self.num_workers, seed=self.seed, windows=self.windows,
                                          pools=self.pre_pools, task=self.task, eval_metrics=self.eval_metrics,
                                          keep_in_mem=self.keep_in_mem, cv_seed=self.seed, dtype=self.dtype)
        if self.lazy_scan:
            ests_output = splitting.fit_transform_lazyscan(x_train, y_train, x_test, y_test)
        else:
            ests_output = splitting.fit_transform(x_train, y_train, x_test, y_test)
        machines = defaultdict(int)
        trees = defaultdict(int)
        machine_time_max = defaultdict(float)
        machine_time_total = defaultdict(float)
        for wi, ests_for_win in enumerate(self.est_for_windows):
            win_est_train = []
            win_est_test = []
            # X_wins[wi] = (60000, 11, 11, 49)
            nh, nw = nhs[wi], nws[wi]
            # (60000, 121, 49)
            y_proba_train_tests = ests_output[est_offsets[wi]:est_offsets[wi + 1]]
            for ei, y_proba_tup in enumerate(y_proba_train_tests):
                y_proba_train = y_proba_tup[0]
                if self.pre_pools is not None:
                    height, width = self.pool_shape(self.pre_pools[wi][ei], (nh, nw))
                else:
                    height, width = nh, nw
                y_proba_train = y_proba_train.reshape((-1, height, width, self.n_class)).transpose((0, 3, 1, 2))
                y_probas_test = y_proba_tup[1]
                assert len(y_probas_test) == 1, 'assume there is only one test set!'
                y_probas_test = y_probas_test[0]
                y_probas_test = y_probas_test.reshape((-1, height, width, self.n_class)).transpose((0, 3, 1, 2))
                # Lack of this line may cause precision issue that is inconsistency of dis and sm
                y_probas_test = check_dtype(y_probas_test, self.dtype)
                win_est_train.append(y_proba_train)
                win_est_test.append(y_probas_test)
                if len(y_proba_tup) == 3 and self.verbose_dis:
                    for log in y_proba_tup[2]:
                        if log[0] == 'INFO':
                            self.LOGGER.info("{}".format(log[1].format(log[2])))
                        elif log[0] == 'WARN':
                            self.LOGGER.warn("{}".format(log))
                        else:
                            if str(log).count('Running on'):
                                machines[log.split(' ')[3]] += 1
                                trees[log.split(' ')[3]] += int(log.split(' ')[0].split(':')[1])
                            elif str(log).count('fit time total:'):
                                machine_time_max[log.split(' ')[0]] = max(machine_time_max[log.split(' ')[0]],
                                                                          float(log.split(' ')[4]))
                                machine_time_total[log.split(' ')[0]] += float(log.split(' ')[4])
                            else:
                                self.LOGGER.info(str(log))

            # TODO: improving keep estimators.
            if self.keep_in_mem:
                self.est_for_windows[wi] = ests_for_win
            else:
                self.est_for_windows[wi] = None
            x_win_est_train.append(win_est_train)
            x_win_est_test.append(win_est_test)
        if len(x_win_est_train) == 0:
            x_wins_train = []
            x_wins_test = []
            for win in self.windows:
                x_wins_train.append(self.scan(win, x_train))
            for win in self.windows:
                x_wins_test.append(self.scan(win, x_test))
            self.LOGGER.info('X_wins of train: {}'.format([win.shape for win in x_wins_train]))
            self.LOGGER.info('X_wins of  test: {}'.format([win.shape for win in x_wins_test]))
            return x_wins_train, x_wins_test
        self.LOGGER.info('x_win_est_train.shape: {}'.format(list2str(x_win_est_train, 2)))
        self.LOGGER.info(' x_win_est_test.shape: {}'.format(list2str(x_win_est_test, 2)))
        if self.cache_in_disk and self.data_save_dir:
            train_path = self._get_disk_path(x_train, 'train')
            test_path = self._get_disk_path(x_test, 'test')
            check_dir(train_path)
            check_dir(test_path)
            save_disk_cache(train_path, x_win_est_train)
            save_disk_cache(test_path, x_win_est_test)
            self.LOGGER.info("[dis] Saving data x_win_est_train to {}".format(train_path))
            self.LOGGER.info("[dis] Saving data x_win_est_test to {}".format(test_path))
        total_task = sum([v for v in machines.values()])
        for key in machines.keys():
            self.LOGGER.info('Machine {} was assigned {}:{} / {}, max {}, total {}'.format(key, machines[key],
                                                                                           trees[key], total_task,
                                                                                           machine_time_max[key],
                                                                                           machine_time_total[key]))
        return x_win_est_train, x_win_est_test

    def fit_transform(self, x_train, y_train, x_test=None, y_test=None):
        """
        Fit and transform.

        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        if x_test is None:
            return self.fit(x_train, y_train), None
        x_train, y_train = self._check_input(x_train, y_train)
        x_test, y_test = self._check_input(x_test, y_test)
        self.LOGGER.debug('x_train size = {}, x_test size = {}'.format(getmbof(x_train[:]), getmbof(x_test[:])))
        self.LOGGER.debug('x_train dtype = {}, x_test dtype = {}'.format(x_train.dtype, x_test.dtype))
        train_path = self._check_disk_cache(x_train, 'train')
        test_path = self._check_disk_cache(x_test, 'test')
        if train_path is not False and test_path is not False:
            self.LOGGER.info("Cache hit! Loading train from {}, skip fit!".format(train_path))
            self.LOGGER.info("Cache hit! Loading test from {}, skip fit!".format(test_path))
            return load_disk_cache(train_path), load_disk_cache(test_path)
        if self.distribute and self.dis_level >= 0:
            return self._dis_fit_transform(x_train, y_train, x_test, y_test)
        x_win_est_train = []
        x_win_est_test = []
        machines = defaultdict(int)
        trees = defaultdict(int)
        machine_time_max = defaultdict(float)
        machine_time_total = defaultdict(float)
        for wi, ests_for_win in enumerate(self.est_for_windows):
            if not isinstance(ests_for_win, (list, tuple)):
                ests_for_win = [ests_for_win]
            win_est_train = []
            win_est_test = []
            x_win_train_wi = self.scan(self.windows[wi], x_train)
            x_win_test_wi = self.scan(self.windows[wi], x_test)
            self.LOGGER.info('X_win_train_{}: {}, size={}'.format(wi, x_win_train_wi.shape, getmbof(x_win_train_wi)))
            self.LOGGER.debug('X_win_train_{}: {}'.format(wi, getmbof(x_win_train_wi)))
            self.LOGGER.info('X_win_test_{}: {}, size={}'.format(wi, x_win_test_wi.shape, getmbof(x_win_test_wi)))
            # X_wins[wi] = (60000, 11, 11, 49)
            _, nh, nw, _ = x_win_train_wi.shape
            # (60000, 121, 49)
            x_win_train_wi = x_win_train_wi.reshape((x_win_train_wi.shape[0], -1, x_win_train_wi.shape[-1]))
            y_win = y_train[:, np.newaxis].repeat(x_win_train_wi.shape[1], axis=1)
            x_win_test_wi = x_win_test_wi.reshape((x_win_test_wi.shape[0], -1, x_win_test_wi.shape[-1]))
            y_win_test = None if y_test is None else y_test[:, np.newaxis].repeat(x_win_test_wi.shape[1], axis=1)
            test_sets = [('testOfWin{}'.format(wi), x_win_test_wi, y_win_test)]
            # fit estimators for this window
            for ei, est in enumerate(ests_for_win):
                if isinstance(est, EstimatorConfig):
                    pool, win_shape = None, None
                    if self.pre_pools and self.distribute:
                        pool = self.pre_pools[wi][ei]
                        win_shape = (nh, nw)
                    est = self._init_estimator(est, wi, ei, win_shape=win_shape, pool=pool)
                # if self.distribute is True, then est is an ActorHandle.
                ests_for_win[ei] = est
            self.LOGGER.debug('y_win.size = {}'.format(getmbof(y_win)))
            if self.distribute:
                x_wins_train_obj_id = ray.put(x_win_train_wi)
                y_win_obj_id = ray.put(y_win)
                y_stratify = ray.put(y_win[:, 0])
                test_sets_obj_id = ray.put(test_sets)
                y_proba_train_tests = ray.get([est.fit_transform.remote(x_wins_train_obj_id, y_win_obj_id,
                                                                        y_stratify, test_sets_obj_id)
                                               for est in ests_for_win])
            else:
                y_proba_train_tests = [est.fit_transform(x_win_train_wi, y_win, y_win[:, 0], test_sets)
                                       for est in ests_for_win]
            self.LOGGER.debug('got y_proba_train_tests size = {}'.format(getmbof(y_proba_train_tests)))
            for ei, y_proba_tup in enumerate(y_proba_train_tests):
                if self.pre_pools is not None:
                    height, width = self.pool_shape(self.pre_pools[wi][ei], (nh, nw))
                else:
                    height, width = nh, nw
                y_proba_train = y_proba_tup[0]
                y_proba_train = y_proba_train.reshape((-1, height, width, self.n_class)).transpose((0, 3, 1, 2))
                y_probas_test = y_proba_tup[1]
                y_probas_test = y_probas_test[0]
                y_probas_test = y_probas_test.reshape((-1, height, width, self.n_class)).transpose((0, 3, 1, 2))
                # Lack of this line may cause precision issue that is inconsistency of dis and sm
                y_probas_test = check_dtype(y_probas_test, self.dtype)
                if len(y_proba_tup) == 3 and self.verbose_dis:
                    for log in y_proba_tup[2]:
                        if log[0] == 'INFO':
                            self.LOGGER.info("{}".format(log[1].format(log[2])))
                        elif log[0] == 'WARN':
                            self.LOGGER.warn("{}".format(log))
                        else:
                            if str(log).count('Running on'):
                                machines[log.split(' ')[3]] += 1
                                trees[log.split(' ')[3]] += int(log.split(' ')[0].split(':')[1])
                            elif str(log).count('fit time total:'):
                                machine_time_max[log.split(' ')[0]] = max(machine_time_max[log.split(' ')[0]],
                                                                          float(log.split(' ')[4]))
                                machine_time_total[log.split(' ')[0]] += float(log.split(' ')[4])
                            else:
                                self.LOGGER.info(str(log))
                win_est_train.append(y_proba_train)
                win_est_test.append(y_probas_test)
            if self.keep_in_mem:
                self.est_for_windows[wi] = ests_for_win
            x_win_est_train.append(win_est_train)
            x_win_est_test.append(win_est_test)
        if len(x_win_est_train) == 0:
            raise EOFError("Nothing in x_win_est_train, training failed!")
        self.LOGGER.info('x_win_est_train.shape: {}'.format(list2str(x_win_est_train, 2)))
        self.LOGGER.info(' x_win_est_test.shape: {}'.format(list2str(x_win_est_test, 2)))
        if self.cache_in_disk and self.data_save_dir:
            train_path = self._get_disk_path(x_train, 'train')
            test_path = self._get_disk_path(x_test, 'test')
            check_dir(train_path)
            check_dir(test_path)
            save_disk_cache(train_path, x_win_est_train)
            save_disk_cache(test_path, x_win_est_test)
            self.LOGGER.info("Saving data x_win_est_train to {}".format(train_path))
            self.LOGGER.info("Saving data x_win_est_test to {}".format(test_path))
        total_task = sum([v for v in machines.values()])
        for key in machines.keys():
            self.LOGGER.info('Machine {} was assigned {}:{} / {}, max {}, total {}'.format(key, machines[key],
                                                                                           trees[key], total_task,
                                                                                           machine_time_max[key],
                                                                                           machine_time_total[key]))
        return x_win_est_train, x_win_est_test

    def transform(self, x_train):
        """
        Transform.

        :param x_train:
        :return:
        """
        assert x_train is not None, 'x_trains should not be None!'
        if isinstance(x_train, (list, tuple)):
            assert len(x_train) == 1, "Multi grain scan Layer only supports exactly one input now!"
            x_train = x_train[0]
        x_wins_train = []
        for win in self.windows:
            x_wins_train.append(self.scan(win, x_train))
        # [[win, win], [win, win], ...], len = len(test_sets)
        self.LOGGER.info('X_wins of train: {}'.format([win.shape for win in x_wins_train]))
        x_win_est_train = []
        for wi, ests_for_win in enumerate(self.est_for_windows):
            if not isinstance(ests_for_win, (list, tuple)):
                ests_for_win = [ests_for_win]
            win_est_train = []
            # X_wins[wi] = (60000, 11, 11, 49)
            _, nh, nw, _ = x_wins_train[wi].shape
            # (60000, 121, 49)
            x_wins_train[wi] = x_wins_train[wi].reshape((x_wins_train[wi].shape[0], -1, x_wins_train[wi].shape[-1]))
            for ei, est in enumerate(ests_for_win):
                # (60000, 121, 10)
                y_proba_train = est.transform(x_wins_train[wi])
                y_proba_train = y_proba_train.reshape((-1, nh, nw, self.n_class)).transpose((0, 3, 1, 2))
                win_est_train.append(y_proba_train)
            x_win_est_train.append(win_est_train)
        if len(x_win_est_train) == 0:
            return x_wins_train
        self.LOGGER.info('[transform] win_est_train.shape: {}'.format(list2str(x_win_est_train, 2)))
        return x_win_est_train

    def predict(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        return self.transform(X)

    def evaluate(self, inputs, labels):
        raise NotImplementedError

    def _check_input(self, x, y):
        if isinstance(x, (list, tuple)):
            assert len(x) == 1, "Multi grain scan Layer only supports exactly one input now!"
            x = x[0]
        if isinstance(y, (list, tuple)):
            assert len(y) == 1, "Multi grain scan Layer only supports exactly one input now!"
            y = y[0]
        return x, y

    def _check_disk_cache(self, x, phase, file_type='pkl'):
        if not self.cache_in_disk or not self.data_save_dir:
            return False
        data_path = self._get_disk_path(x, phase, file_type)
        if osp.exists(data_path):
            return data_path
        return False

    def _get_disk_path(self, x, phase, file_type='pkl'):
        assert isinstance(x, np.ndarray), 'x_train should be numpy.ndarray, but {}'.format(type(x))
        data_name = "x".join(map(str, x.shape))
        data_path = output_disk_path(self.data_save_dir, 'mgs', phase, data_name, file_type)
        return data_path

    def save(self):
        pass

    def __str__(self):
        return self.__class__.__name__

    @property
    def summary_info(self):
        info_str = '[' + ', '.join(map(str, self.windows)) + ']'
        info_str += '\n['
        for ests in self.est_for_windows:
            info_str += '['
            info_str += ', '.join(map(str, ests))
            info_str += ']'
        info_str += ']'
        return info_str


class PoolingLayer(Layer):
    """
    Pooling layer.
    """
    def __init__(self, batch_size=None, dtype=None, name=None, pools=None,
                 cache_in_disk=False, data_save_dir=None):
        """
        Initialize a pooling layer.

        :param batch_size:
        :param dtype:
        :param name:
        :param pools:
        :param cache_in_disk: whether cache pooled data in disk (data_save_dir)
        :param data_save_dir: directory caching into
        """
        super(PoolingLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        # [[pool/7x7/est1, pool/7x7/est2], [pool/11x11/est1, pool/11x11/est1], [pool/13x13/est1, pool/13x13/est1], ...]
        self.pools = pools
        self.cache_in_disk = cache_in_disk
        self.data_save_dir = data_save_dir

    def call(self, x_trains, **kwargs):
        pass

    def __call__(self, x_trains):
        pass

    def _check_disk_cache(self, x_shape, phase):
        if not self.cache_in_disk or not self.data_save_dir:
            return False
        data_path = self._get_disk_path(x_shape, phase)
        if osp.exists(data_path):
            return data_path
        return False

    def _get_disk_path(self, x_shape, phase):
        assert isinstance(x_shape, tuple), 'x_train should be tuple, but {}'.format(type(x_shape))
        data_name = "x".join(map(str, x_shape))
        data_path = output_disk_path(self.data_save_dir, 'pool', phase, data_name)
        return data_path

    def fit(self, x_trains, y_trains=None):
        """
        Fit.

        :param x_trains:
        :param y_trains:
        :return:
        """
        # inputs shape: [[(60000, 10, 11, 11), (60000, 10, 11, 11)], [.., ..], ...]
        if len(self.pools) != len(x_trains):
            raise ValueError('len(pools) does not equal to len(inputs), you must set right pools!')
        x_shape = x_trains[0][0].shape
        # Try to load data from disk.
        train_path = self._check_disk_cache(x_shape, 'train')
        if train_path is not False:
            self.LOGGER.info("Cache hit! Loading data from {}, skip fit!".format(train_path))
            return load_disk_cache(data_path=train_path)
        for pi, pool in enumerate(self.pools):
            if not isinstance(pool, (list, tuple)):
                pool = [pool]
            if len(pool) != len(x_trains[pi]):
                raise ValueError('len(pools[{}]) does not equal to'
                                 ' len(inputs[{}]), you must set right pools!'.format(pi, pi))
            for pj, pl in enumerate(pool):
                x_trains[pi][pj] = pl.fit_transform(x_trains[pi][pj])
        self.LOGGER.info('x_trains pooled: {}'.format(list2str(x_trains, 2)))
        if self.cache_in_disk and self.data_save_dir:
            data_path = self._get_disk_path(x_shape, 'train')
            check_dir(data_path)
            save_disk_cache(data_path, x_trains)
            self.LOGGER.info("Saving data x_win_est_train to {}".format(data_path))
        return x_trains

    def fit_transform(self, x_trains, y_trains=None, x_tests=None, y_tests=None):
        """
        Fit transform.

        :param x_trains:
        :param y_trains:
        :param x_tests:
        :param y_tests:
        :return:
        """
        if x_tests is None:
            return self.fit(x_trains, y_trains), None
        # inputs shape: [[(60000, 10, 11, 11), (60000, 10, 11, 11)], [.., ..], ...]
        if len(self.pools) != len(x_trains):
            raise ValueError('len(pools) = {} does not equal to len(x_trains) = {}, you must set right pools!'.format(
                len(self.pools), len(x_trains)
            ))
        if len(self.pools) != len(x_tests):
            raise ValueError('len(pools) does not equal to len(x_tests), you must set right pools!')
        x_train_shape = x_trains[0][0].shape
        x_test_shape = x_tests[0][0].shape
        # Try to load data from disk.
        # TODO: boundary condition judge if x_trains is not 2d list.
        train_path = self._check_disk_cache(x_train_shape, 'train')
        test_path = self._check_disk_cache(x_test_shape, 'test')
        if train_path is not False and test_path is not False:
            self.LOGGER.info("Cache hit! Loading trains from {}, skip fit!".format(train_path))
            self.LOGGER.info("Cache hit! Loading  tests from {}, skip fit!".format(test_path))
            return load_disk_cache(train_path), load_disk_cache(test_path)
        for pi, pool in enumerate(self.pools):
            if not isinstance(pool, (list, tuple)):
                pool = [pool]
            if len(pool) != len(x_trains[pi]):
                raise ValueError('len(pools[{}]) does not equal to'
                                 ' len(train inputs[{}]), you must set right pools!'.format(pi, pi))
            if len(pool) != len(x_tests[pi]):
                raise ValueError('len(pools[{}]) does not equal to'
                                 ' len(test inputs[{}]), you must set right pools!'.format(pi, pi))
            for pj, pl in enumerate(pool):
                x_trains[pi][pj] = pl.fit_transform(x_trains[pi][pj]).astype(self.dtype)
                x_tests[pi][pj] = pl.fit_transform(x_tests[pi][pj]).astype(self.dtype)
        self.LOGGER.info('x_trains pooled: {}'.format(list2str(x_trains, 2)))
        self.LOGGER.info('x_tests  pooled: {}'.format(list2str(x_tests, 2)))
        # save data into disk.
        if self.cache_in_disk and self.data_save_dir:
            train_path = self._get_disk_path(x_train_shape, 'train')
            test_path = self._get_disk_path(x_test_shape, 'test')
            check_dir(train_path)
            check_dir(test_path)
            save_disk_cache(train_path, x_trains)
            save_disk_cache(test_path, x_tests)
            self.LOGGER.info("Saving data x_trains to {}".format(train_path))
            self.LOGGER.info("Saving data  x_tests to {}".format(test_path))
        return x_trains, x_tests

    def transform(self, xs):
        """
        Transform.

        :param xs:
        :return:
        """
        assert xs is not None, 'x_trains should not be None!'
        if not isinstance(xs, (list, tuple)):
            xs = [xs]
        # inputs shape: [[(60000, 10, 11, 11), (60000, 10, 11, 11)], [.., ..], ...]
        if len(self.pools) != len(xs):
            raise ValueError('len(pools) does not equal to len(inputs), you must set right pools!')
        for pi, pool in enumerate(self.pools):
            if not isinstance(pool, (list, tuple)):
                pool = [pool]
            if len(pool) != len(xs[pi]):
                raise ValueError('len(pools[{}]) does not equal to'
                                 ' len(inputs[{}]), you must set right pools!'.format(pi, pi))
            for pj, pl in enumerate(pool):
                xs[pi][pj] = pl.transform(xs[pi][pj])
        self.LOGGER.info('[transform] x_trains pooled: {}'.format(list2str(xs, 2)))
        return xs

    def evaluate(self, inputs, labels):
        raise NotImplementedError

    def predict(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        return self.transform(X)

    def save(self):
        pass

    @property
    def summary_info(self):
        info_str = '['
        for pool in self.pools:
            info_str += '[' + ', '.join(map(str, pool)) + ']'
        info_str += ']'
        return info_str


class ConcatLayer(Layer):
    """
    Concatenate layer.
    """
    def __init__(self, batch_size=None, dtype=None, name=None, axis=-1,
                 cache_in_disk=False, data_save_dir=None):
        """
        Initialize a concat layer.

        :param batch_size:
        :param dtype:
        :param name:
        :param axis:
        """
        super(ConcatLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        # [[pool/7x7/est1, pool/7x7/est2], [pool/11x11/est1, pool/11x11/est1], [pool/13x13/est1, pool/13x13/est1], ...]
        # to
        # [Concat(axis=axis), Concat(axis=axis), Concat(axis=axis), ...]
        self.axis = axis
        self.cache_in_disk = cache_in_disk
        self.data_save_dir = data_save_dir

    def call(self, X, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def _check_disk_cache(self, x_shape, phase):
        if not self.cache_in_disk or not self.data_save_dir:
            return False
        data_path = self._get_disk_path(x_shape, phase)
        if osp.exists(data_path):
            return data_path
        return False

    def _get_disk_path(self, x_shape, phase):
        assert isinstance(x_shape, tuple), 'x_train should be tuple, but {}'.format(type(x_shape))
        data_name = "x".join(map(str, x_shape))
        data_path = output_disk_path(self.data_save_dir, 'concat', phase, data_name)
        return data_path

    def _check_input(self, xs):
        if not isinstance(xs, (list, tuple)):
            xs = [xs]
        return xs

    def _fit(self, xs):
        """
        fit inner method.

        :param xs:
        :return:
        """
        xs = self._check_input(xs)
        # inputs shape: [[(60000, 10, 6, 6), (60000, 10, 6, 6)], [.., ..], ...]
        concat_results = []
        for bottoms in xs:
            if self.axis == -1:
                for i, bottom in enumerate(bottoms):
                    bottoms[i] = bottom.reshape((bottom.shape[0], -1))
                concat_res = np.concatenate(bottoms, 1)
            else:
                concat_res = np.concatenate(bottoms, self.axis)
            concat_results.append(concat_res)
        return concat_results

    def transform(self, xs):
        """
        Transform.

        :param xs:
        :return:
        """
        xs = self._check_input(xs)
        concat = self._fit(xs)
        self.LOGGER.info("[transform] concatenated shape: {}".format(list2str(concat, 1)))
        return concat

    def evaluate(self, inputs, labels=None):
        raise NotImplementedError

    def fit(self, x_trains, y_trains=None):
        """
        Fit.

        :param x_trains:
        :param y_trains:
        :return:
        """
        x_trains = self._check_input(x_trains)
        x_train_shape = x_trains[0][0].shape
        # Try to load data from disk.
        train_path = self._check_disk_cache(x_train_shape, 'train')
        if train_path is not False:
            self.LOGGER.info("Cache hit! Loading data from {}, skip fit!".format(train_path))
            return load_disk_cache(data_path=train_path)
        concat_train = self._fit(x_trains)
        self.LOGGER.info("concat train shape: {}".format(list2str(concat_train, 1)))
        if self.cache_in_disk and self.data_save_dir:
            data_path = self._get_disk_path(x_train_shape, 'train')
            check_dir(data_path)
            save_disk_cache(data_path, concat_train)
            self.LOGGER.info("Saving data x_win_est_train to {}".format(data_path))
        return concat_train

    def fit_transform(self, x_trains, y_trains, x_tests=None, y_tests=None):
        """
        Fit transform.

        :param x_trains:
        :param y_trains:
        :param x_tests:
        :param y_tests:
        :return:
        """
        if x_tests is None:
            return self.fit(x_trains, y_trains), None
        x_trains = self._check_input(x_trains)
        x_tests = self._check_input(x_tests)
        x_train_shape = x_trains[0][0].shape
        x_test_shape = x_tests[0][0].shape
        # Try to load data from disk.
        # TODO: boundary condition judge if x_trains is not 2d list.
        train_path = self._check_disk_cache(x_train_shape, 'train')
        test_path = self._check_disk_cache(x_test_shape, 'test')
        if train_path is not False and test_path is not False:
            self.LOGGER.info("Cache hit! Loading concat trains from {}, skip fit!".format(train_path))
            self.LOGGER.info("Cache hit! Loading concat tests from {}, skip fit!".format(test_path))
            return load_disk_cache(train_path), load_disk_cache(test_path)
        # inputs shape: [[(60000, 10, 6, 6), (60000, 10, 6, 6)], [.., ..], ...]
        concat_train = self._fit(x_trains)
        self.LOGGER.info("concat train shape: {}".format(list2str(concat_train, 1)))
        concat_test = self._fit(x_tests)
        self.LOGGER.info(" concat test shape: {}".format(list2str(concat_test, 1)))
        # save data into disk.
        if self.cache_in_disk and self.data_save_dir:
            train_path = self._get_disk_path(x_train_shape, 'train')
            test_path = self._get_disk_path(x_test_shape, 'test')
            check_dir(train_path)
            check_dir(test_path)
            save_disk_cache(train_path, concat_train)
            save_disk_cache(test_path, concat_test)
            self.LOGGER.info("Saving data concat_train to {}".format(train_path))
            self.LOGGER.info("Saving data  concat_test to {}".format(test_path))
        return concat_train, concat_test

    def predict(self, X):
        return self.transform(X)

    def predict_proba(self, X):
        return self.transform(X)

    def save(self):
        raise NotImplementedError('ConcatLayer actually has not model. axis is the model')

    @property
    def summary_info(self):
        return 'ConcatLayer(axis={})'.format(self.axis)


class CascadeLayer(Layer):
    def __init__(self, batch_size=None, dtype=None, name=None, task='classification', est_configs=None,
                 layer_id='anonymous', n_classes=None, keep_in_mem=False, data_save_dir=None, model_save_dir=None,
                 num_workers=None, metrics=None, seed=None, distribute=False, verbose_dis=False, dis_level=1,
                 train_start_ends=None, x_train_group_or_id=None, x_test_group_or_id=None):
        """Cascade Layer.
        A cascade layer contains several estimators, it accepts single input, go through these estimators, produces
        predicted probability by every estimators, and stacks them together for next cascade layer.

        :param batch_size: cascade layer do not need batch_size actually.
        :param dtype: data type
        :param name: name of this layer
        :param task: classification or regression, [default = classification]
        :param est_configs: list of estimator arguments, every argument can be `dict` or `EstimatorConfig` instance
                            identify the estimator configuration to construct at this layer
        :param layer_id: layer id, if this layer is an independent layer, layer id is anonymous [default]
        :param n_classes: number of classes to classify
        :param keep_in_mem: identifies whether keep the model in memory, if fit_transform,
                            we recommend set it False to save memory and speed up the application
                            TODO: support dump model to disk to save memory
        :param data_save_dir: directory to save intermediate data into
        :param model_save_dir: directory to save fit estimators into
        :param num_workers: number of workers in the cluster
        :param metrics: str or user-defined Metrics object, evaluation metrics used in training model and evaluating
                         testing data.
                        Support: 'accuracy', 'auc', 'mse', 'rmse',
                         default is accuracy (classification) and mse (regression).
                        Note that user can define their own metrics by extending the
                         class `forestlayer.utils.metrics.Metrics`
        :param seed: random seed, also called random state in scikit-learn random forest
        :param distribute: boolean, whether use distributed training. If use, you should `import ray`
                           and write `ray.init(<redis-address>)` at the beginning of the main program.
        :param verbose_dis: boolean, whether print logging info that generated on different worker machines.
                            default = False.
        :param dis_level: distributed level, or parallelization level, 0 / 1 / 2 / 3
                           0 means lowest parallelization level, parallelization is len(self.est_configs).
                           1 means triple-split.
                           2 means bin-split.
                           3 means avg split

        # Properties
            eval_metrics: evaluation metrics
            fit_estimators: estimator instances after fit
            train_avg_metric: training average metric
            test_avg_metric: testing average metric

        # Raises
            RuntimeError: if estimator.fit_transform returns None data
            ValueError: if estimator.fit_transform returns wrong shape data
        """
        self.est_configs = [] if est_configs is None else est_configs
        self.est_args = [dict() for _ in range(self.n_estimators)]
        # transform EstimatorConfig to dict that represents estimator arguments.
        for eci, est_config in enumerate(self.est_configs):
            if isinstance(est_config, EstimatorConfig):
                self.est_args[eci] = est_config.get_est_args().copy()
        self.layer_id = layer_id
        if not name:
            name = 'layer-{}'.format(self.layer_id)
        super(CascadeLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        self.task = task
        self.n_classes = n_classes
        if self.task == 'regression':
            self.n_classes = 1
        self.keep_in_mem = keep_in_mem
        self.data_save_dir = data_save_dir
        check_dir(self.data_save_dir)  # check dir, if not exists, create the dir
        self.model_save_dir = model_save_dir
        check_dir(self.model_save_dir)
        self.seed = seed
        self.distribute = distribute
        self.verbose_dis = verbose_dis
        self.dis_level = dis_level
        # initialize num_workers if not provided
        if distribute is True:
            if num_workers is None:
                self.init_num_workers()
            else:
                self.num_workers = num_workers
        self.larger_better = True
        self.metrics = metrics
        self.eval_metrics = get_eval_metrics(self.metrics, self.task, self.name)
        self.x_train_group_or_id = x_train_group_or_id
        self.x_test_group_or_id = x_test_group_or_id
        self.train_start_ends = train_start_ends
        # whether this layer the last layer of Auto-growing cascade layer
        self.complete = False
        self.fit_estimators = [None for _ in range(self.n_estimators)]
        self.train_avg_metric = None
        self.test_avg_metric = None
        self.eval_proba_test = None
        self.machines = defaultdict(int)
        self.trees = defaultdict(int)
        self.machine_time_start = defaultdict(float)
        self.machine_time_end = defaultdict(float)

    def call(self, inputs, **kwargs):
        return inputs

    def __call__(self, inputs, **kwargs):
        self.call(inputs, **kwargs)

    def _init_estimators(self, layer_id, est_id):
        """
        Initialize a k_fold estimator.

        :param layer_id:
        :param est_id:
        :return:
        """
        est_args = self.est_args[est_id].copy()
        est_name = 'layer-{}-estimator-{}-{}folds'.format(layer_id, est_id, est_args['n_folds'])
        n_folds = int(est_args['n_folds'])
        est_args.pop('n_folds')
        est_type = est_args['est_type']
        est_args.pop('est_type')
        # seed
        if self.seed is not None:
            seed = (self.seed + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            seed = None
        if self.distribute:
            get_est_func = get_dist_estimator_kfold
        else:
            get_est_func = get_estimator_kfold
        return get_est_func(name=est_name,
                            n_folds=n_folds,
                            task=self.task,
                            est_type=est_type,
                            eval_metrics=self.eval_metrics,
                            seed=seed,
                            dtype=self.dtype,
                            keep_in_mem=self.keep_in_mem,
                            est_args=est_args,
                            cv_seed=seed)

    def assemble(self, x, n_xs, group_or_id):
        if group_or_id is None:
            return x
        x_cur = np.zeros((n_xs, 0), dtype=self.dtype)
        for (start, end) in self.train_start_ends:
            x_cur = np.hstack((x_cur, group_or_id[:, start:end]))
        x_cur = np.hstack((x_cur, x))
        return x_cur

    def fit(self, x_train, y_train):
        # TODO: support lazy assemble later
        """
        Fit and Transform datasets, return one numpy ndarray: train_output
        NOTE: Only one train set and one test set.

        :param x_train: train datasets
        :param y_train: train labels
        :return: train_output
        """
        x_train, y_train, _, _ = _validate_input(x_train, y_train)
        assert x_train.shape[0] == y_train.shape[0], ('x_train.shape[0] = {} not equal to y_train.shape[0]'
                                                      ' = {}'.format(x_train.shape[0], y_train.shape[0]))
        self.LOGGER.info('X_train.shape={}, y_train.shape={}'.format(x_train.shape, y_train.shape))
        n_trains = x_train.shape[0]
        n_classes = self.n_classes  # if regression, n_classes = 1
        if self.task == 'classification' and n_classes is None:
            n_classes = np.unique(y_train)
        if self.task == 'regression' and n_classes is None:
            n_classes = 1
        x_proba_train = np.zeros((n_trains, n_classes * self.n_estimators), dtype=self.dtype)
        eval_proba_train = np.zeros((n_trains, n_classes), dtype=self.dtype)
        # fit and transform
        y_stratify = y_train if self.task == 'classification' else None
        if self.distribute:
            splitting = CascadeSplittingKFoldWrapper(dis_level=self.dis_level, estimators=self.est_args,
                                                     num_workers=self.num_workers, seed=self.seed, task=self.task,
                                                     eval_metrics=self.eval_metrics, keep_in_mem=self.keep_in_mem,
                                                     cv_seed=self.seed, dtype=self.dtype, layer_id=self.layer_id)
            y_proba_trains, split_ests, split_group = splitting.fit_transform(x_train, y_train, y_stratify,
                                                                              test_sets=None)
            # TODO: fill the estimators, utilize est_group
            if self.keep_in_mem:
                estimators = split_ests
            else:
                estimators = None
        else:
            # fit estimators, get probas (classification) or targets (regression)
            estimators = []
            for ei in range(self.n_estimators):
                est = self._init_estimators(self.layer_id, ei)
                estimators.append(est)
            y_proba_trains = [est.fit_transform(x_train, y_train, y_stratify, test_sets=None)
                              for est in estimators]
        for ei, y_proba_train_tup in enumerate(y_proba_trains):
            y_proba_train = y_proba_train_tup[0]
            if len(y_proba_train_tup) == 3 and self.verbose_dis:
                for log in y_proba_train_tup[2]:
                    if log[0] == 'INFO':
                        self.LOGGER.info("{}".format(log[1].format(log[2])))
                    elif log[0] == 'WARN':
                        self.LOGGER.warn("{}".format(log))
                    else:
                        self.LOGGER.info(str(log))

            if y_proba_train is None:
                raise RuntimeError("layer-{}-estimator-{} fit FAILED!,"
                                   " y_proba_train is None!".format(self.layer_id, ei))
            check_shape(y_proba_train, n_trains, n_classes)
            x_proba_train[:, ei * n_classes:ei * n_classes + n_classes] = y_proba_train
            eval_proba_train += y_proba_train
        if self.keep_in_mem:
            if self.distribute:
                self.fit_estimators = ray.get(estimators)
            else:
                self.fit_estimators = estimators
        eval_proba_train /= self.n_estimators
        # now supports one eval_metrics
        metric = self.eval_metrics[0]
        train_avg_acc = metric.calc_proba(y_train, eval_proba_train,
                                          'layer - {} - [train] average'.format(self.layer_id), logger=self.LOGGER)
        self.train_avg_metric = train_avg_acc
        return x_proba_train

    def fit_transform(self, x_train, y_train, x_test=None, y_test=None):
        """
        Fit and Transform datasets, return two numpy ndarray: train_output, test_output
        NOTE: Only one train set and one test set.
        if x_test is None, we invoke _fit_transform to get one numpy ndarray: train_output

        :param x_train: training data
        :param y_train: training label
        :param x_test: testing data
        :param y_test: testing label, can be None,
                       if None, we see that the fit_transform must give the predictions of x_test.
        :return: train_output, test_output
        """
        if x_test is None:
            return self.fit(x_train, y_train), None
        x_train, y_train, x_test, y_test = _validate_input(x_train, y_train, x_test, y_test)
        if y_test is None:
            y_test_shape = (0,)
        else:
            y_test_shape = y_test.shape
        self.LOGGER.debug('X_train.shape={}, size={}, y_train.shape={}, dtype={}'.format(x_train.shape,
                                                                                         getmbof(x_train),
                                                                                         y_train.shape,
                                                                                         x_train.dtype))
        self.LOGGER.debug(' X_test.shape={}, size={},  y_test.shape={}, dtype={}'.format(x_test.shape,
                                                                                         getmbof(x_test),
                                                                                         y_test_shape,
                                                                                         x_test.dtype))
        n_trains = x_train.shape[0]
        n_tests = x_test.shape[0]
        n_classes = self.n_classes  # if regression, n_classes = 1
        if self.task == 'classification' and n_classes is None:
            n_classes = np.unique(y_train)
        if self.task == 'regression' and n_classes is None:
            n_classes = 1
        x_proba_train = np.zeros((n_trains, n_classes * self.n_estimators), dtype=self.dtype)
        x_proba_test = np.zeros((n_tests, n_classes * self.n_estimators), dtype=self.dtype)
        eval_proba_train = np.zeros((n_trains, n_classes), dtype=self.dtype)
        eval_proba_test = np.zeros((n_tests, n_classes), dtype=self.dtype)
        # fit and transform
        y_stratify = y_train if self.task == 'classification' else None
        if self.distribute:
            splitting = CascadeSplittingKFoldWrapper(dis_level=self.dis_level, estimators=self.est_args,
                                                     num_workers=self.num_workers, seed=self.seed, task=self.task,
                                                     eval_metrics=self.eval_metrics, keep_in_mem=self.keep_in_mem,
                                                     cv_seed=self.seed, dtype=self.dtype, layer_id=self.layer_id,
                                                     train_start_ends=self.train_start_ends,
                                                     x_train_group_or_id=self.x_train_group_or_id,
                                                     x_test_group_or_id=self.x_test_group_or_id)
            y_proba_train_tests, split_ests, split_group = splitting.fit_transform(x_train, y_train, y_stratify,
                                                                                   test_sets=[('test', x_test,  y_test)])
            # TODO: fill the estimators
            if self.keep_in_mem:
                estimators = split_ests
            else:
                estimators = None
        else:
            assert isinstance(self.x_train_group_or_id, np.ndarray) and isinstance(self.x_test_group_or_id, np.ndarray)
            # fit estimators, get probas
            y_proba_train_tests = []
            estimators = []
            x_train = self.assemble(x_train, n_trains, self.x_train_group_or_id)
            x_test = self.assemble(x_test, n_tests, self.x_test_group_or_id)
            for ei in range(self.n_estimators):
                est = self._init_estimators(self.layer_id, ei)
                estimators.append(est)
                y_probas = est.fit_transform(x_train, y_train, y_stratify, test_sets=[('test', x_test,  y_test)])
                y_proba_train_tests.append(y_probas)
        for ei, y_proba_train_tup in enumerate(y_proba_train_tests):
            y_proba_train = y_proba_train_tup[0]
            y_proba_test = y_proba_train_tup[1]
            if len(y_proba_train_tup) == 3:
                for log in y_proba_train_tup[2]:
                    if log[0] == 'INFO' and self.verbose_dis:
                        self.LOGGER.info("{}".format(log[1].format(log[2])))
                    elif log[0] == 'WARN' and self.verbose_dis:
                        self.LOGGER.warn("{}".format(log))
                    else:
                        if str(log).count('Running on'):
                            self.machines[log.split(' ')[3]] += 1
                            self.trees[log.split(' ')[3]] += int(log.split(' ')[0].split(':')[1])
                        elif str(log).count('start'):
                            if log.split(' ')[0] not in self.machine_time_start.keys():
                                self.machine_time_start[log.split(' ')[0]] = float(log.split(' ')[2])
                            else:
                                self.machine_time_start[log.split(' ')[0]] = min(self.machine_time_start[log.split(' ')[0]],
                                                                                 float(log.split(' ')[2]))
                        elif str(log).count('end'):
                            self.machine_time_end[log.split(' ')[0]] = max(self.machine_time_end[log.split(' ')[0]],
                                                                           float(log.split(' ')[2]))

            # if only one element on test_sets, return one test result like y_proba_train
            if isinstance(y_proba_test, (list, tuple)) and len(y_proba_test) == 1:
                y_proba_test = y_proba_test[0]
            if y_proba_train is None:
                raise RuntimeError("layer-{}-estimator-{} fit FAILED!,"
                                   " y_proba_train is None".format(self.layer_id, ei))
            check_shape(y_proba_train, n_trains, n_classes)
            if y_proba_test is not None:
                check_shape(y_proba_test, n_tests, n_classes)
                y_proba_test = check_dtype(y_proba_test, self.dtype)
            x_proba_train[:, ei*n_classes:ei*n_classes + n_classes] = y_proba_train
            x_proba_test[:, ei*n_classes:ei*n_classes + n_classes] = y_proba_test
            eval_proba_train += y_proba_train
            eval_proba_test += y_proba_test
        if self.keep_in_mem:
            if self.distribute:
                self.fit_estimators = ray.get(estimators)
            else:
                self.fit_estimators = estimators
        eval_proba_train /= self.n_estimators
        eval_proba_test /= self.n_estimators
        metric = self.eval_metrics[0]
        train_avg_metric = metric.calc_proba(y_train, eval_proba_train,
                                             'layer - {} - [train] average'.format(self.layer_id),
                                             logger=self.LOGGER)
        self.train_avg_metric = train_avg_metric
        # judge whether y_test is None, which means users are to predict test probas
        if y_test is not None:
            test_avg_metric = metric.calc_proba(y_test, eval_proba_test,
                                                'layer - {} - [ test] average'.format(self.layer_id), logger=self.LOGGER)
            self.test_avg_metric = test_avg_metric
        # if y_test is None, we need to generate test prediction, so keep eval_proba_test
        if y_test is None:
            self.eval_proba_test = eval_proba_test
        total_task = sum([v for v in self.machines.values()])
        for key in self.machines.keys():
            self.LOGGER.info('Machine {} was assigned {}:{} / {}, across {},'.format(
                key, self.machines[key], self.trees[key], total_task,
                self.machine_time_end[key] - self.machine_time_start[key]))
        return x_proba_train, x_proba_test

    @property
    def n_estimators(self):
        """
        Number of estimators of this layer.

        :return:
        """
        return len(self.est_configs)

    def transform(self, X):
        """
        Transform datasets, return one numpy ndarray.
        NOTE: Only one train set and one test set.

        :param X: train datasets
        :return:
        """
        if isinstance(X, (list, tuple)):
            X = None if len(X) == 0 else X[0]
        n_trains = X.shape[0]
        n_classes = self.n_classes
        x_proba = np.zeros((n_trains, n_classes * self.n_estimators), dtype=self.dtype)
        # fit estimators, get probas
        for ei, est in enumerate(self.fit_estimators):
            # transform by n-folds CV
            y_proba = est.transform(X)
            if y_proba is None:
                raise RuntimeError("layer-{}-estimator-{} transform FAILED!".format(self.layer_id, ei))
            check_shape(y_proba, n_trains, n_classes)
            x_proba[:, ei * n_classes:ei * n_classes + n_classes] = y_proba
        return x_proba

    @property
    def is_classification(self):
        return self.task == 'classification'

    def predict(self, X):
        """
        Predict data X.

        :param X:
        :return:
        """
        proba_sum = self.predict_proba(X)
        n_classes = self.n_classes
        return np.argmax(proba_sum.reshape((-1, n_classes)), axis=1)

    def predict_proba(self, X):
        """
        Transform datasets, return one numpy ndarray.
        NOTE: Only one train set and one test set.

        :param X: train datasets
        :return:
        """
        if isinstance(X, (list, tuple)):
            X = None if len(X) == 0 else X[0]
        n_trains = X.shape[0]
        n_classes = self.n_classes
        proba_sum = np.zeros((n_trains, n_classes), dtype=self.dtype)
        # fit estimators, get probas
        for ei, est in enumerate(self.fit_estimators):
            # transform by n-folds CV
            y_proba_train = est.transform(X)
            if y_proba_train is None:
                raise RuntimeError("layer-{}-estimator-{} transform FAILED!".format(self.layer_id, ei))
            check_shape(y_proba_train, n_trains, n_classes)
            proba_sum += y_proba_train
        return proba_sum

    def evaluate(self, X, y, eval_metrics=None):
        """
        Evaluate dataset (X, y) with evaluation metrics.

        :param X: data
        :param y: label
        :param eval_metrics: evaluation metrics
        :return: None
        """
        if eval_metrics is None:
            eval_metrics = [Accuracy('evaluate')]
        if isinstance(y, (list, tuple)):
            assert len(y) == 1, 'only support single labels array'
            y = y[0]
        pred = self.predict(X)
        for metric in eval_metrics:
            metric.calc(y, pred, logger=self.LOGGER)

    def save(self):
        pass

    @property
    def summary_info(self):
        info_str = '['
        info_str += ', '.join(map(str, self.est_configs))
        info_str += ']'
        return info_str


class AutoGrowingCascadeLayer(Layer):
    def __init__(self, batch_size=None, dtype=np.float32, name=None, task='classification', est_configs=None,
                 early_stopping_rounds=None, max_layers=0, look_index_cycle=None, data_save_rounds=0,
                 stop_by_test=True, n_classes=None, keep_in_mem=False, data_save_dir=None, model_save_dir=None,
                 metrics=None, keep_test_result=False, seed=None, distribute=False, verbose_dis=False,
                 dis_level=1, num_workers=None):
        """AutoGrowingCascadeLayer
        An AutoGrowingCascadeLayer is a virtual layer that consists of many single cascade layers.
        `auto-growing` means this kind of layer can decide the depth of cascade forest,
         by training error or testing error.

        :param batch_size: cascade layer do not need batch_size actually.
        :param dtype: data type
        :param name: name of this layer
        :param task: classification or regression, [default = classification]
        :param est_configs: list of estimator arguments, every argument can be `dict` or `EstimatorConfig` instance
                            identify the estimator configuration to construct at this layer
        :param early_stopping_rounds: early stopping rounds, if there is no increase in performance (training accuracy
                                      or testing accuracy) over `early_stopping_rounds` layer, we stop the training
                                      process to save time and storage. And we keep first optimal_layer_id cascade
                                      layer models, and predict/evaluate according to these cascade layer.
        :param max_layers: max layers to growing
                           0 means using Early Stopping to automatically find the layer number
        :param look_index_cycle: (2d list): default = None = [[i,] for i in range(n_groups)]
                                 specification for layer i, look for the array in
                                 look_index_cycle[i % len(look_index_cycle)]
                                 .e.g. look_index_cycle = [[0,1],[2,3],[0,1,2,3]]
                                 means layer 1 look for the grained 0,1; layer 2 look for grained 2,3;
                                 layer 3 look for every grained, and layer 4 cycles back as layer 1
        :param data_save_rounds: int [default = 0, means no savings for intermediate results]
        :param stop_by_test: boolean, identifies whether conduct early stopping by testing metric
                             [default = False]
        :param n_classes: number of classes
        :param keep_in_mem: boolean, identifies whether keep model in memory. [default = False] to save memory
        :param data_save_dir: str [default = None]
                              each data_save_rounds save the intermediate results in data_save_dir
                              if data_save_rounds = 0, then no savings for intermediate results
        :param model_save_dir: directory where save fit estimators into
        :param metrics: str or user-defined Metrics object, evaluation metrics used in training model and evaluating
                         testing data.
                        Support: 'accuracy', 'auc', 'mse', 'rmse',
                         default is accuracy (classification) and mse (regression).
                        Note that user can define their own metrics by extending the
                         class `forestlayer.utils.metrics.Metrics`
        :param seed: random seed, also called random state in scikit-learn random forest
        :param distribute: boolean, whether use distributed training. If use, you should `import ray`
                           and write `ray.init(<redis-address>)` at the beginning of the main program.
        :param verbose_dis: boolean, whether print logging info that generated on different worker machines.
                            default = False.
        :param dis_level: distributed level, or parallelization level, 0 / 1 / 2 / 3
                           0 means lowest parallelization level, parallelization is len(self.est_configs).
                           1 means triple-split.
                           2 means bin-split.
                           3 means avg split
        :param num_workers: number of workers in the cluster
        """
        self.est_configs = [] if est_configs is None else est_configs
        super(AutoGrowingCascadeLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        self.task = task
        self.early_stop_rounds = early_stopping_rounds
        self.max_layers = max_layers
        self.n_classes = n_classes
        if self.task == 'regression':
            self.n_classes = 1
        # if look_index_cycle is None, you need set look_index_cycle in fit / fit_transform
        self.look_index_cycle = look_index_cycle
        self.data_save_rounds = data_save_rounds
        self.data_save_dir = data_save_dir
        check_dir(self.data_save_dir)  # check data save dir, if not exists, create the dir
        self.model_save_dir = model_save_dir
        check_dir(self.model_save_dir)
        self.keep_in_mem = keep_in_mem
        self.stop_by_test = stop_by_test
        self.metrics = metrics
        if self.metrics is None:
            if task == 'regression':
                self.metrics = 'mse'
            else:
                self.metrics = 'accuracy'
        self.eval_metrics = get_eval_metrics(self.metrics, self.task)
        self.seed = seed
        self.distribute = distribute
        self.verbose_dis = verbose_dis
        self.dis_level = dis_level
        if distribute is True:
            if num_workers is not None:
                self.num_workers = num_workers
        # properties
        self.layer_fit_cascades = []
        self.n_layers = 0
        self.opt_layer_id = 0
        self.n_group_train = 0
        self.group_starts = []
        self.group_ends = []
        self.group_dims = []
        self.test_results = None
        self.opt_test_metric = 0
        self.opt_train_metric = 0
        self.keep_test_result = keep_test_result

    def _create_cascade_layer(self, est_configs=None, data_save_dir=None, model_save_dir=None,
                              layer_id=None, metrics=None, seed=None, train_start_ends=None, x_train_group_or_id=None,
                              x_test_group_or_id=None):
        """
        Create a cascade layer.

        :param est_configs:
        :param data_save_dir:
        :param model_save_dir:
        :param layer_id:
        :param metrics: str, 'accuracy' or 'rmse' or 'mse' or 'auc'
        :param seed:
        :return: A CascadeLayer Object.
        """
        return CascadeLayer(dtype=self.dtype,
                            task=self.task,
                            est_configs=est_configs,
                            layer_id=layer_id,
                            n_classes=self.n_classes,
                            keep_in_mem=self.keep_in_mem,
                            data_save_dir=data_save_dir,
                            model_save_dir=model_save_dir,
                            num_workers=self.num_workers,
                            metrics=metrics,
                            seed=seed,
                            distribute=self.distribute,
                            verbose_dis=self.verbose_dis,
                            dis_level=self.dis_level,
                            train_start_ends=train_start_ends,
                            x_train_group_or_id=x_train_group_or_id,
                            x_test_group_or_id=x_test_group_or_id)

    def call(self, x_trains):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def add(self, est):
        """
        Add an estimator to the auto growing cascade layer.

        :param est:
        :return:
        """
        if isinstance(est, EstimatorConfig):
            self.est_configs.append(est.get_est_args())
        elif isinstance(est, dict):
            self.est_configs.append(est)
        else:
            raise ValueError("Unknown estimator information {}".format(est))

    @property
    def _percent(self):
        return '%' if isinstance(self.eval_metrics[0], Accuracy) else ''

    @property
    def is_classification(self):
        return self.task == 'classification'

    @property
    def larger_better(self):
        """
        True if the evaluation metric larger is better.
        :return:
        """
        if isinstance(self.eval_metrics[0], (MSE, RMSE)):
            return False
        return True

    def fit(self, x_trains, y_train):
        """
        Fit with x_trains, y_trains.

        :param x_trains:
        :param y_train:
        :return:
        """
        x_trains, y_train, _, _ = self._validate_input(x_trains, y_train)
        if self.stop_by_test is True:
            self.LOGGER.warn('stop_by_test is True, but we do not obey it when fit(x_train, y_train)!')
        self.layer_fit_cascades = []
        n_groups_train = len(x_trains)
        self.n_group_train = n_groups_train
        n_trains = len(y_train)
        # Initialize the groups
        x_train_group = np.zeros((n_trains, 0), dtype=x_trains[0].dtype)
        group_starts, group_ends, group_dims = [], [], []
        # train set
        for i, x_train in enumerate(x_trains):
            assert x_train.shape[0] == n_trains, ('x_train.shape[0]={} not equal to'
                                                  ' n_trains={}'.format(x_train.shape[0], n_trains))
            x_train = x_train.reshape(n_trains, -1)
            group_dims.append(x_train.shape[1])
            group_starts.append(i if i == 0 else group_ends[i - 1])
            group_ends.append(group_starts[i] + group_dims[i])
            x_train_group = np.hstack((x_train_group, x_train))

        self.LOGGER.info('group_starts={}'.format(group_starts))
        self.LOGGER.info('group_dims={}'.format(group_dims))
        self.LOGGER.info('X_train_group={}'.format(x_train_group.shape))
        self.group_starts = group_starts
        self.group_ends = group_ends
        self.group_dims = group_dims
        if self.look_index_cycle is None:
            self.look_index_cycle = [[i, ] for i in range(n_groups_train)]
        else:
            for look_index in self.look_index_cycle:
                if np.max(look_index) >= n_groups_train or np.min(look_index) < 0 or len(look_index) == 0:
                    raise ValueError("look_index invalid! look_index={}".format(look_index))
        x_cur_train = None
        x_proba_train = np.zeros((n_trains, 0), dtype=self.dtype)
        layer_id = 0
        layer_metric_list = []
        opt_data = [None, None]
        try:
            while True:
                if layer_id >= self.max_layers > 0:
                    break
                # clear x_cur_train
                x_cur_train = np.zeros((n_trains, 0), dtype=self.dtype)
                train_ids = self.look_index_cycle[layer_id % len(self.look_index_cycle)]
                for gid in train_ids:
                    x_cur_train = np.hstack((x_cur_train, x_train_group[:, group_starts[gid]:group_ends[gid]]))
                x_cur_train = np.hstack((x_cur_train, x_proba_train))
                data_save_dir = self.data_save_dir
                if data_save_dir is not None:
                    data_save_dir = osp.join(data_save_dir, 'cascade_layer_{}'.format(layer_id))
                model_save_dir = self.model_save_dir
                if model_save_dir is not None:
                    model_save_dir = osp.join(model_save_dir, 'cascade_layer_{}'.format(layer_id))
                cascade = self._create_cascade_layer(est_configs=self.est_configs,
                                                     data_save_dir=data_save_dir,
                                                     model_save_dir=model_save_dir,
                                                     layer_id=layer_id,
                                                     metrics=self.metrics,
                                                     seed=self.seed)
                x_proba_train, _ = cascade.fit_transform(x_cur_train, y_train)
                if self.keep_in_mem:
                    self.layer_fit_cascades.append(cascade)
                layer_metric_list.append(cascade.train_avg_metric)
                # detect best layer id
                opt_layer_id = get_opt_layer_id(layer_metric_list, self.larger_better)
                self.opt_layer_id = opt_layer_id
                self.opt_train_metric = layer_metric_list[opt_layer_id]
                # if this layer is the best layer, set the opt_data
                if opt_layer_id == layer_id:
                    opt_data = [x_cur_train, y_train]
                # early stopping
                if layer_id - opt_layer_id >= self.early_stop_rounds > 0:
                    # log and save the final results of the optimal layer
                    self.LOGGER.info('[Result][Early Stop][Optimal Layer Detected] opt_layer={},'.format(opt_layer_id) +
                                     ' {}_train={:.4f}{},'.format(self.metrics,
                                                                  layer_metric_list[opt_layer_id], self._percent))
                    self.n_layers = layer_id + 1
                    self.save_data(opt_layer_id, True, *opt_data)
                    # wash the fit cascades after optimal layer id to save memory
                    if self.keep_in_mem:
                        for li in range(opt_layer_id + 1, layer_id + 1):
                            self.layer_fit_cascades[li] = None
                    return x_cur_train
                if self.data_save_rounds > 0 and (layer_id + 1) % self.data_save_rounds == 0:
                    self.save_data(layer_id, False, *opt_data)
                layer_id += 1
            # Max Layer Reached
            # opt_data = [x_cur_train, y_train]
            opt_layer_id = get_opt_layer_id(layer_metric_list, larger_better=self.larger_better)
            self.opt_layer_id = opt_layer_id
            self.opt_train_metric = layer_metric_list[opt_layer_id]
            self.LOGGER.info('[Result][Max Layer Reach] max_layer={}, {}_train={:.4f}{},'
                             ' optimal_layer={}, {}_optimal_train={:.4f}{}'.format(
                                self.max_layers,
                                self.metrics, layer_metric_list[-1], self._percent, opt_layer_id,
                                self.metrics, layer_metric_list[opt_layer_id], self._percent))
            self.save_data(opt_layer_id, True, *opt_data)
            self.n_layers = layer_id + 1
            # wash the fit cascades after optimal layer id to save memory
            if self.keep_in_mem:
                for li in range(opt_layer_id + 1, layer_id + 1):
                    self.layer_fit_cascades[li] = None
            return x_cur_train
        except KeyboardInterrupt:
            pass

    def fit_transform(self, x_trains, y_train, x_tests=None, y_test=None):
        """
        NOTE: Only support ONE x_train and one x_test, so y_train is a single numpy array instead of list of it.

        :param x_trains:
        :param y_train:
        :param x_tests:
        :param y_test:
        :return:
        """
        if x_tests is None:
            return self.fit(x_trains, y_train), None
        x_trains, y_train, x_tests, y_test = self._validate_input(x_trains, y_train, x_tests, y_test)
        self.layer_fit_cascades = []
        n_groups_train = len(x_trains)
        self.n_group_train = n_groups_train
        n_groups_test = len(x_tests)
        n_trains = len(y_train)
        n_tests = x_tests[0].shape[0]  # y_test might be None
        if y_test is None and self.stop_by_test is True:
            self.stop_by_test = False
            self.LOGGER.warn('stop_by_test is True, but we do not obey it when fit(x_train, y_train, x_test, None)!')
        assert n_groups_train == n_groups_test, 'n_group_train must equal to n_group_test!,' \
                                                ' but {} and {}'.format(n_groups_train, n_groups_test)
        # Initialize the groups
        # 2018-04-17 change x_trains[0].dtype to self.dtype
        x_train_group = np.zeros((n_trains, 0), dtype=self.dtype)
        x_test_group = np.zeros((n_tests, 0), dtype=self.dtype)
        group_starts, group_ends, group_dims = [], [], []
        # train set
        for i, x_train in enumerate(x_trains):
            assert x_train.shape[0] == n_trains, 'x_train.shape[0] = {} not equal to {}'.format(
                x_train.shape[0], n_trains)
            x_train = x_train.reshape(n_trains, -1)
            group_dims.append(x_train.shape[1])
            group_starts.append(i if i == 0 else group_ends[i - 1])
            group_ends.append(group_starts[i] + group_dims[i])
            x_train_group = np.hstack((x_train_group, x_train))
        # test set
        for i, x_test in enumerate(x_tests):
            assert x_test.shape[0] == n_tests
            x_test = x_test.reshape(n_tests, -1)
            assert x_test.shape[1] == group_dims[i]
            x_test_group = np.hstack((x_test_group, x_test))

        self.LOGGER.info('group_starts={}'.format(group_starts))
        self.LOGGER.info('group_dims={}'.format(group_dims))
        self.LOGGER.info('X_train_group={}, X_test_group={}'.format(x_train_group.shape, x_test_group.shape))
        self.group_starts = group_starts
        self.group_ends = group_ends
        self.group_dims = group_dims
        if self.look_index_cycle is None:
            self.look_index_cycle = [[i, ] for i in range(n_groups_train)]
        else:
            for look_index in self.look_index_cycle:
                if np.max(look_index) >= n_groups_train or np.min(look_index) < 0 or len(look_index) == 0:
                    raise ValueError("look_index invalid! look_index={}".format(look_index))
        x_cur_train, x_cur_test = None, None
        x_proba_train = np.zeros((n_trains, 0), dtype=self.dtype)
        x_proba_test = np.zeros((n_tests, 0), dtype=self.dtype)
        cascade = None  # for save test results
        layer_id = 0
        layer_train_metrics, layer_test_metrics = [], []
        opt_data = [None, None]
        machines = defaultdict(int)
        trees = defaultdict(int)
        machine_time_start = defaultdict(float)
        machine_time_end = defaultdict(float)
        x_train_group_or_id = ray.put(x_train_group) if self.distribute else x_train_group
        x_test_group_or_id = ray.put(x_test_group) if self.distribute else x_test_group
        try:
            while True:
                if layer_id >= self.max_layers > 0:
                    break
                # x_cur_train = np.zeros((n_trains, 0), dtype=self.dtype)
                # x_cur_test = np.zeros((n_tests, 0), dtype=self.dtype)
                train_ids = self.look_index_cycle[layer_id % len(self.look_index_cycle)]
                # for gid in train_ids:
                #     x_cur_train = np.hstack((x_cur_train, x_train_group[:, group_starts[gid]:group_ends[gid]]))
                #     x_cur_test = np.hstack((x_cur_test, x_test_group[:, group_starts[gid]:group_ends[gid]]))
                train_start_ends = [(group_starts[gid], group_ends[gid]) for gid in train_ids]
                # x_cur_train = np.hstack((x_cur_train, x_proba_train))
                # x_cur_test = np.hstack((x_cur_test, x_proba_test))
                assert x_proba_train.dtype == self.dtype, ("x_proba_train dtype = {} != self.dtype({})".format(
                    x_proba_train.dtype, self.dtype))
                x_cur_train = x_proba_train
                x_cur_test = x_proba_test
                data_save_dir = self.data_save_dir
                if data_save_dir is not None:
                    data_save_dir = osp.join(data_save_dir, 'cascade_layer_{}'.format(layer_id))
                model_save_dir = self.model_save_dir
                if model_save_dir is not None:
                    model_save_dir = osp.join(model_save_dir, 'cascade_layer_{}'.format(layer_id))
                cascade = self._create_cascade_layer(est_configs=self.est_configs,
                                                     data_save_dir=data_save_dir,
                                                     model_save_dir=model_save_dir,
                                                     layer_id=layer_id,
                                                     metrics=self.metrics,
                                                     seed=self.seed,
                                                     train_start_ends=train_start_ends,
                                                     x_train_group_or_id=x_train_group_or_id,
                                                     x_test_group_or_id=x_test_group_or_id)
                x_proba_train, x_proba_test = cascade.fit_transform(x_cur_train, y_train, x_cur_test, y_test)
                if self.keep_in_mem:
                    self.layer_fit_cascades.append(cascade)
                layer_train_metrics.append(cascade.train_avg_metric)
                layer_test_metrics.append(cascade.test_avg_metric)
                # detect best layer id
                if self.stop_by_test:
                    opt_layer_id = get_opt_layer_id(layer_test_metrics, self.larger_better)
                    self.opt_test_metric = layer_test_metrics[opt_layer_id]
                else:
                    opt_layer_id = get_opt_layer_id(layer_train_metrics, self.larger_better)
                    self.opt_train_metric = layer_train_metrics[opt_layer_id]
                self.opt_layer_id = opt_layer_id
                # if this layer is the best layer, set the opt_data
                if opt_layer_id == layer_id:
                    opt_data = [x_cur_train, y_train, x_cur_test, y_test]
                    # detected best layer, save test result
                    if y_test is None and cascade is not None:
                        self.save_test_result(x_proba_test=cascade.eval_proba_test)
                        if self.keep_test_result:
                            self.test_results = cascade.eval_proba_test
                # early stopping
                if layer_id - opt_layer_id >= self.early_stop_rounds > 0:
                    # log and save the final results of the optimal layer
                    if y_test is not None:
                        self.LOGGER.info('[Result][Early Stop][Optimal Layer Detected]'
                                         ' opt_layer={},'.format(opt_layer_id) +
                                         ' {}_train={:.4f}{}, {}_test={:.4f}{}'.format(
                                          self.metrics, layer_train_metrics[opt_layer_id],
                                          self._percent, self.metrics, layer_test_metrics[opt_layer_id],
                                          self._percent))
                    else:
                        self.LOGGER.info('[Result][Early Stop][Optimal Layer Detected]'
                                         ' opt_layer={},'.format(opt_layer_id) +
                                         ' {}_train={:.4f}{}'.format(self.metrics,
                                                                     layer_train_metrics[opt_layer_id],
                                                                     self._percent))
                    self.n_layers = layer_id + 1
                    self.save_data(opt_layer_id, True, *opt_data)
                    # wash the fit cascades after optimal layer id to save memory
                    if self.keep_in_mem:  # if not keep_in_mem, self.layer_fit_cascades is None originally
                        for li in range(opt_layer_id + 1, layer_id + 1):
                            self.layer_fit_cascades[li] = None
                    # return the best layer
                    return opt_data[0], opt_data[2]
                if self.data_save_rounds > 0 and (layer_id + 1) % self.data_save_rounds == 0:
                    self.save_data(layer_id, False, *opt_data)
                for key in cascade.machines.keys():
                    machines[key] += cascade.machines[key]
                    trees[key] += cascade.trees[key]
                    if key not in machine_time_start.keys():
                        machine_time_start[key] = cascade.machine_time_start[key]
                    else:
                        machine_time_start[key] = min(machine_time_start[key], cascade.machine_time_start[key])
                    machine_time_end[key] = max(machine_time_end[key], cascade.machine_time_end[key])
                layer_id += 1
            # Max Layer Reached
            # opt_data = [x_cur_train, y_train, x_cur_test, y_test]
            # detect best layer id
            if self.stop_by_test:
                opt_layer_id = get_opt_layer_id(layer_test_metrics, larger_better=self.larger_better)
                self.opt_test_metric = layer_test_metrics[opt_layer_id]
            else:
                opt_layer_id = get_opt_layer_id(layer_train_metrics, self.larger_better)
                self.opt_train_metric = layer_train_metrics[opt_layer_id]
            self.opt_layer_id = opt_layer_id
            if y_test is not None:
                self.LOGGER.info('[Result][Max Layer Reach] max_layer={}, {}_train={:.4f}{}, {}_test={:.4f}{}'
                                 ' optimal_layer={}, {}_optimal_train={:.4f}{},'
                                 ' {}_optimal_test={:.4f}{}'.format(
                                    self.max_layers, self.eval_metrics[0].name, layer_train_metrics[-1], self._percent,
                                    self.metrics, layer_test_metrics[-1], self._percent, opt_layer_id,
                                    self.metrics, layer_train_metrics[opt_layer_id], self._percent,
                                    self.metrics, layer_test_metrics[opt_layer_id], self._percent))
            else:
                self.LOGGER.info('[Result][Max Layer Reach] max_layer={}, {}_train={:.4f}{},'
                                 ' optimal_layer={}, {}_optimal_train={:.4f}{}'.format(
                                  self.max_layers,
                                  self.metrics, layer_train_metrics[-1], self._percent, opt_layer_id,
                                  self.metrics, layer_train_metrics[opt_layer_id], self._percent))
            self.save_data(opt_layer_id, True, *opt_data)
            self.n_layers = layer_id + 1
            # if y_test is None, we predict x_test and save its predictions
            if y_test is None and cascade is not None:
                self.save_test_result(x_proba_test=cascade.eval_proba_test)
                if self.keep_test_result:
                    self.test_results = cascade.eval_proba_test
            # wash the fit cascades after optimal layer id to save memory
            if self.keep_in_mem:
                for li in range(opt_layer_id + 1, layer_id + 1):
                    self.layer_fit_cascades[li] = None
        except KeyboardInterrupt:
            pass
        finally:
            total_task = sum([v for v in machines.values()])
            for key in machines.keys():
                self.LOGGER.info('[SUMMARY] Machine {} was assigned {}:{} / {}, across {}'.format(
                    key, machines[key], trees[key], total_task, machine_time_end[key] - machine_time_start[key]))
            return x_cur_train, x_cur_test

    def transform(self, X, y=None):
        """
        Transform inputs X.

        :param X:
        :param y:
        :return:
        """
        if not isinstance(X, (list, tuple)):
            X = [X]
        n_groups = len(X)
        n_examples = len(X[0])
        # Initialize the groups
        x_test_group = np.zeros((n_examples, 0), dtype=X[0].dtype)
        # test set
        for i, x_test in enumerate(X):
            assert x_test.shape[0] == n_examples
            x_test = x_test.reshape(n_examples, -1)
            assert x_test.shape[1] == self.group_dims[i]
            x_test_group = np.hstack((x_test_group, x_test))

        self.LOGGER.info('[transform] group_starts={}'.format(self.group_starts))
        self.LOGGER.info('[transform] group_dims={}'.format(self.group_dims))
        self.LOGGER.info('[transform] X_test_group={}'.format(x_test_group.shape))

        if self.look_index_cycle is None:
            self.look_index_cycle = [[i, ] for i in range(n_groups)]
        x_proba_test = np.zeros((n_examples, 0), dtype=self.dtype)
        layer_id = 0
        try:
            while layer_id <= self.opt_layer_id:
                self.LOGGER.info('Transforming layer - {} / {}'.format(layer_id, self.n_layers))
                x_cur_test = np.zeros((n_examples, 0), dtype=self.dtype)
                train_ids = self.look_index_cycle[layer_id % n_groups]
                for gid in train_ids:
                    x_cur_test = np.hstack((x_cur_test, x_test_group[:, self.group_starts[gid]:self.group_ends[gid]]))
                x_cur_test = np.hstack((x_cur_test, x_proba_test))
                cascade = self.layer_fit_cascades[layer_id]
                x_proba_test = cascade.transform(x_cur_test)
                layer_id += 1
            return x_proba_test
        except KeyboardInterrupt:
            pass

    def evaluate(self, inputs, labels, eval_metrics=None):
        """
        Evaluate inputs.

        :param inputs:
        :param labels:
        :param eval_metrics:
        :return:
        """
        if eval_metrics is None:
            eval_metrics = [Accuracy('evaluate')]
        if isinstance(labels, (list, tuple)):
            assert len(labels) == 1, 'only support single labels array'
            labels = labels[0]
        predictions = self.predict(inputs)
        for metric in eval_metrics:
            metric.calc(labels, predictions, logger=self.LOGGER)

    def predict_proba(self, X):
        """
        Predict probability of X.

        :param X:
        :return:
        """
        if not isinstance(X, (list, tuple)):
            X = [X]
        x_proba = self.transform(X)
        total_proba = np.zeros((X[0].shape[0], self.n_classes), dtype=self.dtype)
        for i in range(len(self.est_configs)):
            total_proba += x_proba[:, i * self.n_classes:i * self.n_classes + self.n_classes]
        return total_proba

    def predict(self, X):
        """
        Predict with inputs X.

        :param X:
        :return:
        """
        total_proba = self.predict_proba(X)
        if self.is_classification:
            return np.argmax(total_proba.reshape((-1, self.n_classes)), axis=1)
        else:
            return total_proba.reshape((-1, self.n_classes))

    def _depack(self, x, depth):
        if depth == 0:
            return [x]
        elif depth == 1:
            return x
        elif depth == 2:
            x_cp = []
            for bottom in x:
                for xj in bottom:
                    x_cp.append(xj)
            return x_cp
        else:
            raise ValueError('_concat failed. depth should be less than 2!')

    def _validate_input(self, x_train, y_train, x_test=None, y_test=None):
        assert x_train is not None and y_train is not None, 'x_train is None or y_train is None'
        train_depth = 0
        if isinstance(x_train, (list, tuple)):
            train_depth = check_list_depth(x_train)
        x_train = self._depack(x_train, train_depth)
        if x_test is not None:
            test_depth = 0
            if isinstance(x_test, (list, tuple)):
                test_depth = check_list_depth(x_test)
            x_test = self._depack(x_test, test_depth)
        # only supports one y_train
        if isinstance(y_train, (list, tuple)) and len(y_train) > 0:
            y_train = y_train[0]
        if y_test is not None and isinstance(y_test, (list, tuple)):
            y_test = y_test[0]
        return x_train, y_train, x_test, y_test

    @property
    def num_layers(self):
        """
        Number of layers.

        :return:
        """
        return self.n_layers

    def save_data(self, layer_id, opt, x_train, y_train, x_test=None, y_test=None):
        """
        Save the intermediate training data and testing data in this layer.

        :param layer_id:
        :param opt:
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return:
        """
        if self.data_save_dir is None:
            return
        for phase in ['train', 'test']:
            if phase == 'test' and x_test is None:
                return
            data_path = osp.join(self.data_save_dir, "{}{}.pkl".format("opt-" if opt else "", phase))
            check_dir(data_path)
            if phase == 'train':
                data = {"X": x_train, "y": y_train}
            else:
                data = {"X": x_test, "y": y_test if y_test is not None else np.zeros((0,), dtype=self.dtype)}
            self.LOGGER.debug("Saving {} Data in {} ... X.shape={}, y.shape={}".format(
                phase, data_path, data["X"].shape, data["y"].shape))
            with open(data_path, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def save_test_result(self, x_proba_test):
        """
        Save prediction result for testing data without label.

        :param x_proba_test:
        :return:
        """
        if self.data_save_dir is None:
            return
        if x_proba_test is None:
            self.LOGGER.info('x_proba_test is None, DO NOT SAVE!')
            return
        if x_proba_test.shape[1] != self.n_classes:
            self.LOGGER.info('x_proba_test.shape[1] = {} is not equal to n_classes'.format(x_proba_test.shape[1]))
        prefix = datetime.datetime.now().strftime('%m_%d_%H_%M')
        file_name = osp.join(self.data_save_dir, 'submission_' + prefix + '.csv')
        self.LOGGER.info('[Save][Test Output] x_proba_test={}, Saving to {}'.format(x_proba_test.shape, file_name))
        if self.is_classification:
            np.savetxt(file_name, np.argmax(x_proba_test, axis=1), fmt="%d", delimiter=',')
        else:
            np.savetxt(file_name, x_proba_test, fmt="%f", delimiter=',')

    def save(self):
        pass

    @property
    def summary_info(self):
        info_str = "maxlayer={}, esrounds={}".format(self.max_layers, self.early_stop_rounds)
        info_str += '\nEach Level:\n'
        info_str += '['
        info_str += ', '.join(map(str, self.est_configs))
        info_str += ']'
        return info_str


class FinalLayer(Layer):
    def __init__(self, batch_size=None, dtype=np.float32, name=None, task='classification', est_configs=None,
                 n_classes=None, keep_in_mem=False, data_save_dir=None, model_save_dir=None,
                 metrics=None, keep_test_result=False, seed=None, distribute=False, verbose_dis=False,
                 dis_level=1, num_workers=None):
        """
        The final classification layer.
        The estimator(s) of this layer commonly is/are estimator(s) with low bias.

        :param batch_size:
        :param dtype:
        :param name:
        :param est_configs:
        :param n_classes:
        :param keep_in_mem:
        :param data_save_dir:
        :param model_save_dir:
        :param metrics:
        :param keep_test_result:
        :param seed:
        :param distribute:
        :param verbose_dis:
        :param dis_level:
        :param num_workers:
        """
        self.est_configs = [] if est_configs is None else est_configs
        super(FinalLayer, self).__init__(batch_size=batch_size, dtype=dtype, name=name)
        self.task = task
        self.n_classes = n_classes
        if self.task == 'regression':
            self.n_classes = 1
        self.keep_in_mem = keep_in_mem
        self.data_save_dir = data_save_dir
        check_dir(self.data_save_dir)   # check data save dir, if not exists, create the dir
        self.model_save_dir = model_save_dir
        check_dir(self.model_save_dir)  # check model save dir, if not exists, create the dir
        self.metrics = metrics
        if self.metrics is None:
            if task == 'regression':
                self.metrics = 'mse'
            else:
                self.metrics = 'accuracy'
        self.eval_metrics = get_eval_metrics(self.metrics, self.task)
        self.keep_test_result = keep_test_result
        self.seed = seed
        self.distribute = distribute
        self.dis_level = dis_level
        if distribute is True:
            if num_workers is None:
                self.init_num_workers()
            else:
                self.num_workers = num_workers
        self.verbose_dis = verbose_dis
        self.est_args = []
        for est_conf in self.est_configs:
            if isinstance(est_conf, EstimatorConfig):
                self.est_args.append(est_conf.get_est_args().copy())

    def call(self, x_trains):
        raise NotImplementedError

    def __call__(self, x_trains):
        raise NotImplementedError

    def _init_estimator(self, est_id):
        """
        Initialize a k_fold estimator.

        :return:
        """
        est_args = self.est_args[est_id].copy()
        est_name = 'final - estimator - {} - {}folds'.format(est_id, est_args['n_folds'])
        n_folds = int(est_args['n_folds'])
        est_args.pop('n_folds')
        est_type = est_args['est_type']
        est_args.pop('est_type')
        # seed
        if self.seed is not None:
            seed = (self.seed + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            seed = None
        if self.distribute:
            get_est_func = get_dist_estimator_kfold
        else:
            get_est_func = get_estimator_kfold
        return get_est_func(name=est_name,
                            n_folds=n_folds,
                            task=self.task,
                            est_type=est_type,
                            eval_metrics=self.eval_metrics,
                            seed=seed,
                            dtype=self.dtype,
                            keep_in_mem=self.keep_in_mem,
                            est_args=est_args,
                            cv_seed=seed)

    def fit(self, x_trains, y_trains):
        raise NotImplementedError

    def fit_transform_with_file(self, train_file, test_file):
        with open(train_file) as train:
            train_data = pickle.load(train)
        with open(test_file) as test:
            test_data = pickle.load(test)
        x_train, y_train, x_test, y_test = train_data['X'], train_data['y'], test_data['X'], test_data['y']
        return self.fit_transform(x_train, y_train, x_test, y_test)

    def fit_transform(self, x_train, y_train, x_test=None, y_test=None):
        x_train, y_train, x_test, y_test = _validate_input(x_train, y_train, x_test, y_test)
        if y_test is None:
            y_test_shape = (0,)
        else:
            y_test_shape = y_test.shape
        self.LOGGER.info('X_train.shape={}, y_train.shape={}, dtype={}'.format(x_train.shape, y_train.shape,
                                                                               x_train.dtype))
        self.LOGGER.info(' X_test.shape={},  y_test.shape={}, dtype={}'.format(x_test.shape, y_test_shape,
                                                                               x_test.dtype))
        n_trains = x_train.shape[0]
        n_tests = x_test.shape[0]
        n_classes = self.n_classes  # if regression, n_classes = 1
        if self.task == 'classification' and n_classes is None:
            n_classes = np.unique(y_train)
        if self.task == 'regression' and n_classes is None:
            n_classes = 1
        y_stratify = y_train if self.task == 'classification' else None
        est = self._init_estimator(0)
        y_probas = est.fit_transform(x_train, y_train, y_stratify, test_sets=[('test', x_test, y_test)])
        y_proba_train, y_proba_test = y_probas[0], y_probas[1]
        metric = self.eval_metrics[0]
        train_avg_metric = metric.calc_proba(y_train, y_proba_train,
                                             'Final Layer - [train] average'.format(metric.name),
                                             logger=self.LOGGER)
        self.train_avg_metric = train_avg_metric
        # if only one element on test_sets, return one test result like y_proba_train
        if isinstance(y_proba_test, (list, tuple)) and len(y_proba_test) == 1:
            y_proba_test = y_proba_test[0]
        if y_proba_train is None:
            raise RuntimeError("Final Layer - estimator - {} fit FAILED!,"
                               " y_proba_train is None".format(0))
        check_shape(y_proba_train, n_trains, n_classes)
        if y_proba_test is not None:
            check_shape(y_proba_test, n_tests, n_classes)
            # judge whether y_test is None, which means users are to predict test probas
            if y_test is not None:
                test_avg_metric = metric.calc_proba(y_test, y_proba_test,
                                                    'Final Layer - [ test] average',
                                                    logger=self.LOGGER)
                self.test_avg_metric = test_avg_metric
            # if y_test is None, we need to generate test prediction, so keep eval_proba_test
            if y_test is None:
                self.eval_proba_test = y_proba_test
        return y_proba_train, y_proba_test

    def transform(self, inputs):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError

    def predict_proba(self, inputs):
        raise NotImplementedError

    def evaluate(self, inputs, labels):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    @property
    def summary_info(self):
        return self.__str__()


def _to_snake_case(name):
    import re
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure


def get_opt_layer_id(acc_list, larger_better=True):
    """ Return layer id with max accuracy on training data """
    if larger_better:
        opt_layer_id = np.argsort(-np.asarray(acc_list), kind='mergesort')[0]
    else:
        opt_layer_id = np.argsort(np.asarray(acc_list), kind='mergesort')[0]
    return opt_layer_id


def get_eval_metrics(metrics, task='classification', name=''):
    if metrics is None:
        if task == 'regression':
            eval_metrics = [MSE(name)]
        else:
            eval_metrics = [Accuracy(name)]
    elif isinstance(metrics, Metrics):
        eval_metrics = [metrics]
    elif isinstance(metrics, basestring):
        if metrics == 'accuracy':
            eval_metrics = [Accuracy(name)]
        elif metrics == 'auc':
            eval_metrics = [AUC(name)]
        elif metrics == 'mse':
            eval_metrics = [MSE(name)]
        elif metrics == 'rmse':
            eval_metrics = [RMSE(name)]
        else:
            raise ValueError('Unknown metrics : {}'.format(metrics))
    else:
        raise ValueError('Unknown metrics {} of type {}'.format(metrics, type(metrics)))
    return eval_metrics


def check_shape(y_proba, n, n_classes):
    if y_proba.shape != (n, n_classes):
        raise ValueError('output shape incorrect!,'
                         ' should be {}, but {}'.format((n, n_classes), y_proba.shape))


def check_dtype(y_proba, dtype):
    if y_proba.dtype != dtype:
        y_proba = y_proba.astype(dtype)
    return y_proba


def _concat(x, depth):
    """
    Concatenation inner method, to make multiple inputs to be single input, so that to feed it into classifiers.

    :param x: input data, single ndarray(depth=0) or list(depth=1) or 2D list (depth=2), at most 2D.
    :param depth: as stated above, single ndarray(depth=0) or list(depth=1) or 2D list (depth=2)
    :return: concatenated data
    """
    if depth == 0:
        return x
    elif depth == 1:
        for i, bottom in enumerate(x):
            x[i] = bottom.reshape((bottom.shape[0], -1))
        x = np.concatenate(x, 1)
    elif depth == 2:
        for i, bottoms in enumerate(x):
            for j, bot in enumerate(bottoms):
                bottoms[j] = bot.reshape((bot.shape[0], -1))
            x[i] = np.concatenate(bottoms, 1)
        for i, bottom in enumerate(x):
            x[i] = bottom.reshape((bottom.shape[0], -1))
        x = np.concatenate(x, 1)
    else:
        raise ValueError('_concat failed. depth should be less than 2!')
    return x


def _validate_input(x_train, y_train, x_test=None, y_test=None):
    """
    Validate input, check if x_train / x_test s' depth, and do some necessary transform like concatenation.

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    assert x_train is not None and y_train is not None, 'x_train is None or y_train should not be None'
    train_depth = 0
    if isinstance(x_train, (list, tuple)):
        train_depth = check_list_depth(x_train)
    x_train = _concat(x_train, train_depth)
    if x_test is not None:
        test_depth = 0
        if isinstance(x_test, (list, tuple)):
            test_depth = check_list_depth(x_test)
        x_test = _concat(x_test, test_depth)
    if isinstance(y_train, (list, tuple)) and y_train is not None:
        y_train = None if len(y_train) == 0 else y_train[0]
    if isinstance(y_test, (list, tuple)) and y_test is not None:
        y_test = None if len(y_test) == 0 else y_test[0]
    return x_train, y_train, x_test, y_test
