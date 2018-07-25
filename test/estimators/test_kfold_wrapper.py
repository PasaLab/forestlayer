# -*- coding:utf-8 -*-
"""
Test Suites of layers.graph.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import numpy as np
import os.path as osp
import unittest
from forestlayer.estimators.kfold_wrapper import greedy_makespan_split, determine_split
from forestlayer.estimators.estimator_configs import RandomForestConfig, ExtraRandomForestConfig, BinClassXGBConfig
from keras.datasets import mnist


class TestKFoldWrapper(unittest.TestCase):
    def setUp(self):
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        X = np.reshape(x_train, (60000, -1, 28, 28))
        x_train = X[:120, :, :, :]
        y_train = y_train[:120]
        x_test = np.reshape(x_test[:60], (60, -1, 28, 28))
        y_test = y_test[:60]
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        # print('X_train: ', x_train.shape, 'y: ', y_train.shape)
        # print(' X_test: ', x_test.shape, 'y: ', y_test.shape)
        self.estimators = [
            RandomForestConfig().get_est_args(),
            RandomForestConfig().get_est_args(),
            RandomForestConfig().get_est_args(),
            ExtraRandomForestConfig().get_est_args(),
            ExtraRandomForestConfig().get_est_args(),
            ExtraRandomForestConfig().get_est_args()
        ]

    def greedy_makespan_split1(self):
        splits = [
            [167, 33],
            [167, 33],
            [167, 33],
            [167, 33],
            [167, 33]
        ]
        forest_idx_tuples = [
            (167, 0), (33, 0),
            (167, 1), (33, 1),
            (167, 2), (33, 2),
            (167, 3), (33, 3),
            (167, 4), (33, 4),
        ]
        tag, new_splits = greedy_makespan_split(splits, 167, 6, forest_idx_tuples)
        should_splits = [[167, 33], [167, 33], [167, 33], [167, 33], [167, 33]]
        assert (tag is False and should_splits == new_splits)

    def greedy_makespan_split2(self):
        splits = [
            [200], [200], [200], [200],
            [200], [200], [200], [200],
        ]
        forest_idx_tuples = [
            (200, 0), (200, 1), (200, 2), (200, 3),
            (200, 4), (200, 5), (200, 6), (200, 7),
        ]
        tag, new_splits = greedy_makespan_split(splits, 400, 4, forest_idx_tuples)
        # print(new_splits)
        should_splits = [[200], [200], [200], [200], [200], [200], [200], [200]]
        assert (tag is False and should_splits == new_splits), ("new_splits should be {}"
                                                                " but {}".format(should_splits, new_splits))

    def greedy_makespan_split3(self):
        splits = [
            [134, 66], [134, 66]
        ]
        forest_idx_tuples = [
            (134, 0), (66, 0),
            (134, 1), (66, 1)
        ]
        tag, new_splits = greedy_makespan_split(splits, 134, 3, forest_idx_tuples)
        should_splits = [[134, 66], [134, 66]]
        assert (tag is False and should_splits == new_splits)

    def greedy_makespan_split4(self):
        splits = [
            [200], [200], [200]
        ]
        forest_idx_tuples = [
            (200, 0), (200, 1), (200, 2)
        ]
        tag, new_splits = greedy_makespan_split(splits, 300, 2, forest_idx_tuples)
        should_splits = [[200], [200], [100, 100]]
        assert (tag is True and should_splits == new_splits)

    def greedy_makespan_split5(self):
        splits = [
            [134, 66], [134, 66],
            [134, 66], [134, 66],
        ]
        forest_idx_tuples = [
            (134, 0), (66, 0),
            (134, 1), (66, 1),
            (134, 2), (66, 2),
            (134, 3), (66, 3),
        ]
        tag, new_splits = greedy_makespan_split(splits, 134, 6, forest_idx_tuples)
        # print(new_splits)
        should_splits = [[134, 66], [134, 66], [134, 66], [134, 66]]
        assert (tag is False and should_splits == new_splits)

    def test_greedy_makespan_split(self):
        self.greedy_makespan_split1()
        self.greedy_makespan_split2()
        self.greedy_makespan_split3()
        self.greedy_makespan_split4()
        self.greedy_makespan_split5()

    def determine_split1(self, dis_level=3):
        ests = [
            RandomForestConfig(n_estimators=200).get_est_args(),
            ExtraRandomForestConfig(n_estimators=200).get_est_args()
        ]
        should_split, split_scheme = determine_split(dis_level, 3, ests)
        if dis_level == 3:
            assert should_split is True
            assert split_scheme == [[67, 67, 66], [67, 67, 66]]
        elif dis_level == 2:
            assert should_split is True
            # print(split_scheme)
            assert split_scheme == [[100, 100], [100, 100]]

    def determine_split2(self, dis_level=3):
        ests = [
            RandomForestConfig(n_estimators=200).get_est_args(),
            RandomForestConfig(n_estimators=200).get_est_args(),
            ExtraRandomForestConfig(n_estimators=200).get_est_args()
        ]
        should_split, split_scheme = determine_split(dis_level, 1, ests)
        if dis_level == 3:
            assert should_split is True
            assert split_scheme == [[200], [200], [100, 100]]
        elif dis_level == 2:
            assert should_split is True
            # print(split_scheme)
            assert split_scheme == [[100, 100], [100, 100], [100, 100]]

    def determine_split3(self, dis_level=3):
        ests = [
            RandomForestConfig(n_estimators=200).get_est_args(),
            RandomForestConfig(n_estimators=200).get_est_args(),
            ExtraRandomForestConfig(n_estimators=200).get_est_args(),
            ExtraRandomForestConfig(n_estimators=200).get_est_args()
        ]
        should_split, split_scheme = determine_split(dis_level, 3, ests)
        if dis_level == 3:
            assert should_split is True
            assert split_scheme == [[134, 66], [134, 66], [134, 66], [134, 66]]
        elif dis_level == 2:
            assert should_split is True
            # print(split_scheme)
            assert split_scheme == [[100, 100], [100, 100], [100, 100], [100, 100]]

    def determine_split4(self, dis_level=3):
        ests = [
            RandomForestConfig(n_estimators=200).get_est_args(),
            RandomForestConfig(n_estimators=200).get_est_args(),
            RandomForestConfig(n_estimators=200).get_est_args(),
            RandomForestConfig(n_estimators=200).get_est_args(),
            ExtraRandomForestConfig(n_estimators=200).get_est_args(),
            ExtraRandomForestConfig(n_estimators=200).get_est_args(),
            ExtraRandomForestConfig(n_estimators=200).get_est_args(),
            ExtraRandomForestConfig(n_estimators=200).get_est_args()
        ]
        should_split, split_scheme = determine_split(dis_level, 4, ests)
        if dis_level == 3:
            assert should_split is False
            assert split_scheme == [[200], [200], [200], [200], [200], [200], [200], [200]]
        elif dis_level == 2:
            assert should_split is True
            # print(split_scheme)
            assert (split_scheme == [[100, 100], [100, 100], [100, 100], [100, 100],
                                     [100, 100], [100, 100], [100, 100], [100, 100]])

    def determine_split5(self, dis_level=3):
        ests = [
            RandomForestConfig(n_estimators=200).get_est_args(),
        ]
        should_split, split_scheme = determine_split(dis_level, 8, ests)
        if dis_level == 3:
            assert should_split is True
            assert split_scheme == [[13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 5]]
        elif dis_level == 2:
            assert should_split is True
            # print(split_scheme)
            assert split_scheme == [[100, 100]]

    def determine_split6(self, dis_level=3):
        ests = [
            RandomForestConfig(n_estimators=200).get_est_args(),
            BinClassXGBConfig().get_est_args(),
            BinClassXGBConfig().get_est_args(),
            BinClassXGBConfig().get_est_args()
        ]
        should_split, split_scheme = determine_split(dis_level, 8, ests)
        if dis_level == 3:
            assert should_split is True
            assert split_scheme == [[16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 8], [-1], [-1], [-1]]
        elif dis_level == 2:
            assert should_split is True
            # print(split_scheme)
            assert split_scheme == [[100, 100], [-1], [-1], [-1]]

    def determine_split7(self, dis_level=3):
        ests = [
            RandomForestConfig(n_estimators=200).get_est_args(),
            BinClassXGBConfig().get_est_args(),
            BinClassXGBConfig().get_est_args(),
            BinClassXGBConfig().get_est_args()
        ]
        should_split, split_scheme = determine_split(dis_level, 2, ests)
        if dis_level == 3:
            assert should_split is False
            assert split_scheme == [[200], [-1], [-1], [-1]]
        elif dis_level == 2:
            assert should_split is True
            # print(split_scheme)
            assert split_scheme == [[100, 100], [-1], [-1], [-1]]

    def determine_split8(self, dis_level=3):
        ests = [
            RandomForestConfig(n_estimators=200).get_est_args(),
            BinClassXGBConfig().get_est_args(),
            BinClassXGBConfig().get_est_args(),
            BinClassXGBConfig().get_est_args()
        ]
        should_split, split_scheme = determine_split(dis_level, 1, ests)
        if dis_level == 3:
            assert should_split is False
            assert split_scheme == [[200], [-1], [-1], [-1]]
        elif dis_level == 2:
            assert should_split is True
            # print(split_scheme)
            assert split_scheme == [[100, 100], [-1], [-1], [-1]]

    def test_determine_split_uniform(self, dis_level=3):
        ests = [
            RandomForestConfig(n_estimators=500).get_est_args(),
            RandomForestConfig(n_estimators=500).get_est_args(),
            ExtraRandomForestConfig(n_estimators=500).get_est_args(),
            ExtraRandomForestConfig(n_estimators=500).get_est_args()
        ]
        should_split, split_scheme = determine_split(50, 3, ests)
        print(should_split, split_scheme)

    def test_determine_split(self):
        # for i in [2, 3]:
        #     self.determine_split1(i)
        #     self.determine_split2(i)
        #     self.determine_split3(i)
        #     self.determine_split4(i)
        #     self.determine_split5(i)
        #     self.determine_split6(i)
        #     self.determine_split7(i)
        #     self.determine_split8(i)
        pass


if __name__ == '__main__':
    unittest.main()

