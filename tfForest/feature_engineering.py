# -*- coding:utf-8 -*-
"""
Feature Engineering.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import numpy as np


class FeatureParser(object):
    """
    Feature Parser is used to parse feature values according to features metadata.
    It was used to load UCI_ADULT data. See forestlayer.datasets.uci_adult.
    """
    def __init__(self, feature_spec):
        feature_spec = feature_spec.strip()
        if feature_spec == "C":
            self.feature_type = "continuous"
            self.feature_dim = 1
        else:
            self.feature_type = "categorical"
            feature_names = [f.strip() for f in feature_spec.split(',')]
            feature_names = ["?"] + feature_names
            self.name2id = dict(zip(feature_names, range(len(feature_names))))
            self.feature_dim = len(self.name2id)

    def get_continuous(self, f_data):
        """
        Get continuous data.

        :param f_data:
        :return:
        """
        f_data = f_data.strip()
        if self.feature_type == "continuous":
            return float(f_data)
        return float(self.name2id[f_data])

    def get_data(self, f_data):
        """
        Get continuous or categorical data.

        :param f_data:
        :return:
        """
        f_data = f_data.strip()
        if self.feature_type == "continuous":
            return float(f_data)
        data = np.zeros(self.feature_dim, dtype=np.float32)
        data[self.name2id[f_data]] = 1
        return data

    def get_featuredim(self):
        """
        get feature dimension
        """
        return self.feature_dim
