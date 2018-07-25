# -*- coding:utf-8 -*-
"""
Test Suites of sklearn estimators.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import numpy as np
import os.path as osp
import unittest
import pickle
from forestlayer.estimators import get_estimator_kfold
from forestlayer.estimators.sklearn_estimator import FLCRFClassifier
from sklearn.ensemble import RandomForestClassifier

s = get_estimator_kfold('ha')

k = FLCRFClassifier('', None)
print(k)
print(k.__module__)

with open('a.pkl', 'wb') as f:
    pickle.dump(k, f, pickle.HIGHEST_PROTOCOL)

