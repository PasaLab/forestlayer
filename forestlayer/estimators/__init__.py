# -*- coding:utf-8 -*-
"""
Initialize estimators.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from .base_estimator import BaseEstimator
from .kfold_wrapper import get_dist_estimator_kfold, get_estimator_kfold
from .estimator_configs import *

