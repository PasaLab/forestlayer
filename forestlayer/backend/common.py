# -*- coding:utf-8 -*-
"""
Common backend.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

FIT_TIME = 0
KFOLD_TIME = 0
COMB_TIME = 0


def add_fit_time(tim):
    global FIT_TIME
    FIT_TIME += tim
    return FIT_TIME


def add_kfold_time(tim):
    global KFOLD_TIME
    KFOLD_TIME += tim
    return KFOLD_TIME


def add_comb_time(tim):
    global COMB_TIME
    COMB_TIME += tim
    return COMB_TIME
