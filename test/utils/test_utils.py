# -*- coding:utf-8 -*-
"""
Test Suites of utils.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
from forestlayer.utils.utils import check_list_depth

lis = [[1, 2], [2, 3]]
print('before lis', lis)
depth = check_list_depth(lis)
print('after lis', lis)
print('depth', depth)
