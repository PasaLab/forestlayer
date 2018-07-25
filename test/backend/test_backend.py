# -*- coding:utf-8 -*-
"""
Test Suite of forestlayer.backend.backend.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import os.path as osp
import ray
from forestlayer.backend.backend import (set_base_dir, get_base_dir, get_num_nodes)
import unittest


class TestBackend(unittest.TestCase):
    def setUp(self):
        pass

    def test_base_dir(self):
        print("base dir: " + get_base_dir())
        set_base_dir(osp.expanduser(osp.join('~', 'forestlayer')))
        print('after set, base dir = {}'.format(get_base_dir()))

    def test_get_num_nodes(self):
        # Turn this standalone mode to cluster mode, otherwise the num_nodes will always be 1.
        ray.init()
        nodes, num_nodes = get_num_nodes()
        print(nodes, num_nodes)


if __name__ == '__main__':
    unittest.main()

