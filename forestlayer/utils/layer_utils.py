# -*- coding:utf-8 -*-
"""
Base utilities.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function


def check_list_depth(lis):
    """
    Check list's depth.

    :param lis:
    :return:
    """
    if lis is None:
        return 0
    depth = 0
    tmp = lis
    while isinstance(tmp, (list, tuple)):
        depth += 1
        tmp = tmp[0]
    return depth


def print_summary(graph, line_length=None, positions=None, print_fn=print):
    """
    Print summary for a graph.

    :param graph:
    :param line_length:
    :param positions:
    :param print_fn:
    :return:
    """
    line_length = line_length or 100
    positions = positions or [.25, .92, 1.]
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ['Layer', 'Description', 'Param #']

    def print_row(fields, position):
        line = ''
        for i in range(len(fields)):
            if i > 0:
                line = line[:-1] + ' '
            line += str(fields[i])
            line = line[:position[i]]
            line += ' ' * (position[i] - len(line))
        print_fn(line)

    print_fn('_' * line_length)
    print_row(to_display, positions)
    print_fn('=' * line_length)

    def print_layer_summary(layer):
        name = layer.__class__.__name__
        summary_info = layer.summary_info
        summary_info_lis = summary_info.split('\n')
        for i, sum_info in enumerate(summary_info_lis):
            if i == 0:
                fields = [name, sum_info, 'params']
            else:
                fields = ['', sum_info, '']
            print_row(fields, positions)

    layers = graph.layers
    for i, layer in enumerate(layers):
        print_layer_summary(layer)
        if i == len(layers) - 1:
            print_fn('=' * line_length)
        else:
            print_fn('_' * line_length)




