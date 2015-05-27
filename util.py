#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: util.py
# date: Sat May 23 22:40 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""util: utility functions

"""

from __future__ import division

from time import time
import sys
from contextlib import contextmanager
from collections import OrderedDict

from tabulate import tabulate

@contextmanager
def verb_print(msg, verbose=False):
    """Helper for verbose printing with timing around pieces of code.
    """
    if verbose:
        t0 = time()
        msg = msg + '...'
        print msg,
        sys.stdout.flush()
    try:
        yield
    finally:
        if verbose:
            print 'done. time: {0:.3f}s'.format(time() - t0)
            sys.stdout.flush()

class COLORS(object):
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class ProgressPrinter(object):
    """Adapted from nolearn's printlog class
    """
    def __init__(self, color=True):
        self.first_iteration = True
        self.color = color

    def __call__(self, train_history):
        print(self.table(train_history))

    def table(self, train_history):
        info = train_history[-1]

        table = OrderedDict([
            ('epoch', info['epoch'])])
        train_loss = info['train_loss']
        if self.color:
            if info['train_loss_best']:
                color = COLORS.BLUE
                end = COLORS.ENDC
            elif info['train_loss_worse']:
                color = COLORS.RED
                end = COLORS.ENDC
            else:
                color = ''
                end = ''
            table['train_loss'] = '{}{:.5f}{}'.format(
                color, train_loss, end)
        else:
            table['train_loss'] = '{:.5f}'.format(train_loss)
        valid_loss = info['valid_loss']
        if self.color:
            if info['valid_loss_best']:
                color = COLORS.GREEN
                end = COLORS.ENDC
            elif info['valid_loss_worse']:
                color = COLORS.RED
                end = COLORS.ENDC
            else:
                color = ''
                end = ''
            table['valid_loss'] = '{}{:.5f}{}'.format(
                color, valid_loss, end)
        else:
            table['valid_loss'] = '{:.5f}'.format(valid_loss)

        if 'valid_accuracy' in info:
            table['valid_acc'] = info['valid_accuracy']

        if self.color:
            ratio = train_loss / valid_loss
            if ratio < 0.9:
                color = COLORS.RED
            elif ratio > 1.1:
                color = COLORS.BLUE
            else:
                color = COLORS.ENDC
            table['ratio'] = '{}{:.2f}{}'.format(color, ratio, COLORS.ENDC)
        else:
            table['ratio'] = '{:.2f}'.format(ratio)

        table['dur'] = '{:.2f}s'.format(info['duration'])

        tabulated = tabulate([table], headers='keys', floatfmt='.5f')

        out = ''
        if self.first_iteration:
            out = '\n'.join(tabulated.split('\n', 2)[:2])
            out += '\n'
            self.first_iteration = False
        out += tabulated.rsplit('\n', 1)[-1]
        return out
