#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: bottleneck_features.py
# date: Mon May 25 15:21 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""bottleneck_features:

"""

from __future__ import division

from lasagne.layers import \
    get_output, get_all_layers, get_output_shape

from dnn import load_model

def get_bottleneck_features(network, X):
    layers = get_all_layers(network)
    bottleneck = [l for l in layers if l.name == 'bottleneck']
    if len(bottleneck) == 0:
        raise ValueError('network has no bottleneck')
    else:
        bottleneck = bottleneck[0]
    return get_output(bottleneck, X)

if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    model = load_model(fname)
    for layer in get_all_layers(model):
        print layer.name, get_output_shape(layer)
