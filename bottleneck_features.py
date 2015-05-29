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

3. train een binaire classifier 
op de bottleneck features.
"""

from __future__ import division

from lasagne.layers import \
    get_output, get_all_layers, get_output_shape

from dnn import load_model

import numpy as np
from sklearn import svm

def get_bottleneck_features(network, X):
    layers = get_all_layers(network)
    bottleneck = [l for l in layers if l.name == 'bottleneck']
    if len(bottleneck) == 0:
        raise ValueError('network has no bottleneck')
    else:
        bottleneck = bottleneck[0]
    bfeatures = get_output(bottleneck, X)
    #print X.shape
    #print bfeatures.eval().shape
    return bfeatures

def train_SVM(X, y):
    clf = svm.SVC()
    clf.fit(X, y)  

if __name__ == '__main__':
    import sys
    #fname = sys.argv[1]

    fname1 = "/Users/ingeborg/model1/model_0.pkl"
    fname2 = "/Users/ingeborg/model2/model_0.pkl"
    fsamples = "/Users/ingeborg/Desktop/output.npz"
    dataset = np.load(fsamples)
    X, y, labels = dataset['X'], dataset['y'], dataset['labels']

    model = load_model(fname1)
    for layer in get_all_layers(model):
        print layer.name, get_output_shape(layer)
    b_features1 = get_bottleneck_features(model, X)

    #TODO: fix bug: ValueError: setting an array element with a sequence.
    #train_SVM(b_features1, y)

    model = load_model(fname2)
    for layer in get_all_layers(model):
        print layer.name, get_output_shape(layer)
    b_features2 = get_bottleneck_features(model, X)
    #train_SVM(b_features1, y)
