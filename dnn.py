#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: dnn.py
# date: Wed May 27 18:30 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""dnn: some shared functions for neural nets

"""

from __future__ import division

import cPickle as pickle

import theano
import theano.tensor as T
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, softmax, \
    linear
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers import set_all_param_values, get_all_param_values
from lasagne.init import GlorotUniform

_standard_config = dict(
    batch_size=100,
    input_dim=1,
    output_dim=1,
    hidden_pre=(100, 100),
    dropout=0,
    hidden_f='rectify',
    bottleneck_size=5,
    bottleneck_f='linear',
    hidden_post=(),
    output_f='softmax'
)

def load_model(fname):
    with open(fname, 'rb') as fin:
        params, weights = pickle.load(fin)
    return build_model(weights=weights, **params)

def save_model(network, params, fname):
    weights = get_all_param_values(network)
    params['input_dim'] = weights[0].shape[0]
    params['output_dim'] = weights[-1].shape[0]
    params = {k: v for k, v in params.iteritems() if k in _standard_config}
    p = _standard_config.copy()
    p.update(params)
    with open(fname, 'wb') as fout:
        pickle.dump((p, weights), fout, -1)


def build_model(weights=None,
                batch_size=100,
                input_dim=1,
                output_dim=1,
                hidden_pre=(100, 100),
                dropout=0,
                hidden_f='rectify',
                bottleneck_size=5,
                bottleneck_f='linear',
                hidden_post=(),
                output_f='softmax'):
    """
    Build a model from parameters and optionally a set of weights.
    If not weights are given, an untrained model is returned.

    Also optionally, a bottleneck layer is added to the network.

    Parameters
    ----------
    weights : list of ndarrays
        weights for the network, if weights is None, an untrained network
        is returned
    batch_size : interval
        size of batches
    input_dim : int
        number of input dimensions
    output_dim : int
        number of output dimensions
    hidden_pre : sequence of ints
        sizes of dense hidden layers inserted before the bottleneck layer
    dropout : float
        proportion of units to dropout in the dense layers.
    hidden_f : string
        activation function for the hidden layers
    bottleneck_size : int
        size of the bottleneck layer
    bottleneck_f : string
        activation function for the bottleneck layer
    hidden_post : sequence of ints
        sizes of dense hidden layers inserted after the bottleneck
    output_f : string
        activation function for the output layer.

    Returns
    -------
    Layer
        output layer.

    """
    funcs = {'rectify': rectify,
             'sigmoid': T.nnet.hard_sigmoid,
             'tanh': tanh,
             'leaky_rectify': leaky_rectify,
             'softmax': softmax,
             'linear': linear}
    hidden_f = funcs[hidden_f]
    bottleneck_f = funcs[bottleneck_f]
    output_f = funcs[output_f]

    l_in = InputLayer(shape=(batch_size, input_dim), name='input')
    last = l_in
    for ix, size in enumerate(hidden_pre):
        l_hidden = DenseLayer(
            last, num_units=size,
            name='hidden_pre_{}'.format(ix+1),
            nonlinearity=hidden_f,
            W=GlorotUniform())
        if dropout > 0:
            l_dropout = DropoutLayer(
                l_hidden, p=dropout, name='dropout_pre_{}'.format(ix+1))
            last = l_dropout
        else:
            last = l_hidden

    if bottleneck_size > 0:
        last = DenseLayer(
            last, num_units=bottleneck_size,
            name='bottleneck',
            nonlinearity=bottleneck_f,
            W=GlorotUniform())

    for ix, size in enumerate(hidden_post):
        l_hidden = DenseLayer(
            last, num_units=size,
            name='hidden_post_{}'.format(ix+1),
            nonlinearity=hidden_f,
            W=GlorotUniform())
        if dropout > 0:
            last = DropoutLayer(
                l_hidden, p=dropout, name='dropout_post_{}'.format(ix+1))
        else:
            last = l_hidden

    l_out = DenseLayer(
        last, num_units=output_dim,
        nonlinearity=output_f, name='output',
        W=GlorotUniform())

    if not weights is None:
        set_all_param_values(l_out, weights)

    return l_out
