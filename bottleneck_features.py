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

Generate datasets containing bnf instead of MFCCs for all
experimental conditions, both for training and test phase
"""
from __future__ import division
import numpy as np
import glob
import os
import os.path as path
import theano

from lasagne.layers import get_output, get_all_layers, get_output_shape

from dnn import load_model
from util import verb_print

VERBOSE = True


def get_bottleneck_features(network, X):
    """Assume X is of shape (N, 1000, 39)
    """
    X = X.astype(theano.config.floatX)
    n_conditions, batch_size, n_features_in = X.shape

    layers = get_all_layers(network)
    bottleneck = [l for l in layers if l.name == 'bottleneck']
    if len(bottleneck) == 0:
        raise ValueError('network has no bottleneck')
    else:
        bottleneck = bottleneck[0]
    n_features_out = get_output_shape(bottleneck)[1]
    X_out = np.zeros(
        (n_conditions, batch_size, n_features_out),
        dtype=theano.config.floatX
    )

    for i in xrange(n_conditions):
        X_out[i, :, :] = get_output(bottleneck, X[i, :, :]).eval()
    return X_out


if __name__ == '__main__':
    # fmodel1 = '/Users/Research/projects/phonrulemodel/model_69_1.pkl'
    # fmodel2 = '/Users/Research/projects/phonrulemodel/model_69_2.pkl'
    # dir_trainsets = '/Users/Research/projects/phonrulemodel/trainsets'
    # dir_testsets = '/Users/Research/projects/phonrulemodel/testsets'
    # output_dir_train = '/Users/Research/projects/phonrulemodel/bnftrainsets/'
    # output_dir_test = '/Users/Research/projects/phonrulemodel/bnftestsets/'

    datadir = '/Users/mwv/data/'
    fmodel1 = path.join(
        datadir, 'phonrulemodel',
        'acoustic_models_deltas_grid_1', 'model_69.pkl'
    )
    fmodel2 = path.join(
        datadir, 'phonrulemodel',
        'acoustic_models_deltas_grid_2', 'model_69.pkl'
    )
    dir_trainsets = path.join(
        datadir, 'ingeborg_datasets', 'datasets_new', 'trainsets'
    )
    dir_testsets = path.join(
        datadir, 'ingeborg_datasets', 'datasets_new', 'testsets'
    )

    output_dir_train = path.join(
        datadir, 'ingeborg_datasets', 'bnf', 'bnftrainsets'
    )
    output_dir_test = path.join(
        datadir, 'ingeborg_datasets', 'bnf', 'bnftestsets'
    )
    try:
        os.makedirs(output_dir_train)
    except OSError:
        pass
    try:
        os.makedirs(output_dir_test)
    except OSError:
        pass

    # Replace MFCCs in training datasets by bnf's
    # 1. TRAINING
    with verb_print('loading networks', VERBOSE):
        model1 = load_model(fmodel1, deterministic=True, batch_size=1000)
        model2 = load_model(fmodel2, deterministic=True, batch_size=1000)

    print 'TRAINING'
    conditions = glob.iglob(path.join(dir_trainsets, '*.npz'))
    for condition_ix, condition in enumerate(conditions):
        dataset = np.load(condition)
        X, y = dataset['X'], dataset['y']
        for model_ix, model in enumerate([model1, model2]):
            cname = path.splitext(path.basename(condition))[0]
            with verb_print(' computing condition {}, model {}'.format(
                    condition_ix, model_ix), VERBOSE):
                X_bnf = get_bottleneck_features(model, X)
                y_bnf = get_bottleneck_features(model, y)
            with verb_print(' saving condition {}, model {}'.format(
                    condition_ix, model_ix), VERBOSE):
                output_file = path.join(
                    output_dir_train,
                    'train_condition{}model{}'.format(
                        condition_ix+1,
                        model_ix+1
                    )
                )
                np.savez(
                    output_file,
                    X=X_bnf,
                    y=y_bnf,
                    labels=dataset['labels'],
                    info=dataset['info']
                )

    # 2. TESTING
    print 'TESTING'
    conditions = glob.iglob(path.join(dir_testsets, '*.npz'))
    for condition_ix, condition in enumerate(conditions):
        dataset = np.load(condition)
        X1, y1, X2, y2 = dataset['X1'], dataset['y1'], dataset['X2'], \
            dataset['y2']
        for model_ix, model in enumerate([model1, model2]):
            cname = path.splitext(path.basename(condition))[0]
            with verb_print(' computing condition {}, model {}'.format(
                    condition_ix, model_ix), VERBOSE):
                X1_bnf = get_bottleneck_features(model, X1)
                y1_bnf = get_bottleneck_features(model, y1)
                X2_bnf = get_bottleneck_features(model, X2)
                y2_bnf = get_bottleneck_features(model, y2)
            with verb_print(' saving condition {}, model {}'.format(
                    condition_ix, model_ix), VERBOSE):
                output_file = path.join(
                    output_dir_train,
                    'test_condition{}model{}'.format(
                        condition_ix+1, model_ix+1
                    )
                )
                np.savez(
                    output_file,
                    X1=X1_bnf,
                    y1=y1_bnf,
                    X2=X2_bnf,
                    y2=y2_bnf,
                    labels=dataset['labels'],
                    info=dataset['info']
                )
