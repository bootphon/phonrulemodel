#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: regression_predict.py
# date: Mon July 20 04:03 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""regression_predict:

"""

from __future__ import division


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import theano
from lasagne.layers import get_output, get_all_layers, get_output_shape
from sklearn.metrics import mean_squared_error
import pickle

from dnn import load_model


def load_data(fname):
    f = np.load(fname)
    X, Y = f['X'], f['Y']

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    X = X.astype(theano.config.floatX)
    Y = MinMaxScaler(feature_range=(0, 1)).fit_transform(Y)
    Y = Y.astype(theano.config.floatX)

    return X, Y


def get_activations(network, X):
    X = X.astype(theano.config.floatX)
    n_samples, n_features_in = X.shape

    layers = get_all_layers(network)
    if layers[0].shape[1] != n_features_in:
        raise ValueError(
            'expected {} features on network input, got {}'
            .format(n_features_in, layers[0].shape[1])
        )
    batch_size, n_features_out = get_output_shape(network)
    slices = [slice(batch_ix * batch_size, (batch_ix+1) * batch_size)
              for batch_ix in xrange(n_samples // batch_size)]
    return np.vstack(
        (get_output(network, X[sl]).eval()
         for sl in slices)
    )

if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(
            prog='regression_predict.py',
        )
        parser.add_argument(
            'model_file', metavar='MODELFILE',
            nargs=1,
            help='trained network file'
        )
        parser.add_argument(
            'congruent_file', metavar='CONGRUENTFILE',
            nargs=1,
            help='dataset file with congruent data'
        )
        parser.add_argument(
            'incongruent_file', metavar='INCONGRUENTFILE',
            nargs=1,
            help='dataset file with incongruent data'
        )
        parser.add_argument(
            '-b', '--batch-size',
            action='store',
            dest='batch_size',
            default=1000,
            help='batch size'
        )
        parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            default=False,
            help='talk more'
        )
        return vars(parser.parse_args())

    args = parse_args()

    model_file = args['model_file'][0]
    congruent_file = args['congruent_file'][0]
    incongruent_file = args['incongruent_file'][0]
    output_file = args['output_file'][0]
    batch_size = int(args['batch_size'])

    model = load_model(model_file, deterministic=True, batch_size=batch_size)
    X_congruent, Y_congruent = load_data(congruent_file)
    X_incongruent, Y_incongruent = load_data(incongruent_file)

    y_congruent = get_activations(model, X_congruent)
    y_incongruent = get_activations(model, X_incongruent)

    err_congruent = mean_squared_error(
        Y_congruent, y_congruent, multioutput='raw_values'
    )
    err_incongruent = mean_squared_error(
        Y_incongruent, y_incongruent, multioutput='raw_values'
    )

    print 'mse congruent:   {:.5f} (std: {:.5f})'.format(
        err_congruent.mean(), err_congruent.std()
    )
    print 'mse incongruent: {:.5f} (std: {:.5f})'.format(
        err_incongruent.mean(), err_congruent.std()
    )

    with open(output_file, 'wb') as fout:
        pickle.dump(dict(
            Y_congruent=Y_congruent,
            Y_hat_congruent=y_congruent,
            Y_incongruent=Y_incongruent,
            Y_hat_incongruent=y_incongruent,
            err_congruent=err_congruent,
            err_incongruent=err_incongruent
        ))
