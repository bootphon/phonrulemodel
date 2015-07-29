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

import os.path as path
import pickle
import glob
from collections import namedtuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import theano
from lasagne.layers import get_output, get_all_layers, get_output_shape
import pandas as pd

from dnn import load_model


FilePair = namedtuple('FilePair', ['model', 'data'])


def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean(axis=1)


def gather_files(model_dir, data_dir):
    model_files = {
        path.splitext(path.basename(fname))[0].replace('_exposure', ''):
        fname
        for fname in glob.iglob(path.join(model_dir, '*.pkl'))
    }

    data_files = {
        path.splitext(path.basename(fname))[0].replace('_test', ''):
        fname
        for fname in glob.iglob(path.join(data_dir, '*.npz'))
    }
    assert (set(model_files.keys()) == set(data_files.keys()))
    files = {bname: FilePair(model_files[bname], data_files[bname])
             for bname in model_files}
    return files


def load_data(fname):
    f = np.load(fname)
    X, Y = f['X'], f['Y']

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    X = X.astype(theano.config.floatX)
    Y = MinMaxScaler(feature_range=(0, 1)).fit_transform(Y)
    Y = Y.astype(theano.config.floatX)

    return X, Y, f['legend']


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
            'model_dir', metavar='MODELDIR',
            nargs=1,
            help='directory with trained network files'
        )
        parser.add_argument(
            'data_dir', metavar='DATADIR',
            nargs=1,
            help='directory with data files'
        )
        parser.add_argument(
            'output_dir', metavar='OUTPUTDIR',
            nargs=1,
            help='output directory'
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

    model_dir = args['model_dir'][0]
    data_dir = args['data_dir'][0]
    output_dir = args['output_dir'][0]
    batch_size = int(args['batch_size'])
    verbose = args['verbose']

    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    file_pairs = gather_files(model_dir, data_dir)
    for bname, (model_file, data_file) in file_pairs.iteritems():
        X, Y_true, legend = load_data(data_file)
        model = load_model(
            model_file,
            deterministic=True,
            batch_size=batch_size
        )
        Y_pred = get_activations(model, X)
        error = mse(
            Y_true, Y_pred
        )
        df = pd.DataFrame(
            legend,
            columns=['phone1', 'phone2',
                     'stimulusID', 'register',
                     'congruency']
        )
        df['error'] = error
        df.to_csv(path.join(output_dir, bname + '.csv'), index=False)

        with open(path.join(output_dir, bname + '.pkl'), 'wb') as fout:
            pickle.dump(dict(
                Y_true=Y_true,
                Y_pred=Y_pred,
                error=error,
                legend=legend
            ))
