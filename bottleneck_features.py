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
    """Run data X through network up to the layer that is called "bottleneck".
    """
    X = X.astype(theano.config.floatX)
    n_samples, n_features_in = X.shape

    layers = get_all_layers(network)
    if layers[0].shape[1] != n_features_in:
        raise ValueError(
            'expected {} features on network input, got {}'
            .format(n_features_in, layers[0].shape[1])
        )
    bottleneck = [l for l in layers if l.name == 'bottleneck']
    if len(bottleneck) == 0:
        raise ValueError('network has no bottleneck')
    else:
        bottleneck = bottleneck[0]
    batch_size, n_features_out = get_output_shape(bottleneck)

    slices = [slice(batch_ix * batch_size, (batch_ix+1) * batch_size)
              for batch_ix in xrange(n_samples // batch_size)]
    return np.vstack(
        (get_output(bottleneck, X[sl]).eval()
         for sl in slices)
    )


if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(
            prog='bottleneck_features.py',
            description='convert mfcc features to bottleneck features'
        )
        parser.add_argument(
            'model', metavar='MODEL',
            nargs=1,
            help='/path/to/model_file'
        )
        parser.add_argument(
            'datapath', metavar='DATAPATH',
            nargs=1,
            help='/path/to/datapath')
        parser.add_argument(
            'outputpath', metavar='OUTPUTPATH',
            nargs=1,
            help='/path/to/outputpath'
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
            dest='verbose',
            help='talk more'
        )
        return vars(parser.parse_args())

    args = parse_args()

    model_fname = args['model'][0]
    data_path = args['datapath'][0]
    out_path = args['outputpath'][0]
    batch_size = int(args['batch_size'])
    verbose = args['verbose']

    model = load_model(
        model_fname,
        deterministic=True,
        batch_size=batch_size
    )

    try:
        os.makedirs(out_path)
    except OSError:
        pass

    # train sets
    if verbose:
        print 'TRAIN SETS'
    train_path_in = path.join(data_path, 'train')
    train_path_out = path.join(out_path, 'train')
    try:
        os.makedirs(train_path_out)
    except OSError:
        pass
    for infile in glob.iglob(path.join(train_path_in, '*.npz')):
        bname = path.splitext(path.basename(infile))[0]
        with verb_print('loading data for {}'.format(bname), verbose):
            data = np.load(infile)
        with verb_print('computing bnf for {}'.format(bname), verbose):
            X, Y = data['X'], data['Y']
            X_bnf = get_bottleneck_features(model, X)
            Y_bnf = get_bottleneck_features(model, Y)
        with verb_print('saving bnf for {}'.format(bname), verbose):
            np.savez(
                path.join(train_path_out, bname + '.npz'),
                X=X_bnf,
                Y=Y_bnf
            )

    # test sets
    if verbose:
        print 'TEST SETS'
    test_path_in = path.join(data_path, 'test')
    test_path_out = path.join(out_path, 'test')
    try:
        os.makedirs(test_path_out)
    except OSError:
        pass
    for infile in glob.iglob(path.join(test_path_in, '*.npz')):
        bname = path.splitext(path.basename(infile))[0]
        with verb_print('loading data for {}'.format(bname), verbose):
            data = np.load(infile)
        with verb_print('computing bnf for {}'.format(bname), verbose):
            X, Y = data['X'], data['Y']
            X_bnf = get_bottleneck_features(model, X)
            Y_bnf = get_bottleneck_features(model, Y)
        with verb_print('saving bnf for {}'.format(bname), verbose):
            np.savez(
                path.join(test_path_out, bname + '.npz'),
                X=X_bnf,
                Y=Y_bnf
            )
