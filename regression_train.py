#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: regression_bnf.py
# date: Sun July 19 19:27 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""regression_bnf: train dnn regression

"""

from __future__ import division

import os
import os.path as path
import itertools
from collections import OrderedDict
import time

import numpy as np

from sklearn.preprocessing import MinMaxScaler
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import get_all_param_values, set_all_param_values

from dnn import build_model, save_model
from util import verb_print, ProgressPrinter, save_history


def float32(x):
    return np.cast['float32'](x)


def load_data(fname, valid_prop=1/10):
    """Load data set
    """

    f = np.load(fname)

    X, Y = f['X'], f['Y']

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    X = X.astype(theano.config.floatX)
    Y = MinMaxScaler(feature_range=(0, 1)).fit_transform(Y)
    Y = Y.astype(theano.config.floatX)

    X_train, X_valid = train_valid_split(X, valid_prop)
    Y_train, Y_valid = train_valid_split(Y, valid_prop)

    return dict(
        X_train=theano.shared(X_train),
        Y_train=theano.shared(Y_train),
        X_valid=theano.shared(X_valid),
        Y_valid=theano.shared(Y_valid),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        input_dim=X.shape[1],
        output_dim=Y.shape[1]
    )


def train_valid_split(X, valid_prop):
    """split dataset into training and validation parts.

    For the splitting, we're making use of the ordering of the data. If there
    are 128 stimuli and N total samples, the data is laid out as follows.
    The stimuli consist of contiguous blocks of BLKSZ = N/128 samples.
    stim0 = X[0:BLKSZ]
    stim1 = X[BLKSZ: BLKSZ*2]
    ...
    We split into training and test sets by iterating over the blocks
    and picking a proportion for the validation set and its complement
    for the training set.
    """
    nstim = 128  # number of transition stimuli in a dataset
    nsamples, nfeatures = X.shape
    blksz = nsamples // nstim
    blksz_train = (1-valid_prop) * (nsamples // nstim)
    blksz_valid = valid_prop * (nsamples // nstim)
    X_train = np.zeros((blksz_train*nstim, nfeatures), dtype=X.dtype)
    X_valid = np.zeros((blksz_valid*nstim, nfeatures), dtype=X.dtype)
    for blkix in xrange(nstim):
        X_blk = X[blkix*blksz: (blkix+1)*blksz, :]
        X_train[blkix*blksz_train: (blkix+1)*blksz_train, :] = \
            X_blk[:blksz_train]
        X_valid[blkix*blksz_valid: (blkix+1)*blksz_valid, :] = \
            X_blk[blksz_valid:]
    return X_train, X_valid


def create_iter_funcs(dataset, output_layer,
                      tensor_type=T.matrix,
                      batch_size=32000,
                      update='nesterov',
                      learning_rate=0.01,
                      momentum=0.9):
    """Create functions for training and validation to iterate one
       epoch.
    """
    batch_index = T.iscalar('batch_index')
    X_batch = tensor_type('x')
    Y_batch = tensor_type('y')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(
        output_layer,
        loss_function=lasagne.objectives.mse
    )
    loss_train = objective.get_loss(
        X_batch, target=Y_batch
    )
    loss_valid = objective.get_loss(
        X_batch, target=Y_batch, deterministic=True
    )

    all_params = lasagne.layers.get_all_params(output_layer)
    if update == 'sgd':
        updates = lasagne.updates.sgd(
            loss_or_grads=loss_train,
            params=all_params,
            learning_rate=learning_rate
        )
    else:
        updates = lasagne.updates.nesterov_momentum(
            loss_or_grads=loss_train,
            params=all_params,
            learning_rate=learning_rate,
            momentum=momentum
        )

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            Y_batch: dataset['Y_train'][batch_slice],
        }
    )
    iter_valid = theano.function(
        [batch_index], loss_valid,
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            Y_batch: dataset['Y_valid'][batch_slice]
        }
    )
    return dict(
        train=iter_train,
        valid=iter_valid
    )


def train(iter_funcs, dataset, batch_size=32000):
    """Train the model with mini-batch.
    """
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = [
            iter_funcs['train'](batch_ix)
            for batch_ix in range(num_batches_train)
        ]
        batch_valid_losses = [
            iter_funcs['valid'](batch_ix)
            for batch_ix in range(num_batches_valid)
        ]
        yield dict(
            number=epoch,
            train_loss=np.mean(batch_train_losses),
            valid_loss=np.mean(batch_valid_losses)
        )


def train_loop(output_layer,
               iter_funcs,
               dataset,
               batch_size,
               max_epochs,
               patience=1000,
               learning_rate_start=0.01,
               learning_rate_stop=float32(0.001),
               momentum_start=float32(0.9),
               momentum_stop=float32(0.999),
               verbose=True):
    best_valid_loss = np.inf
    best_valid_epoch = 0
    best_train_loss = np.inf
    best_weights = None
    learning_rates = np.logspace(
        np.log10(learning_rate_start),
        np.log10(learning_rate_stop),
        max_epochs
    )
    momentums = np.linspace(
        momentum_start,
        momentum_stop,
        max_epochs
    )

    now = time.time()
    history = []
    if verbose:
        printer = ProgressPrinter(color=True)
    try:
        for epoch in train(iter_funcs, dataset, batch_size):
            epoch_number = epoch['number']
            train_loss = epoch['train_loss']
            valid_loss = epoch['valid_loss']
            info = OrderedDict([
                ('epoch', epoch_number),
                ('train_loss', train_loss),
                ('train_loss_best', train_loss <= best_train_loss),
                ('train_loss_worse', train_loss > history[-1]['train_loss']
                 if len(history) > 0 else False),
                ('valid_loss', valid_loss),
                ('valid_loss_best', valid_loss <= best_valid_loss),
                ('valid_loss_worse', valid_loss > history[-1]['valid_loss']
                 if len(history) > 0 else False),
                ('duration', time.time() - now)])
            history.append(info)
            now = time.time()
            if verbose:
                printer(history)

            # early stopping
            if epoch['valid_loss'] < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_epoch = epoch_number
                best_weights = get_all_param_values(output_layer)
            elif epoch['number'] >= max_epochs:
                break
            elif best_valid_epoch + patience < epoch_number:
                if verbose:
                    print("  stopping early")
                    print("  best validation loss was {:.6f} at epoch {}."
                          .format(best_valid_loss, best_valid_epoch))
                break
            if epoch['number'] >= max_epochs:
                if verbose:
                    print('  last epoch')
                    print('  best validation loss was {:.6f} at epoch {}.'
                          .format(best_valid_loss, best_valid_epoch))
                break

            # adjust learning rate and momentum
            new_learning_rate = float32(learning_rates[epoch_number-1])
            learning_rate_start.set_value(new_learning_rate)
            new_momentum = float32(momentums[epoch_number-1])
            momentum_start.set_value(new_momentum)
    except KeyboardInterrupt:
        pass
    return best_valid_loss, best_valid_epoch, best_weights, history


def main(dataset,
         batch_size=32000,
         hidden_pre=(1000, 1000, 1000),
         dropout=0.5,
         hidden_f='rectify',
         bottleneck_size=0,
         bottleneck_f='linear',
         hidden_post=(),
         output_f='linear',

         max_epochs=100,
         patience=100,
         update='nesterov',
         learning_rate_start=0.01,
         learning_rate_stop=0.001,
         momentum_start=0.9,
         momentum_stop=0.999,
         verbose=True):
    """Build and train a network.

    Parameters
    ----------
    dataset : dict
        as output by load_data
    layers : sequence of ints
        hidden layers
    dropout : float
        proportion of dropout

    #TODO finish documenting this function
    """
    learning_rate_start = float32(learning_rate_start)
    learning_rate_stop = float32(learning_rate_stop)
    momentum_start = float32(momentum_start)
    momentum_stop = float32(momentum_stop)

    output_layer = build_model(
        batch_size=batch_size,
        input_dim=dataset['input_dim'],
        output_dim=dataset['output_dim'],
        hidden_pre=hidden_pre,
        dropout=dropout,
        hidden_f=hidden_f,
        bottleneck_size=bottleneck_size,
        bottleneck_f=bottleneck_f,
        hidden_post=hidden_post,
        output_f=output_f)

    iter_funcs = create_iter_funcs(
        dataset, output_layer,
        batch_size=batch_size,
        update=update,
        learning_rate=learning_rate_start,
        momentum=momentum_start)

    loss, epoch, weights, history = train_loop(
        output_layer, iter_funcs, dataset, batch_size, max_epochs,
        patience, learning_rate_start, learning_rate_stop,
        momentum_start, momentum_stop)

    set_all_param_values(output_layer, weights)
    return loss, epoch, history, output_layer


if __name__ == '__main__':
    args = dict(
        dataset_file=[path.join(
            os.environ['HOME'], 'data', 'ingeborg_datasets',
            'datasets_regression_bnf', 'train',
            'MH-ADS-A_P-IDS-A_exposure.npz'
        )],
        output_file=[path.join(
            'regression_model', 'MH-ADS-A_P-IDS-A_exposure'
        )],
        verbose=True,
    )

    dataset_file = args['dataset_file'][0]
    output_file = args['output_file'][0]

    verbose = args['verbose']

    with verb_print('loading data', verbose):
        dataset = load_data(dataset_file, valid_prop=0.1)

    config = dict(
        # data parameters
        batch_size=32000,

        # network parameters
        hidden_pre=[1000, 1000, 1000],
        dropout=0.5,
        hidden_f='rectify',
        bottleneck_size=0,
        bottleneck_f='linear',
        hidden_post=[],
        output_f='linear',

        # training parameters
        max_epochs=100000,
        patience=1000,
        update='nesterov',
        learning_rate_start=0.01,
        learning_rate_stop=0.001,
        momentum_start=0.9,
        momentum_stop=0.999,
    )

    loss, epoch, history, output_layer = main(dataset, verbose=verbose,
                                              **config)

    save_model(output_layer, config, output_file + '.pkl')
    save_history(history, output_file + '.history')
