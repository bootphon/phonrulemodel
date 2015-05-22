#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: regression.py
# date: Wed May 13 14:32 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""regression:

"""

from __future__ import division

from itertools import count
import time

import numpy as np

from sklearn.datasets import make_regression, make_friedman3, load_boston
from sklearn.metrics import r2_score, explained_variance_score, \
    mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer

import theano
import theano.tensor as T


def train_valid_test_split(X, y, test_prop=0.1, valid_prop=0.2):
    nsamples = X.shape[0]
    ixs = np.random.permutation(nsamples)
    X = np.copy(X)
    X = X[ixs]
    y = np.copy(y)
    y = y[ixs]
    valid_cut = int(valid_prop*nsamples)
    test_cut = int(test_prop*nsamples) + valid_cut
    X_valid, y_valid = X[:valid_cut], y[:valid_cut]
    X_test, y_test = X[valid_cut:test_cut], y[valid_cut:test_cut]
    X_train, y_train = X[test_cut:], y[test_cut:]
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def make_data(n_samples=1000, n_features=1, n_targets=1, informative_prop=1.0,
              noise=0.0, test_prop=0.1, valid_prop=0.3, method='linear'):
    if method == 'linear':
        params = dict(n_features=n_features,
                      n_informative=int(n_features*informative_prop),
                      noise=noise,
                      n_targets=n_targets,
                      n_samples=n_samples,
                      shuffle=False,
                      bias=0.0)
        X, Y = make_regression(**params)
    elif method == 'boston':
        boston = load_boston()
        X = boston.data
        Y = boston.target
    else:
        params = dict(n_samples=n_samples,
                      n_features=n_features)
        X, Y = make_friedman3(n_samples=n_samples, n_features=n_features,
                                 noise=noise)

    X = MinMaxScaler(feature_range=(0.0,1.0)).fit_transform(X)
    X = X.astype(theano.config.floatX)
    Y = MinMaxScaler(feature_range=(0.0,1.0)).fit_transform(Y)
    Y = Y.astype(theano.config.floatX)
    if len(X.shape) > 1:
        n_features = X.shape[1]
    else:
        X = X.reshape(X.shape[0], -1)
        n_features = 1
    if len(Y.shape) > 1:
        n_targets = Y.shape[1]
    else:
        Y = Y.reshape(Y.shape[0], -1)
        n_targets = 1

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = \
        train_valid_test_split(X, Y,
                               test_prop=valid_prop, valid_prop=valid_prop)
    return dict(
        X_train=theano.shared(X_train),
        Y_train=theano.shared(Y_train),
        X_valid=theano.shared(X_valid),
        Y_valid=theano.shared(Y_valid),
        X_test=theano.shared(X_test),
        Y_test=theano.shared(Y_test),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=n_features,
        output_dim=n_targets)


def build_model(input_dim, output_dim,
                hidden_layers=(100, 100, 100),
                batch_size=100, dropout=True):
    l_in = InputLayer(shape=(batch_size, input_dim))
    last = l_in
    for size in hidden_layers[:-1]:
        l_hidden = DenseLayer(last, num_units=size,
                              nonlinearity=lasagne.nonlinearities.leaky_rectify,
                              W=lasagne.init.GlorotUniform())
        if dropout:
            l_dropout = DropoutLayer(l_hidden, p=0.5)
            last = l_dropout
        else:
            last = l_hidden
    l_penult = DenseLayer(last, num_units=hidden_layers[-1],
                          nonlinearity=lasagne.nonlinearities.leaky_rectify,
                          W=lasagne.init.GlorotUniform())
    l_out = DenseLayer(l_penult, num_units=output_dim,
                       nonlinearity=lasagne.nonlinearities.linear)
    return l_out


def create_iter_funcs(dataset, output_layer,
                      tensor_type=T.matrix,
                      batch_size=300,
                      learning_rate=0.01,
                      momentum=0.9):
    batch_index = T.iscalar('batch_index')
    X_batch = tensor_type('x')
    Y_batch = tensor_type('y')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(output_layer,
        loss_function=lasagne.objectives.mse)
    loss_train = objective.get_loss(X_batch, target=Y_batch)
    loss_eval = objective.get_loss(X_batch, target=Y_batch,
                                   deterministic=True)

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.sgd(
        loss_or_grads=loss_train,
        params=all_params,
        learning_rate=learning_rate)

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            Y_batch: dataset['Y_train'][batch_slice],
        }
    )

    iter_valid = theano.function(
        [batch_index], loss_eval,
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            Y_batch: dataset['Y_valid'][batch_slice],
        }
    )

    iter_test = theano.function(
        [batch_index], loss_eval,
        givens={
            X_batch: dataset['X_test'][batch_slice],
            Y_batch: dataset['Y_test'][batch_slice],
        }
    )

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test)

def train(iter_funcs, dataset, batch_size=300, test_every=100):
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size
    num_batches_test = dataset['num_examples_test'] // batch_size

    for epoch in count(1):
        batch_train_losses = []
        for b in xrange(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        for b in xrange(num_batches_valid):
            batch_valid_loss = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)

        avg_valid_loss = np.mean(batch_valid_losses)

        if epoch % test_every == 0:
            batch_test_losses = []
            for b in xrange(num_batches_test):
                batch_test_loss = iter_funcs['test'](b)
                batch_test_losses.append(batch_test_loss)
            avg_test_accuracy = 1 - np.mean(batch_test_losses)
        else:
            avg_test_accuracy = np.nan

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'test_accuracy': avg_test_accuracy
        }

if __name__ == '__main__':
    num_epochs = 100
    batch_size = 1000
    n_samples = 100000
    dataset = make_data(n_samples=n_samples, n_features=10, n_targets=10,
                        method='linear')
    output_layer = build_model(
        input_dim=dataset['input_dim'], output_dim=dataset['output_dim'],
        batch_size=batch_size)
    iter_funcs = create_iter_funcs(dataset, output_layer,
                                   batch_size=batch_size,
                                   learning_rate=0.1, momentum=0.9)
    test_every = 100
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset,
                           batch_size=batch_size, test_every=test_every):
            print('Epoch {} of {} took {:.3f}s'.format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            if epoch['number'] % test_every == 0:
                print('  TEST ACCURACY:\t\t{:.3f} %%'.format(
                    epoch['test_accuracy']*100))
            if epoch['number'] >= num_epochs:
                break
    except KeyboardInterrupt:
        pass
    X_test = dataset['X_test'].get_value()
    slices = [slice(batch_index*batch_size, (batch_index+1)*batch_size)
              for batch_index in xrange(X_test.shape[0] // batch_size)]
    Y_pred = np.vstack((output_layer.get_output(X_test[sl]).eval()
                        for sl in slices))
    # print('Y_pred')
    # print(Y_pred)
    # print(Y_pred.shape)
    Y_test = dataset['Y_test'].get_value()
    # print('Y_test')
    # print(Y_test)
    # print(Y_test.shape)

    # expl_var = explained_variance_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    # print( 'Explained variance: {0:.3f}'.format(expl_var))
    print( 'Mean squared error: {0:.3f}'.format(mse))
    print( 'R^2 score:          {0:.3f}'.format(r2))



# Y_pred = net.predict(X_test)
# expl_var = explained_variance_score(Y_test, Y_pred)
# mse = mean_squared_error(Y_test, Y_pred)
# r2 = r2_score(Y_test, Y_pred)

# print 'Timing:'
# print 'Training time: {0:.3f}s'.format(train_time)
# print 'Test time:     {0:.3f}s'.format(test_time)
# print '---'

# print 'Test performance:'
