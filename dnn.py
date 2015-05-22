#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: dnn.py
# date: Mon May 18 21:13 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
# 
# Licensed under GPLv3
# ------------------------------------
"""dnn:

"""

from __future__ import division

import itertools
import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import theano  
import theano.tensor as T
import lasagne 
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer
from lasagne.nonlinearities import rectify, softmax, sigmoid

def float32(x):
    return np.cast['float32'](x)

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

def load_data(fname, test_prop=1/16, valid_prop=5/16, register='both',testsubset = False):
    """
    If testsubset = True, load_data returns only a small dataset so code can be tested without GPU
    """
    f = np.load(fname)
    X, y, labels = f['X'], f['y'], f['labels']
    if register in ['IDS', 'ADS']:
        sel_ixs = np.in1d(y, np.nonzero(labels[:, 1]==register))
        X = X[sel_ixs]
        y = y[sel_ixs]
    elif register == 'both':
        ix2phone = dict(enumerate(labels[:, 0]))
        phones = sorted(set(ix2phone.values()))
        phone2newix = {p:ix for ix, p in enumerate(phones)}
        y = np.array([phone2newix[ix2phone[i]] for i in y])
        # phone2ix = {k: ix for ix, k in enumerate(labels[:, 0])}
    else:
        raise ValueError('invalid option for register: {0}'.format(register))

    oldix2newix = {old_ix:new_ix for new_ix, old_ix in enumerate(np.unique(y))}
    y = np.array([oldix2newix[i] for i in y])

    print 'number of labels: {0}'.format(len(np.unique(y)))

    X = StandardScaler().fit_transform(X)
    X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
    print X.min(), X.max()
    X = X.astype(theano.config.floatX)
    y = y.astype('int32')
    nclasses = np.unique(y).shape[0]
    nfeatures = X.shape[1]

    if testsubset:
        X = X[1:17] 
        y = y[1:17]

    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        train_valid_test_split(X, y,
                               test_prop=test_prop, valid_prop=valid_prop)
    print X_train.shape, y_train.shape
    print X_valid.shape, y_valid.shape
    print X_test.shape, y_test.shape

    return dict(
        X_train=theano.shared(X_train),
        y_train=theano.shared(y_train),
        X_valid=theano.shared(X_valid),
        y_valid=theano.shared(y_valid),
        X_test=theano.shared(X_test),
        y_test=theano.shared(y_test),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=nfeatures,
        output_dim=nclasses,
        labels=labels
    )


def build_model(input_dim, output_dim,
                hidden_layers=(100,100),
                batch_size=100, dropout=False, bottleneck = True, bsize = 50):
    """
    If bottleneck = True, a bottleneck hiddenlayer of with bsize nodes is added
    """
    l_in = InputLayer(shape=(batch_size, input_dim))
    last = l_in
    for size in hidden_layers:
        l_hidden = DenseLayer(last, num_units=size,
                              # nonlinearity=T.nnet.hard_sigmoid,
                              nonlinearity=lasagne.nonlinearities.leaky_rectify,
                              W=lasagne.init.GlorotUniform())
        if dropout:
            l_dropout = DropoutLayer(l_hidden, p=0.5)
            last = l_dropout
        else:
            last = l_hidden
    
    if bottleneck:
        l_bottleneck = DenseLayer(last, num_units=bsize,
                              # nonlinearity=T.nnet.hard_sigmoid,
                              nonlinearity=lasagne.nonlinearities.leaky_rectify,
                              W=lasagne.init.GlorotUniform())
        last = l_bottleneck

    l_out = DenseLayer(last, num_units=output_dim, nonlinearity=softmax,
                       W=lasagne.init.GlorotUniform())
    return dict(
        l_out = l_out,
        last = last
    )

def create_iter_func(dataset, dataset_all, output_layer,
                     X_tensor_type=T.matrix,
                     batch_size=300,
                     learning_rate=0.01,
                     momentum=0.9):
    """Create functions for training, validation and testing to iterate one
       epoch.
    """
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(output_layer,
        loss_function=lasagne.objectives.categorical_crossentropy)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch,
                                   deterministic=True)

    pred = T.argmax(
        output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(output_layer)
    # updates = lasagne.updates.adadelta(
    #     loss_or_grads=loss_train,
    #     params=all_params,
    #     learning_rate=1.0,
    #     rho=0.95,
    #     epsilon=1e-6
    #     )
    updates = lasagne.updates.sgd(
        loss_or_grads=loss_train,
        params=all_params,
        learning_rate=learning_rate)
    # updates = lasagne.updates.nesterov_momentum(
    #     loss_or_grads=loss_train,
    #     params=all_params,
    #     learning_rate=learning_rate, momentum=momentum)


    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
        },
    )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
        },
    )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
        },
    )

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test
    )

def train(iter_funcs, dataset, batch_size=300, test_every=100):
    """Train the model with `dataset` with mini-batch training. Each
       mini-batch has `batch_size` recordings.
    """
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size
    num_batches_test = dataset['num_examples_test'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        if epoch % test_every == 0:
            batch_test_accuracies = []
            for b in xrange(num_batches_test):
                _, batch_test_accuracy = iter_funcs['test'](b)
                batch_test_accuracies.append(batch_test_accuracy)
            avg_test_accuracy = np.mean(batch_test_accuracies)
        else:
            avg_test_accuracy = np.nan

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
            'test_accuracy': avg_test_accuracy
        }


def load_all_data(fname, register='both',testsubset = False):
    """ 
        Creates dataset for generating new phone representations, without seperating into
        different training, validation and testing tests
        epoch.
        If testsubset = True, load_data returns only a small dataset so code can be tested without GPU
    """
    f = np.load(fname)
    X, y, labels = f['X'], f['y'], f['labels']
    if register in ['IDS', 'ADS']:
        sel_ixs = np.in1d(y, np.nonzero(labels[:, 1]==register))
        X = X[sel_ixs]
        y = y[sel_ixs]
    elif register == 'both':
        ix2phone = dict(enumerate(labels[:, 0]))
        phones = sorted(set(ix2phone.values()))
        phone2newix = {p:ix for ix, p in enumerate(phones)}
        y = np.array([phone2newix[ix2phone[i]] for i in y])
        # phone2ix = {k: ix for ix, k in enumerate(labels[:, 0])}
    else:
        raise ValueError('invalid option for register: {0}'.format(register))

    oldix2newix = {old_ix:new_ix for new_ix, old_ix in enumerate(np.unique(y))}
    y = np.array([oldix2newix[i] for i in y])

    print 'number of labels: {0}'.format(len(np.unique(y)))

    X = StandardScaler().fit_transform(X)
    X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
    print X.min(), X.max()
    X = X.astype(theano.config.floatX)
    y = y.astype('int32')
    nclasses = np.unique(y).shape[0]
    nfeatures = X.shape[1]
    
    if testsubset:
        X = X[1:17] 
        y = y[1:17]

    print X.shape, y.shape
   
    return dict(
        X=theano.shared(X),
        y=theano.shared(y),
        num_examples=X.shape[0],
        input_dim=nfeatures,
        output_dim=nclasses,
        labels=labels
    )

def get_new_representations(iter_funcs, dataset_all, last):
    """Run `dataset_all` through the model and save the second-to-last layer as
       new phone representations
    """ 
    output = lasagne.layers.get_output(last, dataset_all['X']).eval()
    labels = dataset_all['labels']
    #print output
    #print labels
    return dict(
        X = output,
        y = labels   
    )

if __name__ == '__main__':
    num_epochs=10 #return back to 1000
    batch_size=1 #return back to 1000
    dataset = load_data('/Users/ingeborg/Desktop/mfcc.npz',
                        valid_prop=4/16, test_prop=2/16, register='IDS',testsubset = True)
    dataset_all = load_all_data('/Users/ingeborg/Desktop/mfcc.npz', register='IDS',testsubset = True)
    output_layer = build_model(
        input_dim=dataset['input_dim'], output_dim=dataset['output_dim'],
        batch_size=batch_size, bottleneck = True, bsize = 50)
    iter_funcs = create_iter_func(dataset, dataset_all, output_layer['l_out'],
                                  batch_size=batch_size,
                                  learning_rate=0.1, momentum=0.9)
    test_every = 100
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset,
                           batch_size=batch_size, test_every=test_every):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                epoch['valid_accuracy'] * 100))
            if epoch['number'] % test_every == 0:
                print("  TEST ACCURACY:\t\t{:.2f} %%".format(
                    epoch['test_accuracy'] * 100))

            if epoch['number'] >= num_epochs:
                break
    except KeyboardInterrupt:
        pass

    representations = get_new_representations(iter_funcs,dataset_all,output_layer['last'])
    filename = '/Users/ingeborg/Desktop/reprs.npz'
    np.savez(filename, X=representations['X'], y=representations['y'])
        #scrhijf naar .npz file