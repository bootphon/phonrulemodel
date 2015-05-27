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
"""train_acoustic model: train a neural net to predict phone classes from
mfcc tokens.

"""

from __future__ import division

import itertools
import time
import cPickle as pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, \
    get_output, get_all_param_values, set_all_param_values, get_all_params
from lasagne.nonlinearities import linear, leaky_rectify, softmax, \
    tanh, rectify

from util import verb_print, ProgressPrinter


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

def load_data(fname, test_prop=1/16, valid_prop=5/16, register='both',
              test=False):
    """
    If testsubset = True, load_data returns only a small dataset so code can
    be tested without GPU
    """
    f = np.load(fname)
    X, y, labels = f['X'], f['y'], f['labels']

    return build_dataset(X, y, labels, test_prop, valid_prop, register, test)

def build_dataset(X, y, labels, test_prop=0.2, valid_prop=0.2,
                  register='both', test=False):
    if register in ['IDS', 'ADS']:
        sel_ixs = np.in1d(y, np.nonzero(labels[:, 1]==register))
        X = X[sel_ixs]
        y = y[sel_ixs]
    elif register == 'both': # merge IDS and ADS labels per phone
        ix2phone = dict(enumerate(labels[:, 0]))
        phones = sorted(set(ix2phone.values()))
        phone2newix = {p:ix for ix, p in enumerate(phones)}
        y = np.array([phone2newix[ix2phone[i]] for i in y])
    else:
        raise ValueError('invalid option for register: {0}'.format(register))

    oldix2newix = {old_ix:new_ix for new_ix, old_ix in enumerate(np.unique(y))}
    y = np.array([oldix2newix[i] for i in y])

    # X = StandardScaler().fit_transform(X)
    X = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
    X = X.astype(theano.config.floatX)
    y = y.astype('int32')
    nclasses = np.unique(y).shape[0]
    nfeatures = X.shape[1]

    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        train_valid_test_split(X, y,
                               test_prop=test_prop, valid_prop=valid_prop)

    if test:
        X = X_train[100:200]
        y = y_train[100:200]
        X_train = X_train[:100]
        y_train = y_train[:100]
        X_valid = X_valid[:10]
        y_valid = y_valid[:10]
        X_test = X_test[:50]
        y_test = y_test[:50]

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
                hidden_layers=(1000, 1000),
                transfer_func='rectify',
                batch_size=100, dropout=0,
                nbottleneck=25,
                bottleneck_func='linear'):
    """
    If bottleneck = True, a bottleneck hiddenlayer of with bsize nodes is added
    """
    transfer_funcs = {'rectify': rectify,
                      'sigmoid': T.nnet.hard_sigmoid,
                      'tanh': tanh,
                      'leaky_rectify': leaky_rectify,
                      'linear': linear}
    transfer_func = transfer_funcs[transfer_func]
    bottleneck_func = transfer_funcs[bottleneck_func]
    l_in = InputLayer(shape=(batch_size, input_dim))
    last = l_in
    for ix, size in enumerate(hidden_layers[:-1]):
        l_hidden = DenseLayer(
            last, num_units=size,
            nonlinearity=transfer_func,
            W=lasagne.init.GlorotUniform())
        if dropout > 0:
            l_dropout = DropoutLayer(l_hidden, p=dropout)
            last = l_dropout
        else:
            last = l_hidden

    if nbottleneck > 0:
        l_bottleneck = DenseLayer(
            last, num_units=nbottleneck,
            name='bottleneck',
            nonlinearity=bottleneck_func,
            W=lasagne.init.GlorotUniform())
        last = l_bottleneck

    l_last_dense = DenseLayer(
        last, num_units=hidden_layers[-1],
        nonlinearity=transfer_func,
        W=lasagne.init.GlorotUniform())
    if dropout > 0:
        l_last_dense = DropoutLayer(l_last_dense, p=dropout)
    l_out = DenseLayer(l_last_dense, num_units=output_dim,
                       nonlinearity=softmax,
                       W=lasagne.init.GlorotUniform())
    return l_out


def create_iter_funcs(dataset, output_layer,
                      X_tensor_type=T.matrix,
                      batch_size=300,
                      update='nesterov',
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
        get_output(output_layer,
                   X_batch,
                   deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = get_all_params(output_layer)
    # updates = lasagne.updates.adadelta(
    #     loss_or_grads=loss_train,
    #     params=all_params,
    #     learning_rate=1.0,
    #     rho=0.95,
    #     epsilon=1e-6
    #     )
    if update == 'sgd':
        updates = lasagne.updates.sgd(
            loss_or_grads=loss_train,
            params=all_params,
            learning_rate=learning_rate)
    else:
        updates = lasagne.updates.nesterov_momentum(
            loss_or_grads=loss_train,
            params=all_params,
            learning_rate=learning_rate,
            momentum=momentum)


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
    # num_batches_test = dataset['num_examples_test'] // batch_size

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

        # if epoch % test_every == 0:
        #     batch_test_accuracies = []
        #     for b in xrange(num_batches_test):
        #         _, batch_test_accuracy = iter_funcs['test'](b)
        #         batch_test_accuracies.append(batch_test_accuracy)
        #     avg_test_accuracy = np.mean(batch_test_accuracies)
        # else:
        #     avg_test_accuracy = np.nan

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
            # 'test_accuracy': avg_test_accuracy
        }


def train_loop(output_layer, iter_funcs, dataset, batch_size, max_epochs,
               test_every=100, patience=100,
               learning_rate_start=theano.shared(float32(0.03)),
               learning_rate_stop=theano.shared(float32(0.001)),
               momentum_start=theano.shared(float32(0.9)),
               momentum_stop=theano.shared(float32(0.999)),
               verbose=True):
    best_valid_loss = np.inf
    best_valid_epoch = 0
    best_train_loss = np.inf
    best_weights = None
    learning_rates = np.logspace(
        np.log10(learning_rate_start.get_value()),
        np.log10(learning_rate_stop.get_value()),
        max_epochs)
    momentums = np.linspace(
        momentum_start.get_value(), momentum_stop.get_value(), max_epochs)

    now = time.time()
    history = []
    if verbose:
        printer = ProgressPrinter(color=True)
    try:
        for epoch in train(iter_funcs, dataset,
                           batch_size=batch_size, test_every=test_every):
            epoch_number = epoch['number']
            train_loss = epoch['train_loss']
            valid_loss = epoch['valid_loss']
            valid_acc = epoch['valid_accuracy']
            info = dict(
                epoch=epoch_number,
                train_loss=train_loss,
                train_loss_best=train_loss <= best_train_loss,
                train_loss_worse=train_loss > history[-1]['train_loss']
                  if len(history) > 0 else False,
                valid_loss=valid_loss,
                valid_loss_best=valid_loss <= best_valid_loss,
                valid_loss_worse=valid_loss > history[-1]['valid_loss']
                  if len(history) > 0 else False,
                valid_accuracy=valid_acc,
                duration=time.time() - now)
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
         layers=(100, 100),
         dropout=0.5,
         transfer_func='rectify',
         nbottleneck=25,
         bottleneck_func='linear',
         max_epochs=1000,
         batch_size=1000,
         patience=100,
         update='sgd',
         learning_rate_start=0.05,
         learning_rate_stop=0.001,
         momentum_start=0.9,
         momentum_stop=0.999,
         momentum=0.9,
         test_every=100,
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
    learning_rate_start = theano.shared(float32(learning_rate_start))
    learning_rate_stop = theano.shared(float32(learning_rate_stop))
    momentum_start = theano.shared(float32(momentum_start))
    momentum_stop = theano.shared(float32(momentum_stop))

    output_layer = build_model(
        input_dim=dataset['input_dim'],
        output_dim=dataset['output_dim'],
        hidden_layers=layers,
        transfer_func=transfer_func,
        dropout=dropout,
        batch_size=batch_size,
        nbottleneck=nbottleneck,
        bottleneck_func=bottleneck_func)

    iter_funcs = create_iter_funcs(
        dataset, output_layer,
        batch_size=batch_size,
        update=update,
        learning_rate=learning_rate_start,
        momentum=momentum_start)

    loss, epoch, weights, history = train_loop(
        output_layer, iter_funcs, dataset, batch_size, max_epochs,
        test_every, patience, learning_rate_start, learning_rate_stop,
        momentum_start, momentum_stop)

    set_all_param_values(output_layer, weights)
    return loss, epoch, history, output_layer


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='train_acoustic_model.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='train bnf acoustic model',
            epilog="""Example usage:

$ python train_acoustic_model.py /path/to/npzfile /path/to/output 100

will train the model for 100 epochs on the data in npzfile. The input file
needs to have X, y and labels keys.

            """)
        parser.add_argument('input', metavar='INPUT',
                            nargs=1,
                            help='npz input file')
        parser.add_argument('output', metavar='OUTPUT',
                            nargs=1,
                            help='output file')
        parser.add_argument('nlayers', metavar='NLAYERS',
                            nargs=1,
                            help='number of hidden layers')
        parser.add_argument('nunits', metavar='NUNITS',
                            nargs=1,
                            help='number of units in hidden layers')
        parser.add_argument('nbottleneck', metavar='NBOTTLENECK',
                            nargs=1,
                            help='number of bottleneck features')
        parser.add_argument('nepochs', metavar='NEPOCHS',
                            nargs=1,
                            help='number of epochs to train')
        parser.add_argument('batch_size', metavar='BATCH_SIZE',
                            nargs=1,
                            help='batch size.')
        parser.add_argument('--dropout',
                            action='store',
                            dest='dropout',
                            default=0,
                            help='add dropout to hidden layers')
        parser.add_argument('--patience',
                            action='store',
                            dest='patience',
                            default=0,
                            help='convergence patience')
        parser.add_argument('--test',
                            action='store_true',
                            dest='test',
                            default=False,
                            help='small datasets for testing')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            default=False,
                            help='talk more')
        return vars(parser.parse_args())


    args = parse_args()
    num_epochs = int(args['nepochs'][0])
    npzfile = args['input'][0]
    output = args['output'][0]

    batch_size = int(args['batch_size'][0])
    nbottleneck = int(args['nbottleneck'][0])
    nunits = int(args['nunits'][0])
    nlayers = int(args['nlayers'][0])
    dropout = float(args['dropout'])
    patience = int(args['patience'])

    verbose = args['verbose']
    test = args['test']

    with verb_print('loading data', verbose):
        dataset = load_data(
            npzfile,
            valid_prop=4/16, test_prop=2/16,
            register='both',
            test=test)

    config = dict(
        layers=[nunits]*nlayers,
        dropout=dropout,
        transfer_func='rectify',
        nbottleneck=nbottleneck,
        bottleneck_func='linear',
        max_epochs=num_epochs,
        batch_size=batch_size,
        patience=patience)

    loss, epoch, history, output_layer = main(dataset, verbose=verbose,
                                              **config)

    with open(output, 'wb') as fout:
        result = dict(
            descr="""Trained network.""",
            config=config,
            loss=loss,
            epoch=epoch,
            history=history,
            network=output_layer)
        pickle.dump(result, fout, -1)
