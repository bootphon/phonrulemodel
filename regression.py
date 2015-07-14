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

def make_data(dir_train, dir_test, condition, debug, n_samples=1000, n_features=1, n_targets=1, informative_prop=1.0,
              noise=0.0, valid_prop=0.3, method='bnf1'):
#Read in data from bottleneck_features.py
    if method == 'bnf1':
    	train_file = dir_train + 'train_condition' + str(condition) + 'model1.npz'
    	test_file = dir_test + 'test_condition' + str(condition) + 'model1.npz'
        train_dataset = np.load(train_file)
        test_dataset = np.load(test_file)
    elif method == 'bnf2':
    	train_file = dir_train + 'train_condition' + str(condition) + 'model2.npz'
    	test_file = dir_test + 'test_condition' + str(condition) + 'model2.npz'
    	train_dataset = np.load(train_file)
        test_dataset = np.load(test_file)    
    else:
    	raise ValueError('model unknown')

    X_train = train_dataset['X']
    Y_train = train_dataset['y']
    X_test = test_dataset['X']
    Y_test = test_dataset['y']

    #Vanaf hier gaat het fout
    
    #if debug:
    #	X_train = X_train[0:20]
    #	Y_train = Y_train[0:20]
    #	X_test = X_test[0:20]
    #	Y_test = Y_test[0:20]

    print X_train[0]
    print X_train[0].get_value()
    print Y_train[0]


    X_train = MinMaxScaler(feature_range=(0.0,1.0)).fit_transform(X_train)
    X_train = X_train.astype(theano.config.floatX)
    Y_train = MinMaxScaler(feature_range=(0.0,1.0)).fit_transform(Y_train)
    Y_train = Y_train.astype(theano.config.floatX)
    X_test = MinMaxScaler(feature_range=(0.0,1.0)).fit_transform(X_test)
    X_test = X_test.astype(theano.config.floatX)
    Y_test = MinMaxScaler(feature_range=(0.0,1.0)).fit_transform(Y_test)
    Y_test= Y_test.astype(theano.config.floatX)

    #TODO: Ask what this bit does
    if len(X_train.shape) > 1:
        n_features = X_train.shape[1]
    else:
        X_train = X_train.reshape(X_train.shape[0], -1)
        n_features = 1
    if len(Y_train.shape) > 1:
        n_targets = Y_train.shape[1]
    else:
        Y_train = Y+train.reshape(Y_train.shape[0], -1)
        n_targets = 1

    X_train, Y_train, X_valid, Y_valid = \
        train_valid_split(X_train, Y_train,
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
        num_examples_test= X_test.shape[0],
        input_dim=n_features,
        output_dim=n_targets)

def train_valid_split(X, y, valid_prop=0.2):
    nsamples = X.shape[0]
    ixs = np.random.permutation(nsamples)
    X = np.copy(X)
    X = X[ixs]
    y = np.copy(y)
    y = y[ixs]
    valid_cut = int(valid_prop*nsamples)
    X_valid, y_valid = X[:valid_cut], y[:valid_cut]
    X_train, y_train = X[valid_cut:], y[valid_cut:]
    return X_train, y_train, X_valid, y_valid

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
	
    return dict(
        train=iter_train,
        valid=iter_valid)

def train(iter_funcs, dataset, batch_size=300):
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

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

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss
        }

if __name__ == '__main__':
    num_epochs = 10000
    batch_size = 1000 
    dir_train = '/Users/Research/projects/phonrulemodel/bnftrainsets/'
    dir_test = '/Users/Research/projects/phonrulemodel/bnftestsets/'
    #For bnf's generated with model 1: method = bnf1, generated with model 2: method = bnf2
    #Condition: experimental condition (1-8), if debug = True, use subset of data for debugging
    #without GPU
    condition = 1
    debug = True
    dataset = make_data(dir_train, dir_test, condition, debug, n_features=10, n_targets=10,
                        method='bnf1')
    output_layer = build_model(
        input_dim=dataset['input_dim'], output_dim=dataset['output_dim'],
        batch_size=batch_size)
    iter_funcs = create_iter_funcs(dataset, output_layer,
                                   batch_size=batch_size,
                                   learning_rate=0.1, momentum=0.9)
    
    # Exposure phase
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset,
                           batch_size=batch_size):
            print('Epoch {} of {} took {:.3f}s'.format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            if epoch['number'] >= num_epochs:
                break
    except KeyboardInterrupt:
        pass

    # Test phase
    #TODO: adjust for two test items & compare prediction error
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