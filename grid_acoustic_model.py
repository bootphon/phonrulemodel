#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: grid_acoustic_model.py
# date: Mon May 25 16:22 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""grid_acoustic_model: grid search over network parameters for acoustic model

"""

from __future__ import division

import cPickle as pickle
from collections import OrderedDict
import os
import os.path as path

import theano.tensor as T
import numpy as np
from lasagne.layers import get_output
from sklearn.metrics import precision_recall_fscore_support
from sklearn.grid_search import ParameterGrid
from tabulate import tabulate

import train_acoustic_model as tam
import resample
from util import verb_print

def go(estimates, nsamples, dispersal, shrink,
       layers, dropout, transfer_func, nbottleneck, bottleneck_func,
       max_epochs, batch_size, patience, learning_rate, momentum,
       verbose):
    if verbose:
        print '-'*30
    with verb_print('generating samples', verbose):
        X, y, labels = resample.main(
            estimates, nsamples, dispersal, shrink)
    with verb_print('building dataset', verbose):
        dataset = tam.build_dataset(X, y, labels)

    loss, epoch, history, network = tam.main(
        dataset, layers, dropout, transfer_func, nbottleneck, bottleneck_func,
        max_epochs, batch_size, patience, learning_rate, momentum)

    with verb_print('converting data', verbose):
        X_test = dataset['X_test'].get_value()
        y_test = dataset['y_test'].get_value()
    with verb_print('generating predictions', verbose):
        y_pred = T.argmax(get_output(network, X_test, deterministic=True),
                          axis=1).eval()
        # slices = [slice(batch_index*batch_size, (batch_index+1)*batch_size)
        #           for batch_index in xrange(X_test.shape[0] // batch_size+1)]
        # y_pred = np.hstack((
        #     T.argmax(get_output(network, X_test[sl], deterministic=True),
        #          axis=1).eval()
        #     for sl in slices))
    with verb_print('evaluating', verbose):
        prec, recall, fscore, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')
    if verbose:
        print 'prec: {0:.3f}, recall: {1:.3f}, fscore: {2:.3f}'.format(
            prec, recall, fscore)
        print '-'*30
        print

    return loss, epoch, history, network, prec, recall, fscore



if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='grid_acoustic_model.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='grid search over model parameters for acoustic model')
        parser.add_argument('estimation_file', metavar='ESTIMATION_FILE',
                            nargs=1,
                            help='file with distribution parameters per class')
        parser.add_argument('output', metavar='OUTPUT',
                            nargs=1,
                            help='output directory')
        parser.add_argument('-t', '--test',
                            action='store_true',
                            dest='test',
                            default=False,
                            help='run small tests')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='talk more')
        return vars(parser.parse_args())
    args = parse_args()
    est_file = args['estimation_file'][0]
    outdir = args['output'][0]
    if not path.exists(outdir):
        os.makedirs(outdir)
    verbose = args['verbose']
    test = args['test']

    with open(est_file, 'rb') as fin:
        estimates = pickle.load(fin)

    if test:
        param_grid = ParameterGrid(dict(
            nsamples=[100],
            batch_size=[100],
            dispersal=[1, 2],
            shrink=[0],
            layers=[[100, 100]],
            dropout=[0, 0.5],
            transfer_func=['rectify'],
            nbottleneck=[5],
            bottleneck_func=['linear'],
            max_epochs=[10],
            patience=[10],
            learning_rate=[0.01],
            momentum=[0.9]))
    else:
        param_grid = ParameterGrid(dict(
            nsamples=[1000, 5000, 10000],
            dispersal=[1],
            shrink=[0],
            layers=[[100, 100], [100, 100, 100], [100, 100, 100, 100],
                    [500, 500], [500, 500, 500], [100, 100, 100, 100],
                    [1000, 1000], [1000, 1000, 1000], [1000, 1000, 1000, 1000],
                    [1000, 500, 250, 100]],
            dropout=[0, 0.5],
            transfer_func=['rectify'],
            nbottleneck=[5, 10, 15, 20, 25, 50],
            bottleneck_func=['linear', 'rectify'],
            max_epochs=[1000],
            batch_size=[1000],
            patience=[100],
            learning_rate=[0.03, 0.02, 0.01, 0.001, 0.0001],
            momentum=[0.9, 0.95, 0.99]
        ))

    results = []
    for ix, params in enumerate(param_grid):
        best_loss, best_epoch, history, network, prec, recall, fscore = \
            go(estimates, verbose=verbose, **params)
        info = OrderedDict([
            ('idx', ix),
            ('nsamples', params['nsamples']),
            ('disp', params['dispersal']),
            ('shr', params['shrink']),
            ('layers', '\"[{}]\"'.format(':'.join('{}'.format(s)
                                                  for s in params['layers']))),
            ('drop', '{:.2f}'.format(params['dropout'])),
            ('transfer_f', params['transfer_func']),
            ('bnf_s', params['nbottleneck']),
            ('bnf_f', params['bottleneck_func']),
            ('batch', params['batch_size']),
            ('pat', params['patience']),
            ('mom', params['momentum']),
            ('alpha', params['learning_rate']),
            ('max_ep', params['max_epochs']),
            ('best_loss', best_loss),
            ('best_ep', best_epoch),
            ('precision', prec),
            ('recall', recall),
            ('fscore', fscore)])
        results.append(info)
        fname = path.join(outdir, 'model_{}.pkl'.format(ix))
        with open(fname, 'wb') as fout:
            model = dict(
                descr="""Trained network.""",
                config=params,
                loss=best_loss,
                epoch=best_epoch,
                history=history,
                network=network)
            pickle.dump(model, fout, -1)

    print tabulate(results, headers='keys', floatfmt='.5f')
    with open(path.join(outdir, 'results_table.csv'), 'w') as fout:
        fout.write(','.join(results[0].keys()) + '\n')
        fout.write('\n'.join([','.join([str(v) for v in row.values()])
                              for row in results]))
        # fout.write(tabulate(results, headers='keys',
        #            floatfmt='.5f', tablefmt='plain'))
