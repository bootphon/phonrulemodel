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
from pprint import pformat

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
       layers, dropout, transfer_f, bnf_s, bnf_f,
       max_epochs, batch_size, patience, update, learning_rate_start,
       learning_rate_stop, momentum_start, momentum_stop,
       verbose):
    if verbose:
        print '-'*30
    with verb_print('generating samples', verbose):
        X, y, labels = resample.main(
            estimates, nsamples, dispersal, shrink)
    with verb_print('building dataset', verbose):
        dataset = tam.build_dataset(X, y, labels)

    loss, epoch, history, network = tam.main(
        dataset, layers, dropout, transfer_f, bnf_s, bnf_f,
        max_epochs, batch_size, patience, update, learning_rate_start,
        learning_rate_stop, momentum_start, momentum_stop, test_every=100,
        verbose=True)

    with verb_print('converting data', verbose):
        X_test = dataset['X_test'].get_value()
        y_test = dataset['y_test'].get_value()
    with verb_print('generating predictions', verbose):
        y_pred = T.argmax(get_output(network, X_test, deterministic=True),
                          axis=1).eval()
    with verb_print('evaluating', verbose):
        prec, recall, fscore, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')

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
        param_grid = dict(
            nsamples=[1000],
            batch_size=[1000],
            dispersal=[1],
            shrink=[0],
            layers=[[100, 100], [100, 100, 100]],
            dropout=[0.0, 0.5],
            transfer_f=['rectify'],
            bnf_s=[5, 10],
            bnf_f=['linear'],
            max_epochs=[5],
            patience=[10],
            update=['nesterov'],
            learning_rate_start=[0.03, 0.05],
            learning_rate_stop=[0.001],
            momentum_start=[0.9],
            momentum_stop=[0.999])
    else:
        param_grid = dict(
            nsamples=[5000],
            dispersal=[1],
            shrink=[0],
            layers=[[1000, 1000, 1000], [2000, 2000, 2000]],
            dropout=[0.5],
            transfer_f=['rectify'],
            bnf_s=range(10, 20),
            bnf_f=['linear'],
            max_epochs=[5000],
            batch_size=[20000],
            patience=[100],
            update=['nesterov'],
            learning_rate_start=[0.05],
            learning_rate_stop=[0.0001, 0.001],
            momentum_start=[0.9],
            momentum_stop=[0.999])

    # changing parameters
    dyn_params = [p for p in param_grid if len(param_grid[p]) > 1]
    eval_metrics = ['precision', 'recall', 'fscore']
    param_grid = ParameterGrid(param_grid)
    results = []
    for ix, params in enumerate(param_grid):
        info = OrderedDict([
            ('idx', ix),
            ('nsamples', params['nsamples']),
            ('dispersal', params['dispersal']),
            ('shrink', params['shrink']),
            ('layers', '\"[{}]\"'.format(':'.join('{}'.format(s)
                                                  for s in params['layers']))),
            ('dropout', '{:.2f}'.format(params['dropout'])),
            ('transfer_f', params['transfer_f']),
            ('bnf_s', params['bnf_s']),
            ('bnf_f', params['bnf_f']),
            ('batch_size', params['batch_size']),
            ('patience', params['patience']),
            ('update', params['update']),
            ('momentum_start', params['momentum_start']),
            ('momentum_stop', params['momentum_stop']),
            ('learning_rate_start', params['learning_rate_start']),
            ('learning_rate_stop', params['learning_rate_stop']),
            ('max_epochs', params['max_epochs'])])
        print pformat(dict(info))
        best_loss, best_epoch, history, network, prec, recall, fscore = \
            go(estimates, verbose=verbose, **params)
        info.update(OrderedDict([
            ('best_loss', best_loss),
            ('best_ep', best_epoch),
            ('precision', prec),
            ('recall', recall),
            ('fscore', fscore)]))

        results.append(info)
        print tabulate([OrderedDict([(k, v)
                                      for k, v in row.iteritems()
                                      if k in set(dyn_params+eval_metrics)])
                         for row in results],
                        headers='keys',
                        floatfmt='.5f',
                        tablefmt='simple')

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
