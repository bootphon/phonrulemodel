#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: svm.py
# date: Mon May 18 20:41 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""svm:

"""

from __future__ import division
from functools import partial

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score, classification_report

if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='svm.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='train an svm')
        parser.add_argument('datafile', metavar='DATAFILE',
                            nargs=1,
                            help='data in .npz format')
        parser.add_argument('-j', '--jobs',
                            action='store',
                            dest='jobs',
                            type=int,
                            default=1,
                            help='number of parallel jobs')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='talk more')
        return vars(parser.parse_args())
    args = parse_args()

    fname = args['datafile'][0]
    n_jobs = args['jobs']
    verbose = args['verbose']

    f = np.load(fname)
    X, y, labels = f['X'], f['y'], f['labels']

    #######
    # IDS #
    #######
    ids_ixs = np.in1d(y, np.nonzero(labels[:, 1]=='IDS'))
    X_ids = X[ids_ixs]
    y_ids = y[ids_ixs]

    X_train, X_test, y_train, y_test = train_test_split(X_ids, y_ids)
    clf = GridSearchCV(SVC(kernel='rbf'),
                       param_grid={'C':np.logspace(-2, 2, 20)},
                       scoring=make_scorer(partial(f1_score,
                                                   average='weighted')),
                       n_jobs=n_jobs,
                       verbose=0 if verbose else 0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if verbose:
        print 'IDS:'
        print classification_report(
            y_test, y_pred, target_names=labels[labels[:,1]=='IDS'][:, 0])
    else:
        print 'IDS f-score: {0:.3f}'.format(f1_score(y_test, y_pred,
                                                     average='weighted'))


    #######
    # ADS #
    #######
    ads_ixs = np.in1d(y, np.nonzero(labels[:, 1]=='ADS'))
    X_ads = X[ads_ixs]
    y_ads = y[ads_ixs]

    X_train, X_test, y_train, y_test = train_test_split(X_ads, y_ads)

    clf = GridSearchCV(SVC(kernel='rbf'),
                       param_grid={'C':np.logspace(-2, 2, 20)},
                       scoring=make_scorer(partial(f1_score,
                                                   average='weighted')),
                       n_jobs=n_jobs,
                       verbose=0 if verbose else 0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if verbose:
        print 'ADS:'
        print classification_report(
            y_test, y_pred, target_names=labels[labels[:,1]=='ADS'][:, 0])
    else:
        print 'ADS f-score: {0:.3f}'.format(f1_score(y_test, y_pred,
                                                     average='weighted'))

    ########
    # BOTH #
    ########
    ix2phone = dict(enumerate(labels[:, 0]))
    phones = sorted(set(ix2phone.values()))
    phone2newix = {p: ix for ix, p in enumerate(phones)}
    y = np.array([phone2newix[ix2phone[i]] for i in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = GridSearchCV(
        SVC(kernel='rbf'),
        param_grid={'C':np.logspace(-2, 2, 20)},
        scoring=make_scorer(partial(f1_score, average='weighted')),
        n_jobs=n_jobs,
        verbose=0 if verbose else 0
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if verbose:
        print 'BOTH:'
        print classification_report(
            y_test, y_pred, target_names=phones)
    else:
        print 'BOTH f-score: {0:.3f}'.format(f1_score(y_test, y_pred,
                                                      average='weighted'))
