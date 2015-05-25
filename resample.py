#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: resample.py
# date: Mon May 18 14:36 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""resample:

"""

from __future__ import division

from collections import defaultdict
from itertools import chain, product
import cPickle as pickle

import numpy as np
from scipy.stats import multivariate_normal

from util import verb_print


def constcorr(X):
    """Calculate the constant correlation matrix.

    Parameters
    ----------
    X : ndarray (nsamples, nfeatures)
        observations

    Returns
    -------
    constant correlation matrix of X

    Notes
    -----
    "Honey, I shrunk the sample covariance matrix", Ledoit and Wolf

    """
    N, D = X.shape
    X_ = X - X.mean(0) # centered samples
    s = np.dot(X_.T, X_) / (N-1) # sample covariance
    d = np.diag(s)
    sq = np.sqrt(np.outer(d, d))
    d = s / sq # sample correlation
    r = np.triu(d, 1).sum() * 2 / ((N-1)*N) # average correlation
    f = r * sq
    f[np.diag_indices(D)] = np.diag(s)
    return f


def transform_estimates(d, mean_dispersal_factor=1, cov_shrink_factor=0):
    """Manipulate the estimated distributions by shrinking the covariance
    or dispersing the means from their center point.

    Parameters
    ----------
    d : dict from phone to condition to rv_continuous
        estimated distributions
    cov_shrink_factor : float in [0, 1]
        interpolation factor between the estimated covariance and the constant
        correlation matrix. if `shrink_factor`==0, no further shrinkage is
        performed, if 1, the constant correlation matrix is used instead of
        covariance.
    mean_dispersal_factor : float
        the means of the classes can be brought further apart or closer
        together by this factor.

    Returns
    -------
    dict from phone to condition to rv_continuous
        transformed distributions
    """
    if mean_dispersal_factor == 1 and cov_shrink_factor == 0:
        return d
    means = defaultdict(dict)
    covs = defaultdict(dict)
    for phone in d:
        for condition in d[phone]:
            means[phone][condition] = d[phone][condition].mean
            cov = d[phone][condition].cov
            if cov_shrink_factor > 1:
                cov *= (1-cov_shrink_factor)
            covs[phone][condition] = cov
    if mean_dispersal_factor != 1:
        center = np.vstack([means[phone][condition] for phone in means
                            for conditions in means[phone]]).mean(0)
        means = {p: {c: (means[p][c]-center)*mean_dispersal_factor + center
                         for c in means[p]}
                 for p in means}
    return {phone: {cond: multivariate_normal(mean=means[phone][cond],
                                              cov=covs[phone][cond])
                    for cond in d[phone]}
            for phone in d}


def resample(d, n):
    return {p: {c: d[p][c].rvs(size=n)
                for c in d[p]}
            for p in d}


def get_labels(d):
    conditions = sorted(set(chain.from_iterable(v.keys()
                                                for v in d.values())))
    phones = sorted(d.keys())
    return phones, conditions


def reshape(d):
    phones, conditions = get_labels(d)
    X = None
    Y = None
    labels = list(product(phones, conditions))
    cls2ix = {cls: ix for ix, cls in enumerate(labels)}

    for p in phones:
        for c in conditions:
            x = d[p][c]
            y = np.ones(x.shape[0]) * cls2ix[(p, c)]
            if X is None:
                X = x
                Y = y
            else:
                X = np.vstack((X, x))
                Y = np.hstack((Y, y))

    return X, Y, np.array(labels)

def main(estimates, nsamples, mean_dispersal_factor=1, cov_shrink_factor=0,
         verbose=False):
    with verb_print('transforming distributions', verbose=verbose):
        estimates = transform_estimates(estimates, mean_dispersal_factor,
                                        cov_shrink_factor)
    with verb_print('generating samples', verbose=verbose):
        samples = resample(estimates, nsamples)
    with verb_print('reformatting data', verbose=verbose):
        X, y, labels = reshape(samples)
    return X, y, labels


def load_estimates(fname):
    with open(fname, 'rb') as fin:
        e = pickle.load(fin)
    return e

def save_samples(X, y, labels, output):
    np.savez(output, X=X, y=y, labels=labels)


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='resample.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='Resample stimulus classes',
            epilog="""Example usage:

$ python resample.py 1000 estimated.pkl /path/to/output

will generate 1000 samples for each class. The total number of output samples
is nsamples * nclasses.

The output consists of a single .npz file with 3 arrays
X : nsamples*nclasses x ncepstra
y : nsamples*nclasses
labels : nclasses

labels is a 1d array of strings, indicating how the classes (phones,
conditions) map to the label-indices in y.

            """)
        parser.add_argument('nsamples', metavar='NSAMPLES',
                            nargs=1,
                            help='number of samples per class')
        parser.add_argument('estimates', metavar='ESTIMATES',
                            nargs=1,
                            help='estimated normals in .pkl format')
        parser.add_argument('output', metavar='OUTPUT',
                            nargs=1,
                            help='name of output file')
        parser.add_argument('-s', '--shrink',
                            action='store',
                            dest='shrink',
                            default=0,
                            help='covariance shrinkage factor')
        parser.add_argument('-d', '--dispersal',
                            action='store',
                            dest='dispersal',
                            default=1,
                            help='mean dispersal factor')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='talk more')
        return vars(parser.parse_args())

    args = parse_args()

    nsamples = int(args['nsamples'][0])
    input_fname = args['estimates'][0]
    output = args['output'][0]

    shrink = float(args['shrink'])
    dispersal = float(args['dispersal'])
    verbose = args['verbose']

    with verb_print('loading estimates', verbose):
        estimates = load_estimates(input_fname)
    X, y, labels = main(estimates, nsamples, dispersal, shrink, verbose)
    with verb_print('saving samples', verbose):
        save_samples(X, y, labels, output)
