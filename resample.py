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

import os
import os.path as path
import glob
from collections import namedtuple, defaultdict
from itertools import chain, product
import sys
from time import time
from contextlib import contextmanager

import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal as mnorm

import pandas as pd

from scikits.audiolab import wavread



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.covariance import OAS
import sklearn.manifold as manifold

import spectral
import textgrid




@contextmanager
def verb_print(msg, verbose=False):
    """Helper for verbose printing with timing around pieces of code.
    """
    if verbose:
        t0 = time()
        msg = msg + '...'
        print msg,
        sys.stdout.flush()
    try:
        yield
    finally:
        if verbose:
            print 'done. time: {0:.3f}s'.format(time() - t0)
            sys.stdout.flush()


def make_intervals_dataframe(stimulus_dir, fix=False):
    """Return a DataFrame containing the intervals in each file

    Parameters
    ----------
    stimulus_dir : string
        path to stimuli, should contain both audio and textgrid files
    fix : bool
        manually fix some annotations

    Returns
    -------
    DataFrame
        dataframe with columns: filename, start, end, phone, condition, length
    """
    columns = ['filename', 'start', 'end', 'phone', 'condition']
    Interval = namedtuple('Interval', columns)
    # stimuli_dir = path.join(os.environ['HOME'], 'data', 'ingeborg_stimuli')
    intervals = []
    for tgfile in glob.iglob(path.join(stimulus_dir, '*.TextGrid')):
        with open(tgfile, 'r') as fid:
            tg = textgrid.TextGrid.read(fid)
        bname = path.splitext(path.basename(tgfile))[0]
        condition = bname.split('-')[1]
        intervals.extend([Interval(bname, e.start, e.end, e.mark, condition)
                          for e in tg.tiers[0].entries])
    intervals = pd.DataFrame(intervals, columns=columns)
    intervals['length'] = intervals.end - intervals.start

    if fix:
        # manually fix missing values
        intervals.ix[393, 'phone'] = 's'
        intervals.ix[394, 'phone'] = 'e'
        intervals.ix[395, 'phone'] = 'm'
        intervals.ix[504, 'phone'] = 't'
        intervals.ix[505, 'phone'] = 'i'
        intervals.ix[506, 'phone'] = 'l'
    return intervals


def extract_cepstra(df, stimulus_dir, normalize='cmvn', window_shift=0.005):
    """Extract MFCC features.

    Parameters
    ----------
    df : DataFrame
        dataframe with column filename as output by `make_intervals_dataframe`
    stimulus_dir : string
        location of files
    normalize : string ['none', 'cmvn', 'zca']
        normalization method.

    Returns
    -------
    dict from string to ndarray
        MFCCs per file
    """
    if normalize == 'zca':
        try:
            from zca import ZCA
        except:
            raise ValueError('could not import ZCA, use a different '
                             'normalization method')
    mfcc_config = dict(fs=44100, window_length=0.025, window_shift=window_shift,
                       nfft=2048, scale='mel', nfilt=40, taper_filt=True,
                       deltas=False, do_dct=True, nceps=13, log_e=True,
                       lifter=22, energy_compression='log', pre_emph=0.97)
    mfcc_encoder = spectral.Spectral(**mfcc_config)
    cepstra = {} # dict from filename to mfcc features in ndarray
    for bname in df.filename.unique().tolist():
        fname = path.join(stimulus_dir, bname + '.wav')
        sig, fs, _ = wavread(fname)
        if fs != 44100:
            raise ValueError('fs not 44100 in {}'.format(fname))
        cepstrum = mfcc_encoder.transform(sig)
        cepstra[bname] = cepstrum
    if normalize != 'none':
        all_c = np.vstack(cepstra.values())
        if normalize == 'cmvn':
            normer = StandardScaler().fit(all_c)
        elif normalize == 'zca':
            normer = ZCA(regularization=1./all_c.shape[0]).fit(all_c)
        else:
            raise ValueError('invalid value for normalize: {0}'
                             .format(normalize))
        cepstra = {fname: normer.transform(cepstrum)
                   for fname, cepstrum in cepstra.iteritems()}
    return cepstra


def take_mid_frames(cepstra, df, window_shift):
    """Return average of middle 3 frames for each phone and condition.

    Parameters
    ----------
    cepstra : dict from string to ndarray
        MFCCs per filename
    df : DataFrame
        as output by `make_intervals_dataframe`
    window_shift : float
        window shift in seconds

    Returns
    -------
    dict from phone to condition to ndarray
    """
    f = {phone: {condition: None
                 for condition in df.condition.unique()}
         for phone in df.phone.unique()}
    for ix, row in df.iterrows():
        d = cepstra[row.filename]
        phone_end = int(np.ceil(row.end * 1./window_shift))
        phone_start = int(np.floor(row.start * 1./window_shift))
        mid_start = (phone_end - phone_start) // 2 - 1
        frame = d[mid_start: mid_start + 2].mean(0)

        phone, condition = row.phone, row.condition
        if f[phone][condition] is None:
            f[phone][condition] = frame
        else:
            f[phone][condition] = np.vstack((f[phone][condition], frame))
    return f


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


def estimate_normals(d, mean_dispersal_factor=1, cov_shrink_factor=0):
    """Estimate normal distributions for each phone and condition.

    Since we don't have a lot of samples, we take the median as the estimator
    for the mean and the OAS shrunk covariance for the covariance. Optionally,
    the means can be moved further away from their centerpoint and
    the estimated covariance can be shrunk even more towards the constant
    correlation.

    Parameters
    ----------
    d : dict from phone to condition to MFCCs
        data
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
        estimated normals
    """
    phones, conditions = get_labels(d)
    ix2cls = dict(enumerate(sorted(product(phones, conditions))))
    cls2ix = {cls: ix for ix, cls in ix2cls.iteritems()}

    D = d[phones[0]][conditions[0]].shape[1]
    nclasses = len(cls2ix)
    means = np.zeros((nclasses, D))
    covs = np.zeros((nclasses, D, D))

    # estimate means and covars
    for phone in phones:
        for condition in conditions:
            X = d[phone][condition]
            ix = cls2ix[(phone, condition)]
            cov = OAS(assume_centered=False).fit(X).covariance_
            if cov_shrink_factor > 0:
                # corr = constcorr(X)
                cov = (1-cov_shrink_factor)*cov # + cov_shrink_factor*corr
            covs[ix, :, :] = cov
            means[ix, :] = np.median(X, 0)
    if mean_dispersal_factor != 1:
        mu = means.mean(0)
        means = (means-mu)*mean_dispersal_factor + mu

    # make distributions
    normals = {p: {c: mnorm(mean=means[cls2ix[(p,c)]],
                            cov=covs[cls2ix[(p,c)]])
                   for c in d[p]}
               for p in d}
    return normals


def sample(d, n):
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


def resample(input_dir, nsamples, window_shift, dispersal, shrink, verbose):
    with verb_print('extracting intervals', verbose):
        df = make_intervals_dataframe(input_dir, fix=True)
    with verb_print('extracting mfccs', verbose):
        cepstra = extract_cepstra(df, input_dir, window_shift=window_shift)
    with verb_print('taking middle frames', verbose):
        cepstra = take_mid_frames(cepstra, df, window_shift)
    with verb_print('estimating distributions', verbose):
        normals = estimate_normals(cepstra,
                                   mean_dispersal_factor=dispersal,
                                   cov_shrink_factor=shrink)
    with verb_print('sampling', verbose):
        samples = sample(normals, nsamples)
    with verb_print('reformatting data', verbose):
        X, y, labels = reshape(samples)
    return X, y, labels


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='resample.py',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='resample stimulus classes',
            epilog="""Example usage:

$ python resample.py 1000 /path/to/stimuli /path/to/output

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
        parser.add_argument('stimulus_dir', metavar='STIMULUSDIR',
                            nargs=1,
                            help='path to stimuli')
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
        parser.add_argument('--window-shift',
                            action='store',
                            dest='window_shift',
                            default=0.005,
                            help='window shift')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='talk more')
        return vars(parser.parse_args())
    args = parse_args()

    nsamples = int(args['nsamples'][0])
    input_dir = args['stimulus_dir'][0]
    output = args['output'][0]

    shrink = float(args['shrink'])
    dispersal = float(args['dispersal'])
    window_shift = args['window_shift']
    verbose = args['verbose']

    X, y, labels = resample(input_dir, nsamples, window_shift,
                            dispersal, shrink, verbose)

    with verb_print('saving output', verbose):
        np.savez(output, X=X, y=y, labels=labels)
