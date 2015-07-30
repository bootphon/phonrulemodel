#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: make_regression_samples_mfcc.py
# date: Sun July 19 01:55 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""make_regression_samples_mfcc: make samples for regression.

"""

from __future__ import division

import os
import os.path as path
import glob

import re

import numpy as np
import pandas as pd

from util import verb_print

import resample

NFEATURES = 39


def read_train_csv(fname):
    with open(fname) as fin:
        s = fin.read()
    r = []
    for line in re.split(r'[\r\n]', s):
        stim, register = line.strip().split('-')
        stim = stim.lower()
        r.append((stim[0], stim[1], register))
        r.append((stim[1], stim[2], register))
    return pd.DataFrame(r, columns=['phone1', 'phone2', 'register'])


def read_test_csv(fname):
    with open(fname) as fin:
        s = fin.read()
    r = []
    for line in re.split(r'[\r\n]', s):
        part1, part2, congruency_ix, _ = line.strip().split(',')

        stim1, register = part1.split('-')
        stim2, _ = part2.split('-')  # registers always equal
        stim1, stim2 = stim1.lower(), stim2.lower()
        if congruency_ix == 1:
            r.append((stim1[0], stim1[1], register, 'CONGRUENT'))
            r.append((stim2[0], stim2[1], register, 'INCONGRUENT'))
        else:
            r.append((stim1[0], stim1[1], register, 'INCONGRUENT'))
            r.append((stim2[0], stim2[1], register, 'CONGRUENT'))
    return pd.DataFrame(
        r,
        columns=['phone1', 'phone2', 'register', 'congruency']
    )


def gen_train_single_condition(
        cond_file, estimates, nsamples_per_stim, verbose=False):
    df = read_train_csv(cond_file)
    nsamples = len(df) * nsamples_per_stim
    X = np.zeros((nsamples, NFEATURES))
    Y = np.zeros((nsamples, NFEATURES))
    for ix, (phone1, phone2, register) in df.iterrows():
        start_ix = ix * nsamples_per_stim
        end_ix = (ix+1) * nsamples_per_stim
        X[start_ix: end_ix, :] = \
            estimates[phone1][register].rvs(size=nsamples_per_stim)
        Y[start_ix: end_ix, :] = \
            estimates[phone2][register].rvs(size=nsamples_per_stim)
    return X, Y, df


def gen_test_single_condition(
        cond_file, estimates, nsamples_per_stim, verbose=False):
    """Split out test by IDS/ADS, CONGRUENT/INCONGRUENT

    Returns
    X, Y : ndarray (nsamples, nfeatures)
    legend : ndarray (nsamples, 4) dtype='S1'

    """
    df = read_test_csv(cond_file)
    nsamples = len(df) * nsamples_per_stim
    X = np.empty((nsamples, NFEATURES), dtype=np.float32)
    Y = np.empty((nsamples, NFEATURES), dtype=np.float32)
    legend = np.empty((nsamples, 5), dtype=np.string_)

    for ix, (phone1, phone2, register, congruency) in df.iterrows():
        start_ix = ix * nsamples_per_stim
        end_ix = (ix+1) * nsamples_per_stim
        X[start_ix: end_ix, :] = \
            estimates[phone1][register].rvs(size=nsamples_per_stim)
        Y[start_ix: end_ix, :] = \
            estimates[phone2][register].rvs(size=nsamples_per_stim)
        stimulus_id = '{}'.format(ix // 4)
        legend[start_ix: end_ix, :] = \
            np.array([phone1, phone2, stimulus_id, register, congruency])
    return X, Y, legend, df


def gen_train(estimates, nsamples_per_stim, condition_files, output_dir,
              verbose=False):
    for bname, fname in condition_files:
        with verb_print('generating train samples for {}'.format(bname),
                        verbose):
            X, Y, df = gen_train_single_condition(
                fname, estimates, nsamples_per_stim, verbose=verbose
            )
            df.to_csv(path.join(output_dir, bname + '.csv'), index=False)
            np.savez(
                path.join(output_dir, bname + '.npz'),
                X=X,
                Y=Y
            )


def gen_test(estimates, nsamples_per_stim, condition_files, output_dir,
             verbose=False):
    for bname, fname in condition_files:
        with verb_print('generating test samples for {}'.format(bname),
                        verbose):
            X, Y, legend, df = gen_test_single_condition(
                fname, estimates, nsamples_per_stim, verbose=verbose)
            df.to_csv(path.join(output_dir, bname + '.csv'), index=False)
            np.savez(
                path.join(output_dir, bname + '.npz'),
                X=X,
                Y=Y,
                legend=legend
            )


if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(
            prog='make_regression_samples_mfcc.py',
            description='resample MFCC features in regression format'
        )
        parser.add_argument('nsamples', metavar='NSAMPLES',
                            nargs=1,
                            help='number of samples per stimulus')
        parser.add_argument('estimates', metavar='ESTIMATES',
                            nargs=1,
                            help='estimated normals in .pkl format')
        parser.add_argument('condition_dir', metavar='CONDITIONDIR',
                            nargs=1,
                            help='path to stimuli')
        parser.add_argument('output', metavar='OUTPUTDIR',
                            nargs=1,
                            help='output directory')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='talk more')
        return vars(parser.parse_args())
    args = parse_args()

    estimates_file = args['estimates'][0]
    cond_dir = args['condition_dir'][0]
    cond_dir_train = path.join(cond_dir, 'train')
    cond_dir_test = path.join(cond_dir, 'test')
    nsamples_per_stim = int(args['nsamples'][0])
    output_dir = args['output'][0]
    verbose = args['verbose']

    output_dir_train = path.join(output_dir, 'train')
    try:
        os.makedirs(output_dir_train)
    except OSError:
        pass
    output_dir_test = path.join(output_dir, 'test')
    try:
        os.makedirs(output_dir_test)
    except OSError:
        pass

    with verb_print('loading estimates', verbose):
        estimates = resample.load_estimates(estimates_file)

    conditions_train = [
        (path.splitext(path.basename(f))[0], f)
        for f in glob.iglob(path.join(cond_dir_train, '*.csv'))
    ]
    gen_train(
        estimates, nsamples_per_stim, conditions_train,
        output_dir_train, verbose=verbose
    )

    conditions_test = [
        (path.splitext(path.basename(f))[0], f)
        for f in glob.iglob(path.join(cond_dir_test, '*.csv'))
    ]
    gen_test(
        estimates,
        nsamples_per_stim,
        conditions_test,
        output_dir_test, verbose=verbose
    )
