#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: make_sanity_test_set.py
# date: Thu July 30 13:22 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""make_sanity_test_set: make a test set of mfcc's that has positive examples
from the training set and negative examples from the test set.
"""

from __future__ import division

import os
import os.path as path
import re
import glob
from collections import namedtuple

import numpy as np
import pandas as pd

from util import verb_print
import resample

NFEATURES = 39

FilePair = namedtuple('FilePair', ['positive', 'negative'])


def gather_files(cond_dir):
    train_dir = path.join(cond_dir, 'train')
    test_dir = path.join(cond_dir, 'test')

    pos_files = {
        path.splitext(path.basename(fname))[0].replace('_exposure', ''):
        fname
        for fname in glob.iglob(path.join(train_dir, '*.csv'))
    }
    neg_files = {
        path.splitext(path.basename(fname))[0].replace('_test', ''):
        fname
        for fname in glob.iglob(path.join(test_dir, '*.csv'))
    }
    assert (set(pos_files.keys()) == set(neg_files.keys()))
    return {
        bname: FilePair(pos_files[bname], neg_files[bname])
        for bname in pos_files
    }


def read_train_csv(fname):
    with open(fname) as fin:
        s = fin.read()
    r = []
    for line in re.split(r'[\r\n]', s):
        stim, register = line.strip().split('-')
        stim = stim.lower()
        r.append((stim[0], stim[1], register))

    df = pd.DataFrame(r, columns=['phone1', 'phone2', 'register'])
    df['congruency'] = 'CONGRUENT'


def read_test_csv(fname):
    """read only incongruent, i.e. negative, samples
    """

    with open(fname) as fin:
        s = fin.read()
    r = []
    for line in re.split(r'[\r\n]', s):
        part1, part2, congruency_ix, _ = line.strip().split(',')

        stim1, register = part1.split('-')
        stim2, _ = part2.split('-')  # registers always equal
        stim1, stim2 = stim1.lower(), stim2.lower()
        if congruency_ix == 1:
            r.append((stim2[0], stim2[1], register, 'INCONGRUENT'))
        else:
            r.append((stim1[0], stim1[1], register, 'INCONGRUENT'))
    return pd.DataFrame(
        r,
        columns=['phone1', 'phone2', 'register', 'congruency']
    )


def gen_single_condition(
        file_pair, estimates, nsamples_per_stim, verbose=False):
    df_pos = read_train_csv(file_pair.positive)
    df_neg = read_test_csv(file_pair.negative)
    df = pd.concat([df_pos, df_neg])
    nsamples = len(df) * nsamples_per_stim
    X = np.empty((nsamples, NFEATURES), dtype=np.float32)
    Y = np.empty((nsamples, NFEATURES), dtype=np.float32)
    legend = np.empty((nsamples, 4), dtype=np.string_)

    for ix, (phone1, phone2, register, congruency) in df.iterrows():
        start_ix = ix * nsamples_per_stim
        end_ix = (ix+1) * nsamples_per_stim
        X[start_ix: end_ix, :] = \
            estimates[phone1][register].rvs(size=nsamples_per_stim)
        Y[start_ix: end_ix, :] = \
            estimates[phone2][register].rvs(size=nsamples_per_stim)
        legend[start_ix: end_ix] = \
            np.array([phone1, phone2, register, congruency])
    return X, Y, legend, df


def gen_test(estimates, nsamples_per_stim, file_pairs, output_dir,
             verbose=False):
    for bname, file_pair in file_pairs.iteritems():
        with verb_print('generating test samples for {}'.format(bname),
                        verbose):
            X, Y, legend, df = gen_single_condition(
                file_pair, estimates, nsamples_per_stim,
                verbose=False
            )
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
            prog='make_sanity_test_set.py',
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
    nsamples_per_stim = int(args['nsamples'][0])
    output_dir = args['output'][0]
    verbose = args['verbose']

    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    with verb_print('loading estimates', verbose):
        estimates = resample.load_estimates(estimates_file)

    file_pairs = gather_files(cond_dir)
    gen_test(estimates, nsamples_per_stim, file_pairs, output_dir, verbose)
