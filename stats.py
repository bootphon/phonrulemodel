#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: stats.py
# date: Wed July 29 17:33 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""stats:

"""

from __future__ import division

import numpy as np
import pandas as pd
import statsmodels.stats.proportion
from scipy.stats import chi2_contingency
import statsmodels.stats.weightstats
import os
import os.path as path
import glob
from collections import defaultdict

import scipy.stats


def cohens_d(g1, g2):
    l1 = g1.shape[0] - 1
    l2 = g2.shape[0] - 1
    m1 = g1.mean()
    m2 = g2.mean()
    var1 = g1.var()
    var2 = g2.var()
    s = np.sqrt((l1 * var1 + l2 * var2) / (l1 + l2))
    return np.abs(m1-m2) / s


def ttest(g1, g2):
    t, p = scipy.stats.ttest_ind(g1, g2, equal_var=False)
    d = cohens_d(g1, g2)
    return t, p, 1, d


def chi2(correct_ADS, correct_IDS):
    n_ADS = correct_ADS.shape[0]
    n_corr_ADS = correct_ADS.sum()
    n_IDS = correct_IDS.shape[0]
    n_corr_IDS = correct_IDS.sum()

    table = np.array([[n_corr_ADS, n_ADS - n_corr_ADS],
                      [n_corr_IDS, n_IDS - n_corr_IDS]])
    chi2, p, dof, expected = chi2_contingency(table)
    return chi2, p, dof, np.sqrt(chi2/table.sum())


def print_group_stats(congruent, incongruent, label):
    print 'GROUP: {}'.format(label)
    print 'congruent:   {:.5f} (SD: {:.5f})'.format(
        congruent.mean(),
        congruent.std()
    )
    print 'incongruent: {:.5f} (SD: {:.5f})'.format(
        incongruent.mean(),
        incongruent.std()
    )
    print 'difference between ADS/IDS means (t-test):'
    print 't: {:.5f}, p: {:.3f}, dof: {}, cohen\'s d: {:.5f}'.format(
        *ttest(congruent, incongruent)
    )


def binom_std(p, n):
    return np.sqrt(n*p*(1-p))


def print_score_stats(correct_ALL, correct_ADS, correct_IDS):
    mean_ALL = correct_ALL.mean()
    n_ALL = correct_ALL.shape[0]
    print 'ALL score: {:d}/{:d} (SD: {:.5f})'.format(
        int(mean_ALL*n_ALL), n_ALL,
        binom_std(mean_ALL, n_ALL)
    )
    mean_ADS = correct_ADS.mean()
    n_ADS = correct_ADS.shape[0]
    print 'ADS score: {:d}/{:d} (SD: {:.5f})'.format(
        int(mean_ADS * n_ADS), n_ADS,
        binom_std(mean_ADS, n_ADS)
    )
    mean_IDS = correct_IDS.mean()
    n_IDS = correct_IDS.shape[0]
    print 'IDS score: {:d}/{:d} (SD: {:.5f})'.format(
        int(mean_IDS * n_IDS), n_IDS,
        binom_std(mean_IDS, n_IDS)
    )

    print 'difference between ADS/IDS scores (chi^2 test of independence):'
    print 'chi2: {:.5f}, p: {:.3f}, dof: {}, phi: {:.5f}'.format(
        *chi2(correct_ADS, correct_IDS)
    )


def print_stats(df, label=''):
    print '=' * 80
    print label
    print '-' * 80
    print 'group mean error'
    print '-' * 80

    # does the model learn anything at all
    err_con = df[(df.congruency == 'C')].error.values
    err_inc = df[(df.congruency == 'I')].error.values
    print_group_stats(err_con, err_inc, 'ALL')
    print
    congr_ADS = df[
        (df.register == 'A') &
        (df.congruency == 'C')
    ].error.values
    np.random.shuffle(congr_ADS)
    incon_ADS = df[
        (df.register == 'A') &
        (df.congruency == 'I')
    ].error.values
    np.random.shuffle(incon_ADS)
    congr_IDS = df[
        (df.register == 'I') &
        (df.congruency == 'C')
    ].error.values
    np.random.shuffle(congr_IDS)
    incon_IDS = df[
        (df.register == 'I') &
        (df.congruency == 'I')
    ].error.values
    np.random.shuffle(incon_IDS)

    print_group_stats(congr_ADS, incon_ADS, 'ADS')
    print
    print_group_stats(congr_IDS, incon_IDS, 'IDS')

    print '-' * 80
    print 'group scores'
    print '-' * 80
    correct_ADS = congr_ADS < incon_ADS
    correct_IDS = congr_IDS < incon_IDS
    correct_ALL = err_con < err_inc[:err_con.shape[0]]
    print_score_stats(correct_ALL, correct_ADS, correct_IDS)
    print '=' * 80
    print


if __name__ == '__main__':
    err_dir = path.join(os.environ['HOME'], 'data',
                        'ingeborg_datasets', 'sanity_errors')
    consonants = {'b', 'd', 'f', 'p', 's', 't', 'v', 'z'}
    super_df = None
    for fname in sorted(glob.glob(path.join(err_dir, '*.csv'))):
        bname = path.splitext(path.basename(fname))[0]
        df = pd.read_csv(fname)
        df = df[df.phone1.isin(consonants)]
        print_stats(df, bname)

        if super_df is None:
            super_df = df
        else:
            super_df = pd.concat([super_df, df])

    print_stats(super_df, 'ALL MODELS COMBINED')
