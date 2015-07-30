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

consonants = {'b', 'd', 'f', 'p', 's', 't', 'v', 'z'}

if __name__ == '__main__':
    err_dir = path.join(os.environ['HOME'], 'data',
                        'ingeborg_datasets', 'regression_errors')
    super_df = None
    for fname in glob.iglob(path.join(err_dir, '*.csv')):
        bname = path.splitext(path.basename(fname))[0]
        print bname
        df = pd.read_csv(fname)
        df = df[df.phone1.isin(consonants)]
        if super_df is None:
            super_df = df
        else:
            super_df = pd.concat([super_df, df])

        # does the model learn anything at all
        err_con = df[(df.congruency == 'C')].error.values
        err_inc = df[(df.congruency == 'I')].error.values
        t, p, dg, d = ttest(err_con, err_inc)
        print 'ALL (GROUPED): t: {:.5f}, p: {:.3f}, df: {}, d: {:.5f}'.format(
            t, p, dg, d)

        congr_ADS = df[(df.register == 'A') & (df.congruency == 'C')].error.values
        incon_ADS = df[(df.register == 'A') & (df.congruency == 'I')].error.values
        congr_IDS = df[(df.register == 'I') & (df.congruency == 'C')].error.values
        incon_IDS = df[(df.register == 'I') & (df.congruency == 'I')].error.values

        print 'ADS (GROUPED): t: {:.5f}, p: {:.3f}, df: {}, d: {:.5f}'.format(
            *ttest(congr_ADS, incon_ADS))
        print 'IDS (GROUPED): t: {:.5f}, p: {:.3f}, df: {}, d: {:.5f}'.format(
            *ttest(congr_IDS, incon_IDS))

        dec_ADS = (congr_ADS < incon_ADS).astype(np.int)
        dec_IDS = (congr_IDS < incon_IDS).astype(np.int)
        dec_ALL = (err_con < err_inc).astype(np.int)

        print 'ALL (DECISION): {:.5f} ({:d}/{:d}) p: {}'.format(
            dec_ALL.mean(), dec_ALL.sum(), dec_ALL.shape[0],
            statsmodels.stats.proportion.binom_test(
                dec_ALL.sum(), dec_ALL.shape[0]), alternative="larger"
        )
        print 'ADS (DECISION): {:.5f} ({:d}/{:d}) p: {}'.format(
            dec_ADS.mean(), dec_ADS.sum(), dec_ADS.shape[0],
            statsmodels.stats.proportion.binom_test(
                dec_ADS.sum(), dec_ADS.shape[0]), alternative="larger")
        print 'IDS (DECISION): {:.5f} ({:d}/{:d}) p: {}'.format(
            dec_IDS.mean(), dec_IDS.sum(), dec_IDS.shape[0],
            statsmodels.stats.proportion.binom_test(
                dec_IDS.sum(), dec_IDS.shape[0]), alternative="larger")
        chi2, p, _ = statsmodels.stats.proportion.proportions_chisquare(
                [dec_ADS.sum(), dec_IDS.sum()], dec_ADS.shape[0])
        print 'DIFFERENCE: chi2: {:.5f} p: {}'.format(chi2, p)
        print

    print 'ALL TOGETHER'

    err_con = super_df[(super_df.congruency == 'C')].error.values
    err_inc = super_df[(super_df.congruency == 'I')].error.values
    print 'ALL (GROUPED): t: {:.5f}, p: {:.3f}, df: {}, d: {:.5f}'.format(
        *ttest(err_con, err_inc))

    congr_ADS = super_df[(super_df.register == 'A') & (super_df.congruency == 'C')].error.values
    incon_ADS = super_df[(super_df.register == 'A') & (super_df.congruency == 'I')].error.values
    congr_IDS = super_df[(super_df.register == 'I') & (super_df.congruency == 'C')].error.values
    incon_IDS = super_df[(super_df.register == 'I') & (super_df.congruency == 'I')].error.values

    print 'ADS (GROUPED): t: {:.5f}, p: {:.3f}, df: {}, d: {:.5f}'.format(
        *ttest(congr_ADS, incon_ADS))
    print 'IDS (GROUPED): t: {:.5f}, p: {:.3f}, df: {}, d: {:.5f}'.format(
        *ttest(congr_IDS, incon_IDS))

    dec_ADS = (congr_ADS < incon_ADS).astype(np.int)
    dec_IDS = (congr_IDS < incon_IDS).astype(np.int)
    dec_ALL = (err_con < err_inc).astype(np.int)

    print 'ALL (DECISION): {:.5f} ({:d}/{:d}) p: {}'.format(
        dec_ALL.mean(), dec_ALL.sum(), dec_ALL.shape[0],
        statsmodels.stats.proportion.binom_test(
            dec_ALL.sum(), dec_ALL.shape[0]), alternative="larger"
    )
    print 'ADS (DECISION): {:.5f} ({:d}/{:d}) p: {}'.format(
        dec_ADS.mean(), dec_ADS.sum(), dec_ADS.shape[0],
        statsmodels.stats.proportion.binom_test(
            dec_ADS.sum(), dec_ADS.shape[0]), alternative="larger")
    print 'IDS (DECISION): {:.5f} ({:d}/{:d}) p: {}'.format(
        dec_IDS.mean(), dec_IDS.sum(), dec_IDS.shape[0],
        statsmodels.stats.proportion.binom_test(
            dec_IDS.sum(), dec_IDS.shape[0]), alternative="larger")
    chi2, p, _ = statsmodels.stats.proportion.proportions_chisquare(
            [dec_ADS.sum(), dec_IDS.sum()], dec_ADS.shape[0])
    print 'DIFFERENCE: chi2: {:.5f} p: {}'.format(chi2, p)
    print
