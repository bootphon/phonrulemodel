"""
generate datasets
"""
from __future__ import division

from collections import defaultdict
from itertools import chain, product
import cPickle as pickle

import numpy as np
from scipy.stats import multivariate_normal

from util import verb_print

import csv
import glob
import os.path as path
import pandas as pd


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

def generate_test(df_test, estimates, nsamples, output, mean_dispersal_factor=1, 
        cov_shrink_factor=0, verbose=False):
    with verb_print('transforming distributions', verbose=verbose):
        estimates = transform_estimates(estimates, mean_dispersal_factor,
                                        cov_shrink_factor)
       
    samples = resample(estimates, nsamples)  
   
    """
        x1       x2  y  setting x1_c1 x1_v x1_c2 x2_c1 x2_v x2_c2
    0   FIN-ADS  ZUL-ADS  1     ADS   F    I     N     Z    U     L
    """
    df_x1_c1 = df_test['x1_c1'].values
    df_x1_v = df_test['x1_v'].values
    df_x1_c2 = df_test['x1_c2'].values
    df_x2_c1 = df_test['x2_c1'].values
    df_x2_v = df_test['x2_v'].values
    df_x2_c2 = df_test['x2_c2'].values
    df_y = df_test['y'].values
    setting = df_test['setting'].values

    x1_c1s = []
    x1_vs = []
    x1_c2s = []
    x1_ys = []
    x2_c1s = []
    x2_vs = []
    x2_c2s = []
    ys = []
    
    for i in range(len(df_x1_c1)):
        x1_c1 =  samples.get(df_x1_c1[i].lower())
        MFCCs_x1_c1 = x1_c1.get(setting[i])
        x1_c1s.append(MFCCs_x1_c1)
        x1_v =  samples.get(df_x1_v[i].lower())
        MFCCs_x1_v = x1_v.get(setting[i])
        x1_vs.append(MFCCs_x1_v)
        x1_c2 =  samples.get(df_x1_c2[i].lower())
        MFCCs_x1_c2 = x1_c2.get(setting[i])
        x1_c2s.append(MFCCs_x1_c2)

        x2_c1 =  samples.get(df_x2_c1[i].lower())
        MFCCs_x2_c1 = x2_c1.get(setting[i])
        x2_c1s.append(MFCCs_x2_c1)
        x2_v =  samples.get(df_x2_v[i].lower())
        MFCCs_x2_v = x2_v.get(setting[i])
        x2_vs.append(MFCCs_x2_v)
        x2_c2 =  samples.get(df_x2_c2[i].lower())
        MFCCs_x2_c2 = x2_c2.get(setting[i])
        x2_c2s.append(MFCCs_x2_c2)

        y = df_y[i]
        s = setting[i]
        ys.append([y,s])

    x1_c1s = np.array(x1_c1s)
    x1_vs = np.array(x1_vs)
    x1_c2s = np.array(x1_c2s)   
    x2_c1s = np.array(x2_c1s)
    x2_vs = np.array(x2_vs)
    x2_c2s = np.array(x2_c2s)   

    X_x1 = np.column_stack((x1_c1s,x1_vs,x1_c2s))
    X_x2 = np.column_stack((x2_c1s,x2_vs,x2_c2s))
    labels = ['e','i','o','u','b','d','p','f','s','z','v','f']
    labels = np.array(labels)
    X = np.column_stack((X_x1, X_x2))
    y = np.array(ys) #example: ['1' 'ADS']
    np.savez(output, X=X, y = y, labels = labels)
        
def generate_train(df_train, estimates, nsamples, output, mean_dispersal_factor=1, 
        cov_shrink_factor=0, verbose=False):

    with verb_print('transforming distributions', verbose=verbose):
        estimates = transform_estimates(estimates, mean_dispersal_factor,
                                        cov_shrink_factor)

    samples = resample(estimates, nsamples)
    
    df_c1 = df_train['c1'].values
    df_v = df_train['v'].values
    df_c2 = df_train['c2'].values
    setting = df_train['setting'].values

    c1s = []
    vs = []
    c2s = []
    ys = []
    #print samples
    for i in range(len(df_c1)):
        c1 =  samples.get(df_c1[i].lower())
        MFCCs_c1 = c1.get(setting[i])
        c1s.append(MFCCs_c1)
        v =  samples.get(df_v[i].lower())
        MFCCs_v = v.get(setting[i])
        vs.append(MFCCs_v)
        c2 =  samples.get(df_c2[i].lower())
        MFCCs_c2 = c2.get(setting[i])
        c2s.append(MFCCs_c2)
        y = [df_c1[i].lower(),df_v[i].lower(),df_c2[i].lower()]
        s = setting[i]
        ys.append([y,s])

    c1s = np.array(c1s)
    vs = np.array(vs)
    c2s = np.array(c2s)    

    X = np.column_stack((c1s,vs,c2s))
    y = np.array(ys)   #example: [['d', 'o', 'n'] 'ADS']
    
    labels = ['e','i','o','u','b','d','p','f','s','z','v','f']
    labels = np.array(labels)

    np.savez(output, X=X, y = y, labels = labels)
   

def load_estimates(fname):
    with open(fname, 'rb') as fin:
        e = pickle.load(fin)
    return e

def train_data(dir,estimates,output_dir):
    header = ['stimulus']
    counter = 1
    for condition in glob.iglob(path.join(dir, '*.csv')):
        df_train = pd.read_csv(condition, names= header)
        stimulus = df_train['stimulus'].values
        c1 = []
        v = []
        c2 = []
        setting = []
        for stim in stimulus:
            c1.append(stim[0])
            v.append (stim[1])
            c2.append(stim[2])
            setting.append(stim[4:7])
          
        c1 = np.array(c1)
        v = np.array(v)
        c2 = np.array(c2)
        setting = np.array(setting)
        df_train['c1'] = c1     
        df_train['v'] = v
        df_train['c2'] = c2
        df_train['setting'] = setting
 
        output = output_dir + 'train_condition' + str(counter)
        counter = counter + 1
        generate_train(df_train,estimates,nsamples, output, mean_dispersal_factor=1, cov_shrink_factor=0,verbose=False)
        

def test_data(dir,estimates,output_dir):
    counter = 1
    for condition in glob.iglob(path.join(dir, '*.csv')):
        df_test = pd.read_csv(condition, names= ['x1','x2','y','setting'])
        x1 = df_test['x1'].values
        x2 = df_test['x2'].values

        x1_c1 = []
        x1_v = []
        x1_c2 = []
        s = []
        for stim in x1:
            x1_c1.append(stim[0])
            x1_v.append (stim[1])
            x1_c2.append(stim[2])
            s.append(stim[4:7])
        
        x2_c1 = []
        x2_v = []
        x2_c2 = []
        for stim in x2:
            x2_c1.append(stim[0])
            x2_v.append (stim[1])
            x2_c2.append(stim[2])
          
        x1_c1 = np.array(x1_c1)
        x1_v = np.array(x1_v)
        x1_c2 = np.array(x1_c2)
        x2_c1 = np.array(x2_c1)
        x2_v = np.array(x2_v)
        x2_c2 = np.array(x2_c2)
       
        df_test['x1_c1'] = x1_c1     
        df_test['x1_v'] = x1_v
        df_test['x1_c2'] = x1_c2
        df_test['x2_c1'] = x2_c1     
        df_test['x2_v'] = x2_v
        df_test['x2_c2'] = x2_c2
        df_test['setting'] = s

        output = output_dir + 'test_condition' + str(counter)
        counter = counter + 1
        generate_test(df_test,estimates,nsamples, output, mean_dispersal_factor=1, cov_shrink_factor=0,verbose=False)
        

if __name__ == '__main__':
    nsamples = 1000
    input_fname = '/Users/ingeborg/Desktop/estimates.pkl'
    shrink = 0
    dispersal = 1
    train_stimulus_dir = '/Users/ingeborg/phonrulemodel/conditions/train'
    test_stimulus_dir = '/Users/ingeborg/phonrulemodel/conditions/test'
    output_dir = '/Users/ingeborg/Desktop/'

    estimates = load_estimates(input_fname)

    train_data(train_stimulus_dir,estimates, output_dir)
    test_data(test_stimulus_dir,estimates, output_dir)