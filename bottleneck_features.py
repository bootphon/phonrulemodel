#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: bottleneck_features.py
# date: Mon May 25 15:21 2015
# author:
# Maarten Versteegh
# github.com/mwv
# maartenversteegh AT gmail DOT com
#
# Licensed under GPLv3
# ------------------------------------
"""bottleneck_features:

Generate datasets containing bnf instead of MFCCs for all 
experimental conditions, both for training and test phase
"""
from __future__ import division

import glob
import os.path as path

from lasagne.layers import \
    get_output, get_all_layers, get_output_shape

from dnn import load_model

import numpy as np
from sklearn import svm

def get_bottleneck_features(network, X):
    layers = get_all_layers(network)
    bottleneck = [l for l in layers if l.name == 'bottleneck']
    if len(bottleneck) == 0:
        raise ValueError('network has no bottleneck')
    else:
        bottleneck = bottleneck[0]
    bfeatures = get_output(bottleneck, X)
    #print X.shape
    #print bfeatures.eval().shape
    return bfeatures



if __name__ == '__main__':
    #TODO: Check if output is correct
    fmodel1 = '/Users/Research/projects/phonrulemodel/model_69_1.pkl'
    fmodel2 = '/Users/Research/projects/phonrulemodel/model_69_2.pkl'
    dir_trainsets ='/Users/Research/projects/phonrulemodel/trainsets'
    dir_testsets = '/Users/Research/projects/phonrulemodel/testsets'
    output_dir_train = '/Users/Research/projects/phonrulemodel/bnftrainsets/'
    output_dir_test = '/Users/Research/projects/phonrulemodel/bnftestsets/'
    
    #Replace MFCCs in training datasets by bnf's
    counter = 1

    for condition in glob.iglob(path.join(dir_trainsets, '*.npz')):
        dataset = np.load(condition)
        X, y, labels = dataset['X'], dataset['y'], dataset['labels']

        #Split X = [c1,v,c2] into c1, v & c2 - arrays
        c1 = []
        v = []
        c2 = []
        for i in range(len(X)):
            c1.append(X[i][0])
            v.append(X[i][1])
            c2.append(X[i][2])

        #Generate BNFs with model 1
        model = load_model(fmodel1)
        #for layer in get_all_layers(model):
        #    print layer.name, get_output_shape(layer)
         
        bnf_c1 = get_bottleneck_features(model, c1)
        bnf_v = get_bottleneck_features(model, v)
        bnf_c2 = get_bottleneck_features(model, c2)

        #Merge the arrays together
        X = np.column_stack((bnf_c1,bnf_v,bnf_c2))

        #Save new datasets
        output = output_dir_train + 'train_condition' + str(counter) + 'model1'
        np.savez(output, X=X, y = y, labels = labels)

        #Generate BNFs with model 2
        model = load_model(fmodel2)
        #for layer in get_all_layers(model):
        #    print layer.name, get_output_shape(layer)

        bnf_c1 = get_bottleneck_features(model, c1)
        bnf_v = get_bottleneck_features(model, v)
        bnf_c2 = get_bottleneck_features(model, c2)

        #Merge the arrays together
        X = np.column_stack((bnf_c1,bnf_v,bnf_c2))

        #Save new datasets
        output = output_dir_train + 'train_condition' + str(counter) + 'model2'
        np.savez(output, X=X, y = y, labels = labels)
     
        counter = counter + 1

    #repeat for testsets
    counter = 1
    for condition in glob.iglob(path.join(dir_testsets, '*.npz')):
        dataset = np.load(condition)
        X, y, labels = dataset['X'], dataset['y'], dataset['labels']

        #Split X = [[c1,v,c2],[c1,v,c2] into c1, v & c2 - arrays
        c1_1 = []
        v_1 = []
        c2_1 = []
        c1_2 = []
        v_2 = []
        c2_2 = []
        for i in range(len(X)):
            c1_1.append(X[i][0][0])
            v_1.append(X[i][0][1])
            c2_1.append(X[i][0][2])
            c1_2.append(X[i][1][0])
            v_2.append(X[i][1][1])
            c2_2.append(X[i][1][2])

        #Generate BNFs with model 1
        model = load_model(fmodel1)
        #for layer in get_all_layers(model):
        #    print layer.name, get_output_shape(layer)
         
        bnf_c1_1 = get_bottleneck_features(model, c1_1)
        bnf_v_1 = get_bottleneck_features(model, v_1)
        bnf_c2_1 = get_bottleneck_features(model, c2_1)
        bnf_c1_2 = get_bottleneck_features(model, c1_2)
        bnf_v_2 = get_bottleneck_features(model, v_2)
        bnf_c2_2 = get_bottleneck_features(model, c2_2)

        #Merge the arrays together
        bnf_1 = np.column_stack((bnf_c1_1,bnf_v_1,bnf_c2_1))
        bnf_2 = np.column_stack((bnf_c1_2,bnf_v_2,bnf_c2_2))

        X = [bnf_1, bnf_2]

        #Save new datasets
        output = output_dir_test + 'test_condition' + str(counter) + 'model1'
        np.savez(output, X=X, y = y, labels = labels)

        #Generate BNFs with model 2
        model = load_model(fmodel2)
        #for layer in get_all_layers(model):
        #    print layer.name, get_output_shape(layer)

        bnf_c1_1 = get_bottleneck_features(model, c1_1)
        bnf_v_1 = get_bottleneck_features(model, v_1)
        bnf_c2_1 = get_bottleneck_features(model, c2_1)
        bnf_c1_2 = get_bottleneck_features(model, c1_2)
        bnf_v_2 = get_bottleneck_features(model, v_2)
        bnf_c2_2 = get_bottleneck_features(model, c2_2)

        #Merge the arrays together
        bnf_1 = np.column_stack((bnf_c1_1,bnf_v_1,bnf_c2_1))
        bnf_2 = np.column_stack((bnf_c1_2,bnf_v_2,bnf_c2_2))

        X = [bnf_1, bnf_2]

        #Save new datasets
        output = output_dir_test + 'test_condition' + str(counter) + 'model2'
        np.savez(output, X=X, y = y, labels = labels)
     
        counter = counter + 1
   