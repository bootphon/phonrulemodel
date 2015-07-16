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
        print dataset
        X, y, labels, info = dataset['X'], dataset['y'], dataset['labels'], dataset['info']

        #Generate BNFs with model 1
        model = load_model(fmodel1)
        #for layer in get_all_layers(model):
        #    print layer.name, get_output_shape(layer)

        bnf_X = get_bottleneck_features(model, X)
        bnf_Y = get_bottleneck_features(model, y)

        #Save new datasets
        output = output_dir_train + 'train_condition' + str(counter) + 'model1'
        np.savez(output, X=X, y = y, labels = labels, info = info)

        #Generate BNFs with model 2
        model = load_model(fmodel2)
        #for layer in get_all_layers(model):
        #    print layer.name, get_output_shape(layer)

        bnf_X = get_bottleneck_features(model, X)
        bnf_Y = get_bottleneck_features(model, y)

        #Save new datasets
        output = output_dir_train + 'train_condition' + str(counter) + 'model2'
        np.savez(output, X=X, y = y, labels = labels, info = info)
     
        counter = counter + 1

    #repeat for testsets
    counter = 1
    for condition in glob.iglob(path.join(dir_testsets, '*.npz')):
        dataset = np.load(condition)
        X1, X2, y1, y2, labels, info = dataset['X1'], dataset['X2'],dataset['y1'], dataset['y2'], dataset['labels'], dataset['info']

        #Generate BNFs with model 1
        model = load_model(fmodel1)
        #for layer in get_all_layers(model):
        #    print layer.name, get_output_shape(layer)
     
        bnf_X1 = get_bottleneck_features(model, X1)
        bnf_X2 = get_bottleneck_features(model, X2)
        bnf_y1 = get_bottleneck_features(model, y1)
        bnf_y2 = get_bottleneck_features(model, y2)
        #Save new datasets
        output = output_dir_test + 'test_condition' + str(counter) + 'model1'
        np.savez(output, X1=bnf_X1, X2 = bnf_X2, y1 = bnf_y1, y2 = bnf_y2, labels = labels, info = info)

        #Generate BNFs with model 2
        model = load_model(fmodel2)
        #for layer in get_all_layers(model):
        #    print layer.name, get_output_shape(layer)

        bnf_X1 = get_bottleneck_features(model, X1)
        bnf_X2 = get_bottleneck_features(model, X2)
        bnf_y1 = get_bottleneck_features(model, y1)
        bnf_y2 = get_bottleneck_features(model, y2)

        #Save new datasets
        output = output_dir_test + 'test_condition' + str(counter) + 'model2'
        np.savez(output, X1=bnf_X1, X2 = bnf_X2, y1 = bnf_y1, y2 = bnf_y2, labels = labels, info = info)
     
        counter = counter + 1
   