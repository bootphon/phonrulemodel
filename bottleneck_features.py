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
     
        #print X.shape[0]
        #print X[0].shape
        #print X[0][0]
        #print X[127][0]
        #print y.shape
        #print info.shape
        #Generate BNFs with model 1
        model = load_model(fmodel1)
        #for layer in get_all_layers(model):
        #    print layer.name, get_output_shape(layer)
        bnf_X = []
        bnf_Y = []
        bnf_info = []

        length1 = X.shape[0]
        length1 = 1
        length2 = X[0].shape[0]
        length2 = 1
        for i in range(length1):
            for j in range(length2):
                #print j
                bnfx = get_bottleneck_features(model, X[i][j])
                bnf_X.append(bnfx)
                bnfy = get_bottleneck_features(model, y[i][j])
                bnf_Y.append(bnfy)
                bnf_info.append(info[i])

        #Save new datasets
        output = output_dir_train + 'train_condition' + str(counter) + 'model1'
        np.savez(output, X=bnf_X, y = bnf_Y, labels = labels, info = bnf_info)

        #Generate BNFs with model 2
        model = load_model(fmodel2)
        #for layer in get_all_layers(model):
        #    print layer.name, get_output_shape(layer)

        bnf_X = []
        bnf_Y = []
        bnf_info = []
        length1 = X.shape[0]
        #length1 = 1
        length2 = X[0].shape[0]
        #length2 = 1
        for i in range(length1):
            for j in range(length2):
                #print j
                bnfx = get_bottleneck_features(model, X[i][j])
                bnf_X.append(bnfx)
                bnfy = get_bottleneck_features(model, y[i][j])
                bnf_Y.append(bnfy)
                bnf_info.append(info[i])

        #Save new datasets
        output = output_dir_train + 'train_condition' + str(counter) + 'model2'
        np.savez(output, X=bnf_X, y = bnf_Y, labels = labels, info = bnf_info)
     
        counter = counter + 1

    #repeat for testsets
    counter = 1
    for condition in glob.iglob(path.join(dir_testsets, '*.npz')):
        dataset = np.load(condition)
        X1, X2, y1, y2, labels, info = dataset['X1'], dataset['X2'],dataset['y1'], dataset['y2'], dataset['labels'], dataset['info']

       # print X1.shape
       # print X2.shape
       # print y1.shape
       # print y2.shape
       # print info.shape
        #Generate BNFs with model 1
        model = load_model(fmodel1)
        #for layer in get_all_layers(model):
        #    print layer.name, get_output_shape(layer)
        
        bnf_X1 = []
        bnf_Y1 = []
        bnf_X2 = []
        bnf_Y2 = []
        bnf_info = []
        length1 = X1.shape[0]
        #length1 = 1
        length2 = X1[0].shape[0]
        #length2 = 1
        for i in range(length1):
            for j in range(length2):
                #print j
                bnfx1 = get_bottleneck_features(model, X[i][j])
                bnf_X1.append(bnfx1)
                bnfy1 = get_bottleneck_features(model, y[i][j])
                bnf_Y1.append(bnfy1)

                bnfx2 = get_bottleneck_features(model, X[i][j])
                bnf_X2.append(bnfx2)
                bnfy2 = get_bottleneck_features(model, y[i][j])
                bnf_Y2.append(bnfy2)

                bnf_info.append(info[i])

        #Save new datasets
        output = output_dir_test + 'test_condition' + str(counter) + 'model1'
        np.savez(output, X1=bnf_X1, X2 = bnf_X2, y1 = bnf_Y1, y2 = bnf_Y2, labels = labels, info = info)

        #Generate BNFs with model 2
        model = load_model(fmodel2)
        #for layer in get_all_layers(model):
        #    print layer.name, get_output_shape(layer)

        bnf_X1 = []
        bnf_Y1 = []
        bnf_X2 = []
        bnf_Y2 = []
        bnf_info = []
        length1 = X1.shape[0]
        #length1 = 1
        length2 = X1[0].shape[0]
        #length2 = 1
        for i in range(length1):
            for j in range(length2):
                #print j
                bnfx1 = get_bottleneck_features(model, X[i][j])
                bnf_X1.append(bnfx1)
                bnfy1 = get_bottleneck_features(model, y[i][j])
                bnf_Y1.append(bnfy1)

                bnfx2 = get_bottleneck_features(model, X[i][j])
                bnf_X2.append(bnfx2)
                bnfy2 = get_bottleneck_features(model, y[i][j])
                bnf_Y2.append(bnfy2)

                bnf_info.append(info[i])


        #Save new datasets
        output = output_dir_test + 'test_condition' + str(counter) + 'model2'
        np.savez(output, X1=bnf_X1, X2 = bnf_X2, y1 = bnf_Y1, y2 = bnf_Y2, labels = labels, info = info)
     
        counter = counter + 1
   