# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:25:35 2019

@author: sheng
"""

import pandas as pd
import tensorflow as tf
import numpy as np
import yfinance as yf
import stockstats as SS
from tensorflow.contrib import rnn
from tensorflow.keras import datasets, layers, models
import random
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import alp1 # import alpha features in alp1.py

def SingleAlpha(X,Y, alphasize):
    '''Single alpha machine to select alphas for actual predicting model'''
    X_train, X_test = splitterX(X)
    Y_train, Y_test = splitterY(Y)
    X_train, Y_train = check(X_train, Y_train)
    X_test, Y_test = check(X_test, Y_test)
    alphas = X_train.columns
    alphaAcc = []
    for i in alphas:
        #Train and Test on single alphas
        modeli, iacc, t = trainSingleAlpha(X_train[i], X_test[i], Y_train, Y_test)
        alphaAcc+= [[iacc, i, t]]
        print(i)
    alphaAcc.sort(reverse=True)
    #print(alphaAcc)
    selectedAlphas = []
    for j in range(len(alphaAcc)):
        if alphaAcc[j][2]==1:
            selectedAlphas += [alphaAcc[j]]
    if len(selectedAlphas) >=alphasize:
        selectedAlphas = selectedAlphas[:alphasize]
    alphaIndex = extractAlpha(selectedAlphas)
    return alphaIndex

def extractAlpha(lis):
    '''get alpha names from 2-d list'''
    res = []
    for i in lis:
        res+= [i[1]]
    return res

def splitterX(dist):
    '''Concat dataset for training and testing'''
    newT = pd.DataFrame()
    newR = pd.DataFrame()
    for i in dist.keys():
        cut = int(dist[i].shape[0]*0.8)
        train = dist[i].iloc[:cut]
        test = dist[i].iloc[cut:]
        newT = pd.concat([newT, train], axis=0, ignore_index=True)
        newR = pd.concat([newR, test], axis=0, ignore_index=True)
    return newT.copy(), newR.copy()

def splitterY(dist):
    '''Concat dataset for training and testing'''
    newT = np.asarray([])
    newR = np.asarray([])
    for i in dist.keys():
        cut = int(dist[i].shape[0]*0.8)
        train = dist[i][:cut]
        test = dist[i][cut:]
        newT = np.append(newT, train)
        newR = np.append(newR, test)
    return newT.copy(), newR.copy()

def check(X, Y):
    ''' to keep X and Y in the same length'''
    lx = len(X)
    ly = len(Y)
    if lx != ly:
        temp = min(lx,ly)
        return (X[:temp]), (Y[:temp])
    else:
        return X, Y
    
def trainSingleAlpha(X_train, X_test, Y_train, Y_test):
    '''Train and test a regular neural network'''
    model = models.Sequential()
    model.add(layers.Flatten(input_shape = [1]))
    model.add(layers.Dense(512, activation = tf.nn.relu))
    model.add(layers.Dense(256, activation = tf.nn.relu))
    model.add(layers.Dense(16, activation=tf.nn.relu))
    model.add(layers.Dense(8, activation=tf.nn.softmax))
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=5, verbose = 0)
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose = 0)
    temp = 1
    if pd.isna(test_loss):
        temp = 0
    return model, test_acc, temp

def train(X_train, X_test, Y_train, Y_test):
    '''Train and test a regular neural network'''
    model = models.Sequential()
    model.add(layers.Flatten(input_shape = [len(X_train.columns)]))
    #model.add(layers.Dense(28000, activation = tf.nn.relu))
    #model.add(layers.Dense(20000, activation = tf.nn.relu))
    model.add(layers.Dense(512, activation = tf.nn.relu))
    #model.add(layers.Dense(12000, activation = tf.nn.relu))
    #model.add(layers.Dense(8000, activation = tf.nn.relu))
    #model.add(layers.Dense(6000, activation = tf.nn.relu))
    #model.add(layers.Dense(4000, activation = tf.nn.relu))
    #model.add(layers.Dense(2000, activation = tf.nn.relu))
    #model.add(layers.Dense(1000, activation = tf.nn.relu))
    #model.add(layers.Dense(500, activation = tf.nn.relu))
    model.add(layers.Dense(256, activation = tf.nn.relu))
    model.add(layers.Dense(16, activation=tf.nn.relu))
    model.add(layers.Dense(8, activation=tf.nn.softmax))
    # model.summary()
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=5, verbose = 0)
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose = 0)
    
    # print('Correct Prediction (%): ', accuracy_score(Y_test, model.predict(X_test), normalize=True)*100.0)
    return model, test_acc
