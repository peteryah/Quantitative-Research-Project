# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:22:45 2019

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
import Testing2
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import keras
import time as t
import warnings
warnings.filterwarnings("ignore")

def AvgedPredict(dist, X, Y, alphaIndex, n, portSize,date_str):
    '''Get averaged portfolio for future prediction'''
    res = []
    for i in dist:
        temp=i
        break
    indices = list(dist[temp].index.values)
    #print('test')
    date_object = datetime.datetime.strptime(date_str, '%m-%d-%Y').date()
    today = np.datetime64(date_object - relativedelta(days = date_object.day-1))
    #print(today)
    st = indices[indices.index(today)-3]
    ed = indices[indices.index(today)-1]
    #print(X['MMM'].loc[st:ed])
    raw = pd.DataFrame(dist['MMM'].loc[st:ed])
    processed = pd.DataFrame(X['MMM'].loc[st:ed])
    raw.to_excel('RawData'+str(ed)[:10]+'.xlsx')
    processed.to_excel('ProcData'+str(ed)[:10]+'.xlsx')
    inputX = {}
    inputY = {}
    for i in dist.keys():
        indices = list(dist[i].index.values)
        if (st in indices) and (ed in indices) and (today in indices):
            inputX[i] = X[i].loc[st:ed]
            inputY[i] = Y[i][indices.index(st):indices.index(ed)]
    tempX = splitterX2(inputX)
    tempY = splitterY2(inputY)
    #print(st, ed)
    mm1=t.time()
    modelss = Testing2.AvgPort(tempX, tempY, alphaIndex,n)
    mm2=t.time()
    mm=mm2-mm1
    print('one training took: '+str(mm) +' sec')
    scores = []
    m1=t.time()
    for i in dist:
        indices = list(dist[i].index.values)
        if (st in indices) and (ed in indices) and (today in indices):
            b = Testing2.AvgScore(modelss, X, alphaIndex, st,ed,i)

            scores += [[b[-1][0], i]]
    scores.sort(reverse=True)
    sscores=scores.copy()
    m2=t.time()
    m=m2-m1
    print('one model scoring took: '+str(m)+' sec')
    comps = []


    shortcomp=len(scores)-portSize
    scomps=[]
    for i in range(shortcomp, len(scores)):
        scomps+=[sscores[i][1]]

    i = 0
    j = 0
    while (len(comps)<portSize and i<len(scores)):

        comps += [scores[i][1]]
        j+=1
        i+=1
    c=pd.DataFrame(comps)
    c.to_excel('comps'+str(ed)[:10]+'.xlsx')
    return comps, scomps

def LatestPredict(dist, X, Y, alphaIndex, n):
    '''Return the result for current month'''
    today1 = date_object - relativedelta(days = date_object.day-1)
    startDate = today1 - relativedelta(months=3)
    endDate =  today1
    inputX = {}
    inputY = {}
    poped = []
    for i in dist.keys():
        indices = dist[i].index.values
        if (np.datetime64(startDate) in indices) and (np.datetime64(endDate) in indices):
            inputX[i] = X[i].loc[np.datetime64(startDate):np.datetime64(endDate-relativedelta(months=1))]
            inputY[i] = Y[i][-4:-1]
        else:
            poped += [i]
    #print(inputX)
    inputX = splitterX2(inputX)
    inputY = splitterY2(inputY)
    #print(inputX.shape, inputY.shape)
    model = train2(inputX[alphaIndex], inputY)
    to_xl(model)
    scores = []
    for i in dist.keys():
        if not (i in poped):
            b = model.predict(X[i][alphaIndex].loc[np.datetime64(startDate):np.datetime64(endDate)])
            scores += [[b[-1][0], i]]
    scores.sort(reverse=True)
    res = []
    for j in scores[:n]:
        res+=[j[1]]
    
    return res

def splitterX2(dist):
    '''Concat dataset for rolling predicting and training'''
    newT = pd.DataFrame()
    for i in dist.keys():
        train = dist[i]
        newT = pd.concat([newT, train], axis=0, ignore_index=True)
    return newT.copy()

def splitterY2(dist):
    '''Concat dataset for rolling predicting and training'''
    newT = np.asarray([])
    for i in dist.keys():
        train = dist[i]
        newT = np.append(newT, train)
    return newT.copy()

def train2(X,Y):
    '''Train and test a regular neural network'''
    X, Y = check(X,Y)
    model = models.Sequential()
    model.add(layers.Flatten(input_shape = [len(X.columns)]))
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
    model.fit(X, Y, epochs=5, verbose = 0)
    # print('Correct Prediction (%): ', accuracy_score(Y_test, model.predict(X_test), normalize=True)*100.0)
    return model

def check(X, Y):
    ''' to keep X and Y in the same length'''
    lx = len(X)
    ly = len(Y)
    if lx != ly:
        temp = min(lx,ly)
        return (X[:temp]), (Y[:temp])
    else:
        return X, Y

def to_xl(model):
    df =pd.DataFrame(model.get_weights())
    for i in range(8):
        temp = pd.DataFrame(df.copy()[0][i])
    #temp.to_csv('RCmodel.csv', index=False)
        print(temp)
        print(temp.shape)
    return
