# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:16:39 2019

@author: sheng
"""
from numba import cuda
import pandas as pd
import tensorflow as tf
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

from tensorflow.keras import datasets, layers, models
import random
import datetime
from dateutil.relativedelta import relativedelta
#import matplotlib.pyplot as plt

def Designated(Target, dist, X, Y, alphaIndex, portSize):
    startDate = Target - relativedelta(months = 3)
    endDate = Target - relativedelta(months = 1)
    inputX = {}
    inputY = {}
    poped = []
    for i in Y.keys():
        indices = dist[i].index.values
        if (np.datetime64(startDate) in indices) and (np.datetime64(endDate) in indices) and (np.datetime64(Target) in indices):
            #print(i)
            st, en = getIndex(dist[i], startDate, endDate)
            #print(st,en)
            inputX[i] = X[i].loc[np.datetime64(startDate):np.datetime64(endDate)]
            inputY[i] = Y[i][st:en]
        else:
            poped += [i]
    inputX = splitterX2(inputX)
    inputY = splitterY2(inputY)
    model = train2(inputX[alphaIndex], inputY)
    scores = []

    for i in X.keys():
        if not (i in poped):
            #print(i)
            b = model.predict(X[i][alphaIndex].loc[np.datetime64(startDate):np.datetime64(Target)])
            #print(b)
            scores += [[b[-1][0], i]]
    scores.sort(reverse=True)
    res = []
    for j in scores[:10]:
        res+=[j[1]]
    temp = []
    for k in res:
        Target2 = Target + relativedelta(months = 1)
        a = dist[k]['close'][Target]
        b = dist[k]['close'][Target2]
        pctr = (b-a)/a
        temp += [pctr]
    
    return sum(temp)/len(temp)

def getIndex(dist, start, end):
    indices = dist.index.values
    startd = np.where(indices == np.datetime64(start))[0][0]
    endd = np.where(indices == np.datetime64(end))[0][0]
    return startd, endd

def AvgPort(X, Y, alphaIndex,n):
    res = []
    for i in range(n):
        res+=[train2(X[alphaIndex],Y)]
    return res


def Backtest(numOfMonth, alphaIndex, initial, dist, X, Y, portSize, stDate):
    '''test on model and alphas on its performance on a period of time'''
    actinitial = initial
    rminitial = initial
    handinitial = initial
    #today1 = datetime.date.today() - relativedelta(days = datetime.date.today().day-1)
    dataX = [0]
    dataY = [initial]
    rmX = [0]
    rmY = [initial]
    indices = list(dist['GOOG'].index.values)
    rmports = RandomPort(numOfMonth, list(dist.keys()))
    date2 = indices.index(np.datetime64(stDate))
    for i in range(date2+1,date2+numOfMonth+1):
        
        startD = i
        startDate = indices[i]
        endDate =  indices[i+2]
        endD = i+2
        print(startDate,endDate)
        tempDist, tempX, tempY = ExtractDist(dist.copy(), X.copy(), Y.copy(), startDate, endDate, startD, endD)
        tempX = splitterX2(tempX)
        tempY = splitterY2(tempY)
        
            
        
        if tempX.empty == False:
            models = AvgPort(tempX, tempY, alphaIndex,10)
            #model = train2(tempX[alphaIndex], tempY)
            
            Portfolio, AvgPctr = SelectAndPCTR(models, dist, X.copy(), alphaIndex, startD, endD, startDate, endDate, portSize)
            print(Portfolio, AvgPctr) 
           
            

            
            if AvgPctr <0.2:
                initial = initial*(1+AvgPctr)
                
                dataX += [i-date2]
                dataY += [initial]
                
            else:
                dataX += [i]
                dataY += [initial]
                
    '''
    plt.plot(np.asarray(handX), np.asarray(handY), label = 'Projected Return')
    plt.xlabel('Months')
    plt.ylabel('Money')
    plt.title("Change of money on the past 2 years")
    plt.legend()
    plt.show()
    print('Initial Money:', actinitial, 'Resulting Money:', initial)
    '''
    
    return dataX, dataY, initial

def CalcPctr(companies, date, dist):
    '''calculate percentage return of certain portfolio on certain date'''
    temp = 0
    for j in companies:
        if (np.datetime64(date) in dist[j].index.values) and (np.datetime64(date+relativedelta(days=1)) in dist[j].index.values):
            pctr = (dist[j]['close'][np.datetime64(date+relativedelta(days = 1))] - dist[j]['close'][np.datetime64(date)]) / dist[j]['close'][np.datetime64(date)]
            #print(j, pctr)
            temp += pctr / len(companies)
    return temp

def ExtractDist(dist, X, Y, startDate, endDate, startD, endD):
    '''Extract needed data for rolling window of training set'''
    tempDist = {}
    tempX = {}
    tempY = {}
    for j in dist.keys():
        indices = dist[j].index.values
        #if (np.datetime64(startDate) in indices) and (np.datetime64(endDate) in indices):
        
        #startD, endD = caliDate(startD, endD, startDate, endDate, indices)
        tempDist[j] = dist[j].loc[startDate:endDate]
        tempX[j] = X[j].loc[startDate:endDate]
        tempY[j] = Y[j][startD:endD+1]
    return tempDist.copy(), tempX.copy(), tempY.copy()

def checkdate(start, startc, end, endc):
    '''check if data of index is available for certain company'''
    start = np.datetime64(start)
    end = np.datetime64(end)
    if start >= startc and end <= endc:
        return True
    else:
        return False
    
def AvgScore(models, X, alphaIndex, d1, d2,i):
    total=0
    for j in range(len(models)):
        score = models[j].predict(X[i][alphaIndex].loc[d1:d2])
        total+=score
    return total / len(models)

def SelectAndPCTR(models, dist, X, alphaIndex, startD, endD, startDate, endDate, portSize):
    '''Use model to predict on alpha values and select portfolio'''
    scores = []
    for i in dist:
        indices = list(dist[i].index.values)
        if endDate in indices:
            ind = indices.index(endDate)
            if (len(indices)-ind)>2:
                
                a = (dist[i]['close'].loc[indices[ind+2]] - dist[i]['close'].loc[indices[ind+1]])/dist[i]['close'].loc[indices[ind+1]]
                b = AvgScore(models, X, alphaIndex, endDate,indices[ind+1],i)
                #b = model.predict(X[i][alphaIndex].loc[endDate:indices[ind+1]])
                scores += [[b[-1][0], i, a]]
    scores.sort(reverse=True)
    #print(b[-1][0])
    temp = 0
    comps = []
    i = 0
    j = 0
    while (len(comps)<portSize and i<len(scores)):
        if scores[i][2]<0.5:
            temp+=scores[i][2]/portSize
            comps += [scores[i][1]]
            j+=1
        i+=1
    return comps, temp

def dateCF(date, n):
    return np.datetime64(date+relativedelta(days=n))

def caliDate(startD, endD, startDate, endDate, indices):
    '''calibrate date of index'''
    sd = np.datetime64(startDate)
    ind = np.where(indices == sd)[0]
    if len(ind) == 0:
        return startD, endD
    else:
        return ind[0], ind[0]+2
    
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

def check(X, Y):
    ''' to keep X and Y in the same length'''
    lx = len(X)
    ly = len(Y)
    if lx != ly:
        temp = min(lx,ly)
        return (X[:temp]), (Y[:temp])
    else:
        return X, Y
    
def train2(X,Y):
    '''Train and test a regular neural network'''
    X, Y = check(X,Y)
    model = models.Sequential()
    model.add(layers.Flatten(input_shape = [len(X.columns)]))
    model.add(layers.Dense(512, activation = tf.nn.relu))
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

def RandomPort(months, keys):
    '''Generate Random Portfolios'''
    syms = keys
    ports = []
    for i in range(months):

        random.shuffle(syms)
        ports += [syms[:10]]
    return ports
