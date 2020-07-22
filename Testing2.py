# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:16:39 2019

@author: sheng
"""
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
            st, en = getIndex(dist[i], startDate, endDate)
            inputX[i] = X[i].loc[np.datetime64(startDate):np.datetime64(endDate)]
            inputY[i] = Y[i][st:en]
        else:
            poped += [i]
    
    inputX = splitterX2(inputX)
    inputY = splitterY2(inputY)
    models = []
    for i in range(10):
        models += [train2(inputX[alphaIndex], inputY)]
    scores = []
    for i in X.keys():
        if not (i in poped):
            b = AvgScore(models, X, alphaIndex, startDate, endDate,i)
            scores += [[b[-1][0], i]]
    scores.sort(reverse=True)
    res = []
    for j in scores[:portSize]:
        res+=[j[1]]
    '''
    temp = []
    for k in res:
        Target2 = Target + relativedelta(months = 1)
        a = dist[k]['close'][Target]
        b = dist[k]['close'][Target2]
        pctr = (b-a)/a
        temp += [pctr]
    scor = []
    for l in range(portSize):
        scor += [scores[l][0]]
    '''
    
    return res

def getIndex(dist, start, end):
    indices = dist.index.values
    startd = np.where(indices == np.datetime64(start))[0][0]
    endd = np.where(indices == np.datetime64(end))[0][0]
    return startd, endd

def Backtest(numOfMonth, alphaIndex, initial, dist, X, Y, portSize,n=10, stDate = datetime.date.today() - relativedelta(days = datetime.date.today().day-1)):
    '''test on model and alphas on its performance on a period of time'''
    actinitial = initial
    rminitial = initial
    dataX = [0]
    dataY = [initial]
    rmX = [0]
    rmY = [initial]
    
    rmports = RandomPort(numOfMonth-1, list(dist.keys()))
    Dates = []
    Ports = []
    Pctrs = []
    for i in range(1,numOfMonth-1):

        startD = i
        startDate = stDate - relativedelta(months=numOfMonth-i-1)
        endDate =  stDate - relativedelta(months=numOfMonth-i-3)
        endD = i+2
        print(startDate,endDate)
        tempDist, tempX, tempY = ExtractDist(dist.copy(), X.copy(), Y.copy(), startDate, endDate, startD, endD)
        tempX = splitterX2(tempX)
        tempY = splitterY2(tempY)
        #tempX, tempY = splitter22(X,Y)
        if tempX.empty == False:
            models = AvgPort(tempX, tempY, alphaIndex,n)
            Portfolio, AvgPctr = SelectAndPCTR(models, dist, X.copy(), alphaIndex, startD, endD, startDate, endDate, portSize)
            Ports+=[PortConcat(Portfolio)]
            Dates+= [(stDate - relativedelta(months=numOfMonth-i-4))]
            Pctrs+=[AvgPctr]
            print(Portfolio, AvgPctr) 
            rmport = rmports[i]
            rmpctr = CalcPctr(rmport, endDate + relativedelta(months = 1), dist)
            
            


            
            if AvgPctr <0.2:
                initial = initial*(1+AvgPctr)
                rminitial  = rminitial * (1+rmpctr)
                dataX += [i]
                dataY += [initial]
                rmX += [i]
                rmY += [rminitial]
            else:
                dataX += [i]
                dataY += [initial]
                rmX += [i]
                rmY += [rminitial]
    '''
    plt.plot(np.asarray(handX), np.asarray(handY), label = 'Projected Return')
    plt.xlabel('Months')
    plt.ylabel('Money')
    plt.title("Change of money on the past 2 years")
    plt.legend()
    plt.show()
    print('Initial Money:', actinitial, 'Resulting Money:', initial)
    '''
    dff = pd.DataFrame()
    
    dff['Date'] = Dates
    dff['Portfolio'] = Ports
    dff['Percentage Return'] = Pctrs
    dff.to_excel('Metrics.xlsx',index=False)
    
    return dataX, dataY, initial, rmX, rmY

def PortConcat(lis):
    res = ''
    for i in lis:
        res+=i+', '
    return res[:-2]

def AvgPort(X, Y, alphaIndex,n):
    res = []
    for i in range(n):
        res+=[train2(X[alphaIndex],Y)]
    return res
    

def CalcPctr(companies, date, dist):
    '''calculate percentage return of certain portfolio on certain date'''
    temp = 0
    for j in companies:
        if (np.datetime64(date) in dist[j].index.values) and (np.datetime64(date+relativedelta(months=1)) in dist[j].index.values):
            pctr = (dist[j]['close'][np.datetime64(date+relativedelta(months = 1))] - dist[j]['close'][np.datetime64(date)]) / dist[j]['close'][np.datetime64(date)]
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
            if (np.datetime64(startDate) in indices) and (np.datetime64(endDate) in indices):
                if checkdate(startDate, indices[0], endDate+relativedelta(months=2), indices[-1]):
                    '''Data is valid for selected time'''
                    
                    startD, endD = caliDate(startD, endD, startDate, endDate, indices)
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

def SelectAndPCTR(models, dist, X, alphaIndex, startD, endD, startDate, endDate, portSize):
    '''Use model to predict on alpha values and select portfolio'''
    scores = []
    for i in dist:
        indices = dist[i].index.values
        if (dateCF(startDate,0) in indices) and (dateCF(endDate,0) in indices) and (dateCF(endDate,1) in indices) and (dateCF(endDate,2) in indices):
            if checkdate(startDate, indices[0], dateCF(endDate,2), indices[-1]):
                
                a = (dist[i]['close'].loc[dateCF(endDate,2)] - dist[i]['close'].loc[dateCF(endDate,1)])/dist[i]['close'].loc[dateCF(endDate,1)]
                b = AvgScore(models, X, alphaIndex, dateCF(endDate,0),dateCF(endDate,1),i)
                #b = model.predict(X[i][alphaIndex].loc[dateCF(endDate,0):dateCF(endDate,1)])
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

def AvgScore(models, X, alphaIndex, d1, d2,i):
    total=0
    for j in range(len(models)):
        score = models[j].predict(X[i][alphaIndex].loc[d1:d2])
        total+=score
    return total / len(models)
    

def dateCF(date, n):
    return np.datetime64(date+relativedelta(months=n))

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

def splitter22(X, Y):
    newX = pd.DataFrame()
    newY = np.asarray([])
    for i in X.keys():
        tempX = X[i].copy()
        tempY = Y[i].copy()
        tempDateX = []
        tempDateY = []
        indices = list(tempX.index.values)
        
        for j in range(len(indices)):
            if tempY[j] == -1:
                tempDateX+=[indices[j]]
                tempDateY+=[j]
        
        tempX = tempX.drop(tempDateX).copy()
        
        tempY = np.delete(tempY, tempDateY).copy()
        
        newX = pd.concat([newX, tempX], axis=0, ignore_index=True)
        newY = np.append(newY, tempY)
    return newX, newY
        

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
        ports += [syms[:20]]
    return ports
