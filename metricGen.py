
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
import Testing2 as T
import Prediction
import Alphas
import Process
import warnings
import json
import Main


def metric2x2(c):
    
    dist = Main.YFI(0)
    X, Y = Process.ProcessData(dist, 2)
    Main.data_to_local(X,Y,dist)
    
    X,Y,dist = Main.local_to_data()
    goodAlphas = ['alpha083','alpha101','alpha024','alpha042','alpha028','alpha025','alpha018','alpha010','alpha047','alpha033','alpha009','alpha005','alpha051']
    alphaIndex = goodAlphas
    Date = datetime.datetime(year = 2019, month = 6, day = 1)
    #scores, port, pctrs = T.Designated(Date, dist, X, Y, alphaIndex,10)
   
    suc = 0
    if c==1:
        for i in range(100):
            a,b,c = T.Designated(Date, dist, X, Y, alphaIndex,10)
            c=sum(c)/len(c)
            if c>=0:
                suc += 1
        print('20 attempt to predict, ' + str(suc) + 'successes on ' + Date.isoformat())
        return
    suc = 0
    if c==2:
        for i in range(200):
            a,b,c = T.Designated(Date, dist, X, Y, alphaIndex,10)
            c = classify(c)
            suc+=sum(c)
        print('Among 2000 selected companies, ' + str(suc) + ' of them has positive percentage return')
    return

def classify(lis):
    temp = []
    for i in lis:
        if i >= 0:
            temp+=[1]
        else:
            temp+=[0]
    return temp

def metric4x4(c):
    
    dist = Main.YFI(0)
    X, Y = Process.ProcessData(dist, 4)
    Main.data_to_local(X,Y,dist)
    
    X,Y,dist = Main.local_to_data()
    goodAlphas = ['alpha083','alpha101','alpha024','alpha042','alpha028','alpha025','alpha018','alpha010','alpha047','alpha033','alpha009','alpha005','alpha051']
    alphaIndex = goodAlphas
    Date = datetime.datetime(year = 2019, month = 6, day = 1)
    #scores, port, pctrs = T.Designated(Date, dist, X, Y, alphaIndex,10)
    '''
    suc = 0
    if c==1:
        for i in range(200):
            a,b,c = T.Designated(Date, dist, X, Y, alphaIndex,10)
            c=sum(c)/len(c)
            if c>=0:
                suc += 1
        print('200 attempt to predict, ' + str(suc) + 'successes on ' + Date.isoformat())
        return
    suc = 0
    if c==2:
        for i in range(200):
            a,b,c = T.Designated(Date, dist, X, Y, alphaIndex,10)
            c = classify(c)
            suc+=sum(c)
        print('Among 2000 selected companies, ' + str(suc) + ' of them has positive percentage return')
    '''
    a,b,c = T.Designated(Date, dist, X, Y, alphaIndex,10)
    print(a)
    print(b)
    print(c)
    print(sum(c)/len(c))
    return

metric2x2(1)
