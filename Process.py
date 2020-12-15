# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:29:47 2019

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
import time as timi
import warnings
warnings.filterwarnings("ignore")

def ProcessData(dist, choice):
    '''Prepare X and Y for model'''
    yx=timi.time()
    print('Calculating alpha values......')
    print('This may take up to 4.5 minutes......')
    print()
    X = GetAlphasAll(dist)
    yx2=timi.time()
    xx=int(yx2-yx)/60
    print('Calculation finished and took ' +str(xx)+' minutes')
    yt=timi.time()
    print('Starting to rate companies by 0 or 1......')
    Y = TrueYTransform(dist, choice)
    yt2=timi.time()
    yy=yt2-yt
    print('Rating finished and took '+str(int(yy))+' secs')
    
    return X.copy(), Y.copy()

def TrueYTransform(dist, choice):
    '''rescale price (Y) into 0/1'''
    if choice == 0:
        '''Decrease in price = 0, increase or equal = 1'''
        new = choice0(dist)
    elif choice == 1:
        '''If percentage return is higher than that of S&P 500, Y = 1, otherwise Y = 0'''
        new = choice1(dist)
    elif choice == 2:
        '''Top x% pctr companies have Y = 1, others have Y = 0'''
        new = choice2(dist, 0.5)
    elif choice == 3:
        '''PCTR > rolling S&P = 1'''
        new = choice3(dist)
    elif choice == 4:
        new = choice4(dist)
    elif choice == 5:
        new = choice5(dist)
        
    return new


def choice0(dist):
    '''Decrease in price = 0, increase or equal = 1'''
    new = {}
    for i in dist.keys():
        new[i]=np.asarray([])
        for j in range(dist[i].shape[0]-1):
            temp1 = dist[i]['close'][j]
            temp2 = dist[i]['close'][j+1]
            if temp1 < temp2:
                new[i] = np.append(new[i], [0])
            else:
                new[i] = np.append(new[i], [1])
    return new

def choice1(dist):
    '''If percentage return is higher than that of S&P 500, Y = 1, otherwise Y = 0'''
    new = {}
    sppctrs = SPpctr()
    for i in dist.keys():
        new[i]=np.asarray([])
        indices = dist[i].index.values
        #if indices[0] == np.datetime64('2014-08-01T00:00:00.000000000'):
        #    indices = indices[1:]
        #print(i, indices)
        #print(list(sppctrs))
        for j in indices[:-1]:
            k = list(indices).index(j)
            temp1 = dist[i]['close'].loc[j]
            temp2 = dist[i]['close'].loc[indices[k+1]]
            pctr = (temp2-temp1)/temp1
            
            if pctr <= sppctrs.loc[j]:
                new[i] = np.append(new[i], [0])
            else:
                new[i] = np.append(new[i], [1])
    return new

def choice2(dist, cut):
    '''Top x% pctr companies have Y = 1, others have Y = 0'''
    pctrs = {}
    temp=''
    for i in dist.keys():
        pctrs[i] = GetAlphas(dist[i])['pctr']
        temp=i
    #new = {}
    #temp = pctrs['AES'].shape[0]
    indices = pctrs[temp].index.values
    for j in range(len(indices)):
        allpctr = []
        availCompanies = []
        for i in dist.keys():
            indi = pctrs[i].index.values
            if indices[j] in indi:
                allpctr += [pctrs[i].loc[indices[j]]]
                availCompanies += [i]
        allpctr.sort(reverse = True)
        splitter = int(cut*len(allpctr))
        cutter = allpctr[splitter]
        for i in availCompanies:
            tem = pctrs[i].loc[indices[j]]
            if tem < cutter:
                pctrs[i].loc[indices[j]] = 1
            else:
                pctrs[i].loc[indices[j]] = 0
    for i in pctrs.keys():
        pctrs[i] = np.asarray(pctrs[i])
    return pctrs

def choice3(dist):
    '''PCTR > rolling S&P = 1'''
    cutter =( ((3000-2470)/2470) ** (1/float(24))) -1
    new = {}
    for i in dist.keys():
        new[i]=np.asarray([])
        indices = dist[i].index.values
        if indices[0] == np.datetime64('2014-07-01T00:00:00.000000000'):
            indices = indices[1:]
        #print(i, indices)
        for j in range(indices.shape[0]-1):
            temp1 = dist[i]['close'][j]
            temp2 = dist[i]['close'][j+1]
            pctr = (temp2-temp1)/temp1
            
            if pctr <= cutter:
                new[i] = np.append(new[i], [0])
            else:
                new[i] = np.append(new[i], [1])
    return new

def choice4(dist):
    
    pctrs = {}
    for i in dist.keys():
        pctrs[i] = GetAlphas(dist[i])['pctr']
    #new = {}
    #temp = pctrs['AES'].shape[0]  
    indices = pctrs['MSFT'].index.values
    for j in range(len(indices)):
        allpctr = []
        availCompanies = []
        for i in dist.keys():
            indi = pctrs[i].index.values
            if indices[j] in indi:
                allpctr += [pctrs[i].loc[indices[j]]]
                availCompanies += [i]
        allpctr.sort(reverse = True)
        splitter1 = int(0.2*len(allpctr))
        splitter2 = int(0.4*len(allpctr))
        splitter3 = int(0.6*len(allpctr))
        splitter4 = int(0.8*len(allpctr))
        cutter1 = allpctr[splitter1]
        cutter2 = allpctr[splitter2]
        cutter3 = allpctr[splitter3]
        cutter4 = allpctr[splitter4]
        for i in availCompanies:
            tem = pctrs[i].loc[indices[j]]
            if tem > splitter1:
                pctrs[i].loc[indices[j]] = 0
            elif tem > splitter2:
                pctrs[i].loc[indices[j]] = 1
            elif tem > splitter3:
                pctrs[i].loc[indices[j]] = 2
            elif tem > splitter4:
                pctrs[i].loc[indices[j]] = 3
            else:
                pctrs[i].loc[indices[j]] = 4
    for i in pctrs.keys():
        pctrs[i] = np.asarray(pctrs[i])
    return pctrs

def choice5(dist):
    
    pctrs = {}
    for i in dist.keys():
        pctrs[i] = GetAlphas(dist[i])['pctr']
        
    #new = {}
    #temp = pctrs['AES'].shape[0]  
    indices = pctrs['MSFT'].index.values
    for j in range(len(indices)):
        allpctr = []
        availCompanies = []
        for i in dist.keys():
            indi = pctrs[i].index.values
            if indices[j] in indi:
                allpctr += [pctrs[i].loc[indices[j]]]
                availCompanies += [i]
        allpctr.sort(reverse = True)
        splitter1 = int(0.25*len(allpctr))
        splitter2 = int(0.75*len(allpctr))
        splitter3 = int(0.5*len(allpctr))
        cutter1 = allpctr[splitter1]
        cutter2 = allpctr[splitter2]
        cutter3 = allpctr[splitter3]
        for i in availCompanies:
            tem = pctrs[i].loc[indices[j]]
            if tem < cutter2:
                pctrs[i].loc[indices[j]] = -1
            elif tem > cutter1:
                pctrs[i].loc[indices[j]] = -1
            elif tem >=cutter3:
                pctrs[i].loc[indices[j]] = 0
            elif tem >=cutter3:
                pctrs[i].loc[indices[j]] = 1
    for i in pctrs.keys():
        pctrs[i] = np.asarray(pctrs[i])
    return pctrs
def GetAlphasAll(dist):
    '''Return dataframe of companies with corresponding alpha values'''
    df = {}
    for i in dist.keys():
        temp = GetAlphas(dist[i].copy())
        if (temp.empty==False):
            df[i] = alp1.get_alpha(temp).drop(['adj close', 'close', 'high', 'low', 'open', 'volume', 'amount', 'pctr'], axis=1).fillna(value = 0)
    return df

def GetAlphas(df):
    '''return the percentage return and quantum'''
    new = df.copy()[:-1]
    pctr = []
    amount = []
    for i in range(df.shape[0]-1):
        pctr += [(df['close'][i+1]-df['close'][i])/df['close'][i]] 
        amount += [df['close'][i]*df['volume'][i]]
    new['pctr'] = pctr
    new['amount'] = amount
    return new

def SPpctr():
    '''Get list of percentage return of S&P 500'''
    prices = yf.download('^GSPC', period = '2y', interval = '1mo', auto_adjust = False)
    indices = prices.index.values
    #print(indices[0])
    prices = prices[prices.Close.isna() == False]
    for j in indices:
        if str(j)[8:10] != '01':
            prices = prices.drop(j)
    new = []
    indices = prices.index.values
    for k in range(len(indices)-1):
        a = prices['Close'][k]
        b = prices['Close'][k+1]
        pctr = (b-a)/a
        new+=[pctr]
    indices = prices.index.values
    prices = prices.drop(indices[-1])
    prices['pctr'] = new
    return prices.drop(['Close', 'Open', 'High','Low','Adj Close','Volume'], axis = 1)['pctr']


