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
import Testing2 as Testing
import Prediction
import Alphas
import Process
import warnings
warnings.filterwarnings("ignore")

import json
from multiprocessing import Pool
import time as timi
def LTD():
    temp = []
    with open('symbols.txt', 'r') as symbols:
        for line in symbols.readlines():
            temp += [line[:-1]]
    dist = {}
    for i in temp:
        filenameD = 'dataset/D/' + i + ' D.json'
        with open(filenameD) as json_D:
            tempD = pd.read_json(json_D)
        dist[i] = tempD

    return dist

dist=LTD()

def DBM(indices):
    tempCru = []
    tempp=[]
    tempCru+=[indices[0]]
    for i in range(len(indices)-1):
        
        if int(np.datetime_as_string(indices[i],unit='D')[5:7]) != int(np.datetime_as_string(indices[i+1],unit='D')[5:7]):
            tempCru += [indices[i+1].copy()]
            tempp+=[i]
            
    return tempp,tempCru
    
def monthly(dist):
    print('loop 1 strt')
    t1=timi.time()
    count=1
    for i in dist:
        it=timi.time()
        indices = list(dist[i].index.values)
        ida,cdate=DBM(indices)
        for j in range(len(ida)):
            if j==0:
                dist[i]['high'][0]=max(dist[i]['high'][:ida[j]+1])
                dist[i]['low'][0]=min(dist[i]['low'][:ida[j]+1])
                dist[i]['close'][0]=dist[i]['close'][ida[j]]
            else:
                dist[i]['high'][ida[j-1]+1]=max(dist[i]['high'][ida[j-1]+1:ida[j]+1])
                dist[i]['low'][ida[j-1]+1]=min(dist[i]['low'][ida[j-1]+1:ida[j]+1])
                dist[i]['close'][ida[j-1]+1]=dist[i]['close'][ida[j]]
        print(str(count)+' out of 473 done')
        itt=(timi.time()-it)
        print(str(itt)+' sec')
        
        count+=1

        
    print('loop1 end')
    t2=(timi.time()-t1)/60
    print(str(t2)+' min')

    print('l2 s')
    t1=timi.time()
    for i in dist:
        tt,cdate=DBM(indices)
        dist[i]=dist[i].loc[cdate]
        dist[i].reset_index(drop=True, inplace=True)

        newDates = []
        for l in cdate:
            newDates += [np.datetime64(np.datetime_as_string(l,unit='D')[:-2]+'01')]
        dist[i]['Date'] = newDates.copy()

        dist[i].set_index('Date',inplace=True, drop=True)
    print('l2 e')
    t2=(timi.time()-t1)/60

    print(str(t2)+' min')
    return dist

dist=monthly(dist)

    
