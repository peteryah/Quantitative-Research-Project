
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
import Testing
import Prediction
import Alphas
import Process
import warnings
warnings.filterwarnings("ignore")
symbols = pd.read_excel('SP500.xlsx')
symbols = list(symbols['Symbol'])
#print(symbols)


# Available types from Yahoo Finance
tem = ['adj close', 'close', 'high', 'low', 'open', 'volume']
def YFI(choice, startDate = '2010-01-01', endDate = '2015-01-01'):
    '''import stock data from yahoo finance'''
    # Get data from Yahoo Finance
    # Turn into Stockstats dataframe, did some initial cleaning
    dist = {}
    
    # get monthly data from yahoo finance 
    if choice == 0:
        for i in symbols:
            dist[i] = SS.StockDataFrame.retype(yf.download(i, period = '2y', interval = '1mo', auto_adjust = False))
    else:
        for i in symbols:
            dist[i] = SS.StockDataFrame.retype(yf.download(i, start = startDate, end = endDate, interval = '1mo', auto_adjust = False))
    # Data cleaning
    # delete empty dataframes or dataframes with large amount of NAs.

    na = []
    for k in dist:
        if(dist[k].empty):
            na = na+[k]
    for i in na:
        del dist[i]
    # delete invalid data
    for i in dist.keys():
        dist[i] = dist[i][dist[i].close.isna() == False]

    for i in dist.keys():
        temp = dist[i].index.values
        for j in temp:
            if str(j)[8:10] != '01':
                dist[i] = dist[i].drop(j)

    for i in dist.keys():
        for j in ['close', 'open', 'high', 'low']:
            for k in range(dist[i][j].shape[0]-1):
                a = dist[i][j][k]
                b = dist[i][j][k+1].copy()
                if a==b:
                    dist[i][j][k+1] = b+0.001

    na = []
    for k in dist:
        if(dist[k].empty):
            na = na+[k]
    for i in na:
        del dist[i] 

    return dist

def main():
    '''main function for generating portfolio'''
    pp = []
    choices = []
    goodAlphas = ['alpha083','alpha101','alpha024','alpha042','alpha028','alpha025','alpha018','alpha010','alpha047','alpha033','alpha009','alpha005','alpha051']
    dist = YFI(0)

    X, Y = Process.ProcessData(dist, 2)
    

    #alphaIndex = Alphas.SingleAlpha(X, Y, 13)
    for i in range(5):
        result = Prediction.AvgedPredict(dist, X, Y, goodAlphas[:13], 30, 8)
        print(result)
    
   
    return result


    

    
#test area
print(main())
