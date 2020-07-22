from numba import cuda
import time
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
import json
warnings.filterwarnings("ignore")
symbols = pd.read_excel('SP500.xlsx')
symbols = list(symbols['Symbol'])
#print(symbols)


# Available types from Yahoo Finance
tem = ['adj close', 'close', 'high', 'low', 'open', 'volume']
def YFI(choice, startDate = '2018-03-01', endDate = '2020-03-02'):
    '''import stock data from yahoo finance'''
    # Get data from Yahoo Finance
    # Turn into Stockstats dataframe, did some initial cleaning
    dist = {}
   
    # get monthly data from yahoo finance 
    if choice == 0:
        for i in symbols:
            
               
            dist[i] = SS.StockDataFrame.retype(yf.download(i, period = '1y', auto_adjust = False))
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
    '''
    for i in dist.keys():
        temp = dist[i].index.values
        for j in temp:
            if str(j)[8:10] != '01':
                dist[i] = dist[i].drop(j)
    '''
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
    '''main function for backtest etc'''
   
    if check_4_update():
        dist = YFI(0)
        X, Y = Process.ProcessData(dist, 2)
            
        data_to_local(X,Y,dist)
    X,Y,dist = local_to_data()
    #print(dist['GOOG'])
    Y = reverseY(Y)
    #print(Y)
    '''
    dist2 = YFI(1)
    X2, Y2 = Process.ProcessData(dist2, 2)
    for i in Y2.keys():
        a = Y2[i].copy()
        a = np.where(a != 0, 1, 0)
        a = np.where(a != 1, 0, 1)

        Y2[i] = a
    '''
    goodAlphas = ['alpha083','alpha101','alpha024','alpha042','alpha028','alpha025','alpha018','alpha010','alpha047','alpha033','alpha009','alpha005','alpha051']
    goodAlphas1 = goodAlphas[:7]
    goodAlphas2 = goodAlphas[7:]
    #allAlphas = list(X['MSFT'].columns)
    
    
    alphaIndex = goodAlphas
    
    #today = datetime.date.today().isoformat()
    #Date = datetime.datetime(year = int(today[:4]), month = int(today[-5:-3])-1, day = 1)
    Date = datetime.datetime(year = 2019, month = 10, day = 1)
    #sp5 = list(SP(1,2))[:-1]
    
    dataX, dataY, result = Testing.Backtest(60, alphaIndex, 2000, dist, X, Y, 10 ,Date)
    #print('Test and train on 2010-2015:', result)
    #plt.plot(np.asarray(dataX), np.asarray(sp5[2:]), label = ('S&P500'))
    plt.plot(np.asarray(dataX), np.asarray(dataY), label = ('Prediction'))
    #plt.plot(np.asarray(rmX), np.asarray(rmY), label = ('Random'))


    plt.xlabel('Months')
    plt.ylabel('Deposit')
    plt.legend()
    plt.show()

    '''
    #Calculate percentage return on particular month
    
    target = datetime.datetime(year = 2020, month = 4, day = 1)
    #a = Testing.Designated(target, dist, X, Y, alphaIndex, 10)
    
    
    
    result = Prediction.LatestPredict(dist, X,Y,alphaIndex,10)
    print(result)
    '''
    return result

def reverseY(Y):
    newY = {}
    for i in list(Y.keys()):
        temp1 = Y[i].copy()
        temp2 = []
        for k in temp1:
            temp2+=[abs(k-1)]
        newY[i] = temp2
    return newY

def plotSingleAlpha(alphas, X, Y, dist):
    sums = []
    sp5 = list(SP(1,2))[:-1]
    for i in alphas:
        print(i)
        dataX, dataY, result, rmX, rmY = Testing.Backtest(6, [i], sp5[3], dist, X, Y, 10)
        plt.plot(np.asarray(dataX), np.asarray(dataY), label =( 'Alpha = ' + str(i)))
        #plt.plot(np.asarray(rmX), np.asarray(rmY))
                
        
        sums += [[result, i]]
    sums.sort(reverse = True)
    dataX, dataY, result, rmX, rmY = Testing.Backtest(24, alphas, sp5[3], dist, X, Y, 10)
    plt.plot(np.asarray(dataX), np.asarray(dataY), label =('Cumulative'))
    plt.plot(np.asarray(dataX), np.asarray(sp5[4:]), label = ('SP500'))


    plt.xlabel('Months')
    plt.ylabel('Money')
    plt.legend()
    plt.show()
    return result

def SP(startD, endD):
    dist = SS.StockDataFrame.retype(yf.download('^GSPC', period = '2y', interval = '1mo', auto_adjust = False))
    return dist['close']
    
def data_to_local(X,Y,dist):
    temp = list(X.keys())
    with open('symbols.txt','w') as symbols:
        for i in temp:
            symbols.write('%s\n' % i)
    for i in temp:
        filenameX = 'datasetD\X\\' +i+" X.json"
        filenameY = 'datasetD\Y\\' +i+' Y.txt'
        filenameD = 'datasetD\D\\' +i+' D.json'
        with open(filenameX, 'w') as f:
            
            out = pd.DataFrame(X[i]).to_json()
            f.write(out)
        with open(filenameY, 'w') as f:
            for k in Y[i]:
                f.write('%s\n' % str(k))
        with open(filenameD, 'w') as f:
            out = pd.DataFrame(dist[i]).to_json()
            f.write(out)
        
        
    return
    

def local_to_data():
    temp=[]
    with open('symbols.txt','r') as symbols:
        for line in symbols.readlines():
            temp+=[line[:-1]]
    
    dataX = {}
    dist = {}
    dataY = {}
    for i in temp:
        filenameX = 'datasetD\X\\' +i+" X.json"
        filenameY = 'datasetD\Y\\' +i+' Y.txt'
        filenameD = 'datasetD\D\\' +i+' D.json'
        tempY = []
        with open(filenameX) as json_X:
            tempX = pd.read_json(json_X)
            #print(tempX)
        with open(filenameY) as txt_Y:
            for line in txt_Y.readlines():
                tempY+=[int(float(line))]
        with open(filenameD) as json_D:
            tempD = pd.read_json(json_D)
        dataX[i] = tempX
        dataY[i] = tempY
        dist[i] = tempD
        
        
    return dataX, dataY, dist
    
    

def check_4_update():
    today = datetime.date.today().isoformat()
    with open('last_update_day.txt') as update:
        update1 = update.readlines()
        temp = update1[0][-2:]
        write_today()
        if today[-2:] != temp:
            
            return True
        else:
            
            return False
   

def write_today():
    today = datetime.date.today().isoformat()
    with open('last_update_day.txt','w') as f:
        f.write(today)
    return
    
def test():
    pp = []
    choices = []
    if check_4_update():
        dist = YFI(0)
        X, Y = Process.ProcessData(dist, 2)
            
        data_to_local(X,Y,dist)
    X,Y,dist = local_to_data()
    i = "GOOG"
    raw = pd.DataFrame(dist[i])
    processed = pd.DataFrame(X[i])
    raw.to_excel('RawData.xlsx')
    processed.to_excel('ProcData.xlsx')
    
    
    return


def autoMain():
    goodAlphas = ['alpha083','alpha101','alpha024','alpha042','alpha028','alpha025','alpha018','alpha010','alpha047','alpha033','alpha009','alpha005','alpha051']
    alphaIndex = goodAlphas
    if check_4_update():
        dist = YFI(0)
        X, Y = Process.ProcessData(dist, 2)
        data_to_local(X,Y,dist)
            
    X,Y,dist = local_to_data()
    print(dist['MSFT'])
    Ports = []
    Pctrs = []
    
    indices = list(dist['GOOG'].index.values)
    Date = indices.index(np.datetime64(datetime.datetime(year = 2020, month = 5, day = 20)))
    
    while (indices[Date])!=indices[-2]:
        
        
        tempPort = Designated(Date,dist,X,Y,alphaIndex, 10)
        Ports+=[PortConcat(tempPort)]
       
        tempPctr = getPctr(dist, tempPort, Date)
        Pctrs+=[tempPctr]
        Date +=1
    
    
    tempDate = indices[Date]
    tempPort = Designated(Date, dist, X, Y, alphaIndex, 10)
    Ports+=[PortConcat(tempPort)]
    
    
    while True:
        
        if check_4_update():
            dist = YFI(0)
            X, Y = Process.ProcessData(dist, 2)
            data_to_local(X,Y,dist)
            
        X,Y,dist = local_to_data()
        indices = list(dist['GOOG'].index.values)
        #print(indices)
        if indices[-2]==indices[Date]:
            tempPctr = getPctr(dist,tempPort,Date)
            Pctrs+=[tempPctr]
        
            Date += 1
            
            tempPort = Designated(Date, dist, X, Y, alphaIndex, 10)
            
            Ports+=[PortConcat(tempPort)]
            
            df = pd.DataFrame()
            
            df['Date'] = indices[indices.index(np.datetime64(datetime.datetime(year = 2020, month = 5, day = 20))):]
            df['Portfolio'] = Ports
            df['Return'] = Pctrs+[0]
            
            print(df)
        else:
            print('Stock market close day')
        return
        while np.datetime64(datetime.datetime.now()) <= (indices[Date]+np.timedelta64(1,'D')):
            time.sleep(60*60*24)
        print(datetime.datetime.now())
        print(indices[Date])
    return

def Designated(Target, dist, X, Y, alphaIndex, portSize):
    indices = list(dist['GOOG'].index.values)
    initDate = Target
    startDate = indices[initDate-3]
    endDate = indices[initDate-1]
    inputX = {}
    inputY = {}
    
    for i in dist.keys():
        indices = list(dist[i].index.values)
        if startDate in indices and endDate in indices:
            inputX[i] = X[i].loc[startDate:endDate]
            inputY[i] = Y[i][indices.index(startDate):indices.index(endDate)]
        
    inputX = Testing.splitterX2(inputX)
    inputY = Testing.splitterY2(inputY)
    models = []
    
    for i in range(10):
        
        models+= [Testing.train2(inputX[alphaIndex], inputY)]
    scores = []

    for i in X.keys():
        
        if X[i].loc[indices[initDate-1]:indices[initDate]].empty == False:
            
            b = AvgScore(models, X, alphaIndex, indices[initDate-1],indices[initDate],i)
            
            scores += [[b[-1][0], i]]
    scores.sort(reverse=True)
    res = []
    for j in scores[:portSize]:
        res+=[j[1]]
    
    
    return res

def getPctr(dist, port, date):
    indices = list(dist['GOOG'].index.values)
    date2 = date
    temp=0
    for i in port:
        a = dist[i]['close'].iloc[date2]
        b = dist[i]['close'].iloc[date2+1]
        temp+=(b-a)/a
    return temp/len(port)

def AvgScore(models, X, alphaIndex, d1, d2,i):
    total=0
    
    for j in range(len(models)):
        score = models[j].predict(X[i][alphaIndex].loc[d1:d2])
        #print(score)
        total+=score
    return total / len(models)

def PortConcat(lis):
    res = ''
    for i in lis:
        res+=i+', '
    return res[:-2]

def test():
    Date = np.datetime64('2020-05-12')
    X,Y,dist = local_to_data()
    goodAlphas = ['alpha083','alpha101','alpha024','alpha042','alpha028','alpha025','alpha018','alpha010','alpha047','alpha033','alpha009','alpha005','alpha051']
    alphaIndex = goodAlphas
    indices = list(dist['GOOG'].index.values)
    tempPort = Designated(indices.index(Date), dist, X, Y, alphaIndex, 10)
    return tempPort
    

#test area
print(autoMain())


