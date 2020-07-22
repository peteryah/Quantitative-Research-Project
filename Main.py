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
import json
warnings.filterwarnings("ignore")
symbols = pd.read_excel('SP500_2018.xlsx')
symbols = list(symbols['Symbol'])
#print(symbols)


# Available types from Yahoo Finance
tem = ['adj close', 'close', 'high', 'low', 'open', 'volume']
def YFI(choice, startDate = '2018-07-01', endDate = '2020-07-02'):
    '''import stock data from yahoo finance'''
    # Get data from Yahoo Finance
    # Turn into Stockstats dataframe, did some initial cleaning
    dist = {}
    k=['APH', 'CPRI']
    # get monthly data from yahoo finance 
    if choice == 0:
        for i in symbols:
            if i not in k:
                #print(i)
                dist[i] = SS.StockDataFrame.retype(yf.download(i, period = '3y', interval = '1mo', auto_adjust = False))
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
    '''main function for backtest etc'''
    goodAlphas = ['alpha083','alpha101','alpha024','alpha042','alpha028','alpha025','alpha018','alpha010','alpha047','alpha033','alpha009','alpha005','alpha051']
    goodAlphas1 = goodAlphas[:7]
    goodAlphas2 = goodAlphas[7:]
    #allAlphas = list(X['MSFT'].columns)
    sums = []
    
    alphaIndex = goodAlphas[1:-1]
    if check_4_update():
        dist = YFI(0)
        X, Y = Process.ProcessData(dist, 2)
            
        data_to_local(X,Y,dist)
    X,Y,dist = local_to_data()
    '''
    raw = pd.DataFrame(dist['GOOG'])
    
    raw.to_excel('LastRunLog.xlsx')
    
    
    
    
    #alphaIndex = list(X['GOOG'].columns)
    
    today = datetime.date.today().isoformat()
    Date = datetime.datetime(year = int(today[:4]), month = int(today[-5:-3])-2, day = 1)
    sp5 = list(SP(1,2))[:-1]
    
    dataX, dataY, result, rmX, rmY = Testing.Backtest(24, alphaIndex, int(sp5[0]), dist, X, Y, 10, n=10,stDate = Date)
    
    plt.plot(np.asarray(dataX), np.asarray(sp5[:-2]), label = ('S&P500'))
    plt.plot(np.asarray(dataX), np.asarray(dataY), label = ('Prediction'))
    plt.plot(np.asarray(rmX), np.asarray(rmY), label = ('Random'))


    plt.xlabel('Months')
    plt.ylabel('Deposit')
    plt.legend()
    plt.show()

    '''
    #Calculate percentage return on particular month
    goodAlphas = ['alpha083','alpha101','alpha024','alpha042','alpha028','alpha025','alpha018','alpha010','alpha047','alpha033','alpha009','alpha005','alpha051']
    alphaIndex = goodAlphas
    #target = datetime.datetime(year = 2020, month = 7, day = 1)
    #result = Testing.Designated(target, dist, X, Y, alphaIndex, 10)
    
    print(X)
    type(X)
    result={}
    raw={}
    Processed={}
    dlist=['02-01-2019','03-01-2019','04-01-2019']
##    for i in dlist:
##        result[i],raw[i],Processed[i] = Prediction.AvgedPredict(dist, X,Y,alphaIndex,10,10,i)
##
##    dresult=[result[dlist[0]],result[dlist[1]],result[dlist[2]]]
##    dr=pd.concat(dresult)
##    dr.to_excel('Companies_SM.xlsx')
##    
##    draw=[raw[dlist[0]],raw[dlist[1]],raw[dlist[2]]]
##    dra=pd.concat(draw)
##    dra.to_excel('raw_data_SM.xlsx')
##
##    dpro=[Processed[dlist[0]],Processed[dlist[1]],Processed[dlist[2]]]
##    dp=pd.concat(dpro)
##    dp.to_excel('alpha_values_SM.xlsx')
    
    result1,result2,result3=Prediction.AvgedPredict(dist, X,Y,alphaIndex,10,10,'07-01-2020')
    r3=pd.DataFrame(result3)
    r3.to_excel('alpha_values.xlsx')
    
    return result1,result2
    

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
        dataX, dataY, result, rmX, rmY = Testing.Backtest(24, [i], sp5[3], dist, X, Y, 10)
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
        filenameX = 'dataset/X/' +i+" X.json"
        filenameY = 'dataset/Y/' +i+' Y.txt'
        filenameD = 'dataset/D/' +i+' D.json'
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
        filenameX = 'dataset/X/' +i+" X.json"
        filenameY = 'dataset/Y/' +i+' Y.txt'
        filenameD = 'dataset/D/' +i+' D.json'
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
    with open('last_update.txt') as update:
        update1 = update.readlines()
        temp = update1[0][-5:-3]
        write_today()
        if today[-5:-3] != temp:
            
            return True
        else:
            
            return False
   

def write_today():
    today = datetime.date.today().isoformat()
    with open('last_update.txt','w') as f:
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
    
#test area
res1,res2=main()
print(main())
r2=pd.DataFrame(res2)
r2['Rank']=r2.index
for i in range(len(r2)):
    r2['Rank'][i]=r2['Rank'][i]+1
r2=r2.rename(columns={1:'Company', 0:'Score'})
r2=r2.set_index("Company")
r2.to_excel('Rank_list.xlsx')

