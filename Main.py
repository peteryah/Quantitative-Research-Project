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
import itertools
warnings.filterwarnings("ignore")

import json
from multiprocessing import Pool
import time as timi

symbols = pd.read_excel('SP500_2018.xlsx')
symbols = list(symbols['Symbol'])

print('Welcome to the machine of MF2019 !')
# timi.sleep(2)
print('Parallelization will start in less than 3 secs......')
# timi.sleep(3)

print()
print()

entire1=timi.time()

s1=symbols[:126]
s2=symbols[126:252]
s3=symbols[252:378]
s4=symbols[378:len(symbols)]
#
# s1=symbols[:4]
# s2=symbols[4:8]
# s3=symbols[8:12]
# s4=symbols[12:16]
#print(symbols)

def reverseY(Y):
    newY = {}
    for i in list(Y.keys()):
        temp1 = Y[i].copy()
        temp2 = []
        for k in temp1:
            temp2 += [abs(k - 1)]
        newY[i] = temp2
    return newY


def plotSingleAlpha(alphas, X, Y, dist):
    sums = []
    sp5 = list(SP(1, 2))[:-1]
    for i in alphas:
        print(i)
        dataX, dataY, result, rmX, rmY = Testing.Backtest(24, [i], sp5[3], dist, X, Y, 10)
        plt.plot(np.asarray(dataX), np.asarray(dataY), label=('Alpha = ' + str(i)))
        # plt.plot(np.asarray(rmX), np.asarray(rmY))

        sums += [[result, i]]
    sums.sort(reverse=True)
    dataX, dataY, result, rmX, rmY = Testing.Backtest(24, alphas, sp5[3], dist, X, Y, 10)
    plt.plot(np.asarray(dataX), np.asarray(dataY), label=('Cumulative'))
    plt.plot(np.asarray(dataX), np.asarray(sp5[4:]), label=('SP500'))

    plt.xlabel('Months')
    plt.ylabel('Money')
    plt.legend()
    plt.show()
    return result


def SP(startD, endD):
    dist = SS.StockDataFrame.retype(yf.download('^GSPC', period='2y', interval='1mo', auto_adjust=False))
    return dist['close']


def data_to_local(X, Y, dist):
    temp = list(X.keys())
    with open('symbols.txt', 'w') as symbols:
        for i in temp:
            symbols.write('%s\n' % i)
    for i in temp:
        filenameX = 'dataset/X/' + i + " X.json"
        filenameY = 'dataset/Y/' + i + ' Y.txt'
        filenameD = 'dataset/D/' + i + ' D.json'
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
    temp = []
    with open('symbols.txt', 'r') as symbols:
        for line in symbols.readlines():
            temp += [line[:-1]]
    dataX = {}
    dist = {}
    dataY = {}
    for i in temp:
        filenameX = 'dataset/X/' + i + ' X.json'
        filenameY = 'dataset/Y/' + i + ' Y.txt'
        filenameD = 'dataset/D/' + i + ' D.json'
        tempY = []
        with open(filenameX) as json_X:
            tempX = pd.read_json(json_X)
            # print(tempX)
        with open(filenameY) as txt_Y:
            for line in txt_Y.readlines():
                tempY += [int(float(line))]
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
    with open('last_update.txt', 'w') as f:
        f.write(today)
    return


def test():
    pp = []
    choices = []
    #if check_4_update():
     #   dist = YFI(0)
      #  X, Y = Process.ProcessData(dist, 2)

       # data_to_local(X, Y, dist)
    X, Y, dist = local_to_data()
    i = "GOOG"
    raw = pd.DataFrame(dist[i])
    processed = pd.DataFrame(X[i])
    raw.to_excel('RawData.xlsx')
    processed.to_excel('ProcData.xlsx')
    return


def clean(dist):
    if (dist.empty):
        del dist
    # delete invalid data

    temp = dist.index.values
    for j in temp:
        if str(j)[8:10] != '01':
            dist = dist.drop(j)

    for j in ['close', 'open', 'high', 'low']:
        for k in range(dist[j].shape[0] - 1):
            a = dist[j][k]
            b = dist[j][k + 1].copy()
            if a == b:
                dist[j][k + 1] = b + 0.001
    return

def DBM(indices):
    tempCru = []
    tempp = []
    #print(indices)
    tempCru += [indices[0]]
    for k in range(len(indices) - 1):
        if int(np.datetime_as_string(indices[k], unit='D')[5:7]) != int(np.datetime_as_string(indices[k + 1], unit='D')[5:7]):

            tempCru += [indices[k + 1].copy()]
            tempp += [k]

    return tempp, tempCru

# Available types from Yahoo Finance
tem = ['adj close', 'close', 'high', 'low', 'open', 'volume']
def YFI(choice, startDate = '2018-07-01', endDate = '2020-07-02'):
    '''import stock data from yahoo finance'''
    # Get data from Yahoo Finance
    # Turn into Stockstats dataframe, did some initial cleaning
    k=['APH', 'CPRI']
    # get monthly data from yahoo finance
    if choice == 0:
        print('process 1 starts')
        d1={}
        for i in s1:
            if i not in k:
                #print(i)

                d1[i] = SS.StockDataFrame.retype(yf.download(i, period = '6mo', interval = '1d', auto_adjust = False))
                if d1[i].empty:
                    continue
                #print(d1[i])
                indices = list(d1[i].index.values)
                ida, ttt = DBM(indices)
                for j in range(len(ida)):
                    if j == 0:
                        d1[i]['high'][0] = max(d1[i]['high'][:ida[j] + 1])
                        d1[i]['low'][0] = min(d1[i]['low'][:ida[j] + 1])
                        d1[i]['close'][0] = d1[i]['close'][ida[j]]
                        d1[i]['volume'][0]=sum(d1[i]['volume'][:ida[j] + 1])
                    else:
                        d1[i]['high'][ida[j - 1] + 1] = max(d1[i]['high'][ida[j - 1] + 1:ida[j] + 1])
                        d1[i]['low'][ida[j - 1] + 1] = min(d1[i]['low'][ida[j - 1] + 1:ida[j] + 1])
                        d1[i]['close'][ida[j - 1] + 1] = d1[i]['close'][ida[j]]
                        d1[i]['volume'][ida[j - 1] + 1] = sum(d1[i]['volume'][ida[j - 1] + 1:ida[j] + 1])

                tt, cdate = DBM(indices)
                d1[i] = d1[i].loc[cdate]
                d1[i].reset_index(drop=True, inplace=True)

                newDates = []
                #print(cdate)
                for l in cdate:
                    newDates += [np.datetime64(np.datetime_as_string(l, unit='D')[:-2] + '01')]
                d1[i]['Date'] = newDates.copy()

                d1[i].set_index('Date', inplace=True, drop=True)

        print('process 1 ends')
        return d1
    elif choice==1:
        print('process 2 starts')

        d2={}
        for i in s2:
            if i not in k:
                #print(i)
                d2[i] = SS.StockDataFrame.retype(yf.download(i, period = '6mo', interval = '1d', auto_adjust = False))
                if d2[i].empty:
                    continue
                indices = list(d2[i].index.values)
                ida, ttt = DBM(indices)
                for j in range(len(ida)):
                    if j == 0:
                        d2[i]['high'][0] = max(d2[i]['high'][:ida[j] + 1])
                        d2[i]['low'][0] = min(d2[i]['low'][:ida[j] + 1])
                        d2[i]['close'][0] = d2[i]['close'][ida[j]]
                        d2[i]['volume'][0] = sum(d2[i]['volume'][:ida[j] + 1])
                    else:
                        d2[i]['high'][ida[j - 1] + 1] = max(d2[i]['high'][ida[j - 1] + 1:ida[j] + 1])
                        d2[i]['low'][ida[j - 1] + 1] = min(d2[i]['low'][ida[j - 1] + 1:ida[j] + 1])
                        d2[i]['close'][ida[j - 1] + 1] = d2[i]['close'][ida[j]]
                        d2[i]['volume'][ida[j - 1] + 1] = sum(d2[i]['volume'][ida[j - 1] + 1:ida[j] + 1])

                tt, cdate = DBM(indices)
                d2[i] = d2[i].loc[cdate]
                d2[i].reset_index(drop=True, inplace=True)

                newDates = []
                for l in cdate:
                    newDates += [np.datetime64(np.datetime_as_string(l, unit='D')[:-2] + '01')]
                d2[i]['Date'] = newDates.copy()

                d2[i].set_index('Date', inplace=True, drop=True)
        print('process 2 ends')

        return d2
    elif choice==2:
        print('process 3 starts')

        d3={}
        for i in s3:
            if i not in k:
                #print(i)
                d3[i] = SS.StockDataFrame.retype(yf.download(i, period = '6mo', interval = '1d', auto_adjust = False))
                #clean(d3[i])
                if d3[i].empty:
                    continue
                indices = list(d3[i].index.values)
                ida, ttt = DBM(indices)

                for j in range(len(ida)):
                    if j == 0:
                        d3[i]['high'][0] = max(d3[i]['high'][:ida[j] + 1])
                        d3[i]['low'][0] = min(d3[i]['low'][:ida[j] + 1])
                        d3[i]['close'][0] = d3[i]['close'][ida[j]]
                        d3[i]['volume'][0] = sum(d3[i]['volume'][:ida[j] + 1])
                    else:
                        d3[i]['high'][ida[j - 1] + 1] = max(d3[i]['high'][ida[j - 1] + 1:ida[j] + 1])
                        d3[i]['low'][ida[j - 1] + 1] = min(d3[i]['low'][ida[j - 1] + 1:ida[j] + 1])
                        d3[i]['close'][ida[j - 1] + 1] = d3[i]['close'][ida[j]]
                        d3[i]['volume'][ida[j - 1] + 1] = sum(d3[i]['volume'][ida[j - 1] + 1:ida[j] + 1])

                tt, cdate = DBM(indices)
                d3[i] = d3[i].loc[cdate]
                d3[i].reset_index(drop=True, inplace=True)

                newDates = []
                for l in cdate:
                    newDates += [np.datetime64(np.datetime_as_string(l, unit='D')[:-2] + '01')]
                d3[i]['Date'] = newDates.copy()

                d3[i].set_index('Date', inplace=True, drop=True)

        print('process 3 ends')
        return d3
    elif choice==3:
        print('process 4 starts')

        d4={}
        for i in s4:
            if i not in k:
                #print(i)
                d4[i] = SS.StockDataFrame.retype(yf.download(i, period = '6mo', interval = '1d', auto_adjust = False))
                #clean(d4[i])
                if d4[i].empty:
                    continue
                indices = list(d4[i].index.values)
                ida, ttt = DBM(indices)
                for j in range(len(ida)):
                    if j == 0:
                        d4[i]['high'][0] = max(d4[i]['high'][:ida[j] + 1])
                        d4[i]['low'][0] = min(d4[i]['low'][:ida[j] + 1])
                        d4[i]['close'][0] = d4[i]['close'][ida[j]]
                        d4[i]['volume'][0] = sum(d4[i]['volume'][:ida[j] + 1])
                    else:
                        d4[i]['high'][ida[j - 1] + 1] = max(d4[i]['high'][ida[j - 1] + 1:ida[j] + 1])
                        d4[i]['low'][ida[j - 1] + 1] = min(d4[i]['low'][ida[j - 1] + 1:ida[j] + 1])
                        d4[i]['close'][ida[j - 1] + 1] = d4[i]['close'][ida[j]]
                        d4[i]['volume'][ida[j - 1] + 1] = sum(d4[i]['volume'][ida[j - 1] + 1:ida[j] + 1])

                tt, cdate = DBM(indices)
                d4[i] = d4[i].loc[cdate]
                d4[i].reset_index(drop=True, inplace=True)

                newDates = []
                for l in cdate:
                    newDates += [np.datetime64(np.datetime_as_string(l, unit='D')[:-2] + '01')]
                d4[i]['Date'] = newDates.copy()

                d4[i].set_index('Date', inplace=True, drop=True)

        print('process 4 ends')

        return d4

if __name__=='__main__':
    '''main function for backtest etc'''
    goodAlphas = ['alpha083','alpha101','alpha024','alpha042','alpha028','alpha025','alpha018','alpha010','alpha047','alpha033','alpha009','alpha005','alpha051']
    goodAlphas1 = goodAlphas[:7]
    goodAlphas2 = goodAlphas[7:]
    #allAlphas = list(X['MSFT'].columns)
    sums = []
    
    alphaIndex = goodAlphas[1:-1]
    if 1==1:
        startDate = '2018-08-01'
        endDate = '2020-08-01'
        num_proc = 4
        pool = Pool(num_proc)
        t1 = timi.time()
        output = pool.starmap(YFI, ((0, startDate, endDate), (1, startDate, endDate), (2, startDate, endDate), (3, startDate, endDate)))
        pool.close()

        dist = {}
        dist = output[0]
        dist.update(output[1])
        dist.update(output[2])
        dist.update(output[3])

        t2 = timi.time()
        tt = int(t2 - t1)
        tt = tt / 60
        to = int(((12 - tt) / 12) * 100)

        na = []
        for i in dist:
            temp = i

        na = []
        for i in dist:
            temp=i

        for k in dist:
            if (dist[k].empty):
                na = na + [k]
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
                for k in range(dist[i][j].shape[0] - 1):
                    a = dist[i][j][k]
                    b = dist[i][j][k + 1].copy()
                    if a == b:
                        dist[i][j][k + 1] = b + 0.001
        na = []
        for k in dist:
            if (dist[k].empty):
                na = na + [k]
        for i in na:
            del dist[i]

        print()
        print()

        print('We have now downloaded data for '+str(len(dist))+' companies in the S&P 500 List')
        print()

        t2 = timi.time()
        tt = int(t2 - t1)
        tt = int(tt) / 60
        to = int(((24 - tt) / 24) * 100)
        print('Downloading took ' + str(tt) + ' minutes')
        print('Multiprocessing reduced the program downloading time by ' + str(to) + ' %')
        print()
    else:
        print('Starting to load data......')
        X, Y, dist = local_to_data()
        print('data loading finished')
        print()
        print()

    print('Starting to process data...')
    X, Y = Process.ProcessData(dist, 2)
    print('Processing finished')
    print()
    print()

    print('Starting to save data......')
    data_to_local(X, Y, dist)
    print('data saving finished')
    print()
    print()

    #Calculate percentage return on particular month
    goodAlphas = ['alpha083','alpha101','alpha024','alpha042','alpha028','alpha025','alpha018','alpha010','alpha047','alpha033','alpha009','alpha005','alpha051']
    alphaIndex = goodAlphas
    #target = datetime.datetime(year = 2020, month = 7, day = 1)
    #result = Testing.Designated(target, dist, X, Y, alphaIndex, 10)
    print('Starting to train neural networks......')
    n1=timi.time()
    resl={}
    ress={}

    for i in range(10):
        reslo, ressh = Prediction.AvgedPredict(dist, X, Y, alphaIndex, 10, 10, '08-01-2020')
        for j in reslo:
            if j in resl:
                resl[j] += 1
            else:
                resl[j] = 1
        for k in ressh:
            if k in ress:
                ress[k] += 1
            else:
                ress[k] = 1
        print(str(i)+' out of 10 done')

    n2=int(timi.time()-n1)/60
    print('Entire Training took'+str(n2)+' min')
    resl = sorted(resl.items(), key=lambda x: x[1], reverse=True)
    ress= sorted(ress.items(), key=lambda x: x[1], reverse=True)

    longp=[]
    for i in resl:
        if len(longp)>=10:
            break
        longp+=[i[0]]

    shortp = []
    for i in ress:
        if len(shortp) >= 10:
            break
        shortp += [i[0]]


    t3=timi.time()
    ttt=(t3-t2)/60
    tw=int(t3-entire1)/60
    print('Done')
    print()

    print('Data processing took '+str(ttt) +' minutes')
    print()

    print('Entire run took ' + str(tw) + ' minutes')
    print()

    #r3=pd.DataFrame(result3)
    #r3.to_excel('alpha_values.xlsx')

    #return result
    

#test area
    print("Here's the Long portfolio you are waiting for: "+str(longp))
    print("and here's the Short portfolio: "+str(shortp))
    #print(result)

