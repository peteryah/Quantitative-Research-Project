import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool
#from muliprocessing
import time as timi

import yfinance as yf
import stockstats as SS

import warnings

##import


symbols = pd.read_excel('SP500_2018.xlsx')
symbols = list(symbols['Symbol'])

# s1=symbols[:126]
# s2=symbols[126:252]
# s3=symbols[252:378]
# s4=symbols[378:len(symbols)]

s1=symbols[:4]
s2=symbols[4:8]
s3=symbols[8:12]
s4=symbols[12:16]

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
                d1[i] = SS.StockDataFrame.retype(yf.download(i, period = '1y', interval = '1mo', auto_adjust = False))
        print('process 1 ends')
        return d1
    elif choice==1:
        print('process 2 starts')

        d2={}
        for i in s2:
            if i not in k:
                #print(i)
                d2[i] = SS.StockDataFrame.retype(yf.download(i, period = '1y', interval = '1mo', auto_adjust = False))
        print('process 2 ends')

        return d2
    elif choice==2:
        print('process 3 starts')

        d3={}
        for i in s3:
            if i not in k:
                #print(i)
                d3[i] = SS.StockDataFrame.retype(yf.download(i, period = '1y', interval = '1mo', auto_adjust = False))
        print('process 3 ends')
        return d3
    elif choice==3:
        print('process 4 starts')

        d4={}
        for i in s4:
            if i not in k:
                #print(i)
                d4[i] = SS.StockDataFrame.retype(yf.download(i, period = '1y', interval = '1mo', auto_adjust = False))
        print('process 4 ends')

        return d4


if __name__=='__main__':
    warnings.filterwarnings("ignore")

    startDate = '2018-07-01'
    endDate = '2020-07-02'

    num_proc=4
    pool=Pool(num_proc)
    t1=timi.time()
    output=pool.starmap(YFI,((0, startDate, endDate),(1, startDate, endDate),(2, startDate, endDate),(3, startDate, endDate)))
    pool.close()
    print(output)
    dist={}
    dist=output[0]
    dist.update(output[1])
    dist.update(output[2])
    dist.update(output[3])
    t2=timi.time()
    tt=int(t2-t1)
    tt=tt/60
    to=int(((55-tt)/55)*100)
    print('Entire run took '+str(tt)+' minutes')
    print('Multiprocessing reduced the program runtime by '+str(to)+' %')


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



def data_to_localm(X, Y, dist):
    temp = list(X.keys())
    with open('symbols.txt', 'w') as symbols:
        for i in temp:
            symbols.write('%s\n' % i)
    for i in temp:
        filenameD = 'dataset/D/' + i + ' D.json'
        with open(filenameD, 'w') as f:
            out = pd.DataFrame(dist[i]).to_json()
            f.write(out)
    return


