import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool
#多的就这个 from muliprocessing
import time as timi
import Process
import yfinance as yf
import stockstats as SS

import warnings

##import 多的只有muliocessing，其他按照main里的就好了。


symbols = pd.read_excel('SP500_2018.xlsx')
symbols = list(symbols['Symbol'])

s1=symbols[:126]
s2=symbols[126:252]
s3=symbols[252:378]
s4=symbols[378:len(symbols)]
s5 = ['MSFT','GOOG','GOOGL']
# s1=symbols[:4]
# s2=symbols[4:8]
# s3=symbols[8:12]
# s4=symbols[12:16]

def divideByMonth(indices):
    tempCru = []
    tempp=[]
    for i in range(len(indices)-1):
        if int(np.datetime_as_string(indices[i],unit='D')[5:7]) != int(np.datetime_as_string(indices[i+1],unit='D')[5:7]):
            tempCru += [indices[i].copy()]
            tempp+=[i]
    return tempCru,tempp



def ThirtyOne(year):
    d1 = {}

    for i in s5:
        temp = SS.StockDataFrame.retype(yf.download(i, period='3mo', interval='1d', auto_adjust=False))

        indices = list(temp.index.values)
        print(indices)
        temp2 = pd.DataFrame()
        crucialDate, ida = divideByMonth(indices)
        print(ida)
        for k in ida:
            print(temp.iloc[k])
        #print(crucialDate)
        dataList = temp.loc[crucialDate]
        dataList.reset_index(drop=True, inplace=True)


        newDates = []
        for j in crucialDate:
            newDates += [np.datetime64(np.datetime_as_string(j,unit='D')[:-2]+'01')]
        dataList['Date'] = newDates.copy()

        dataList.set_index('Date',inplace=True, drop=True)

        d1[i] = dataList
        
    return d1

#print(np.datetime_as_string(np.datetime64('2019-01-01'),unit='D'))
print(ThirtyOne(1))
dist=ThirtyOne(1)

