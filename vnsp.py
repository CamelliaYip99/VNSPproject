# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:36:08 2021

@author: Administrator
"""
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def construct_VNSP(data_df,n=60):
    
    
    data_df['gain'] = data_df['n_return']
    data_df['loss'] = data_df['n_return']
    
    data_df['1-vol'] = 1 - data_df['vol']
    
    for i in range(len(data_df['n_return'])):

        weight = []
        gain = 0
        loss = 0
        for j in range(1,n):
            weight.append( data_df['vol'][i-n+j] * data_df['1-vol'][i-n+j+1:i-1].prod())
            gl_rate = (data_df['n_return'][i] - data_df['n_return'][i-n+j])/ data_df['n_return'][i]
            if gl_rate >= 0:
                gain += gl_rate *weight[j-1]
            elif gl_rate < 0:
                loss += gl_rate *weight[j-1]
        k = sum(weight)

        data_df['gain'][i] = gain/k
        data_df['loss'][i] = loss/k
        
    data_df['VNSP'] = data_df['gain'] - 0.5*data_df['loss']
    
    return data_df
            
def trading_signal(data_df):
    signal_df = pd.DataFrame(index = data_df.index)
    signal_df['sign'] = np.zeros(len(data_df))
    
    for i in range(len(data_df)):
        if data_df['VNSP'][i-1] > data_df['VNSP'][i-2] :
            signal_df['sign'][i] = 1
        elif data_df['VNSP'][i-1] < data_df['VNSP'][i-2]:
            signal_df['sign'][i] = -1
    signal_df['position'] = signal_df['sign']

    return signal_df
    
    
if __name__ == '__main__':
    # data_df = download_data()
    # sigma = 0.5
    # n = 60
    # trading_cost = 0.0005
    SPY = yf.download('SPY', start='2017-01-01', end= '2021-01-01' )
    SPY.fillna('ffill')
    SPY['return'] =  SPY['Adj Close']/SPY['Adj Close'].shift(1) - 1
    SPY = SPY.fillna(0)
    
    mkt_cap_df = pd.read_csv('D:\BU\Programming\MF703 Fall21\project\daily_mkt_cap_data.csv',index_col=0,header=0)
    tickers = mkt_cap_df.columns.tolist()+['SPY']
    names = globals()
    for ticker in tickers:
        names[ticker] = yf.download(ticker, start ='2017-01-01', end = '2021-01-01' )
        names[ticker].fillna('ffill')
        names[ticker]['return'] =  names[ticker]['Adj Close']/names[ticker]['Adj Close'].shift(1) - 1
        names[ticker] = names[ticker].fillna(0)
        '''construct data_df with n_return and turnove rate'''
        names[ticker+'_df'] = pd.DataFrame(index = names[ticker].index)
        names[ticker+'_df'] ['Adj Close'] = names[ticker]['Adj Close'] 
        names[ticker+'_df']['vol'] = np.array(names[ticker]['Volume'])/np.array(mkt_cap_df[ticker])
        '''calculate t-n daily log return'''
        names[ticker+'_df']['n_return'] = np.log(names[ticker+'_df']['Adj Close'].shift(60) / names[ticker+'_df']['Adj Close'])
        
        
        names[ticker+'_df'] = construct_VNSP(names[ticker+'_df'])
        for i in range(len(names[ticker+'_df'])):
            if abs(names[ticker+'_df']['VNSP'][i]) >= 10:
                names[ticker+'_df']['VNSP'][i] = names[ticker+'_df']['VNSP'][i-1]
        names[ticker+'_signal_df'] = trading_signal(names[ticker+'_df'])
        names[ticker+'_signal_df']['result'] = names[ticker+'_signal_df']['position']*names[ticker]['return']
    
    
        # names[ticker+'_result_df'] = pd.DataFrame()
        # names[ticker+'_result_df'] ['Market Return'] = SPY['return'].cumsum()
        # # names[ticker+'_result_df']['Stock Return'] = names[ticker]['return'].cumsum()
        # names[ticker+'_result_df'] ['Strategy Return'] = names[ticker+'_signal_df']['result'].cumsum()
        # names[ticker+'_result_df'] .plot()
        # plt.title(f'{ticker} Accumulative Return')
        # plt.figure()
    
    # portfolio_result_df = pd.DataFrame()
    # portfolio_result_df['Market Return'] = SPY['return'].cumsum()
    # portfolio_result_df['Stock Return'] = pd.concat([CHTR['return']['2017-01-01':'2017-09-30'],PENN['return']['2017-10-01':'2017-12-31'],
    #                                                  RCL['return']['2018-01-01':'2018-03-31'],CHTR['return']['2018-04-01':'2018-06-30'],
    #                                                  FANG['return']['2018-07-01':'2018-09-30'],KHC['return']['2018-10-01':'2018-12-31'],
    #                                                  CHTR['return']['2019-01-01':'2019-06-30'],NCLH['return']['2019-07-01':'2020-06-30'],
    #                                                  CCL['return']['2020-07-01':'2020-09-30']]).cumsum()
    # portfolio_result_df['Strategy Return'] = pd.concat([CHTR_signal_df['result']['2017-01-01':'2017-09-30'],PENN_signal_df['result']['2017-10-01':'2017-12-31'],
    #                                                   RCL_signal_df['result']['2018-01-01':'2018-03-31'],CHTR_signal_df['result']['2018-04-01':'2018-06-30'],
    #                                                   FANG_signal_df['result']['2018-07-01':'2018-09-30'],KHC_signal_df['result']['2018-10-01':'2018-12-31'],
    #                                                   CHTR_signal_df['result']['2019-01-01':'2019-06-30'],NCLH_signal_df['result']['2019-07-01':'2020-06-30'],
    #                                                   CCL_signal_df['result']['2020-07-01':'2020-09-30']]).cumsum()
    # portfolio_result_df.plot()
    # plt.title('Accumulative Return')
    # plt.figure()
    
    temp_df = pd.DataFrame()
    temp_df['Market Return'] = SPY['return'].cumsum()
    temp_df['NSP value'] = pd.concat([CHTR_df['VNSP']['2017-01-01':'2017-09-30'],PENN_df['VNSP']['2017-10-01':'2017-12-31'],
                                                      RCL_df['VNSP']['2018-01-01':'2018-03-31'],CHTR_df['VNSP']['2018-04-01':'2018-06-30'],
                                                      FANG_df['VNSP']['2018-07-01':'2018-09-30'],KHC_df['VNSP']['2018-10-01':'2018-12-31'],
                                                      CHTR_df['VNSP']['2019-01-01':'2019-06-30'],NCLH_df['VNSP']['2019-07-01':'2020-06-30'],
                                                      CCL_df['VNSP']['2020-07-01':'2020-09-30']])
    fig,ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(temp_df.index,temp_df['Market Return'])
    ax1.set_ylabel('Market Return')
    ax2.plot(temp_df.index,temp_df['NSP value'] )
    ax2.set_ylabel('NSP Factor Values')
    plt.title('NSP Factor Values')
    plt.figure()
    # CCL_period_df = pd.DataFrame()
    # CCL_period_df['Market Return'] = SPY['return']['2020-07-01':'2020-09-30'].cumsum()
    # CCL_period_df['Stock Return'] = CCL['return']['2020-07-01':'2020-09-30'].cumsum()
    # CCL_period_df['Strategy Return'] = CCL_signal_df['result']['2020-07-01':'2020-09-30'].cumsum()
    # CCL_period_df.plot()
    # plt.title('CCL-2020Q3 Accumulative Return')
    # plt.figure()
    
    # CHTR_period_df = pd.DataFrame()
    # CHTR_period_df['Market Return'] = SPY['return']['2017-04-01':'2017-09-30'].cumsum()
    # CHTR_period_df['Stock Return'] = CHTR['return']['2017-04-01':'2017-09-30'].cumsum()
    # CHTR_period_df['Strategy Return'] = CHTR_signal_df['result']['2017-04-01':'2017-09-30'].cumsum()
    # CHTR_period_df.plot()
    # plt.title('CHTR-2017-Q23 Accumulative Return')
    # plt.figure()
    
    # CHTR_period_df = pd.DataFrame()
    # CHTR_period_df['Market Return'] = SPY['return']['2018-04-01':'2018-06-30'].cumsum()
    # CHTR_period_df['Stock Return'] = CHTR['return']['2018-04-01':'2018-06-30'].cumsum()
    # CHTR_period_df['Strategy Return'] = CHTR_signal_df['result']['2018-04-01':'2018-06-30'].cumsum()
    # CHTR_period_df.plot()
    # plt.title('CHTR-2018Q2 Accumulative Return')
    # plt.figure()
    
    # CHTR_period_df = pd.DataFrame()
    # CHTR_period_df['Market Return'] = SPY['return']['2019-01-01':'2019-06-30'].cumsum()
    # CHTR_period_df['Stock Return'] = CHTR['return']['2019-01-01':'2019-06-30'].cumsum()
    # CHTR_period_df['Strategy Return'] = CHTR_signal_df['result']['2019-01-01':'2019-06-30'].cumsum()
    # CHTR_period_df.plot()
    # plt.title('CHTR-2019Q12 Accumulative Return')
    # plt.figure()
    
    # FANG_period_df = pd.DataFrame()
    # FANG_period_df['Market Return'] = SPY['return']['2018-07-01':'2018-09-30'].cumsum()
    # FANG_period_df['Stock Return'] = FANG['return']['2018-07-01':'2018-09-30'].cumsum()
    # FANG_period_df['Strategy Return'] = FANG_signal_df['result']['2018-07-01':'2018-09-30'].cumsum()
    # FANG_period_df.plot()
    # plt.title('FANG-2018Q3 Accumulative Return')
    # plt.figure()
    
    # KHC_period_df = pd.DataFrame()
    # KHC_period_df['Market Return'] = SPY['return']['2018-10-01':'2018-12-31'].cumsum()
    # KHC_period_df['Stock Return'] = KHC['return']['2018-10-01':'2018-12-31'].cumsum()
    # KHC_period_df['Strategy Return'] = KHC_signal_df['result']['2018-10-01':'2018-12-31'].cumsum()
    # KHC_period_df.plot()
    # plt.title('KHC-2018Q4 Accumulative Return')
    # plt.figure()
    
    # NCLH_period_df = pd.DataFrame()
    # NCLH_period_df['Market Return'] = SPY['return']['2019-07-01':'2020-06-30'].cumsum()
    # NCLH_period_df['Stock Return'] = NCLH['return']['2019-07-01':'2020-06-30'].cumsum()
    # NCLH_period_df['Strategy Return'] = NCLH_signal_df['result']['2019-07-01':'2020-06-30'].cumsum()
    # NCLH_period_df.plot()
    # plt.title('NCLH-2019Q3-2020Q2 Accumulative Return')
    # plt.figure()
    
    # RCL_period_df = pd.DataFrame()
    # RCL_period_df['Market Return'] = SPY['return']['2018-01-01':'2018-03-31'].cumsum()
    # RCL_period_df['Stock Return'] = RCL['return']['2018-01-01':'2018-03-31'].cumsum()
    # RCL_period_df['Strategy Return'] = RCL_signal_df['result']['2018-01-01':'2018-03-31'].cumsum()
    # RCL_period_df.plot()
    # plt.title('RCL-2018Q1 Accumulative Return')
    # plt.figure()
    