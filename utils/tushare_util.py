import os
import sys
import tushare as ts
import pandas as pd
import numpy as np
import datetime as dt
import arrow
import time
from .base.stock import *
from .tools import *


class DailyDownloader():
    '''
    日线行情数据下载器
    包括：
        每日的股价数据以及基本数据
        每日的上市公司财务信息
        每日市场基本信息
        每日宏观经济 利率信息

    参数：
        start_date: YYYYMMDD
        end_date: YYYYMMDD
        stock_code: 6位股票代码，字符串形式
    '''

    def __init__(self, start_date, end_date, stock_code, save_dir=None):
        if isinstance(start_date, str) == False:
            try:
                start_date = str(start_date)
            except Exception as e:
                raise TypeError('[ERROR] %s , start date should be a string.' %e)
        if isinstance(end_date, str) == False:
            try:
                end_date = str(end_date)
            except Exception as e:
                raise TypeError('[ERROR] %s , end date should be a string.' %e)
        if isinstance(stock_code, str) == False:
            try:
                stock_code = str(stock_code)
            except Exception as e:
                raise TypeError('[ERROR] %s , stock code should be a string.' %e)
        assert len(start_date) == len(end_date)

        if len(stock_code) == 6 and stock_code[-3:] != '.SH':
            stock_code = stock_code + '.SH'
        assert stock_code[-3:] == '.SH'

        self.start_date = dt.datetime.strptime(start_date, '%Y%m%d')
        self.end_date = dt.datetime.strptime(end_date, '%Y%m%d')
        self.stock_code = stock_code
        self.save_dir = save_dir
        self.paralist = []
        for i in range(self.start_date.year, self.end_date.year + 1):
            para = Parameters(ts_code=self.stock_code,
                              start_date=str(i)+'0101',
                              end_date=str(i)+'1231')
            self.paralist.append(para)

    @info
    def getDailyStock(self, save=False):
        '''
        获取每日的股价数据以及基本数据
        一边获取数据 一边修改列名
        '''
        total = pd.DataFrame()
        for para in self.paralist:
            stockdata = StockData(para)
            cal = stockdata.getTradeCalender().drop(
                columns=['exchange', 'is_open'])
            daily = stockdata.getDaily().drop(
                columns='ts_code').rename(columns=lambda x: 'daily_'+x)
            daily_indicator = stockdata.getDailyIndicator().drop(
                columns='ts_code').rename(columns=lambda x: 'daily_indicator_'+x)
            moneyflow = stockdata.getMoneyflow().drop(
                columns='ts_code').rename(columns=lambda x: 'moneyflow_'+x)

            daily_total = pd.merge(
                pd.merge(
                    pd.merge(cal, daily, left_on='cal_date',
                             right_on='daily_trade_date', how='left'),
                    daily_indicator, left_on='cal_date', right_on='daily_indicator_trade_date', how='left'),
                moneyflow, left_on='cal_date', right_on='moneyflow_trade_date', how='left')
            # 整合个股每日行情、资金信息
            res_qfq = stockdata.getRestoration(adj='qfq').drop(
                columns='ts_code').rename(columns=lambda x: 'res_qfq_'+x)
            res_hfq = stockdata.getRestoration(adj='hfq').drop(
                columns='ts_code').rename(columns=lambda x: 'res_hfq_'+x)

            restoration = pd.merge(
                res_hfq, res_qfq, left_on='res_hfq_trade_date', right_on='res_qfq_trade_date')
            # 整合复权信息
            df = pd.merge(daily_total, restoration, left_on='cal_date',
                          right_on='res_hfq_trade_date', how='left')
            total = total.append(df.sort_values(
                by='cal_date', ascending=True), ignore_index=True)
        print('Get {0} stock market data at {1} dimentions and {2} rows.'.format(
            self.stock_code, total.shape[1], total.shape[0]))
        if save:
            total.to_csv(self.save_dir + '\\Dailystock-'+self.stock_code+'.csv')
        return total

    @info
    def getDailyFinance(self, save=False):
        '''
        获取每日的上市公司财务信息
        '''
        total = pd.DataFrame()
        for para in self.paralist:
            stockdata = StockData(para)
            cal = stockdata.getTradeCalender().drop(
                columns=['exchange', 'is_open'])
            stockfinance = StockFinance(para)
            income = stockfinance.getIncome().drop(
                columns=['ts_code', ]).rename(columns=lambda x: 'income_'+x)
            balancesheet = stockfinance.getBalanceSheet().drop(
                columns='ts_code').rename(columns=lambda x: 'balancesheet_'+x)
            cashflow = stockfinance.getCashflow().drop(
                columns='ts_code').rename(columns=lambda x: 'cashflow_'+x)
            forecast = stockfinance.getForecast().drop(columns=[
                'ts_code', 'type', 'summary', 'change_reason']).rename(columns=lambda x: 'forecast_'+x)
            express = stockfinance.getExpress().drop(
                columns='ts_code').rename(columns=lambda x: 'express_'+x)
            dividend = stockfinance.getDividend().drop(
                columns=['ts_code', 'div_proc']).rename(columns=lambda x: 'dividend_'+x)
            financeindicator = stockfinance.getFinacialIndicator().drop(
                columns='ts_code').rename(columns=lambda x: 'financeindicator_'+x)

            finance_total = pd.merge(
                cal, income, left_on='cal_date', right_on='income_ann_date', how='left')
            finance_total = pd.merge(finance_total, financeindicator, left_on='cal_date',
                                     right_on='financeindicator_ann_date', how='left')
            finance_total = pd.merge(
                finance_total, balancesheet, left_on='cal_date', right_on='balancesheet_ann_date', how='left')
            finance_total = pd.merge(
                finance_total, cashflow, left_on='cal_date', right_on='cashflow_ann_date', how='left')
            finance_total = pd.merge(
                finance_total, forecast, left_on='cal_date', right_on='forecast_ann_date', how='left')
            finance_total = pd.merge(
                finance_total, express, left_on='cal_date', right_on='express_ann_date', how='left')
            finance_total = pd.merge(
                finance_total, dividend, left_on='cal_date', right_on='dividend_ann_date', how='left')

            total = total.append(finance_total.sort_values(
                by='cal_date', ascending=True), ignore_index=True)
        print('Get {0} stock finance data at {1} dimentions and {2} rows.'.format(
            self.stock_code, total.shape[1], total.shape[0]))
        if save:
            finance_total.to_csv(
                'dataset\\DailyFinance-' + self.stock_code + '.csv')
        return finance_total

    @info
    def getDailyMarket(self, save=False):
        '''
        获取每日市场基本信息
        '''
        total = pd.DataFrame()
        for para in self.paralist:
            stockdata = StockData(para)
            cal = stockdata.getTradeCalender().drop(
                columns=['exchange', 'is_open'])
            market = Market(para)
            HSGTflow = market.getMoneyflow_HSGT().rename(columns=lambda x: 'HSGTflow_'+x)
            margin = market.getMargin().drop(columns='exchange_id').rename(
                columns=lambda x: 'margin_'+x)
            if margin.shape[0]:  # 如果有记录数据 才进行聚合操作 否则会损失column数据
                margin = margin.groupby(
                    'margin_trade_date').mean().reset_index()
            pledge = market.getPledgeState().drop(
                columns='ts_code').rename(columns=lambda x: 'pledge_'+x)
            if pledge.shape[0]:
                pledge = pledge.groupby('pledge_end_date').mean().reset_index()
            repurchase = market.getRepurchase().drop(
                columns=['end_date', 'proc', 'exp_date']).rename(columns=lambda x: 'repurchase_'+x)
            if repurchase.shape[0]:
                repurchase = repurchase.groupby(
                    'repurchase_ann_date').mean().reset_index()
            desterilization = market.getDesterilization().drop(
                columns=['holder_name', 'share_type']).rename(columns=lambda x: 'desterilization_'+x)
            if desterilization.shape[0]:
                desterilization = desterilization.groupby(
                    'desterilization_ann_date').mean().reset_index()
            block = market.getBlockTrade().drop(
                columns=['buyer', 'seller']).rename(columns=lambda x: 'block_'+x)
            if block.shape[0]:
                block = block.groupby('block_trade_date').sum().reset_index()

            # 为限售解禁和大宗交易添加两列数据 便于接下来合并数据

            market_total = cal.merge(HSGTflow,
                                     left_on='cal_date', right_on='HSGTflow_trade_date', how='left').merge(margin,
                                                                                                           left_on='cal_date', right_on='margin_trade_date', how='left').merge(pledge,
                                                                                                                                                                               left_on='cal_date', right_on='pledge_end_date', how='left').merge(repurchase,
                                                                                                                                                                                                                                                 left_on='cal_date', right_on='repurchase_ann_date', how='left').merge(desterilization,
                                                                                                                                                                                                                                                                                                                       left_on='cal_date', right_on='desterilization_ann_date', how='left').merge(block,
                                                                                                                                                                                                                                                                                                                                                                                                  left_on='cal_date', right_on='block_trade_date', how='left')
            # print(market_total)
            total = total.append(market_total.sort_values(
                by='cal_date', ascending=True), ignore_index=True)
        print('Get {0} daily market data at {1} dimentions and {2} rows.'.format(
            self.stock_code, total.shape[1], total.shape[0]))
        if save:
            total.to_csv('dataset\\Dailymarket-'+self.stock_code+'.csv')
        return total

    @info
    def getDailyInterest(self, save=False):
        '''
        获取每日宏观经济 利率信息
        '''
        total = pd.DataFrame()
        for para in self.paralist:
            stockdata = StockData(para)
            cal = stockdata.getTradeCalender().drop(
                columns=['exchange', 'is_open'])

            interest = Interest(para)
            shibor = interest.getShibor().rename(columns=lambda x: 'shibor_'+x)
            shiborquote = interest.getShiborQuote().drop(
                columns='bank').rename(columns=lambda x: 'shiborquote_'+x)
            if shiborquote.shape[0]:
                shiborquote = shiborquote.groupby(
                    'shiborquote_date').mean().reset_index()
            shiborLPR = interest.getShibor_LPR().rename(columns=lambda x: 'shiborLPR_'+x)
            libor = interest.getLibor().drop(
                columns='curr_type').rename(columns=lambda x: 'libor_'+x)
            hibor = interest.getHibor().rename(columns=lambda x: 'hibor_'+x)
            wen = interest.getWenZhouIndex().rename(columns=lambda x: 'wen_'+x)

            interest_total = cal.merge(shibor,
                                       left_on='cal_date', right_on='shibor_date', how='left').merge(shiborquote,
                                                                                                     left_on='cal_date', right_on='shiborquote_date', how='left').merge(shiborLPR,
                                                                                                                                                                        left_on='cal_date', right_on='shiborLPR_date', how='left').merge(libor,
                                                                                                                                                                                                                                         left_on='cal_date', right_on='libor_date', how='left').merge(hibor,
                                                                                                                                                                                                                                                                                                      left_on='cal_date', right_on='hibor_date', how='left').merge(wen,
                                                                                                                                                                                                                                                                                                                                                                   left_on='cal_date', right_on='wen_date', how='left')
            # print(market_total)
            total = total.append(interest_total.sort_values(
                by='cal_date', ascending=True), ignore_index=True)
        print('Get {0} interest data at {1} dimentions and {2} rows.'.format(
            self.stock_code, total.shape[1], total.shape[0]))
        if save:
            total.to_csv('dataset\\Dailyinterest-'+self.stock_code+'.csv')
        return total

    @info
    def downloadDaily(self, save=False):
        stock_total = self.getDailyStock()
        finance_total = self.getDailyFinance()
        market_total = self.getDailyMarket()
        interest_total = self.getDailyInterest()

        total = stock_total.merge(finance_total,
                                  on='cal_date', how='left').merge(market_total,
                                                                   on='cal_date', how='left').merge(interest_total,
                                                                                                    on='cal_date', how='left')
        print('[Download] Get {0} daily total data at {1} dimentions and {2} rows.'.format(
            self.stock_code, total.shape[1], total.shape[0]))
        if save:
            total.to_csv(self.save_dir + '\\daily_total_' + self.stock_code[:-3]+'.csv')
        return total


class MinuteDownloader():
    '''
    分钟级行情数据下载器（目前限制为每分钟2次）

    包括：
        股票、基金、指数、期货分钟行情数据

    参数：
        start_date=None,  开始日期
        end_date=None,  结束日期
        stock_code=None,  股票（基金、指数、期货）代码
        category=None,  类型：股票 E（默认），指数 I，基金 FD，期货 FT
        save=False, 
        freq='1min'

    '''
    def __init__(self, start_date=None, end_date=None, stock_code=None, category='E', freq='1min'):
        self.start_date = start_date
        self.end_date = end_date
        if start_date is not None:
            self.start_date = dt.datetime.strptime(start_date, '%Y%m%d')
        if end_date is not None:
            self.end_date = dt.datetime.strptime(end_date, '%Y%m%d')
        if isinstance(stock_code, str) == False:
            try:
                stock_code = str(stock_code)
            except Exception as e:
                raise TypeError('[ERROR] %s , stock code should be a string.' %e)
        self.stock_code = stock_code

        assert category in ['E', 'I', 'FD', 'FT']

        self.config = {
                       'freq':freq, 
                       'span':6, 
                       'asset':category, 
                       'adj':None,
                       'adjfactor':False,
                       'factors':['tor', 'vr'], 
                       'ma':[7, 21],
                       }

    @info
    def downloadMinutes(self, save=False):
        '''
        概述：
            将Parameter按照时间切割，通过API接口获取数据后再拼接数据，返回一个parameters的列表；
            受API限制，每次只能返回7000行数据，考虑到每天数据量为4个小时，240分钟，每个月数据量为5000-6000行，
            所以把时间按月拆分，但是由于是短期交易，所以并不需要太长的历史数据
            每分钟的限制是5次
            默认获取最近6个月，如果指定开始结束时间，则在时间范围内获取。
        参数：
            ts_code：股票代码
            start_date:开始时间
            end_date:结束时间
            freq：交易频率：1min 5min 15min 30min 60min
            span：时间范围，默认6个月
            asset：查询资产的类型：E股票 I沪深指数 FT期货 FD基金 默认E
            adj：复权类型(只针对股票)：None未复权 qfq前复权 hfq后复权 , 默认None
            adjfactor：复权因子，在复权数据是，如果此参数为True，返回的数据中则带复权因子，默认为False。
            factors：股票因子（asset='E'有效）支持 tor换手率 vr量比
            ma:均线，支持任意合理int数值
            sleep:每次请求数据之间的延迟，单位秒

        返回：
            DataFrame类型数据
        '''
        datalist = pd.DataFrame()
        if self.start_date is None:
        # 如果没有指定开始结束时间，则获取最近6个月
            now = arrow.now()
            for i in range(self.config['span']):
                trade_date_start = str(now.shift(months=-(i+1)).date())
                trade_date_end = str(now.shift(months=-i).date())
                data = General_API(ts_code=self.stock_code, 
                                    start_date=trade_date_start, 
                                    end_date=trade_date_end,
                                    asset=self.config['asset'], 
                                    adj=self.config['adj'],
                                    adjfactor=self.config['adjfactor'],
                                    factors=self.config['factors'], 
                                    freq=self.config['freq'], 
                                    ma=self.config['ma'],
                                    ).getMinuteStock()
                time.sleep(31)
                try:
                    datalist = pd.concat([datalist, data], axis=0, ignore_index=True)
                except Exception as e:
                    print('[ERROR] Get minutes stock failed.\n', e)
        else:
            # 如果指定了开始结束时间，则在这个时间段内，按月获取数据并拼接
            start = arrow.get(self.start_date)
            end = arrow.get(self.end_date)
            for inter in arrow.Arrow.interval('month', start, end, 1):
                trade_date_start = str(inter[0].date())
                trade_date_end = str(inter[1].date())
                data = General_API(ts_code=self.stock_code, 
                                    start_date=trade_date_start, 
                                    end_date=trade_date_end,
                                    asset=self.config['asset'], 
                                    adj=self.config['adj'],
                                    adjfactor=self.config['adjfactor'],
                                    factors=self.config['factors'], 
                                    freq=self.config['freq'], 
                                    ma=self.config['ma'],
                                    ).getMinuteStock()
                time.sleep(31)
                try:
                    datalist = pd.concat([datalist, data], axis=0, ignore_index=True)
                except Exception as e:
                    print('[ERROR] Get minutes stock failed.\n', e)

        if save:
            datalist = datalist.sort_values(by='trade_time', ascending=True).reset_index(drop=True)
            datalist.to_csv('dataset\\minutes_total_' + self.stock_code + '.csv')
        return datalist
