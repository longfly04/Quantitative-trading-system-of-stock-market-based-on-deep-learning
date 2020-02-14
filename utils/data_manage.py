import numpy as np 
import pandas as pd 
import time
import os
from vnpy.trader.constant import *
from vnpy.trader.object import *

from utils.tushare_util import DailyDownloader


class DataDownloader(object):
    """
    数据下载器，包含完整下载和增量下载，数据输出至本地数据库
    """
    def __init__(self, data_path, stock_pool:list, download_mode:str, ):
        self.data_path = data_path
        self.stock_pool = stock_pool
        self.donwload_mode = download_mode

    def download_stock(self, date:str):
        """
        更新股票池中的记录（截止最新）
        """

    def download_portfolio(self, ):
        """
        从经纪人获取账户最新资产情况，存入本地文件
        """

    def download_canlender(self,):
        """
        下载最新的交易日历
        """


class StockManager(object):
    """
    股票数据管理器，为预测模型和决策模型提供数据
    """
    def __init__(self, data_path, stock_pool:list, trade_calender=None, date_col=None, quote_col=None):
        """
        参数：
            data_path: 文件路径
            stock_pool：股票池列表
            trade_calender：交易日历（索引）
            date_col:交易日期 列名称
            quote_col:行情 列名称
        """
        self.data_path = data_path
        self.stock_pool = stock_pool
        self.trade_calender = trade_calender
        self.date_col = date_col
        self.quote_col = quote_col
        self.stock_data_list = []
        self.preprocessed = False

        self.load_data()

    def load_data(self, ):
        """
        加载最新数据
        """
        for st in self.stock_pool:
            for f in os.listdir(self.data_path):
                f_path = os.path.join(self.data_path, f)
                if st in f_path:
                    his_data = pd.read_csv(f_path)
                    self.stock_data_list.append(his_data)
        self.preprocessed = False

    def global_preprocess(self, ):
        """
        对整个股票数据进行预处理
        """
        processed_list = []
        assert self.trade_calender is not None

        for data in self.stock_data_list:
            # 验证时间索引唯一性
            date_col = self.date_col
            history = pd.DataFrame(self.trade_calender, columns=[
                                   date_col], dtype='int64')
            assert pd.unique(history[date_col]).shape[0] == history.shape[0]
            self.trade_calender = history

            if data[date_col].is_unique:
                data_ = data
            else:
                data_ = data.drop_duplicates(subset=[date_col], keep='first')
                data_ = pd.merge(data_, history, how='outer', left_on=data_[
                                 date_col], right_on=history[date_col])
                print("Data drop duplications at %d rows." %
                      (data.shape[0] - data_.shape[0]))

            # 处理空值
            data = data_
            data = data.T.drop_duplicates(keep='first').T
            data = data.fillna(axis=0, method='ffill', inplace=True)
            data = data.fillna(0, inplace=True)
            processed_list.append(data)

        self.stock_data_list = processed_list
        self.preprocessed = True

    def get_history_data(self, ):
        """
        获取股票数据
        """
        if self.preprocessed:
            return self.stock_data_list
        else:
            print("Please preprocess the data list firstly.")
            return None

    def get_trade_calender(self,):
        """
        获取交易日历
        """
        if self.preprocessed:
            return self.trade_calender
        else:
            print("Please preprocess the data list firstly.")
            return None

    def get_quote(self, ):
        """
        获取全部股票池行情，传递给Portfoliomanager
        """
        quote_columns = self.quote_col
        quote_list = []

        assert len(self.stock_pool) == len(self.stock_data_list)
        for i in range(len(self.stock_data_list)):
            data_quote = self.stock_data_list[i][quote_columns]
            data_quote = data_quote.rename(lambda x: x+self.stock_pool[i])
            quote_list.append(data_quote)

        total_quote = pd.concat(quote_list, axis=1, join='outer', ignore_index=False)

        return total_quote


class PortfolioManager(object):
    """
    资产管理器，提供Gym环境的资产向量
    """
    def __init__(self, config):
        self.data_cfg = config['data']
        self.quote_columns = self.data_cfg['daily_quotes'] # 行情数据列，传递给Env

    def _step(self,):
        """"""

    def _reset(self,):
        """"""

    def load_portfolio(self,):
        """
        从本地数据文件获取最新的资产情况
        """


