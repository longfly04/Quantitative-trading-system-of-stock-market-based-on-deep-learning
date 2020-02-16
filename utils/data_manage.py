import numpy as np 
import pandas as pd 
import datetime
import arrow
import os
from vnpy.trader.constant import *
from vnpy.trader.object import *

from utils.tushare_util import DailyDownloader
from utils.base.stock import Parameters, StockData


class DataDownloader(object):
    """
    数据下载器，包含完整下载和增量下载，数据输出至本地数据库
    """
    def __init__(self, data_path, stock_list_file:str,):
        """
        参数：
            data_path：存放数据路径
            stock_list_file:股票列表文件，从文件读取一揽子股票
            download_mode：下载模型，增量或者全量
        """
        self.data_path = data_path
        self.stock_list_file = stock_list_file
        self.current_date = datetime.strftime(datetime.now(), format="%Y%m%d")

    def download_stock(self, download_mode:str, start_date:str, date_col):
        """
        更新股票池中的记录（截止最新）
        """
        self.download_mode = download_mode
        self.start_date = start_date
        self.date_col = date_col

        with open(self.stock_list_file, encoding='UTF-8') as f:
            stock_dict = {}
            for line in f.readlines():
                if line == '\n':
                    break
                l = list(line.rstrip('\n').split())
                stock_dict[l[0]] = l[1][1:-1]

        if self.download_mode == 'total':
            # 全量下载
            for k,v in stock_dict.items():
                # md = MinuteDownloader(start_date='20190101', end_date='20191231', stock_code=str(v))
                # minutes_data = md.downloadMinutes(save=False)
                dd = DailyDownloader(start_date=self.start_date, 
                                     end_date=self.current_date, 
                                     stock_code=str(v), 
                                     save_dir=self.data_path
                                     )
                daily_data = dd.downloadDaily(save=True)
                print('Complete %s %s total downloading from %s to %s.' %(k, v, self.start_date, self.current_date))

        elif self.download_mode == 'additional':
            # 增量下载，可以提高速度
            for k,v in stock_dict.items():
                try:
                    path_ = search_file(self.data_path, v)
                    if len(path_) == 0:
                        # 文件夹中不含有这个文件 则启动对这个文件的全量下载
                        dd = DailyDownloader(start_date=self.start_date, 
                                     end_date=self.current_date, 
                                     stock_code=str(v), 
                                     save_dir=self.data_path
                                     )
                        daily_data = dd.downloadDaily(save=True)
                        print('Complete %s %s total downloading from %s to %s.' %(k, v, self.start_date, self.current_date))
                    else:
                        old_data = pd.read_csv(path_[0])
                        old_cal_date = str(old_data[self.date_col].iloc[-1])
                        new_start = arrow.get(old_cal_date, 'YYYYMMDD').shift(days=1).format('YYYYMMDD')
                        dd = DailyDownloader(start_date=new_start, 
                                             end_date=self.current_date, 
                                             stock_code=str(v),
                                            )
                        additional_data = dd.downloadDaily(save=False)
                        additional_data.to_csv(path_[0], mode='a+', header=False, index=True)
                        print('Complete %s %s additional downloading from %s to %s.' %(k, v, old_cal_date, self.current_date))
                except Exception as e:
                    print(e)

    def download_portfolio(self, ):
        """
        从经纪人获取账户最新资产情况，存入本地文件
        """

    def get_calender(self, start_date=None):
        """
        下载最新的交易日历
        """
        end_date = self.current_date

        st_code = '600000'
        para = Parameters(ts_code=st_code, start_date=start_date, end_date=end_date)
        stockdata = StockData(para=para)
        stock_calender = stockdata.getTradeCalender()
        self.calender = [arrow.get(i, 'YYYYMMDD') for i in stock_calender['cal_date'].unique()]

        return self.calender

    def connect_vnpy(self,):
        """
        通过客户端的形式访问vnpy并获取账户资金
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
            path_ = search_file(self.data_path, st)
            try:
                data = pd.read_csv(path_[0])
                self.stock_data_list.append(data)
            except Exception as e:
                print(e,)
        self.preprocessed = False

    def global_preprocess(self, ):
        """
        对整个股票数据进行预处理，时间唯一性、列去重、填充空值(使用0.001填充空值，防止出现无穷)
        """
        processed_list = []
        assert self.trade_calender is not None

        for data in self.stock_data_list:
            # 验证时间索引唯一性
            date_col = self.date_col
            history = pd.DataFrame([i.format('YYYYMMDD') for i in self.trade_calender], columns=[
                                   date_col], dtype='int64')
            assert pd.unique(history[date_col]).shape[0] == history.shape[0]

            if data[date_col].is_unique:
                data_ = data
            else:
                data_ = data.drop_duplicates(subset=[date_col], keep='first')
                data_ = pd.merge(data_, history, how='outer', left_on=data_[
                                 date_col], right_on=history[date_col])
                print("Data drop duplications at %d rows." %
                      (data.shape[0] - data_.shape[0]))

            # 处理空值和重复列
            data = data_.T.drop_duplicates(keep='first').T
            data.fillna(axis=0, method='ffill', inplace=True)
            data.fillna(0.001, inplace=True)
            processed_list.append(data)

        self.stock_data_list = processed_list
        self.preprocessed = True

    def get_history_data(self, ):
        """
        获取股票数据
        """
        if self.preprocessed:
            for data in self.stock_data_list:
                print('History data shape is ' , data.shape)
            return self.stock_data_list
        else:
            print("Please preprocess the data list firstly.")
            return None

    def get_trade_calender(self,):
        """
        获取交易日历
        """
        if self.preprocessed:
            print('Calender length is ', len(self.trade_calender))
            return [i.format('YYYYMMDD') for i in self.trade_calender]
        else:
            print("Please preprocess the data list firstly.")
            return None

    def get_quote(self, ):
        """
        获取全部股票池行情，传递给Portfoliomanager
        """
        quote_columns = self.quote_col[1:]
        quote_index = self.quote_col[0]
        quote_list = []
        assert len(self.stock_pool) == len(self.stock_data_list)
        '''
        # 行情数据延展为二维拼接
        for i in range(len(self.stock_data_list)):
            data_quote = self.stock_data_list[i][quote_columns]
            data_quote = data_quote.rename(columns=lambda x: x + '_' + self.stock_pool[i])
            quote_list.append(data_quote)

        total_quote = pd.concat(quote_list, axis=1, join='outer', ignore_index=False)
        '''
        for i in range(len(self.stock_data_list)):
            data_quote = self.stock_data_list[i][quote_columns].values
            quote_list.append(data_quote)
        total_quote = np.array(quote_list)
        print('Total quote shape is ', total_quote.shape )
        return total_quote


class PortfolioManager(object):
    """
    资产管理器，提供Gym环境的资产向量，回测情况下，通过行情历史计算，实盘情况下，通过交易接口获取账户信息
    """
    def __init__(self, config):
        self.data_cfg = config['data']
        self.quote_columns = self.data_cfg['daily_quotes'] # 行情数据列，传递给Env
        self.stock_list = self.data_cfg['stock_code'] # 从股票池中选择进行投资的股票

    def _step(self,):
        """"""

    def _reset(self,):
        """"""

    def load_quote(self, total_quote=None, calender=None, history_data=None,):
        """
        从本地数据文件获取最新的股票行情
        """
        # 从data manager获得的历史数据、交易日历和行情数据
        self.total_quote = total_quote
        self.calender = calender
        self.history_data = history_data

    def load_trade(self,):
        """
        获取成交的订单情况，并更新资产
        """


def search_file(path=None, filename=None):
    """
    递归查询文件下包含指定字符串的文件
    """
    res = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            search_file(item_path, filename)
        elif os.path.isfile(item_path):
            if filename in item_path:
                res.append(item_path)
    return res
