import sys
import os
from .stock import *

class StockHistory():
    '''
    股票历史数据类
    '''
    def __init__(self, config):
        self.cfg = config
        self.data_cfg = config['data']
        self.pre_cfg = config['preprocess']
        self.stock_code_list = self.data_cfg['stock_code']
        self.stock_info = self._get_stock_info()
        self.stock_calender = self._get_trade_calender()
        self.stock_history = self._get_history_data()

    def _get_trade_calender(self,):
        '''
        获取股票交易日历
        '''
        start_date = self.data_cfg['date_range'][0]
        end_date = self.data_cfg['date_range'][1]
        assert isinstance(self.stock_code_list, list)
        st_code = self.stock_code_list[0]
        para = Parameters(ts_code=st_code, start_date=start_date, end_date=end_date)
        stockdata = StockData(para=para)
        stock_calender = stockdata.getTradeCalender()
        st_cal = pd.to_datetime(stock_calender['cal_date']).unique()

        return st_cal

    def _get_stock_info(self,):
        '''
        获取当前分析的股票基本信息
        '''
        para = Parameters()
        stockdata = StockData(para)
        stock_list = stockdata.getStockList()
        stock_info_list = []
        for i in self.stock_code_list:
            curr_stock = stock_list[stock_list['symbol'] == i]
            stock_info = [v for k,v in curr_stock.T.to_dict().items()][0]
            stock_info_list.append(stock_info)

        return stock_info_list

    def _get_history_data(self,):
        '''
        从数据集中获取指定的历史数据
        '''
        data_dir = self.data_cfg['data_dir']
        history_data_list = []
        for st in self.stock_code_list:
            for f in os.listdir(data_dir):
                f_path = os.path.join(data_dir, f)
                if st in f_path:
                    his_data = pd.read_csv(f_path)
                    history_data_list.append(his_data)

        return history_data_list



