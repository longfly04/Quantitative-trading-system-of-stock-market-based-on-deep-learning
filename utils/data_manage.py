import numpy as np 
import time

from utils.tushare_util import DailyDownloader

class StockManager(object):
    """
    股票数据管理器，输入为本地数据库或者文件，输出为单一股票或者多股票的slice
    """
    def __init__(self, config):
        self.cfg = config

    def _step(self, ):
        """"""

    def _reset(self, ):
        """"""

    def load_data(self, ):
        """"""
    

class PortfolioManager(object):
    """
    资产管理器
    """
    def __init__(self, config):
        self.cfg = config

    def _step(self,):
        """"""

    def _reset(self,):
        """"""

    


class DataDownloader(object):
    """
    数据下载器，包含完整下载和增量下载，数据输出至本地数据库
    """
    def __init__(self, config):
        self.cfg = config

    def download(self, date:str):
        """"""

