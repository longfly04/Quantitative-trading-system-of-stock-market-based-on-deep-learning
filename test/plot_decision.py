import os,sys
sys.path.insert(0, 'D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\quantitative_analysis_with_deep_learning')

from utils.tools import search_file

import seaborn as sns
import pandas as pd 
import numpy as np 

def main():
    """"""
    path = os.path.join(sys.path[0], 'output')
    statistics_list = search_file(path, 'statistics')
    order_list = search_file(path, 'order')
    portfolio_list = search_file(path, 'portfolio')

    plot_statistics(statistics_list)
    plot_order(order_list)
    plot_portfolio(portfolio_list)


def plot_statistics(data=None):
    """
    总资产变化情况
    """
    

def plot_portfolio(data=None):
    """
    资产向量变化
    """

def plot_order(data=None):
    """
    订单变化
    """

if __name__ == '__main__':
    main()