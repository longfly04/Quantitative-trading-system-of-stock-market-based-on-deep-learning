import os,sys
sys.path.insert(0, 'D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\quantitative_analysis_with_deep_learning')
import arrow
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

from utils.tools import search_file


def main():
    """"""
    path = os.path.join(sys.path[0], 'output')
    # 存档路径
    archive_path = os.path.join(path, '0312')
    # 以上证50为参考
    ref_index = search_file(path, '000016.SH')

    statistics_list = search_file(archive_path, 'statistics')
    order_list = search_file(archive_path, 'order')
    portfolio_list = search_file(archive_path, 'portfolio')

    # 获取数据
    ref_index = pd.read_csv(ref_index[0], index_col=0)
    ref_index = ref_index.set_index(pd.Series([arrow.get(str(i), 'YYYYMMDD').date() for i in ref_index['trade_date'].values]))

    statistics_list = [pd.read_csv(i, index_col=0) for i in statistics_list]
    order_list = [pd.read_csv(i, index_col=0) for i in order_list]
    portfolio_list = [pd.read_csv(i, index_col=0) for i in portfolio_list]

    plot_statistics(statistics_list, reference=ref_index)
    plot_order(order_list)
    plot_portfolio(portfolio_list)


def plot_statistics(data_list=None, reference=None):
    """
    总资产变化情况
    """
    plot_data_col = ['accumulated_reward','reward']

    plt.figure(figsize=(16,10), dpi=160)
    index_low = arrow.now().date()
    index_high = arrow.get(0).date()

    plt.title("Growth in total assets")

    # 绘制各个数据的增长情况
    for data,_ in zip(data_list, range(len(data_list))):
        data.set_index(pd.Series([arrow.get(j, 'YYYY-MM-DD').date() for j in data['current_date'].values]), inplace=True)
        plt.plot(data[plot_data_col[0]], alpha=0.5 , label='Training %d iterations' %((_+1)*50000))
        if data.index.values[0] < index_low:
            index_low = data.index.values[0]
        if data.index.values[-1] > index_high:
            index_high = data.index.values[-1]
    plt.legend()
    # 绘制参考的上证综指情况
    plot_index = reference.loc[index_low:index_high]
    plt.plot(plot_index['close'] / plot_index['close'].loc[index_low] )
    save_path = os.path.join(sys.path[0], 'saved_figures')
    plt.savefig(os.path.join(save_path,'61_Growth_in_total_assets.png'))
    plt.show()

    plt.figure(figsize=(16,10), dpi=160)
    plt.title("Single investment yield")
    for data,_ in zip(data_list, range(len(data_list))):
        plt.bar(data.index.values, data[plot_data_col[1]], alpha=0.5, label='Training %d iterations' %((_+1)*5000))
    plt.legend()
    plt.savefig(os.path.join(save_path,'62_Single_investment_yield.png'))
    plt.show()


def plot_portfolio(data_list=None):
    """
    资产向量变化
    """

def plot_order(data_list=None):
    """
    订单变化
    """

if __name__ == '__main__':
    main()