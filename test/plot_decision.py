import os,sys
sys.path.insert(0, 'D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\quantitative_analysis_with_deep_learning')
import arrow
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import json

from utils.tools import search_file


def main():
    """"""

    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    path = os.path.join(sys.path[0], 'output')
    # 存档路径
    archive_path = os.path.join(path, '031502')
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

    # plot_statistics(statistics_list, reference=ref_index, save=False)
    plot_order(order_list)
    plot_portfolio(portfolio_list, save=False, stock_list=config['data']['stock_code'])


def plot_statistics(data_list=None, reference=None, save=True):
    """
    总资产变化情况
    """
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    episode_steps = config['training']['episode_steps']

    plot_data_col = ['accumulated_reward','reward','sharpe_of_reward','mdd_of_reward']
    save_path = os.path.join(sys.path[0], 'saved_figures')
    
    plt.figure(figsize=(16,10), dpi=160)
    index_low = arrow.now().date()
    index_high = arrow.get(0).date()

    plt.title("Growth in total assets")

    # 绘制各个数据的增长情况
    for data,_ in zip(data_list, range(len(data_list))):
        data.set_index(pd.Series([arrow.get(j, 'YYYY-MM-DD').date() for j in data['current_date'].values]), inplace=True)
        plt.plot(data[plot_data_col[0]], alpha=0.5 , label='Training %d iterations' %((_+1)*5*episode_steps))
        if data.index.values[0] < index_low:
            index_low = data.index.values[0]
        if data.index.values[-1] > index_high:
            index_high = data.index.values[-1]
    plt.legend()

    # 绘制参考的上证综指情况
    plot_index = reference.loc[index_low:index_high]
    plt.plot(plot_index['close'] / plot_index['close'].loc[index_low] ,label='SH50 Index')
    plt.legend()

    if save:
        plt.savefig(os.path.join(save_path,'61_Growth_in_total_assets.png'))
    plt.show()

    # 单步投资收益
    plt.figure(figsize=(16,10), dpi=160)
    plt.title("Single investment yield")
    for data,_ in zip(data_list, range(len(data_list))):
        plt.bar(data.index.values, data[plot_data_col[1]], alpha=0.5, label='Training %d iterations' %((_+1)*5*episode_steps))
    plt.legend()
    if save:
        plt.savefig(os.path.join(save_path,'62_Single_investment_yield.png'))
    plt.show()
    
    # sharp率和最大回撤率
    fig = plt.figure(figsize=(16,10), dpi=160)
    plt.title("Sharpe Ratio and Max Drawdown")
    # 合并
    sharp_data = []
    mdd_data = []
    for data,_ in zip(data_list, range(len(data_list))):
        data.set_index(pd.Series([arrow.get(j, 'YYYY-MM-DD').date() for j in data['current_date'].values]), inplace=True)
        # 最后一个sharpe值才是整个投资周期的夏普率
        sharp_data.append(data[plot_data_col[2]].values[-1])
        mdd_data.append(data[plot_data_col[3]].values[-1])

    ax = plt.subplot(1,1,1)
    ax2 = ax.twinx()
    x = range(len(data_list))
    ax.plot(sharp_data, label='sharpe ratio', c='orange', alpha=0.8)
    ax.fill_between(x,sharp_data,color='orange', alpha=0.1)
    ax2.plot(mdd_data, label='max drawdown', c='green', alpha=0.8)
    ax2.fill_between(x,mdd_data,color='green', alpha=0.1)
    ax.set_xlabel(' N*5000 iterations ')
    ax.set_ylabel('sharpe ratio')
    ax2.set_ylabel('max drawdown (%) ')
    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    
    if save:
        plt.savefig(os.path.join(save_path,'63_Sharp_Ratio_Max_Drawdown.png'))
    plt.show()



def plot_portfolio(data_list=None, save=False, stock_list=None):
    """
    资产向量变化

    stackplot
    """
    sns.set(style="white", palette="muted", color_codes=True)

    save_path = os.path.join(sys.path[0], 'saved_figures')
    plt.figure(figsize=(16,10), dpi=160)
    plt.title("Portfolio Distribution")

    data = data_list[-9]

    plot_data = data[[col for col in data.columns.values if col.startswith('A1_')]]
    x = [arrow.get(i,'YYYY-MM-DD').date() for i in plot_data.index.values]
    plot_data = plot_data.set_index(pd.Series(x))
    y_label = ['position'] + stock_list
    assert len(y_label) == plot_data.shape[-1]

    y_list = [plot_data['A1_'+col].values for col in y_label]

    plt.stackplot(x, y_list[0], y_list[1],y_list[2],y_list[3],y_list[4],y_list[5],labels=y_label, alpha=0.6,
                    colors=['red', 'orange','green',  'cyan', 'blue','purple', ])

    plt.legend()
    if save:
        plt.savefig(os.path.join(save_path,'65_Portfolio_Distribution.png'))
    plt.show()


def plot_order(data_list=None):
    """
    订单变化
    """

def sharpe(returns, freq=250, rfr=0.02):
    """
    夏普比率
    
    """
    return (np.sqrt(freq) * np.mean(returns - rfr)) / (np.std(returns - rfr) + 1e-7)


if __name__ == '__main__':
    main()