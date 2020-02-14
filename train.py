"""
    全局训练流程：
        1.读取股票池，拉取行情，数据增量存储，
        2.读取账户，拉取资金、成交订单
        3.调用资产管理，计算资产向量
            (时间事件——决策时间)
            4.调用算法，训练：
                1.算法检测新数据，启动数据获取与处理
                2.传入资产向量、成交订单、股票数据到环境中
                3.训练t1批数据进行预测，得到预测向量
                4.训练t2批数据进行策略提升，得到交易向量
                5.返回交易向量到订单处理
            5.订单处理接收订单，处理并等待提交
            (时间事件——订单交易)
            6.提交订单
        7.更新资产向量
        8.循环

    环境：
        1.每个交易日之前，被冻结资金会自动返还（触发资产管理事件）
        2.手续费在成交时自动扣除
        3.奖励设置为一个投资周期（episode）的总资产增长率，投资周期为全局参数，默认63日（一季度，全年约250个交易日）
        4.评估方法还有最大回撤，夏普比率（Sharp）等

    问题：
        1.模型训练时，判断订单成交的依据：提交订单价格在high~low之内，否则不成交
        2.模型决策的内容，包括成交量和成交价，分别为Box行为，其中成交量单位-手，价格单位0.01元
        3.每次observe是延迟到次日才能得到结果（关于成交和冻结资金处理问题）
        4.配置文件在训练阶段作为参数传入各个组件中，而不是在组件定义时与配置文件绑定，减少耦合性

    配置文件：
        1.数据配置文件：记录最新数据时间，避免重复训练，记录数据时间索引，训练用参数文件索引等
        2.时序预测模型配置文件：记录超参数
        3.环境配置文件：记录环境配置参数，强化学习算法参数
"""
import json
from utils.data_manage import StockManager, PortfolioManager, DataDownloader
from utils.data_process import DataProcessor
from utils.order_process import OrderProcessor, TradeSimulator

def prepare_train(config=None):
    """
    数据准备
    """
    data_cfg = config['data']

    data_downloader = DataDownloader(data_path=data_cfg['data_dir'],
                                     stock_list_file=data_cfg['SH50_list_path'],
                                     download_mode='additional',
                                     start_date=data_cfg['date_range'][0],
                                     date_col=data_cfg['date_col'])
    trade_calender = data_downloader.get_calender()

    stock_mgr = StockManager(data_path=data_cfg['data_dir'],
                            stock_pool=data_cfg['stock_code'],
                            trade_calender=trade_calender,
                            date_col=data_cfg['date_col'],
                            quote_col=data_cfg['daily_quotes'])



def train_forecasting():
    """
    训练预测模型
    """

def train_decision():
    """
    训练决策模型
    """

def trade_process():
    """
    处理订单
    """

def connect_vnpy():
    """
    通过vnpy发送订单
    """

def main():
    """"""
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    prepare_train(config)


    print("A lot of work to do ...")


if __name__ == '__main__':
    main()