"""
    Portfolio_Prediction_Env

    依据观察和预测进行决策的资产组合交易环境。包括PortfolioManager和StockManager
    根据论文中的设定，一个step包括两个变化，持有期内股价变化，交易期内资产向量变化，交易时产生交易摩擦和交易阻碍


    参数：
        data source：
            股票池行情Q_t                        shape:(股票池宽度N, 历史时间T, 行情列数quote_col_num)
            预测股价的step by step历史Prd_t       shape:(股票池宽度N, 预测历史时间T', 预测长度pred_len)
            预测均方误差（转化为风险系数）Var_t    shape:(股票池宽度N, 预测历史时间T', 1)
            资产价格P_t                          shape:(历史时间T, n_asset+1, )
            持有量历史V_t（用于计算总资产）        shape:(历史时间T, n_asset+1, )
            资产分配比例W_t（用于计算action）      shape:(历史时间T, n_asset+1, )

        action space：
            Dict{
                Weight：Box((0,1), shape=n_asset+1),      # 资产分配比例，用于计算交易方向和交易量
                Percentage：Box((-1,1), shape=n_asset+1)  # 交易价格相比当日价格浮动比例，用于计算交易价格,其中现金比例固定为1
                }

        observation space：
            对应于data source的全部数据，以窗口化形式迭代，shape:(reshape(data_source), obs_window_length)：
            （实际上，只需要对n_asset预测即可，通过配置确定从池中选择哪些股票进行投资）
                Dict{
                    Quote: Box(shape=(n_asset, obs_window_length, 行情列数quote_col_num)),
                    Pred_price:Box(shape=(n_asset, 预测长度pred_len)),
                    Pred_var:Box(shape=(n_asset, 准确率和损失acc+loss)),
                    Portfolio_price：Box(shape=(obs_window_length, n_asset+1, )),
                    Portfolio_volumns:Box(shape=(obs_window_length, n_asset+1, )),
                    Portfolio_weight:Box(shape=(obs_window_length, n_asset+1, )),
                    }

        constraint condition:
            1.总资产 = 资产持有量 * 资产价格        A_t = V_t * P_t
            2.sum(资产分配比例) == 1               sum(W_t_i) == 1
            3.交易向量 = 资产分配比例差值           T_t = Weight_t+1 - W_t
            4.挂单价格 = 当前资产价格乘以波动率      Order_price = P_t * Percentage [1:]
        
        order generation:
            1.处理Agent得到的交易向量和挂单价格向量。
            2.交易金额 = 总资产 * 交易向量          Order_amount = A_t * T_t
            3.对交易金额每个分量训练判断：
                round：四舍五入
                买入 buy_t+1_i ：round(Order_amount_i / Order_price_i)/100 > 0
                卖出 sell_t+1_i ：round(Order_amount_i / Order_price_i)/100 < 0
                持有 ：round(Order_amount_i / Order_price_i)/100 == 0
            4.订单历史记录本地

        trade process：用于模拟回测
            1.处理订单，默认在每个新交易日开始挂单
                买入订单：Order_price_i > low_price 判断成交
                卖出订单：Order_price_i < high_price 判断成交
            2.交易损耗（手续费）：成交额损耗 u = 0.001，不分买入卖出
                买入，资金减少量 = buy_t+1_i * Order_price_i * (1 + u)
                卖出，资金增加量 = sell_t+1_i * Order_price_i * (1 - u)
            3.订单完成之后，改变持有量，持有量每个分量都是100的倍数（除了现金），默认每个订单都完整成交

        step:
            obs, reward, done, info
            reward：论文中使用log比例，也有分为浮动收益和固定收益，我们总体上希望 log(A_t+1/A_0) 最大
                    或者在一个episode内，最终的A_t最大。

    By LongFly

"""

import gym
import gym.spaces
import pandas as pd 
import numpy as np 
import arrow

from utils.data_process import DataProcessor 


class QuotationManager(object):
    """
    股价行情管理器，关注多资产的股价量价信息
    """
    def __init__(self,  config, 
                        calender, 
                        stock_history, 
                        window_len=32,
                        start_trade_date='20140901'
                        ):
        """
        参数：
            config, 配置文件
            calender, 交易日历
            stock_history, 历史数据
            window_len, 历史数据窗口
        """
        self.config = config
        self.calender = calender
        self.stock_history = stock_history
        self.window_len = window_len
        self.start_trade_date = arrow.get(start_trade_date, 'YYYYMMDD').date()

        self.stock_list = config['data']['stock_code']
        self.quotation_col = config['data']['daily_quotes']
        self.date_col = config['data']['date_col']
        self.target_col = config['data']['target']
        self.train_pct = config['preprocess']['train_pct']

        assert len(self.stock_list) == len(self.stock_history)

        self.data_pro = DataProcessor(  date_col=self.date_col,
                                        daily_quotes=self.quotation_col,
                                        target_col=self.target_col,
                                        window_len=self.window_len,
                                        pct_scale=config['preprocess']['pct_scale'])
        # 对时间进行编码
        (date_list, self.embeddings_list) = self.data_pro.encode_date_embeddings(calender)
        self.date_index = pd.to_datetime(date_list, format='%Y%m%d').date

        # 确定决策训练的时间范围
        self.decision_daterange = self.date_index[self.date_index >= self.start_trade_date]
        
        # 定义存放行情信息的字典
        self.stock_quotation = {}
        for name, data in zip(self.stock_list, self.stock_history):
            # 计算日行情
            daily_quotes = self.data_pro.cal_daily_quotes(data)
            daily_quotes['idx'] = range(len(daily_quotes))
            try:
                daily_quotes = daily_quotes.set_index(self.date_index, drop=True)
            except Exception as e:
                print(e)
            self.stock_quotation[name] = daily_quotes

        self.reset()
        
    def _step(self,):
        """
        向前迭代一步，产生一个window的数据窗口
        """


    def reset(self,):
        """"""
        reset_date = self.decision_daterange[0]
        quotation = self.get_quotation(reset_date)

        return quotation


    def get_quotation(self, date):
        """
        获取date，股价行情
        """
        for k,v in self.stock_quotation.items():
            pass
    


class PortfolioManager(object):
    """
    资产管理器，提供Gym环境的资产向量，回测情况下，通过行情历史计算，实盘情况下，通过交易接口获取账户信息
    """
    def __init__(self,  config, 
                        calender, 
                        stock_history, 
                        init_asset=100000,
                        tax_rate=0.0025,
                        start_trade_date='20140901',
                        window_len=32,
                        ):
        """
        参数：
            config, 配置文件
            calender, 交易日历
            stock_history, 历史数据
            init_asset，初始化资产（现金）
            tax_rate，交易税率
            start_trade_date，开始交易时间
            window_len，观察窗口长度

        """
        self.config = config
        self.stock_list = config['data']['stock_code']
        self.init_asset = init_asset
        self.tax_rate = tax_rate
        self.start_trade_date = start_trade_date
        self.window_len = window_len

        self.n_asset = len(self.stock_list)

        self.reset()


    def _step(self, offer, W1, trade_date):
        """
        每个交易期的资产量变化，一步迭代更新状态

        参数：
            offer,W1：agent计算出的报价向量和分配向量
            trade_date:交易日期
        """
        P0 = self.P0
        V0 = self.V0
        A0 = self.A0
        W0 = self.W0

        # 订单计算和处理，交易期内视为股价无变化
        



    def reset(self,):
        """
        初始化资产向量和持有量
        """
        self.info = []
        # 定义价格向量
        self.P0 = np.array([1.0] + [1.0] * self.n_asset)
        # 定义持有量向量
        self.V0 = np.array([self.init_asset] + [0.0] * self.n_asset)
        # 定义总资产
        self.A0 = self.P0 * self.V0
        # 定义资产分配比例
        self.W0 = self.A0 / self.A0.sum()


    def save_history(self, save=True):
        """
        保存资产组合变化历史数据
        """





class Portfolio_Prediction_Env(gym.Env):
    """
    基于股市预测的组合资产管理模拟环境
    """
    def __init__(self, config, 
                    calender, 
                    stock_history, 
                    window_len, 
                    prediction_history, 
                    start_trade_date='20140901',
                    save=True):
        """
        参数：
            config, 配置文件
            calender, 交易日历
            stock_history, 股价历史数据
            prediction_history，预测历史数据

        method：
            step,
            reset,
            
        """
        self.config = config
        self.calender = calender
        self.stock_history = stock_history
        self.prediction_history = prediction_history
        self.window_len =window_len
        self.start_trade_date = start_trade_date
        self.save = save

        self.quotation_mgr = QuotationManager(  config=config,
                                                calender=calender,
                                                stock_history=stock_history,
                                                window_len=window_len)
        self.portfolio_mgr = PortfolioManager(  config=config,
                                                stock_history=stock_history,
                                                calender=calender,
                                                window_len=window_len,
                                                save=save)



    def step(self,):
        """"""

    def reset(self,):
        """"""




