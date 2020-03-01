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
from gym.spaces import Box, Dict
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
                        start_trade_date=None,
                        prediction_history=None,
                        ):
        """
        参数：
            config, 配置文件
            calender, 交易日历
            stock_history, 历史数据
            window_len, 历史数据窗口

            predict_history,预测的历史与行情数据相同的处理方式
        """
        self.config = config
        self.calender = calender
        self.stock_history = stock_history
        self.window_len = window_len
        self.start_trade_date = start_trade_date
        self.stock_list = config['data']['stock_code']
        self.quotation_col = config['data']['daily_quotes']
        self.date_col = config['data']['date_col']
        self.target_col = config['data']['target']
        self.train_pct = config['preprocess']['train_pct']
        
        # prediction数据和列名
        self.prediction_history = prediction_history
        col_names = []
        for i in range(config['preprocess']['predict_len']):
            col_name = 'pred_' + str(i)
            col_names.append(col_name)
        self.prediction_col = col_names + ['epoch_loss', 'epoch_val_loss', 'epoch_acc', 'epoch_val_acc']

        assert len(self.stock_list) == len(self.stock_history)

        self.data_pro = DataProcessor(  date_col=self.date_col,
                                        daily_quotes=self.quotation_col,
                                        target_col=self.target_col,
                                        window_len=self.window_len,
                                        pct_scale=config['preprocess']['pct_scale'])

        # 定义存放行情信息的字典
        self.stock_quotation = {}
        for name,data in self.stock_history.items():
            # 计算日行情
            daily_quotes = self.data_pro.cal_daily_quotes(data)
            daily_quotes['idx'] = range(len(daily_quotes))
            calender_index = pd.Series(self.calender)
            try:
                daily_quotes = daily_quotes.rename(index=calender_index)
            except Exception as e:
                print(e)
            self.stock_quotation[name] = daily_quotes

        self._reset()
        
    def _step(self, step_date,):
        """
        向前迭代一步，按照step date产生一个window的数据窗口

        参数：
            step_date,迭代步的日期索引
        """
        self.current_date = step_date

        quotation = self.get_window_quotation(self.current_date)
        prediction = self.get_prediction(self.current_date)
        high_low_price = self.get_high_low_price(self.current_date)

        return quotation, prediction, high_low_price


    def _reset(self,):
        """"""
        self.current_date = self.start_trade_date
        quotation = self.get_window_quotation(self.current_date)
        prediction = self.get_prediction(self.current_date)

        self.quotation_shape = quotation.shape
        self.prediction_shape = prediction.shape

        return quotation, prediction, None


    def get_window_quotation(self, current_date):
        """
        获取股价行情，时间范围：[current_date - window_len, current_date]
        """
        window_quotation = []
        window_start = [i for i in self.calender if i <= current_date][-self.window_len]
        for k,v in self.stock_quotation.items():
            try:
                quote = v[v.index >= window_start].iloc[:self.window_len]
            except Exception as e:
                print(e)

            window_quotation.append(quote.values)

        return np.concatenate(window_quotation, axis=1)
    
    def get_prediction(self, current_date):
        """
        获取预测：[current_date, current_date + predict_len]
        """
        prediction_list = []
        for k,v in self.prediction_history.items():
            try:
                prediction = v[v.index > current_date].iloc[0]
            except Exception as e:
                print(e)

            # 将有效数据列放入window中
            prediction_list.append(prediction[self.prediction_col].values)
        
        # 横向拼接行情数据
        return np.array(prediction_list)

    def get_high_low_price(self, current_date):
        """
            获取当日的最高最低价，用于计算订单
        """
        high_low_price = []
        for k,v in self.stock_quotation.items():
            try:
                high_low = v[['daily_high', 'daily_low', 'daily_open', 'daily_close']].loc[current_date]
            except Exception as e:
                print(e)
                high_low['stock'] = k
                high_low_price.append(high_low)

        return pd.concat(high_low_price)


class PortfolioManager(object):
    """
    资产管理器，提供Gym环境的资产向量，回测情况下，通过行情历史计算，实盘情况下，通过交易接口获取账户信息
    """
    def __init__(self,  config, 
                        calender, 
                        stock_history, 
                        init_asset=100000,
                        tax_rate=0.0025,
                        start_trade_date=None,
                        window_len=32,
                        save=True
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
        self.calender = calender
        self.stock_history = stock_history
        self.stock_list = config['data']['stock_code']
        self.init_asset = init_asset
        self.tax_rate = tax_rate
        self.start_trade_date = start_trade_date
        self.window_len = window_len
        self.save = save

        self.date_col = config['data']['date_col']
        self.target_col = config['data']['target']

        self.n_asset = len(self.stock_list)
        
        # 订单列表，存储次日的订单
        self.order_list = []

        self._reset()


    def _step(self, offer, W1, step_date):
        """
        每个交易期的资产量变化，一步迭代更新状态

        参数：
            offer,W1：agent计算出的报价向量和分配向量, 报价向量是波动的百分比
            trade_date:交易日期

        步骤：

        """
        P0 = self.P0
        V0 = self.V0
        A0 = self.A0
        W0 = self.W0

        # 订单计算和处理，交易期内视为股价无变化
        offer_price = P0 * offer + P0   # 出价是在原有的价格基础上乘以 (1+offer向量)
        delta_A = (W1 - W0) * A0        # 需要交易的资产数
        delta_position = delta_A[0]     # 

        for A_i in delta_A:
            pass


    def _reset(self,):
        """
        初始化资产向量和持有量
        """
        # 存储额外信息的全局infos
        self.infos = []
        # 定义价格向量
        self.P0 = self.get_price_vector(self.start_trade_date)
        # 定义持有量向量
        self.V0 = np.array([self.init_asset] + [0.0] * self.n_asset)
        # 定义总资产
        self.A0 = self.P0 * self.V0
        # 定义资产分配比例
        self.W0 = self.A0 / self.A0.sum()

        return self.P0, self.V0, self.W0


    def save_history(self, save=True):
        """
        保存资产组合变化历史数据
        """

    def get_price_vector(self, current_date):
        """
        获取指定日期的价格向量
        """
        price_list = []
        for stock, history in self.stock_history.items():
            try:
                price = history[self.target_col].loc[current_date]
            except Exception as e:
                print(e)
            price_list.append(price)

        P = np.array([[1.0] + price_list]).reshape((-1))

        return P

    def update_portfolio(self, trade_list):
        """
        根据已经成交的交易列表trade_list，更新资产向量
        """
        



class Portfolio_Prediction_Env(gym.Env):
    """
    基于股市预测的组合资产管理模拟环境
    """
    def __init__(self, config, 
                    calender, 
                    stock_history, 
                    prediction_history, 
                    init_asset=100000.0,
                    tax_rate=0.0025,
                    window_len=32, 
                    start_trade_date=None,
                    save=True):
        """
        参数：
            config, 配置文件
            calender, 交易日历 datetime对象的list
            stock_history, 股价历史数据
            prediction_history，预测历史数据

        说明：
            1.模拟环境在交易日收盘之后运行，预测未来价格，并做出投资决策
            2.每个step，首先处理上次step增加的订单，或是成交或者退出资金
            3.成交的订单，计算手续费后，加入总资产
            4.清空订单列表后，计算本次订单，加入列表
            5.冻结资金一并算入总资产
            6.未来与vnpy的回测引擎对接，可以直接包装为一个backtester类，读取每日的订单并下单
        """
        self.config = config
        self.stock_list = config['data']['stock_code']
        # 将calender转换为datetime
        self.calender = [i.date() for i in calender]
        
        # 将history中的索引转换为calender
        self.stock_history = {k:v.rename(index=pd.Series(self.calender)) for k,v in zip(self.stock_list, stock_history)}
        self.prediction_history = prediction_history
        self.window_len = window_len
        self.n_asset = len(stock_history)
        self.init_asset = init_asset
        self.tax_rate = tax_rate

        if start_trade_date is not None:
            self.decision_daterange = [i for i in self.calender if i >= arrow.get(start_trade_date, 'YYYYMMDD').date()]
        else:
            self.decision_daterange = self.calender[int(len(self.calender) * config['preprocess']['train_pct']) + self.window_len :]

        self.save = save

        self.quotation_mgr = QuotationManager(  config=config,
                                                calender=self.calender,
                                                stock_history=self.stock_history,
                                                window_len=window_len,
                                                prediction_history=prediction_history,
                                                start_trade_date=self.decision_daterange[0],
                                                )
        
        self.portfolio_mgr = PortfolioManager(  config=config,
                                                stock_history=self.stock_history,
                                                calender=self.calender,
                                                window_len=window_len,
                                                start_trade_date=self.decision_daterange[0],
                                                save=save)
        # 定义行为空间，offer的scale为100
        self.action_space = Box(low=np.array([[0.0] * (self.n_asset + 1), [-10.0]* (self.n_asset + 1)]), high=np.array([[1.0] * (self.n_asset + 1), [10.0]* (self.n_asset + 1)]))

        # 定义观察空间
        self.observation_space = Dict({
            'Quotation': Box(low=-1000.0, high=1000.0, shape=(self.n_asset, self.window_len, self.quotation_mgr.quotation_shape[-1])),
            'Prediction': Box(low=-100.0, high=100.0,shape=(self.n_asset, self.quotation_mgr.prediction_shape[-1])),
            'Portfolio_Price': Box(low=0.0, high=1000.0, shape=(self.n_asset + 1,)),
            'Portfolio_Volumns': Box(low=0.0, high=self.init_asset, shape=(self.n_asset + 1,)),
            'Portfolio_Weight': Box(low=0.0, high=1.0, shape=(self.n_asset + 1,)),
        })
        
        self.reset()


    def step(self, offer, W,):
        """
        步骤：
            1.更新行情：获取最新行情并生成新的数据窗口，并且获得当日的最高最低价用于订单计算
            2.更新订单列表：根据高低价计算当日订单成交情况，清空列表
            3.更新资产组合：根据订单成交情况，更新资产向量，计算新的订单，并加入订单列表
        """
        step_date = [i for i in self.calender if i > self.current_date][0]

        quotation, prediction, high_low_price = self.quotation_mgr._step(step_date)

        while len(self.order_list) > 0:
            order = self.order_list.pop()
            self.order_process(order)

        



    def reset(self,):
        """"""
        self.current_date = self.decision_daterange[0]

        quotation, prediction, _ = self.quotation_mgr._reset()
        P, V, W = self.portfolio_mgr._reset()

        self.order_list = []

        self.infos = []

        observation = {
            'Quotation':quotation,
            'Prediction': prediction,
            'Portfolio_Price': P,
            'Portfolio_Volumns': V,
            'Portfolio_Weight': W,
        }


        return observation




