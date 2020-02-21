import gym
import gym.spaces

from utils.data_manager import StockManager, PortfolioManager


class Portfolio_Prediction_Env(gym.Env):
    """
    基于股市预测的组合资产管理模拟环境
    """
    def __init__(self, n_asset, ):
        """
        参数：

        data source：
            股票池行情Q_t                        shape:(股票池宽度N, 历史时间T, 行情列数quote_col_num)
            预测股价的step by step历史Prd_t       shape:(股票池宽度N, 预测历史时间T', 预测长度pred_len)
            预测方差（转化为风险系数）Var_t        shape:(股票池宽度N, 预测历史时间T', 1)
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
                    Pred_var:Box(shape=(n_asset, )),
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
            
        """
       self.n_asset = n_asset

    def step(self,):
        """"""

    def reset(self,):
        """"""

