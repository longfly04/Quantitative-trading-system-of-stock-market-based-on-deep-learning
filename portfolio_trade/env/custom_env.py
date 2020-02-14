import gym
import gym.spaces

from utils.data_manager import StockManager, PortfolioManager


class Portfolio_Prediction_Env(gym.Env):
    """
    自定义的带股市预测的组合资产管理模拟环境
    """
    def __init__(self, ):
       """
       数据来自于StockManager、predict模型，包括整个股票池和账户持有的数据
       行为空间是Box类型，范围是（0，1），维度是asset+1，代表每次行为之后各个资产持有比例。
       观察空间包括行情、持有量、预测、风险
       """

    def step(self,):
        """"""

    def reset(self,):
        """"""

