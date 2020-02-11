from .vnpy.event.engine import EventEngine
from .vnpy.trader.gateway import LocalOrderManager
from .vnpy.trader.event import (EVENT_TICK,
                                EVENT_ORDER,
                                EVENT_TRADE,
                                EVENT_POSITION,
                                EVENT_ACCOUNT,)


class PortfolioManager:
    """
    本地资产计算和管理中间件，传入资产和价格之后计算资产向量
    """
    def __init__(self, gateway:XTPGateway, event_engine: EventEngine):
        """"""
        self.gateway = gateway
        self.event_engine = event_engine
        self.stock_list = []
        self.holdings = []
        self.frozen = []
        self.price_list = []
        self.history = dict()

    def on_tick(self, tick_event: EVENT_TICK):
        """
        获取tick事件并更新股票价格
        """

    def on_order(self, order_event: EVENT_ORDER): 
        """
        获取订单事件并更新持有量和冻结量
        """

    def on_trade(self, trade_event: EVENT_TRADE): 
        """
        获取交易事件并更新持有量和冻结量
        """

    def on_account(self, account_event: EVENT_ACCOUNT): 
        """
        获取账户事件并更新资金余额等
        """

    def on_position(self, position_event: EVENT_POSITION):
        """
        获取头寸事件并更新什么呢？
        """

    def describe(self,): 
        """
        执行全部信息更新，并计算总资产
        """

    def _flush_history(self,):
        """
        更新现有数据，并将历史数据存入history
        """


class OrderProcesser:
    """
    算法结果与平台之间的中间件，传入决策向量并生成订单
    """
    def __init__(self, local_order_mgr:LocalOrderManager, 
                       portfolio_mgr: PortfolioManager, 
                       event_engine: EventEngine):
        """"""
        self.local_order_mgr = local_order_mgr
        self.event_engine = event_engine
        self.portfolio_mgr = portfolio_mgr
        self.order_list = []

    def order_transform(self,):
        """
        计算交易向量并生成可执行订单列表
        """

    def check_order(self,):
        """
        检查订单是否合法
        """

    def send_order(self, ):
        """
        将处理之后合法订单交给平台进行交易
        """

 
