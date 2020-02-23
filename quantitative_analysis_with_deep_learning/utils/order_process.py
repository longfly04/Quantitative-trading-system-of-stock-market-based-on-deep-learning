import numpy as np 



class TradeSimulator:
    """
    用于回测的交易模拟
    """
    def __init__(self, config):
        self.cfg = config