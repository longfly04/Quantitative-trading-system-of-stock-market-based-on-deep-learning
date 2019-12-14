import math
import json
from utils.data_process import *
from utils.base.baseobj import *

def main():
    with open('config.json', 'r', encoding='utf-8') as cfg:
        config = json.load(cfg)

    stock_his = StockHistory(config)
    data_pre = DataProcessor(config)
    

    pass

if __name__ == "__main__":
    main()