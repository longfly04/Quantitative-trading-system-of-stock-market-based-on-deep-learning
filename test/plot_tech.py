import math
import json
import numpy as np 
import os,sys
sys.path.insert(0,'D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\quantitative_analysis_with_deep_learning')

from sklearn.preprocessing import StandardScaler

from utils.data_process import DataProcessor, DataVisualiser
from utils.base.baseobj import StockHistory

def main():
    with open('config.json', 'r', encoding='utf-8') as cfg:
        config = json.load(cfg)

    # 获取历史数据，如果股票信息为列表，则获取列表
    stock_his = StockHistory(config)
    # 定义数据处理器
    data_pro = DataProcessor(date_col=config['data']['date_col'],
                             daily_quotes=config['data']['daily_quotes'],
                             target_col=config['data']['target'])
    # 对时间进行编码
    (date_list, embeddings_list) = data_pro.encode_date_embeddings(stock_his.stock_calender)
    
    # 对多只股票的数据进行训练
    for idx, data in enumerate(stock_his.stock_history):
        # 参考交易日日历，对数据集的日期时间进行验证
        data_ = data_pro.datetime_validation(data=data, history_calendar=date_list)
        # 去重和填充空值
        data_ = data_pro.drop_dup_fill_nan(data_)
        # 计算技术指标
        data_tec = data_pro.cal_technical_indicators(data_, date_index=date_list, plot=True, save=True)
        # 计算傅里叶变换
        data_fft = data_pro.cal_fft(data_, plot=True, save=True)

if __name__ == "__main__":
    main()