import math
import json
from sklearn.preprocessing import MinMaxScaler
from model.baseline import *
from utils.data_process import *
from utils.base.baseobj import *

def main():
    with open('config.json', 'r', encoding='utf-8') as cfg:
        config = json.load(cfg)

    # 获取历史数据，如果股票信息为列表，则获取列表
    stock_his = StockHistory(config)
    # 定义数据处理器
    data_pre = DataProcessor(config)

    date_col = config['data']['date_col']

    for data in stock_his.stock_history:
        # 描述数据的统计特征
        # data_pre.get_data_statistics(data_)
        
        # 对时间进行编码
        (date_list, embeddings_list) = data_pre.encode_date_embeddings(timeseries=data[date_col])
        
        data_ = data_pre.drop_dup_fill_nan(data)

        # 计算技术指标
        data_tec = data_pre.cal_technical_indicators(data_)

        # 计算傅里叶变换
        data_fft = data_pre.cal_fft(data_)

        # 计算日行情
        daily_quotes = data_pre.cal_daily_quotes(data_)

        # 将数据列中日行情列剔除，日期列剔除
        daily_quotes_features_col = config['data']['daily_quotes']
        full_features_col = config['data']['features']
        other_features_col = [x for x in data_.columns.values if x not in daily_quotes_features_col]
        # 剔除多余的日期列
        del_col = ['cal_date','Unnamed: 0']
        for d in del_col:
            other_features_col.remove(d)

        # 对特征进行拼接，去重，变换等，pca_comp指定了PCA降维主成分的方差和所占的最小比例阈值
        daily_features = data_pre.concat_features([data_[other_features_col].values,
                                                   data_tec, 
                                                   data_fft],
                                                   pca_comp=config['preprocess']['pca_comp'])

        mmscaler = MinMaxScaler(feature_range=(-1, 1))
        scalar_daily_features = mmscaler.fit_transform(daily_features)
        scalar_daily_quotes = mmscaler.fit_transform(daily_quotes)

        # 组合成训练使用的特征和标签
        y = daily_quotes[0]
        x = np.concatenate([scalar_daily_quotes, embeddings_list, scalar_daily_features], axis=1)

    model = LSTM_Model(config)
    


    pass

if __name__ == "__main__":
    main()