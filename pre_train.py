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
    data_pro = DataProcessor(config)

    # 定义模型
    model = LSTM_Model(config)
    model.build_model()
    # 可以将之前的训练权重载入再继续训练
    if config['after_training']['continue']:
        model_file = os.path.join(config['after_training']['well_trained_dir'],
                              config['after_training']['well_trained_model'])
        model.load_model(model_file)

    # 对多只股票的数据进行训练
    for data in stock_his.stock_history:
        # 对时间进行编码
        (date_list, embeddings_list) = data_pro.encode_date_embeddings(timeseries=data[config['data']['date_col']])
        # 去重和填充空值
        data_ = data_pro.drop_dup_fill_nan(data)
        # 计算技术指标
        data_tec = data_pro.cal_technical_indicators(data_)
        # 计算傅里叶变换
        data_fft = data_pro.cal_fft(data_)
        # 计算日行情
        daily_quotes = data_pro.cal_daily_quotes(data_)
        # 将数据列中日行情列剔除，日期列剔除
        daily_quotes_features_col = config['data']['daily_quotes']
        full_features_col = config['data']['features']
        other_features_col = [x for x in data_.columns.values if x not in daily_quotes_features_col]
        del_col = ['cal_date','Unnamed: 0']
        for d in del_col:
            other_features_col.remove(d)
        # 对特征进行拼接，去重，变换等，pca_comp指定了PCA降维主成分的方差和所占的最小比例阈值
        daily_features = data_pro.concat_features([data_[other_features_col].values,
                                                   data_tec, 
                                                   data_fft],
                                                   pca_comp=config['preprocess']['pca_comp'])
        mmscaler = MinMaxScaler(feature_range=(-1, 1))
        scalar_daily_features = mmscaler.fit_transform(daily_features)
        scalar_daily_quotes = mmscaler.fit_transform(daily_quotes)
        # 组合成训练使用的特征和标签，以及日期序列索引
        y = daily_quotes[0]
        x = np.concatenate([scalar_daily_quotes, embeddings_list, scalar_daily_features], axis=1)
        date_index = date_list





    


    pass

if __name__ == "__main__":
    main()