import math
import json

from sklearn.preprocessing import StandardScaler

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
    # 对时间进行编码
    (date_list, embeddings_list) = data_pro.encode_date_embeddings(stock_his.stock_calender)
    
    # 对多只股票的数据进行训练
    for idx, data in enumerate(stock_his.stock_history):
        # 参考交易日日历，对数据集的日期时间进行验证
        data_ = data_pro.datetime_validation(data=data, history_calendar=date_list)
        # 去重和填充空值
        data_ = data_pro.drop_dup_fill_nan(data_)
        # 计算技术指标
        data_tec = data_pro.cal_technical_indicators(data_, date_index=date_list)
        # 计算傅里叶变换
        data_fft = data_pro.cal_fft(data_, plot=True, save=True)
        # 计算日行情
        daily_quotes = data_pro.cal_daily_quotes(data_)
        # 将数据列中日行情列剔除，日期列剔除
        daily_quotes_features_col = config['data']['daily_quotes']
        
        other_features_col = [x for x in data_.columns.values if x not in daily_quotes_features_col]
        del_col = [x for x in other_features_col if x.endswith('date')]
        del_col = ['Unnamed: 0'] + del_col
        for d in del_col:
            other_features_col.remove(d)
        # 对除了每日行情之外的特征进行拼接，去重，变换等，pca_comp指定了PCA降维主成分的方差和所占的最小比例阈值
        daily_features = data_pro.concat_features([data_[other_features_col].values,
                                                   data_tec, 
                                                   data_fft],
                                                   pca_comp=config['preprocess']['pca_comp'])
        # 组合成训练使用的标签，以及日期价格序列
        real_price = daily_quotes.values[:, 0]
        y = real_price
        if config['preprocess']['predict_type'] == 'real':
            pass
        elif config['preprocess']['predict_type'] == 'diff':
            y = daily_quotes.values[:, 1]
        elif config['preprocess']['predict_type'] == 'pct':
            y = daily_quotes.values[:, 2]

        # 建立时间和股价的索引，作为该数据集的全局索引，
        date_index = pd.to_datetime(date_list, format='%Y%m%d').date
        date_price_index = pd.DataFrame(np.concatenate([date_list.reshape((-1, 1)), real_price.reshape((-1, 1))],
                                  axis=1), columns=['date', 'price'], index=date_index)
        date_price_index['num'] = range(len(date_price_index))
        
        # 全局标准化，还可以选择窗口数据标准化
        if config['preprocess']['norm_type'] == 'global':
            sscaler = StandardScaler()
            daily_features = pd.DataFrame(sscaler.fit_transform(daily_features))
            daily_quotes = pd.DataFrame(sscaler.fit_transform(daily_quotes))
        else:
            pass
        # 拼接特征
        x = np.concatenate([daily_quotes.values, embeddings_list, daily_features.values], axis=1)
        # 定义训练、测试、验证、强化学习数据集范围
        date_range_dict = data_pro.split_train_test_date(date_price_index)
        # 训练数据生成
        train_gen = data_pro.batch_data_generator(x, y, date_price_index, date_range_dict, 'train')
        val_gen = data_pro.batch_data_generator(x, y, date_price_index, date_range_dict, 'validation')
        reinforcement = data_pro.batch_data_generator(x, y, date_price_index, date_range_dict, 'reinforcement')
        
        # 定义模型
        stock_name = stock_his.stock_info[idx]['symbol']
        model = LSTM_Model(config, name=stock_name)
        # 定义输入输出维度
        input_shape = (config['preprocess']['window_len'], x.shape[-1])
        output_shape = (config['preprocess']['predict_len'], )
        batch_size = config['training']['batch_size']
        # 定义每一代训练、验证的次数
        epoch_steps = (date_range_dict['train'].shape[0] - batch_size, date_range_dict['validation'].shape[0] - batch_size)
        # 根据输入输出维度，每一代的训练次数，构建模型
        model.build_model(input_shape=input_shape, output_shape=output_shape, epoch_steps=epoch_steps)
        
        if config['training']['train']:
            # 训练模型或者载入模型，路径是与股票代码对应的列表
            if config['training']['load']:
                model_file = config['training']['load_path'][idx]
                model.load_model(model_file)
                if config['training']['continue']:
                    # 载入上次训练的权重并继续训练
                    model.train_model_generator(train_gen, val_gen)
            else:
                model.train_model_generator(train_gen, val_gen)
        elif config['prediction']['well_trained']:
            # 不训练模型，使用已经训练好的，well trained权重
            model_file = config['prediction']['well_trained_model'][idx]
            model.load_model(model_file)
        # 预测结果的长度是标签长度与预测步数的乘积
        predict_len = output_shape[0] * config['prediction']['predict_steps']
        assert predict_len <= date_range_dict['predict'].shape[0]
        predict_data = data_pro.predict_data_x(x, date_price_index, date_range_dict['predict'][:predict_len])
        results = model.predict_future(predict_data)
        
        real_results = data_pro.cal_daily_price(date_price_index, date_range_dict['train'][-1], results)

        data_vis = DataVisualiser(config, name=stock_name)
        data_vis.plot_prediction(date_range_dict=date_range_dict, prediction=real_results, date_price_index=date_price_index)

        try:
            # 预测数据集之外的时间的股价，也就是真正意义上的“未来股价预测”，仅供参考。
            window_len = config['preprocess']['window_len']
            last_window = x[-window_len:]
            last_day = date_price_index.index[-1]
            unknown_future = model.predict_unknown(last_window)
            real_unknown_future = data_pro.cal_daily_price(date_price_index, last_day, unknown_future, unknown=True)
            last_day = dt.datetime.strftime(last_day, format='%Y%m%d')
            print("[UNKNOWN] The future 5 days' close price from %s of stock %s is as follows." %(last_day, stock_name))
            print(real_unknown_future)
        except Exception as e:
            print(e)
    pass

if __name__ == "__main__":
    main()