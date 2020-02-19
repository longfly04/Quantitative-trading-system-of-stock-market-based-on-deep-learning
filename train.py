"""
    全局训练流程：
        1.读取股票池，拉取行情，数据增量存储，
        2.读取账户，拉取资金、成交订单
        3.调用资产管理，计算资产向量
            (时间事件——决策时间)
            4.调用算法，训练：
                1.算法检测新数据，启动数据获取与处理
                2.传入资产向量、成交订单、股票数据到环境中
                3.训练t1批数据进行预测，得到预测向量
                4.训练t2批数据进行策略提升，得到交易向量
                5.返回交易向量到订单处理
            5.订单处理接收订单，处理并等待提交
            (时间事件——订单交易)
            6.提交订单
        7.更新资产向量
        8.循环

    环境：
        1.每个交易日之前，被冻结资金会自动返还（触发资产管理事件）
        2.手续费在成交时自动扣除
        3.奖励设置为一个投资周期（episode）的总资产增长率，投资周期为全局参数，默认63日（一季度，全年约250个交易日）
        4.评估方法还有最大回撤，夏普比率（Sharp）等

    问题：
        1.模型训练时，判断订单成交的依据：提交订单价格在high~low之内，否则不成交
        2.模型决策的内容，包括成交量和成交价，分别为Box行为，其中成交量单位-手，价格单位0.01元
        3.每次observe是延迟到次日才能得到结果（关于成交和冻结资金处理问题）
        4.配置文件在训练阶段作为参数传入各个组件中，而不是在组件定义时与配置文件绑定，减少耦合性

    配置文件：
        1.数据配置文件：记录最新数据时间，避免重复训练，记录数据时间索引，训练用参数文件索引等
        2.时序预测模型配置文件：记录超参数
        3.环境配置文件：记录环境配置参数，强化学习算法参数
"""
import json
import numpy as np 
import pandas as pd 
import arrow
from sklearn.decomposition import PCA

from utils.data_manage import StockManager, PortfolioManager, DataDownloader
from utils.data_process import DataProcessor
from utils.order_process import OrderProcessor, TradeSimulator
from utils.data_process import DataProcessor

from model.baseline import LSTM_Model

from utils.tools import search_file, parse_filename, add_to_df


def prepare_train(config=None, download=False):
    """
    数据准备
    """
    data_cfg = config['data']

    # 初始化数据下载器 更新行情
    data_downloader = DataDownloader(data_path=data_cfg['data_dir'],
                                     stock_list_file=data_cfg['SH50_list_path'],
                                     )
    if download:
        data_downloader.download_stock(download_mode='additional',
                                       start_date=data_cfg['date_range'][0],
                                       date_col=data_cfg['date_col']
                                       )

    trade_calender = data_downloader.get_calender(start_date=data_cfg['date_range'][0])

    stock_mgr = StockManager(data_path=data_cfg['data_dir'],
                            stock_pool=data_cfg['stock_code'],
                            trade_calender=trade_calender,
                            date_col=data_cfg['date_col'],
                            quote_col=data_cfg['daily_quotes'])

    stock_mgr.global_preprocess()
    history = stock_mgr.get_history_data()
    all_quote = stock_mgr.get_quote()
    calender = stock_mgr.get_trade_calender()

    return calender, history, all_quote


def train_forecasting(config=None, save=False, calender=None, history=None, forecasting_deadline=None):
    """
    训练预测模型

        参数：
            config：配置文件
            save：是否保存结果至本地
            calender：交易日历
            history：股票历史行情
            forecasting_deadline：指定预测模型预测的截止时间
    """
    assert config is not None

    data_pro = DataProcessor(date_col=config['data']['date_col'],
                             daily_quotes=config['data']['daily_quotes'],
                             target_col=config['data']['target'])
    stock_list = config['data']['stock_code']
    assert len(stock_list) == len(history)
    # 对时间进行编码
    (date_list, embeddings_list) = data_pro.encode_date_embeddings(calender)

    # 预测结果的字典：索引、涨跌、方差
    predict_results_dict = {}

    # 对投资标的的历史数据进行建模
    for idx, data in zip(stock_list, history):

        # 计算技术指标、填充空值
        data_tec = data_pro.cal_technical_indicators(data, date_index=date_list)
        data_tec = data_pro.fill_nan(data_tec)

        # 计算傅里叶变换、填空
        data_fft = data_pro.cal_fft(data,)
        data_fft = data_pro.fill_nan(data_fft)

        # 计算日行情
        daily_quotes = data_pro.cal_daily_quotes(data)

        # 分离其他特征、填空
        daily_other_features = data_pro.split_quote_and_others(data)
        daily_other_features = data_pro.fill_nan(daily_other_features)

        assert data_tec.shape[0] == data_fft.shape[0] == daily_other_features.shape[0]

        # 将技术指标、傅里叶变换和除了额外指标进行拼接
        extra_features = np.concatenate([data_tec, data_fft, daily_other_features], axis=1).astype(float)
        
        # 处理无穷数
        extra_features_no_nan_inf = data_pro.fill_inf(extra_features)

        # 超限数量级，对数压缩
        scaled_extra_features = data_pro.convert_log(pd.DataFrame(extra_features_no_nan_inf), 
                                                         trigger=config['data']['log_threshold'])

        # 获取标签列
        real_price = daily_quotes.values[:, 0]
        if config['preprocess']['predict_type'] == 'real':
            y = real_price
        elif config['preprocess']['predict_type'] == 'diff':
            y = daily_quotes.values[:, 1]
        elif config['preprocess']['predict_type'] == 'pct':
            y = daily_quotes.values[:, 2]
        else:
            raise ValueError('Please input right prediction type: real/diff/pct .')

        # 建立时间和股价的索引，作为该数据集的全局索引，
        date_index = pd.to_datetime(date_list, format='%Y%m%d').date
        date_price_index = pd.DataFrame({
                                            'date':date_list, 
                                            'price': real_price,
                                            'idx':range(len(date_index))
                                        },
                                        index=date_index)
        # 拼接特征，顺序是[行情数据，时间编码，额外特征] ，标签是[标签]
        assert len(daily_quotes) == len(embeddings_list) == len(scaled_extra_features)
        x = np.concatenate([daily_quotes.values, embeddings_list, scaled_extra_features.values], axis=1)
        # 确定训练集和测试集时间范围，模型在测试集中迭代训练并预测
        date_range_dict = data_pro.split_train_test_date(date_price_index=date_price_index,
                                                         train_pct=config['preprocess']['train_pct'],
                                                         validation_pct=config['preprocess']['validation_pct'])
        # 分解训练、验证数据的时间范围
        total_train_daterange = date_range_dict['train']
        validation_daterange = date_range_dict['validation']
        step_by_step_train_daterange = date_range_dict['predict']
        """
        定义参数文件命名方式：
            YYYYMMDD_hhmmss-loss-val_loss-acc-val_acc-stock_symbol-end_date.h5
            loss:训练误差，val loss:验证误差，acc：准确率，val acc：验证准确率，stock：代码：end date：训练数据截止日期

        训练流程：
            1.从save model path中查找权重文件，有则解析文件名，加载最新，无则直接【全量训练】
            2.从最新文件名，获得end_date，根据已有数据的latest date，计算出还需要预测几个window
            3.加载最新权重，以1个batch为单位，调用【增量训练】，每次增量训练之后预测1个window，写入文件或return
            4.直到预测到latest date为止，保存权重。
            5.预测一定是step by step的，为了避免信息泄露，确保时序因果性
            
            【全量训练】：在training period中，只训练，直到predict period，从predict 的start date开始到
                        latest date，按照增量训练的方式，1个batch预测1个window
            【增量训练】：使用1个batch训练，参数来自文件
        """

        # 获取已经保存的模型参数文件,查找字符串：股票代码idx
        model_para_path = search_file(config['training']['save_model_path'], idx)

        if len(model_para_path) > 0:
            # 已有相关权重文件，解析最新文件
            try:
                parser_list = [parse_filename(filename=filename) for filename in model_para_path]
            except Exception as e:
                print(e)
            parser_list = [s for s in parser_list if s is not None]
            
            # 查找最近训练的权重
            tmp = arrow.get(0)
            for d in parser_list:
                if tmp < d['end_date']:
                    tmp = d['end_date']
                    latest_file = d
            # 定位最新训练文件
            timestamps = latest_file['train_date'].format('YYYYMMDD_HHmmss')
            latest_date = arrow.get(latest_file['end_date'], 'YYYYMMDD').date
            latest_file = search_file(config['training']['save_model_path'], timestamps)[0]
        else:
            latest_date = total_train_daterange[-1]
            latest_file = None
        # 确定step预测的起始时间
        step_by_step_start_date = arrow.get(latest_date).shift(days=1)
        if forecasting_deadline is not None:
            step_by_step_end_date = arrow.get(forecasting_deadline, 'YYYYMMDD')
        else:
            step_by_step_end_date = arrow.get(step_by_step_train_daterange[-1])

        # 定义模型
        stock_name = idx
        model = LSTM_Model(config, name=stock_name)

        # 定义输入输出维度
        input_shape = (config['preprocess']['window_len'], x.shape[-1])
        output_shape = (config['preprocess']['predict_len'], )
        batch_size = config['training']['batch_size']

        # 定义每一代训练、验证的次数
        epoch_steps = (total_train_daterange.shape[0] - batch_size, validation_daterange.shape[0])
        
        # 根据输入输出维度，每一代的训练次数，构建模型
        model.build_model(input_shape=input_shape, output_shape=output_shape, epoch_steps=epoch_steps)

        # 定义存放结果数据的dataframe，包括预测涨跌，训练均方误差和精确度
        col_names = []
        for i in range(config['preprocess']['predict_len']):
            col_name = 'pred_' + str(i)
            col_names.append(col_name)
        col_names = ['predict_date'] + col_names + ['epoch_loss', 'epoch_val_loss', 'epoch_acc', 'epoch_val_acc']
        results_df = pd.DataFrame(columns=col_names)
        
        # 全量训练，改为使用普通方法训练，节省时间
        if latest_file is None:
            # 训练数据生成
            train_gen = data_pro.batch_data_generator(x, y, date_price_index, total_train_daterange, 'train')
            val_gen = data_pro.batch_data_generator(x, y, date_price_index, validation_daterange, 'validation')
            
            epoch_loss, epoch_val_loss, epoch_acc, epoch_val_acc = \
                model.train_model_generator(train_gen, val_gen, 
                                            save_model=True, 
                                            end_date=arrow.get(latest_date).format('YYYYMMDD'))
            pred_x = data_pro.get_step_predict_X(x, date_price_index, step_by_step_start_date)
            result = model.predict_one_step(pred_x, )
            row_data = [step_by_step_start_date.format('YYYYMMDD')] + list(result) + [epoch_loss, epoch_val_loss, epoch_acc, epoch_val_acc]
            # 将一次预测的结果存入
            results_df = add_to_df(results_df, col_names, row_data)
        else:
            # 加载已有的权重，按步训练，按步预测
            model.load_model_weight(latest_file)


        for date_step in step_by_step_train_daterange[1:]:
            pred_x = data_pro.get_step_predict_X(x, date_price_index, date_step)
            results = model.predict_one_step(date_step, pred_x,)

    return None


def forecast_step_by_step(config=None, 
                          X=None,
                          Y=None,
                          input_dim=None,
                          data_processor=None,
                          h5_file=None,
                          date_price_index=None,
                          latest_date=None,
                          deadline=None,
                          stock=None,
                          save=True,
                          ):
    """
    训练一步预测一步，最后保存模型权重，输出预测结果的列表

    参数：
        config: 模型配置
        mode：模式，total完整训练，add增量训练
        h5_file：权重参数文件
        date_range_dict：范围划分
        X：训练特征
        Y：标签
        input_dim:模型输入数据维度
        date_price_index：日期索引
        latest_date：训练截止日期 = 预测开始日期
        stock：股票
        save:保存与否
        deadline:指定模型step by step 预测到什么时间
    返回：
        预测股价，loss，acc，权重文件
    """



    
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
    predict_data = data_processor.predict_data_x(x, date_price_index, date_range_dict['predict'][:predict_len])
    results = model.predict_future(predict_data)
    
    real_results = data_processor.cal_daily_price(date_price_index, date_range_dict['train'][-1], results)
    data_vis = DataVisualiser(config, name=stock_name)
    data_vis.plot_prediction(date_range_dict=date_range_dict, prediction=real_results, date_price_index=date_price_index)




def train_decision():
    """
    训练决策模型
    """

def trade_process():
    """
    处理订单
    """

def connect_vnpy():
    """
    通过vnpy发送订单
    """

def main():
    """"""
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    # 准备交易行情和日历
    calender, history, all_quote = prepare_train(config, download=False)
    # 训练预测模型，得到预测向量和风险向量
    predict_price = train_forecasting(config, 
                                      calender=calender, 
                                      history=history, 
                                      forecasting_deadline='20180101')
    # 训练决策模型，初始化资金，得到



    print("A lot of work to do ...")


if __name__ == '__main__':
    main()