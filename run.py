"""
    基于深度学习预测和决策的量化交易系统

    数据准备：更新至截至目前最新的股票数据
    训练预测模型：模型加载上一次训练权重继续训练，预测时保证时间因果性
    训练决策模型：加载预测模型的数据，根据股票数据进行决策
    全局配置文件：config

        Author：LongFly
        2020.03.01
"""
import json
import os,sys
sys.path.insert(0,'D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\quantitative_analysis_with_deep_learning')

from preparation import prepare_train
from train_forecasting import train_forecasting
from train_decision import train_decision


def main():
    """"""
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    # 准备交易行情和日历
    calender, history, all_quote = prepare_train(config, download=True)
    # 训练预测模型，得到预测向量和风险向量
    predict_results_dict = train_forecasting(   config, 
                                                calender=calender, 
                                                history=history, 
                                                forecasting_deadline='20150110')
    # 训练决策模型，初始化资金，得到
    train_decision( config=config,
                    save=True, 
                    calender=calender, 
                    history=history, 
                    predict_results_dict=predict_results_dict)



    print("A lot of work to do ...")


if __name__ == '__main__':
    main()