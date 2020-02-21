"""
数据准备：更新至截至目前最新的股票数据
训练预测模型：模型加载上一次训练权重继续训练，并将
"""
import json

from preparaion import prepare_train
from train_forecasting import train_forecasting
from train_decision import train_decision


def main():
    """"""
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    # 准备交易行情和日历
    calender, history, all_quote = prepare_train(config, download=False)
    # 训练预测模型，得到预测向量和风险向量
    train_forecasting(config, 
                      calender=calender, 
                      history=history, 
                      forecasting_deadline='20150101')
    # 训练决策模型，初始化资金，得到
    train_decision()



    print("A lot of work to do ...")


if __name__ == '__main__':
    main()