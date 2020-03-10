import json
import os,sys
sys.path.insert(0,'D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\quantitative_analysis_with_deep_learning')
import random
import arrow
import pandas as pd 

from preparation import prepare_train
from train_decision import train_decision
from utils.tools import search_file


def main():
    """"""
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    # 准备交易行情和日历
    calender, history, all_quote = prepare_train(config, download=False)

    # 读取已经保存好的训练结果
    stock_list = config['data']['stock_code']
    results_path = os.path.join(sys.path[0], 'saved_results')
    predict_results_dict = {}
    for item in stock_list:
        csv_files = search_file(results_path, item)
        data = pd.read_csv(csv_files[0])
        predict_results_dict[item] = data

    # 全局训练范围，在这个范围内随机指定时间段进行训练
    # global_stop_date = arrow.get(config['training']['train_deadline'], 'YYYYMMDD')
    global_stop_date = arrow.get('20151231', 'YYYYMMDD')
    global_start_date = calender[int(config['preprocess']['train_pct'] * len(calender))]
    global_training_range = [i for i in calender if i < global_stop_date and i >= global_start_date]
    # 约定决策训练的时间长度
    train_len = 200

    # 随机在整个训练周期内挑选时间段训练，时间长度为200天
    for _ in range(50):
        choose_start = random.choice(global_training_range[:-train_len])
        choose_range = [i for i in global_training_range if i >= choose_start][:train_len]

        # 训练决策模型，初始化资金，得到
        train_decision( config=config,
                        save=True, 
                        calender=calender, 
                        history=history, 
                        predict_results_dict=predict_results_dict,
                        test_mode=False,
                        start_date=choose_start.date(),
                        stop_date=choose_range[-1].date(),
                        load=True
                        )



    print("A lot of work to do ...")


if __name__ == '__main__':
    main()