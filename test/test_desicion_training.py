import json
import os,sys
sys.path.insert(0,'D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\quantitative_analysis_with_deep_learning')

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
    
    # 训练决策模型，初始化资金，得到
    train_decision( config=config,
                    save=True, 
                    calender=calender, 
                    history=history, 
                    predict_results_dict=predict_results_dict,
                    test_mode=True)



    print("A lot of work to do ...")


if __name__ == '__main__':
    main()