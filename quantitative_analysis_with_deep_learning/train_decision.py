import numpy as np 
import pandas as pd 
import arrow

import gym
from stable_baselines import PPO2, DDPG
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import CnnLstmPolicy,LstmPolicy

from portfolio_trade.env.custom_env import Portfolio_Prediction_Env, QuotationManager, PortfolioManager


def train_decision(config=None, save=False, load=False, calender=None, history=None, predict_results_dict=None):
    """
    训练决策模型，从数据库读取数据并进行决策训练

    参数：
        config:配置文件, 
        save：保存结果, 
        calender：交易日日历, 
        history：行情信息, 
        all_quotes:拼接之后的行情信息
        predict_results_dict：预测结果信息
    """
    # 首先处理预测数据中字符串日期
    predict_dict = {}
    for k,v in predict_results_dict.items():
        assert isinstance(v['predict_date'].iloc[0], str)
        tmp = v['predict_date'].apply(lambda x: arrow.get(x, 'YYYY-MM-DD').date())
        predict_dict[k] = v.rename(index=tmp)

    env = Portfolio_Prediction_Env( config=config,
                                    calender=calender, 
                                    stock_history=history, 
                                    window_len=32, 
                                    prediction_history=predict_dict,
                                    save=save)

    if load:
        model = DDPG.load('DDPG')
    else:
        model = DDPG(   policy=LstmPolicy,
                        env=env,
                        )

    model.learn(total_timesteps=10000,)

    obs = env.reset()

    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, info , done = env.step(action[0], action[1])
        env.render()

    env.save()
    env.close()


def order_process_trade():
    """
    处理订单（实盘环境）
    """

def connect_vnpy():
    """
    通过vnpy发送订单
    """
