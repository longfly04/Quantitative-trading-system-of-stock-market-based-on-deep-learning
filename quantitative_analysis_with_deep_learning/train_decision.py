import numpy as np 
import pandas as pd 
import arrow
import random
import os,sys
import gym

from stable_baselines import DDPG, PPO1, TD3, HER
from stable_baselines.common.noise import NormalActionNoise,OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy, LstmPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPG_MlpPolicy
from stable_baselines.td3.policies import MlpPolicy as TD3_MlpPolicy
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.common.env_checker import check_env
from portfolio_trade.policy.custom_policy import CustomDDPGPolicy, CustomTD3Policy

from portfolio_trade.env.custom_env import Portfolio_Prediction_Env, QuotationManager, PortfolioManager

from utils.tools import search_file


def train_decision( config=None, 
                    save=False, 
                    load=False, 
                    calender=None, 
                    history=None, 
                    predict_results_dict=None, 
                    test_mode=False,
                    start_date=None,
                    stop_date=None,
                    episode_steps=1000,
                    model='DDPG'):
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

    MODEL = model

    predict_dict = {}
    for k,v in predict_results_dict.items():
        assert isinstance(v['predict_date'].iloc[0], str)
        tmp = v['predict_date'].apply(lambda x: arrow.get(x, 'YYYY-MM-DD').date())
        predict_dict[k] = v.rename(index=tmp)

    env = Portfolio_Prediction_Env( config=config,
                                    calender=calender, 
                                    stock_history=history, 
                                    window_len=1, 
                                    prediction_history=predict_dict,
                                    start_trade_date=start_date,
                                    stop_trade_date=stop_date,
                                    save=save)
    
    # 测试模式
    if test_mode:
        obs = env.reset()
        # check_env(env)
        for i in range(1000):
            W = np.random.uniform(0.0, 1.0, size=(6,))
            offer = np.random.uniform(-10.0, 10.0, size=(6,))
            obs, reward, done, infos = env.step(np.hstack((W, offer)))
            # env.render()
            if done:
                env.save_history()
                break
        env.close()
        
    # 训练模式
    if MODEL == "DDPG":
        # 添加噪声
        n_actions = env.action_space.shape
        param_noise = None
        # 适合于惯性系统控制的OU噪声
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
        
        model_path = search_file(os.path.join(sys.path[0],'saved_models', MODEL), MODEL)
        if len(model_path) > 0 and load:
            model = DDPG.load(  model_path[0], 
                            env=env,
                            policy=CustomDDPGPolicy,
                            param_noise=param_noise,
                            action_noise=action_noise,
                            # tensorboard_log='./tb_log',
                            ) 
        else:
            model = DDPG(policy=CustomDDPGPolicy,
                        env=env,
                        verbose=1,
                        param_noise=param_noise,
                        action_noise=action_noise,
                        # tensorboard_log='./tb_log',
                        )
        # 训练步数
        model.learn(total_timesteps=episode_steps,)
        model.save(os.path.join(sys.path[0],'saved_models',MODEL,MODEL + '.h5'))

    elif MODEL == 'TD3':
        n_actions = env.action_space.shape[-1]
        # 适合于惯性系统控制的OU噪声
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

        model_path = search_file(os.path.join(sys.path[0],'saved_models', MODEL), MODEL)
        if len(model_path) > 0 and load:
            model = TD3.load(model_path[0], 
                            env=env,
                            policy=CustomTD3Policy,
                            action_noise=action_noise,
                            # tensorboard_log='./tb_log',
                            ) 
        else:
            model = TD3(policy=CustomTD3Policy,
                        env=env,
                        verbose=1,
                        action_noise=action_noise,
                        # tensorboard_log='./tb_log',
                        )
        # 训练步数
        model.learn(total_timesteps=episode_steps,)
        model.save(os.path.join(sys.path[0],'saved_models',MODEL, MODEL + '.h5'))

    elif MODEL == "HER":
        """
        env必须是GoalEnv
        """
        model_class = DDPG

        # Available strategies (cf paper): future, final, episode, random
        goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

        # Wrap the model
        model = HER(policy=CustomDDPGPolicy, 
                    env=env, 
                    model_class=model_class, 
                    n_sampled_goal=4, 
                    goal_selection_strategy=goal_selection_strategy,
                    verbose=1
                    )
        model.learn(total_timesteps=episode_steps,)
        model.save(os.path.join(sys.path[0],'saved_models', MODEL, MODEL + '.h5'))

    obs = env.reset()
    # 实测模式
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info  = env.step(action)
        # env.render(info=info)
        if done:
            if save:
                env.save_history()
            env.reset()
            break

    env.close()


def order_process_trade():
    """
    处理订单（实盘环境）
    """

def connect_vnpy():
    """
    通过vnpy发送订单
    """
