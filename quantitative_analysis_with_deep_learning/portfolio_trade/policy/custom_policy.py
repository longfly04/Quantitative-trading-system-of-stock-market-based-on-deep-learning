import os,sys
sys.path.insert(0,'D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\quantitative_analysis_with_deep_learning')

import gym

from stable_baselines.common.policies import register_policy
from stable_baselines.ddpg.policies import FeedForwardPolicy as DDPG_FeedForwardPolicy
from stable_baselines.td3.policies import FeedForwardPolicy as TD3_FeedForwardPolicy

# 自定义策略和价值网络
class CustomDDPGPolicy(DDPG_FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__( *args, **kwargs,
                                            layers=[256, 128, 128, 64],
                                            feature_extraction="mlp")


# Register the policy, it will check that the name is not already taken
register_policy('CustomDDPGPolicy', CustomDDPGPolicy)

class CustomTD3Policy(TD3_FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__( *args, **kwargs,
                                            layers=[256, 128, 128, 64],
                                            feature_extraction="mlp")

register_policy('CustomTD3Policy', CustomTD3Policy)


