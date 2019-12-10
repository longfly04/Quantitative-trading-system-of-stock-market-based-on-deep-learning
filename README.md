# Quantitative-analysis-with-Deep-Learning
Quantitative analysis with deep learning prediction and reinforcement learning transactions.

本项目尝试使用深度学习模型预测股价波动，并结合强化学习算法训练一个交易代理，让交易代理通过对模型预测的未来股价、市场现有的交易情况（环境）和先验知识（人工策略）三个方面进行综合决策。

---

## 股价预测模型

使用基于feature pattern注意力机制的LSTM网络对A股市场股票进行分析，预测未来走势情况。

## 自动量化交易模型

强化学习领域以是否对环境建模，分为model free算法和model based算法。根据本项目中股价预测模型(Feature pattern attetion LSTM)，我们使用model based算法，将强化学习的agent与模型进行交互，同时为了引入人工先验知识，我们尝试将人工策略的行为作为agent的battle，agent同时学习对手的行为方式，从环境模型、策略反馈和对手三个方面获取信息优化自身的策略，提出一套基于battle的actor critic算法模型。

---

## 参考项目

Awesome DL/RL/SL in Quantitative Finance
https://github.com/georgezouq/awesome-deep-reinforcement-learning-in-finance#trading-system-back-test--live-trading





## 参考文献

PGPortfolio: Policy Gradient Portfolio, the source code of "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"(https://arxiv.org/pdf/1706.10059.pdf).

Uncertainty-driven Imagination for Continuous Deep Reinforcement Learning (http://proceedings.mlr.press/v78/kalweit17a/kalweit17a.pdf)

Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning  (https://arxiv.org/abs/1803.00101)

Value Prediction Network  (https://arxiv.org/abs/1707.03497)

Imagination-Augmented Agents for Deep Reinforcement Learning  (https://arxiv.org/pdf/1707.06203.pdf)


Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion  (https://arxiv.org/abs/1807.01675)