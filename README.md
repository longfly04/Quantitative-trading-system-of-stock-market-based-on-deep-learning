# Quantitative trading system of stock market based on deep learning

本项目设计了一种通过深度学习进行股价时间序列预测，并利用股价预测结果训练深度强化学习交易算法，实现多资产量化交易的模拟和实时交易系统，整个算法部分嵌入到VNPY量化交易事件驱动框架中，可以轻松的实现深度学习算法交易。

Quantitative analysis with deep learning prediction and reinforcement learning transactions.

I am trying to utilise LSTM model to predict stock motivations as a foresight of the agent, which will observe a package of portfolio asset , and trade between them. 

The prediction model serves as a statistician where the agent, who will act in reinforcement learning manners , will trial errors to yield more profit . Two models will bound together to achieve the goal.


The system is embedded in the VNPY framewark, which is the biggest open-source quantative trading tool in python and is widely used across quantative field.

---
## Data

Stock market data contains daily quote, market quote, financial report, foreign capitals, nation interset and so on, I will make full use of all data. 

## Prediction models

Based on Keras, I prefer LSTM with temporal attentions.

## Trade Agent

I customed a new Gym Env named "Portfolio Management with Prediction", which will have a foresight of future , also it is more flexible and real than other envs, which will not only decide buying or selling , but also asking price.That means the agent will dicide in two diminsions, when to trade and how much of it. 

The reinforcement algo is between DDPG, PPO, A3C, the trade experiments are limited in Shanghai Stock Exchange.

---

## References

Awesome DL/RL/SL in Quantitative Finance
https://github.com/georgezouq/awesome-deep-reinforcement-learning-in-finance#trading-system-back-test--live-trading



PGPortfolio: Policy Gradient Portfolio, the source code of "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"(https://arxiv.org/pdf/1706.10059.pdf).

Uncertainty-driven Imagination for Continuous Deep Reinforcement Learning (http://proceedings.mlr.press/v78/kalweit17a/kalweit17a.pdf)

Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning  (https://arxiv.org/abs/1803.00101)

Value Prediction Network  (https://arxiv.org/abs/1707.03497)

Imagination-Augmented Agents for Deep Reinforcement Learning  (https://arxiv.org/pdf/1707.06203.pdf)


Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion  (https://arxiv.org/abs/1807.01675)