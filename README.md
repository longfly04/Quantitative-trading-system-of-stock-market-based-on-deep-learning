# Quantitative trading system of stock market based on deep learning

本项目设计了一种通过深度学习进行股价时间序列预测，并利用股价预测结果训练深度强化学习交易算法，实现多资产量化交易的模拟和实时交易系统，整个算法部分嵌入到VNPY量化交易事件驱动框架中，可以轻松的实现深度学习算法交易。


Quantitative analysis with deep learning prediction and reinforcement learning transactions.

I am trying to utilise LSTM model to predict stock motivations as a foresight of the agent, which will observe a package of portfolio asset , and trade between them. 

The prediction model serves as a statistician where the agent, who will act in reinforcement learning manners , will trial errors to yield more profit . Two models will bound together to achieve the goal.


The system is embedded in the VNPY framewark, which is the biggest open-source quantative trading tool in python and is widely used across quantative field.

---
## Data

本项目中的数据来源于Tushare金融大数据社区。

Stock market data contains daily quote, market quote, financial report, foreign capitals, nation interset and so on, I will make full use of all data. 

## Prediction models

Based on Keras, I prefer LSTM with temporal attentions.

## Trade Agent

I customed a new Gym Env named "Portfolio Management with Prediction", which will have a foresight of future , also it is more flexible and real than other envs, which will not only decide buying or selling , but also asking price.That means the agent will dicide in two diminsions, when to trade and how much of it. 

The reinforcement algo is between DDPG, PPO, A3C, the trade experiments are limited in Shanghai Stock Exchange.

---

## Thanks

VNPY量化交易框架：https://www.vnpy.com/

Tushare金融大数据：https://tushare.pro/

中泰证券XTP量化平台：https://xtp.zts.com.cn/home

深度强化学习资产管理系统：http://www-scf.usc.edu/~zhan527/post/cs599/
