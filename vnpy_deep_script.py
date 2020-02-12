"""
将深度学习的预测结果通过脚本形式传递给vnpy的脚本引擎，这个脚本需要按照规则编写
"""

from time import sleep
from vnpy.app.script_trader import ScriptEngine
# from .utils.vnpy_util import PortfolioManager, OrderProcessor


def run(engine: ScriptEngine):
    """
    脚本策略的主函数说明：
        1. 唯一入参是脚本引擎ScriptEngine对象，通用它来完成查询和请求操作
        2. 该函数会通过一个独立的线程来启动运行，区别于其他策略模块的事件驱动
        3. while循环的维护，请通过engine.strategy_active状态来判断，实现可控退出

    流程：
        1.读取股票池，拉取行情
        2.读取账户，拉取资金、成交订单
        3.调用资产管理，计算资产向量
            (时间事件——决策时间)
            4.调用算法，训练：
                1.算法检测新数据，启动数据获取与处理
                2.传入资产向量、成交订单、股票数据到环境中
                3.训练t1批数据进行预测，得到预测向量
                4.训练t2批数据进行策略提升，得到交易向量
                5.返回交易向量到订单处理
            5.订单处理接收订单，处理并等待提交
            (时间事件——订单交易)
            6.提交订单
        7.更新资产向量
        8.循环

    环境：
        1.每个交易日之前，被冻结资金会自动返还（触发资产管理事件）
        2.手续费在成交时自动扣除
        3.奖励设置为一个投资周期（episode）的总资产增长率，投资周期为全局参数，默认63日（一季度，全年约250个交易日）
        4.评估方法还有最大回撤，夏普比率（Sharp）等

    问题：
        1.模型训练时，判断订单成交的依据：提交订单价格在high~low之内，否则不成交
        2.模型决策的内容，包括成交量和成交价，分别为Box行为，其中成交量单位-手，价格单位0.01元
        3.每次observe是延迟到次日才能得到结果（关于成交和冻结资金处理问题）
        4.
    """

    vt_symbols = ["600196.SSH", "600585.SSH"]

    # 订阅行情
    engine.subscribe(vt_symbols)

    msg = f"订阅股票为：{vt_symbols}"

    # 获取合约信息
    for vt_symbol in vt_symbols:
        contract = engine.get_contract(vt_symbol)
        msg = f"合约信息，{contract}"
        engine.write_log(msg)

    # 持续运行，使用strategy_active来判断是否要退出程序
    while engine.strategy_active:
        # 轮询获取行情
        for vt_symbol in vt_symbols:
            tick = engine.get_tick(vt_symbol)
            msg = f"最新行情, {tick}"
            engine.write_log(msg)

        # 等待3秒进入下一轮
        sleep(3)


if __name__ == '__main__':
    s_engine = ScriptEngine
    run(s_engine)