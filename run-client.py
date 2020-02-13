"""
启动客户端client，并利用rpc服务监听主程序的事件，也可以直接发送委托。
"""

import sys
sys.path.append('D:\\GitHub\\vnpy')

import json

import multiprocessing
from time import sleep
from datetime import datetime, time
from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine
from vnpy.app.script_trader import ScriptEngine

from vnpy.gateway.rpc import RpcGateway
from vnpy.gateway.xtp.xtp_gateway import XtpGateway


def run(engine: ScriptEngine):
    """
    脚本策略的主函数说明：
        1. 唯一入参是脚本引擎ScriptEngine对象，通用它来完成查询和请求操作
        2. 该函数会通过一个独立的线程来启动运行，区别于其他策略模块的事件驱动
        3. while循环的维护，请通过engine.strategy_active状态来判断，实现可控退出
    """
    # 连接XTP
    with open("D:\\Program Files\\vnpy_runtime\\.vntrader\\connect_xtp.json", encoding='UTF-8') as f:
        setting = json.load(f)
    engine.connect_gateway(setting, "XTP")

    sleep(3)

    # 获取股票列表
    with open('D:\\GitHub\\Quantitative-analysis-with-Deep-Learning\\dataset\\上证50成分股.txt', encoding='UTF-8') as f:
        stock_dict = {}
        for line in f.readlines():
            if line == '\n':
                break
            l = list(line.rstrip('\n').split())
            stock_dict[l[0]] = l[1][1:-1]

    vt_symbols = [v + '.SSE' for k,v in stock_dict.items()][:10]
    # 订阅行情
    engine.subscribe(vt_symbols)

    msg = f"订阅股票为：{vt_symbols}"
    engine.write_log(msg)

    # 获取合约信息
    for vt_symbol in vt_symbols:
        contract = engine.get_contract(vt_symbol)
        msg = f"合约信息 {contract}"
        engine.write_log(msg)

    # 持续运行，使用strategy_active来判断是否要退出程序
    while engine.strategy_active:

        all_positions = engine.get_all_positions(use_df=True)
        all_accounts = engine.get_all_accounts(use_df=True)
        all_active_orders = engine.get_all_active_orders(use_df=True)
        all_ticks = engine.get_ticks(vt_symbols=vt_symbols, use_df=True) 

        sleep(3)
        msg = f"获取全部头寸\n {all_positions}"
        engine.write_log(msg)
        
        sleep(3)
        msg = f"获取全部账户\n {all_accounts}"
        engine.write_log(msg)

        sleep(3)
        msg = f"获取全部活动订单\n {all_active_orders}"
        engine.write_log(msg)

        sleep(3)
        msg = f"获取全部行情\n {all_ticks}"
        engine.write_log(msg)

        # 等待3秒进入下一轮
        sleep(3)



def run_child():
    """
    Running in the child process.
    """
    SETTINGS["log.file"] = True
    SETTINGS["log.active"] = True
    SETTINGS["log.level"] = INFO
    SETTINGS["log.console"] = True

    rpc_setting = {
            "主动请求地址": "tcp://127.0.0.1:2014",
            "推送订阅地址": "tcp://127.0.0.1:4102"
        }

    event_engine = EventEngine()
    
    main_engine = MainEngine(event_engine)
    script_engine = ScriptEngine(main_engine, event_engine)
    main_engine.add_gateway(RpcGateway)
    main_engine.add_gateway(XtpGateway)
    # cta_engine = main_engine.add_app(CtaStrategyApp)
    main_engine.write_log("主引擎创建成功")

    log_engine = main_engine.get_engine("log")
    event_engine.register(EVENT_CTA_LOG, log_engine.process_log_event)
    main_engine.write_log("注册日志事件监听")

    while True:
        sleep(20)
        main_engine.connect(rpc_setting, "RPC")
        # run(script_engine)

def run_parent():
    """
    Running in the parent process.
    """
    print("启动client守护父进程")

    # Chinese futures market trading period (day/night)
    DAY_START = time(8, 45)
    DAY_END = time(15, 30)

    NIGHT_START = time(20, 45)
    NIGHT_END = time(2, 45)

    child_process = None

    while True:
        current_time = datetime.now().time()
        trading = False

        # Check whether in trading period
        if (
            (current_time >= DAY_START and current_time <= DAY_END)
            or (current_time >= NIGHT_START)
            or (current_time <= NIGHT_END)
        ):
            trading = True

        # Start child process in trading period
        if trading and child_process is None:
            print("启动子进程")
            child_process = multiprocessing.Process(target=run_child)
            child_process.start()
            print("子进程启动成功")

        # 非记录时间则退出子进程
        if not trading and child_process is not None:
            print("关闭子进程")
            child_process.terminate()
            child_process.join()
            child_process = None
            print("子进程关闭成功")

        sleep(5)


if __name__ == "__main__":
    run_parent()
