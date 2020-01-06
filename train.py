from time import sleep
from vnpy.app.script_trader import ScriptEngine
from vnpy.app.script_trader import init_cli_trading
from vnpy.gateway.xtp import XtpGateway
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine

def run(engine: ScriptEngine):
    """"""
    vt_symbols = ["600036.SSE", "601601.SSE", "000060.SZSE", "000002.SZSE"]

    # 订阅行情
    # engine.subscribe(vt_symbols)

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

def main():
    setting = {
    "账号": "53191000110",
    "密码": "Lw6t1JVS",
    "客户号": 1,
    "行情地址": "120.27.164.138",
    "行情端口": 6002,
    "交易地址": "120.27.164.69",
    "交易端口": 6001,
    "行情协议": "TCP",
    "授权码": "b8aa7173bba3470e390d787219b2112e"
    }
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.add_gateway(XtpGateway)
    # main_engine.start()

    engine = init_cli_trading([XtpGateway])
    engine.connect_gateway(setting,"XTP")
    # main_engine.add_engine(engine)
    
    pass
    # run(engine)


if __name__ == "__main__":
   main()