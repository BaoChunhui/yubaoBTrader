import warnings
warnings.filterwarnings("ignore")

from yubaotrader.app.cta_strategy.backtesting import BacktestingEngine, OptimizationSetting
from yubaotrader.trader.object import Interval
from datetime import datetime
from strategies.turtle_strategy import TurtleSignal

import pdb


engine = BacktestingEngine()
engine.set_parameters(
    vt_symbol="ETH-USDT-OKX.OKX",
    interval=Interval.MINUTE,
    start=datetime(2018, 1, 11),
    end=datetime(2024, 8, 11),
    rate=3.6/10000, # 手续费率
    slippage=0.05, # 滑点
    size=1, # 合约乘数
    pricetick=0.01, # 最小价差
    capital=10000, # 初始资金
)


# 旧的btc参数
# setting = {
#     "entry_window": 70,
#     "exit_window": 26,
#     "n_window": 2,
#     "n_min_each_bar": 290,
#     "risk_level": 0.002,
#     "capital_change_ratio": 0.4,  # 涨跌多大比例后改变capital
#     "stop_loss": 2,  # 开仓后几倍atr止损
#     "max_hold": 4  # 最多持有几份仓位
# }



# btc最优参数
# setting = {
#     "entry_window": 69,
#     "exit_window": 25,
#     "n_window": 1,
#     "n_min_each_bar": 300,
#     "risk_level": 0.002,
#     "capital_change_ratio": 0.4,  # 涨跌多大比例后改变capital
#     "stop_loss": 2,  # 开仓后几倍atr止损
#     "max_hold": 4,  # 最多持有几份仓位
#     "minimum_volume": 0.001,  # 最小挂单量
#     "max_gearing_ratio": 0.5, # 最大杠杆倍数
# }


# 旧的eth参数
# setting = {
#     "entry_window": 66,
#     "exit_window": 27,
#     "n_window": 8,
#     "n_min_each_bar": 135,
#     "risk_level": 0.0025,
#     "capital_change_ratio": 0.4,  # 涨跌多大比例后改变capital
#     "stop_loss": 2,  # 开仓后几倍atr止损
#     "max_hold": 4  # 最多持有几份仓位
# }



# eth最优参数
setting = {
    "entry_window": 65,
    "exit_window": 28,
    "n_window": 5,
    "n_min_each_bar": 85,
    "risk_level": 0.0025,
    "capital_change_ratio": 0.4,  # 涨跌多大比例后改变capital
    "stop_loss": 2,  # 开仓后几倍atr止损
    "max_hold": 4,  # 最多持有几份仓位
    "minimum_volume": 0.004,  # 最小挂单量
    "max_gearing_ratio": 0.5, # 最大杠杆倍数
    # "cci_window": 50,
    # "cci_signal": 20
}

engine.add_strategy(TurtleSignal, setting)
engine.load_data()
engine.run_backtesting()
df = engine.calculate_result()
engine.calculate_statistics()
# engine.calculate_statistics_for_optimization()
engine.show_chart()



# setting = OptimizationSetting()
# setting.set_target("r_cubic")
# setting.set_target("sharpe_ratio")
# setting.add_parameter("entry_window", 30, 100, 1)
# setting.add_parameter("exit_window", 5, 50, 1)
# setting.add_parameter("n_window", 1, 10, 1)
# setting.add_parameter("n_min_each_bar", 60, 180, 5)
# # setting.add_parameter("risk_level", 0.002, 0.03, 0.002)
# # setting.add_parameter("capital_change_ratio", 0.1, 0.8, 0.1)
# # setting.add_parameter("stop_loss", 1.0, 3.0, 0.1)
# setting.add_parameter("max_hold", 4, 6, 1)
# # setting.add_parameter("cci_window", 1, 70, 1)
# # setting.add_parameter("cci_signal", 10, 100, 10)



# result = engine.run_ga_optimization(setting)  # 优化策略参数
# result = engine.run_optimization(setting)  # 优化策略参数
# print(result)



# from pandas import DataFrame
# import pandas as pd


# rm = engine.strategy.rm
# df2 = DataFrame.from_dict([r.__dict__ for r in rm.get_results()])
# df2["balance"] = df2["pnl"].cumsum()
# pd.set_option('display.max_rows', None)
# abnormal = []
# for i in range(df2.shape[0]):
#     # print(len(df2.iloc[i, :]['trades']))
#     if len(df2.iloc[i, :]['trades']) > 5:
#         abnormal.append(i)

# print(abnormal)
