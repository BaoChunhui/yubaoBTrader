from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Callable
from itertools import product
from functools import lru_cache
from time import time
import multiprocessing
import random
import traceback
import pickle
from typing import Callable, List, Dict, Optional, Type, Tuple
import numpy as np
from pandas import DataFrame
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from deap import creator, base, tools, algorithms

from yubaoBtrader.trader.constant import (Direction, Offset, Exchange,
                                  Interval, Status)
from yubaoBtrader.trader.database import get_database, BaseDatabase
from yubaoBtrader.trader.object import OrderData, TradeData, BarData, TickData
from yubaoBtrader.trader.utility import round_to
from decimal import Decimal

database: BaseDatabase = get_database()

from .base import (
    BacktestingMode,
    EngineType,
    STOPORDER_PREFIX,
    StopOrder,
    StopOrderStatus,
    INTERVAL_DELTA_MAP
)
from .template import CtaTemplate

from .tools import ResultManager, DailyResult

import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path

# 创建数据文件目录
from yubaoBtrader.trader.utility import TEMP_DIR
data_path: Path = TEMP_DIR.joinpath('data')
if not data_path.exists():
    data_path.mkdir()


# Set deap algo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class OptimizationSetting:
    """
    Setting for runnning optimization.
    """

    def __init__(self):
        """"""
        self.params = {}
        self.target_name = ""

    def add_parameter(
        self, name: str, start: float, end: float = None, step: float = None
    ):
        """"""
        if not end and not step:
            self.params[name] = [start]
            return

        if start >= end:
            print("start value should be greater than end value")
            return

        if step <= 0:
            print("step value should be greater than zero")
            return

        value = start
        value_list = []

        while value <= end:
            value_list.append(value)
            value += step

        self.params[name] = value_list

    def set_target(self, target_name: str):
        """"""
        self.target_name = target_name

    def generate_setting(self):
        """"""
        keys = self.params.keys()
        values = self.params.values()
        products = list(product(*values))

        settings = []
        for p in products:
            setting = dict(zip(keys, p))
            settings.append(setting)

        return settings

    def generate_setting_ga(self):
        """"""
        settings_ga = []
        settings = self.generate_setting()
        for d in settings:
            param = [tuple(i) for i in d.items()]
            settings_ga.append(param)
        return settings_ga


class BacktestingEngine:
    """
    optimized BacktestingEngine
    """

    engine_type: EngineType = EngineType.BACKTESTING
    gateway_name: str = "BACKTESTING"

    def __init__(self, risk_free = None) -> None:
        """"""
        self.vt_symbol: str = ""
        self.symbol: str = ""
        self.exchange: Exchange = None
        self.start: datetime = None
        self.end: datetime = None
        self.rate: float = 0
        self.slippage: float = 0
        self.size: float = 1
        self.pricetick: float = 0
        self.capital: float = 1_000_000
        self.mode: BacktestingMode = BacktestingMode.BAR
        self.annual_days: int = 365
        self.inverse: bool = False

        self.strategy_class: Type[CtaTemplate] = None
        self.strategy: CtaTemplate = None
        self.tick: TickData
        self.bar: BarData
        self.datetime: datetime = None

        self.interval: Interval = None
        self.days: int = 0
        self.callback: Callable = None
        self.history_data: list = []

        self.stop_order_count: int = 0
        self.stop_orders: Dict[str, StopOrder] = {}
        self.active_stop_orders: Dict[str, StopOrder] = {}

        self.limit_order_count: int = 0
        self.limit_orders: Dict[str, OrderData] = {}
        self.active_limit_orders: Dict[str, OrderData] = {}

        self.trade_count: int = 0
        self.trades: Dict[str, TradeData] = {}

        self.logs: list = []

        self.daily_results: Dict[date, DailyResult] = {}
        self.daily_df: DataFrame = None
        
        if not risk_free:
            self.risk_free: float = 0.0
        else:
            self.risk_free: float = risk_free

        self.rm = ResultManager()

    def clear_data(self) -> None:
        """
        Clear all data of last backtesting.
        """
        self.strategy = None
        self.tick = None
        self.bar = None
        self.datetime = None

        self.stop_order_count = 0
        self.stop_orders.clear()
        self.active_stop_orders.clear()

        self.limit_order_count = 0
        self.limit_orders.clear()
        self.active_limit_orders.clear()

        self.trade_count = 0
        self.trades.clear()

        self.logs.clear()
        self.daily_results.clear()

    def set_parameters(
        self,
        vt_symbol: str,
        interval: Interval,
        start: datetime,
        rate: float,
        slippage: float,
        size: float,
        pricetick: float,
        capital: int = 0,
        end: datetime = None,
        mode: BacktestingMode = BacktestingMode.BAR,
        inverse: bool = False,
        annual_days: int = 365
    ):
        """"""
        self.mode = mode
        self.vt_symbol = vt_symbol
        self.interval = interval
        self.rate = rate
        self.slippage = slippage
        self.size = size
        self.pricetick = pricetick
        self.start = start

        self.symbol, exchange_str = self.vt_symbol.split(".")
        self.exchange = Exchange(exchange_str)

        self.capital = capital
        self.end = end
        self.mode = mode
        self.inverse = inverse
        self.annual_days = annual_days

    def add_strategy(self, strategy_class: Type[CtaTemplate], setting: dict) -> None:
        """"""
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, self.vt_symbol, setting
        )

    def load_data(self) -> None:
        """"""
        self.output("开始加载历史数据")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output("起始日期必须小于结束日期")
            return
        
        self.history_data.clear()       # Clear previously loaded history data

        # Load 30 days of data each time and allow for progress update
        total_days: int = (self.end - self.start).days
        progress_days: int = max(int(total_days / 10), 1)
        progress_delta: timedelta = timedelta(days=progress_days)
        interval_delta: timedelta = INTERVAL_DELTA_MAP[self.interval]
        
        # 尝试加载缓存
        self.history_data = load_cache(self.vt_symbol, self.interval, self.start, self.end)
        
        # 如果没有缓存，才从数据库中取
        if not self.history_data:

            start: datetime = self.start
            end: datetime = self.start + progress_delta
            progress = 0

            while start < self.end:
                progress_bar: str = "#" * int(progress * 10 + 1)
                self.output(f"加载进度：{progress_bar} [{progress:.0%}]")

                end: datetime = min(end, self.end)  # Make sure end time stays within set range

                if self.mode == BacktestingMode.BAR:
                    data: List[BarData] = load_bar_data(
                        self.symbol,
                        self.exchange,
                        self.interval,
                        start,
                        end
                    )
                else:
                    data: List[TickData] = load_tick_data(
                        self.symbol,
                        self.exchange,
                        start,
                        end
                    )

                self.history_data.extend(data)

                if not total_days:
                  return

                progress += progress_days / total_days
                progress = min(progress, 1)

                start = end + interval_delta
                end += progress_delta

            # 保存缓存数据文件
            save_cache(self.vt_symbol, self.interval, self.start, self.end, self.history_data)
        self.output(f"历史数据加载完成，数据量：{len(self.history_data)}")

    def run_backtesting(self) -> None:
        """"""
        if self.mode == BacktestingMode.BAR:
            func = self.new_bar
        else:
            func = self.new_tick

        self.strategy.on_init()

        # Use the first [days] of history data for initializing strategy
        day_count: int = 0
        ix: int = 0

        for ix, data in enumerate(self.history_data):
            if self.datetime and data.datetime.day != self.datetime.day:
                day_count += 1
                if day_count >= self.days:
                    break

            self.datetime = data.datetime

            try:
                self.callback(data)
            except Exception:
                self.output("raise exception, stop backtesting")
                self.output(traceback.format_exc())
                return

        self.strategy.inited = True
        self.output("initialize strategy")

        self.strategy.on_start()
        self.strategy.trading = True  # 这里置为True之后，CtaTemplate的send_order函数才开始调用BacktestingEngine的send_order函数
        self.output("start backtesting")

        # Use the rest of history data for running backtesting
        for data in self.history_data[ix:]:
            try:
                func(data)
            except Exception:
                self.output("raise exception, stop backtesting")
                self.output(traceback.format_exc())
                return

        self.strategy.on_stop()
        self.output("finish backtesting")

    def calculate_result(self):
        """"""
        self.output("start calculating pnl")

        if not self.trades:
            self.output("there is no trades，can't calculate")
            return

        # Add trade data into daily reuslt.
        for trade in self.trades.values():
            d: date = trade.datetime.date()
            daily_result: DailyResult = self.daily_results[d]
            daily_result.add_trade(trade)

            self.rm.update_trade(trade)

        # Calculate daily result by iteration.
        pre_close = 0
        start_pos = 0

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_close,
                start_pos,
                self.size,
                self.rate,
                self.slippage,
                self.inverse
            )

            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos

        # Generate dataframe
        results = defaultdict(list)

        for daily_result in self.daily_results.values():
            for key, value in daily_result.__dict__.items():
                results[key].append(value)

        self.daily_df = DataFrame.from_dict(results).set_index("date")

        self.output("finish calculating pnl ")
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True):
        """"""
        self.output("start calculating strategy's performance")

        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            # Set all statistics to 0 if no trade.
            start_date: str = ""
            end_date: str = ""
            total_days: int = 0
            profit_days: int = 0
            loss_days: int = 0
            end_balance: float = 0
            max_drawdown: float = 0
            max_ddpercent: float = 0
            max_drawdown_duration: int = 0
            total_net_pnl: float = 0
            daily_net_pnl: float = 0
            total_commission: float = 0
            daily_commission: float = 0
            total_slippage: float = 0
            daily_slippage: float = 0
            total_turnover: float = 0
            daily_turnover: float = 0
            total_trade_count: int = 0
            daily_trade_count: int = 0
            total_return: float = 0
            annual_return: float = 0
            daily_return: float = 0
            return_std: float = 0
            sharpe_ratio: float = 0
            return_drawdown_ratio: float = 0
        else:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital  # 策略每天的净值
            df["log_balance"] = np.log(df["balance"])  # 对数坐标下策略每天的净值
            # 对数坐标下策略每天的净值做线性回归
            x = df.reset_index().reset_index()["index"].values.reshape((-1, 1))
            y = df["log_balance"].values
            model = LinearRegression()
            model.fit(x, y)
            y_pred = model.predict(x)
            df["LinearRegression_log_balance"] = y_pred

            df["return"] = np.log(df["balance"] / df["balance"].shift(1)).fillna(0)  # 逐日盯市盈亏率，log盈亏率，底为e
            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )  # 从开始交易的第一天（策略完成初始化的那一天）到当前这天净值到达过的最高位
            df["drawdown"] = df["balance"] - df["highlevel"]  # 回撤
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100  # 百分比回撤

            # 取交易配对信息
            df_trades = pd.DataFrame.from_dict([r.__dict__ for r in self.rm.get_results()])
            df_trades["duration"] = df_trades["close_dt"] - df_trades["open_dt"]
            df_trades["balance"] = df_trades["pnl"].cumsum() + self.capital
            df_trades["return"] = ((df_trades["balance"] / df_trades["balance"].shift(1) - 1) * 100).fillna(0)  # 主笔对冲盈亏率
            df_trades["return"][0] = (df_trades["pnl"][0] / self.capital) * 100
            # df_trades["MAE"], df_trades["MFE"] = df_trades.apply(lambda x:calculate_MAE(x['open_dt'], x['close_dt'], x['trades']), axis = 1)
            # 处理配对后的交易信息
            mfe_list, mae_list = [], []
            maximum_number_of_consecutive_losses, number_of_consecutive_losses = 0, 0  # 最大连续亏损交易次数
            for i in range(df_trades.shape[0]):
                # 统计最大连续亏损交易次数
                pnl = df_trades.iloc[i, :]["pnl"]
                if pnl < 0:
                    number_of_consecutive_losses += 1
                else:
                    maximum_number_of_consecutive_losses = max(maximum_number_of_consecutive_losses, number_of_consecutive_losses)
                    number_of_consecutive_losses = 0

                # 计算最大有利变化幅度MFE(maximum favorable excursion)和最大不利变化幅度MAE(maximum adverse excursion)
                holding_bars: list[BarData] = self.history_data[int((df_trades.iloc[i, :]['open_dt'].to_pydatetime() - self.history_data[0].datetime)/timedelta(minutes=1)):
                                                                int((df_trades.iloc[i, :]['close_dt'].to_pydatetime() - self.history_data[0].datetime)/timedelta(minutes=1) + 1)]
                highest = max([bar.high_price for bar in holding_bars])
                lowest = min([bar.low_price for bar in holding_bars])
                open_price = df_trades.iloc[i, :]['trades'][0].price
                direction = df_trades.iloc[i, :]['trades'][0].direction
                if direction == Direction.LONG:
                    mfe = round((abs(highest-open_price)/open_price) * 100, 2)  # 最大有利变化幅度
                    mae = round((abs(lowest-open_price)/open_price) * 100, 2)  # 最大不利变化幅度
                else:
                    mfe = round((abs(lowest-open_price)/open_price) * 100, 2)  # 最大有利变化幅度
                    mae = round((abs(highest-open_price)/open_price) * 100, 2)  # 最大不利变化幅度
                mfe_list.append(mfe)
                mae_list.append(mae)
            df_trades["MFE"] = mfe_list  # 最大有利变化幅度MFE(maximum favorable excursion)
            df_trades["MAE"] = mae_list  # 最大不利变化幅度MAE(maximum adverse excursion)

            # 增加一些指标
            average_duration: pd._libs.tslibs.timedeltas.Timedelta = df_trades["duration"].mean()  # 平均持仓周期
            min_duration: pd._libs.tslibs.timedeltas.Timedelta = df_trades["duration"].min()  # 最小持仓时间
            max_duration: pd._libs.tslibs.timedeltas.Timedelta = df_trades["duration"].max()  # 最大持仓时间
            total_open_to_close_trade_count: int = df_trades.shape[0]  # 交易次数，从开仓到全部平仓算一次
            profit_trade_count: int = df_trades[df_trades['pnl'] > 0].shape[0]  # 盈利的交易次数
            loss_trade_count: int = df_trades[df_trades['pnl'] < 0].shape[0]  # 亏损的交易次数
            winning_percentage: float = round((profit_trade_count/total_open_to_close_trade_count) * 100, 2)  # 胜率
            total_profit: float = df_trades[df_trades['pnl'] > 0]['pnl'].sum()  # 总盈利金额
            total_loss: float = df_trades[df_trades['pnl'] < 0]['pnl'].sum()  # 总亏损金额
            profit_loss_sharing_ratio: float = abs(total_profit / total_loss)  # 盈亏比
            average_profit_return: float = df_trades[df_trades['return'] > 0]['return'].mean()  # 盈利的交易平均百分比收益
            average_loss_return: float = df_trades[df_trades['return'] < 0]['return'].mean()  # 亏损的交易平均百分比收益
            max_profit_return: float = df_trades[df_trades['return'] > 0]['return'].max()  # 一笔交易最大盈利多少
            max_loss_return: float = df_trades[df_trades['return'] < 0]['return'].min()  # 一笔交易最大亏损多少 
            average_return: float = average_profit_return * winning_percentage / 100 + average_loss_return * (100 - winning_percentage) / 100  # 均值收益率

            e_ratio: float = df_trades["MFE"].sum() / df_trades["MAE"].sum()  # e比率
            # 最长衰落期
            longest_decline_period = (df[df["drawdown"] >= 0].reset_index()["date"] - df[df["drawdown"] >= 0].reset_index()["date"].shift(1).fillna(df[df["drawdown"] >= 0].reset_index()["date"][0])).max().days

            # Calculate statistics value
            start_date = df.index[0]  # 开始交易的第一天（策略完成初始化的那一天）
            end_date = df.index[-1]  # 回测数据的最后一天
            total_days: int = len(df)  # 总的天数
            profit_days: int = len(df[df["net_pnl"] > 0])  # 盈利天数
            loss_days: int = len(df[df["net_pnl"] < 0])  # 亏损天数

            end_balance = df["balance"].iloc[-1]  # 最终权益
            max_drawdown = df["drawdown"].min()  # 最大回撤
            max_ddpercent = df["ddpercent"].min()  # 最大回撤率
            max_drawdown_time = df["ddpercent"].idxmin()  # 最大回撤的日期

            if isinstance(max_drawdown_time, date):
                max_drawdown_start = df["balance"][:max_drawdown_time].idxmax()  # 最大衰退之前到达最高点的日期
                new_high = df[max_drawdown_time:][df["drawdown"] >= 0].index
                if not new_high.empty:
                    max_drawdown_end = new_high[0]
                else:
                    max_drawdown_end = end_date
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days # 最大衰退持续时间
            else:
                max_drawdown_duration = 0
            
            max_ddpercents = [max_ddpercent]
            max_drawdown_durations = [max_drawdown_duration]
            new_df = df.drop(pd.date_range(max_drawdown_start+timedelta(days=1), max_drawdown_end), axis=0)
            for i in range(4):
                new_max_ddpercent = new_df["ddpercent"].min()
                new_max_drawdown_time = new_df["ddpercent"].idxmin()
                if isinstance(new_max_drawdown_time, date):
                    new_max_drawdown_start = df["balance"][:new_max_drawdown_time].idxmax()  # 最大衰退之前到达最高点的日期
                    new_high = df[new_max_drawdown_time:][df["drawdown"] >= 0].index
                    if not new_high.empty:
                        new_max_drawdown_end = new_high[0]
                    else:
                        new_max_drawdown_end = end_date
                    new_max_drawdown_duration = (new_max_drawdown_end - new_max_drawdown_start).days # 最大衰退持续时间
                else:
                    new_max_drawdown_duration = 0

                max_ddpercents.append(new_max_ddpercent)
                max_drawdown_durations.append(new_max_drawdown_duration)
                new_df = new_df.drop(pd.date_range(new_max_drawdown_start+timedelta(days=1), new_max_drawdown_end))

            top_5_ddpercent = np.mean(max_ddpercents)
            top_5_drawdown_duration = np.mean(max_drawdown_durations)

            total_net_pnl = df["net_pnl"].sum()  # 总收益
            daily_net_pnl = total_net_pnl / total_days  # 日均总收益

            total_commission = df["commission"].sum()  # 总手续费
            daily_commission = total_commission / total_days  # 日均手续费

            total_slippage = df["slippage"].sum()  # 总滑点成本
            daily_slippage = total_slippage / total_days  # 日均滑点成本

            total_turnover = df["turnover"].sum()  # 总成交额
            daily_turnover = total_turnover / total_days  # 日均成交额

            total_trade_count = df["trade_count"].sum()  # 总交易笔数
            daily_trade_count = total_trade_count / total_days  # 日均交易笔数

            total_return = (end_balance / self.capital - 1) * 100  # 总收益率
            # annual_return = total_return / total_days * self.annual_days
            annual_return = (np.exp(self.annual_days / total_days * np.log(end_balance / self.capital)) - 1) * 100  # 年化收益率
            linear_regression_annual_return = (np.exp(self.annual_days / total_days * np.log(np.exp(df["LinearRegression_log_balance"].iloc[-1]) / np.exp(df["LinearRegression_log_balance"].iloc[0]))) - 1) * 100  # 回归年度回报率
            daily_return = df["return"].mean() * 100  # 日均收益率，log盈亏率，底为e
            return_std = df["return"].std() * 100  # 日均收益率的标准差

            if return_std:
                # sharpe_ratio = daily_return / return_std * np.sqrt(365) # 夏普比率
                # daily_risk_free: float = self.risk_free / np.sqrt(self.annual_days)
                daily_risk_free: float = (np.exp(np.log(1+self.risk_free) / self.annual_days) - 1) * 100  # 由年化百分比收益率计算日均对数收益率，再乘100
                sharpe_ratio: float = (daily_return - daily_risk_free) / return_std * np.sqrt(self.annual_days) # 计算夏普比率
            else:
                sharpe_ratio = 0

            return_drawdown_ratio = -total_return / max_ddpercent # 收益回撤比
            mra_ratio = -annual_return / max_ddpercent # MRA比率

            r_cubic = linear_regression_annual_return / (-top_5_ddpercent * (top_5_drawdown_duration / self.annual_days))

        # Output
        if output:
            self.output("-" * 30)
            self.output(f"start date：\t{start_date}")
            self.output(f"end date：\t{end_date}")

            self.output(f"total days：\t{total_days}")
            self.output(f"profit days：\t{profit_days}")
            self.output(f"loss days：\t{loss_days}")

            self.output(f"capital：\t{self.capital:,.2f}")
            self.output(f"end balance：\t{end_balance:,.2f}")

            self.output(f"total return：\t{total_return:,.2f}%")
            self.output(f"annual return：\t{annual_return:,.2f}%")
            self.output(f"linear regression annual return：\t{linear_regression_annual_return:,.2f}%")
            self.output(f"max drawdown: \t{max_drawdown:,.2f}")
            self.output(f"max drawdown percent: \t{max_ddpercent:,.2f}%")
            self.output(f"max drawdown duration: \t{max_drawdown_duration} days")

            self.output(f"total net pnl：\t{total_net_pnl:,.2f}")
            self.output(f"total commission：\t{total_commission:,.2f}")
            self.output(f"total slippage：\t{total_slippage:,.2f}")
            self.output(f"total turnover：\t{total_turnover:,.2f}")
            self.output(f"total trade count：\t{total_trade_count}")

            self.output(f"daily net pnl：\t{daily_net_pnl:,.2f}")
            self.output(f"daily commission：\t{daily_commission:,.2f}")
            self.output(f"daily slippage：\t{daily_slippage:,.2f}")
            self.output(f"daily turnover：\t{daily_turnover:,.2f}")
            self.output(f"daily trade count：\t{daily_trade_count}")

            self.output(f"daily return：\t{daily_return:,.2f}%")
            self.output(f"return std：\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"return drawdown ratio：\t{return_drawdown_ratio:,.2f}")
            self.output(f"mra ratio：\t{mra_ratio:,.2f}")
            self.output(f"r cubic：\t{r_cubic:,.2f}")

            self.output(f"average duration：\t{average_duration}")
            self.output(f"min duration：\t{min_duration}")
            self.output(f"max duration：\t{max_duration}")
            self.output(f"total open to close trade count：\t{total_open_to_close_trade_count}")
            self.output(f"profit trade count：\t{profit_trade_count}")
            self.output(f"loss trade count：\t{loss_trade_count}")
            self.output(f"maximum number of consecutive losses：\t{maximum_number_of_consecutive_losses}")
            self.output(f"longest decline period：\t{longest_decline_period} days")
            self.output(f"winning percentage：\t{winning_percentage:,.2f}")
            self.output(f"total profit：\t{total_profit:,.2f}")
            self.output(f"total loss：\t{total_loss:,.2f}")
            self.output(f"profit loss sharing ratio：\t{profit_loss_sharing_ratio:,.2f}")
            self.output(f"average profit return：\t{average_profit_return:,.2f}%")
            self.output(f"average loss return：\t{average_loss_return:,.2f}%")
            self.output(f"max profit return：\t{max_profit_return:,.2f}%")
            self.output(f"max loss return：\t{max_loss_return:,.2f}%")
            self.output(f"average return：\t{average_return:,.2f}%")
            self.output(f"e ratio：\t{e_ratio:,.2f}")

        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "linear_regression_annual_return": linear_regression_annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
            "mra_ratio": mra_ratio,
            "r_cubic": r_cubic,     
            "average_duration": average_duration,             # 新增指标
            "min_duration": min_duration,
            "max_duration": max_duration,
            "total_open_to_close_trade_count": total_open_to_close_trade_count,
            "profit_trade_count": profit_trade_count,
            "loss_trade_count": loss_trade_count,
            "winning_percentage": winning_percentage,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "profit_loss_sharing_ratio": profit_loss_sharing_ratio,
            "average_profit_return": average_profit_return,
            "average_loss_return": average_loss_return,
            "max_profit_return": max_profit_return,
            "max_loss_return": max_loss_return,
            "average_return": average_return,
            "e_ratio": e_ratio
        }

        # Filter potential error infinite value
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        self.output("finish calculating strategy's performance")
        return statistics
    
    def calculate_statistics_for_optimization(self, df: DataFrame = None, output=True):
        """"""
        self.output("start calculating strategy's performance")

        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            # Set all statistics to 0 if no trade.
            start_date: str = ""
            end_date: str = ""
            total_days: int = 0
            profit_days: int = 0
            loss_days: int = 0
            end_balance: float = 0
            max_drawdown: float = 0
            max_ddpercent: float = 0
            max_drawdown_duration: int = 0
            total_net_pnl: float = 0
            daily_net_pnl: float = 0
            total_commission: float = 0
            daily_commission: float = 0
            total_slippage: float = 0
            daily_slippage: float = 0
            total_turnover: float = 0
            daily_turnover: float = 0
            total_trade_count: int = 0
            daily_trade_count: int = 0
            total_return: float = 0
            annual_return: float = 0
            daily_return: float = 0
            return_std: float = 0
            sharpe_ratio: float = 0
            return_drawdown_ratio: float = 0
            r_cubic: float = 0
        else:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital  # 策略每天的净值
            df["log_balance"] = np.log(df["balance"])  # 对数坐标下策略每天的净值
            # 对数坐标下策略每天的净值做线性回归
            x = df.reset_index().reset_index()["index"].values.reshape((-1, 1))
            y = df["log_balance"].values
            model = LinearRegression()
            model.fit(x, y)
            y_pred = model.predict(x)
            df["LinearRegression_log_balance"] = y_pred

            df["return"] = np.log(df["balance"] / df["balance"].shift(1)).fillna(0)  # 逐日盯市盈亏率，log盈亏率，底为e
            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )  # 从开始交易的第一天（策略完成初始化的那一天）到当前这天净值到达过的最高位
            df["drawdown"] = df["balance"] - df["highlevel"]  # 回撤
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100  # 百分比回撤

            # Calculate statistics value
            start_date = df.index[0]  # 开始交易的第一天（策略完成初始化的那一天）
            end_date = df.index[-1]  # 回测数据的最后一天
            total_days: int = len(df)  # 总的天数
            profit_days: int = len(df[df["net_pnl"] > 0])  # 盈利天数
            loss_days: int = len(df[df["net_pnl"] < 0])  # 亏损天数

            end_balance = df["balance"].iloc[-1]  # 最终权益
            max_drawdown = df["drawdown"].min()  # 最大回撤
            max_ddpercent = df["ddpercent"].min()  # 最大回撤率
            max_drawdown_time = df["ddpercent"].idxmin()  # 最大回撤的日期

            if isinstance(max_drawdown_time, date):
                max_drawdown_start = df["balance"][:max_drawdown_time].idxmax()  # 最大衰退之前到达最高点的日期
                new_high = df[max_drawdown_time:][df["drawdown"] >= 0].index
                if not new_high.empty:
                    max_drawdown_end = new_high[0]
                else:
                    max_drawdown_end = end_date
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days # 最大衰退持续时间
            else:
                max_drawdown_duration = 0
            
            max_ddpercents = [max_ddpercent]
            max_drawdown_durations = [max_drawdown_duration]
            new_df = df.drop(pd.date_range(max_drawdown_start+timedelta(days=1), max_drawdown_end), axis=0)
            for i in range(4):
                new_max_ddpercent = new_df["ddpercent"].min()
                new_max_drawdown_time = new_df["ddpercent"].idxmin()
                if isinstance(new_max_drawdown_time, date):
                    new_max_drawdown_start = df["balance"][:new_max_drawdown_time].idxmax()  # 最大衰退之前到达最高点的日期
                    new_high = df[new_max_drawdown_time:][df["drawdown"] >= 0].index
                    if not new_high.empty:
                        new_max_drawdown_end = new_high[0]
                    else:
                        new_max_drawdown_end = end_date
                    new_max_drawdown_duration = (new_max_drawdown_end - new_max_drawdown_start).days # 最大衰退持续时间
                else:
                    new_max_drawdown_duration = 0

                max_ddpercents.append(new_max_ddpercent)
                max_drawdown_durations.append(new_max_drawdown_duration)
                new_df = new_df.drop(pd.date_range(new_max_drawdown_start+timedelta(days=1), new_max_drawdown_end))

            top_5_ddpercent = np.mean(max_ddpercents)
            top_5_drawdown_duration = np.mean(max_drawdown_durations)

            total_net_pnl = df["net_pnl"].sum()  # 总收益
            daily_net_pnl = total_net_pnl / total_days  # 日均总收益

            total_commission = df["commission"].sum()  # 总手续费
            daily_commission = total_commission / total_days  # 日均手续费

            total_slippage = df["slippage"].sum()  # 总滑点成本
            daily_slippage = total_slippage / total_days  # 日均滑点成本

            total_turnover = df["turnover"].sum()  # 总成交额
            daily_turnover = total_turnover / total_days  # 日均成交额

            total_trade_count = df["trade_count"].sum()  # 总交易笔数
            daily_trade_count = total_trade_count / total_days  # 日均交易笔数

            total_return = (end_balance / self.capital - 1) * 100  # 总收益率
            # annual_return = total_return / total_days * self.annual_days
            annual_return = (np.exp(self.annual_days / total_days * np.log(end_balance / self.capital)) - 1) * 100  # 年化收益率
            linear_regression_annual_return = (np.exp(self.annual_days / total_days * np.log(np.exp(df["LinearRegression_log_balance"].iloc[-1]) / np.exp(df["LinearRegression_log_balance"].iloc[0]))) - 1) * 100  # 回归年度回报率
            daily_return = df["return"].mean() * 100  # 日均收益率，log盈亏率，底为e
            return_std = df["return"].std() * 100  # 日均收益率的标准差

            if return_std:
                daily_risk_free: float = (np.exp(np.log(1+self.risk_free) / self.annual_days) - 1) * 100  # 由年化百分比收益率计算日均对数收益率，再乘100
                sharpe_ratio: float = (daily_return - daily_risk_free) / return_std * np.sqrt(self.annual_days) # 计算夏普比率
            else:
                sharpe_ratio = 0

            return_drawdown_ratio = -total_return / max_ddpercent # 收益回撤比
            mra_ratio = -annual_return / max_ddpercent # MRA比率

            r_cubic = linear_regression_annual_return / (-top_5_ddpercent * (top_5_drawdown_duration / self.annual_days))

        # Output
        if output:
            self.output("-" * 30)
            self.output(f"start date：\t{start_date}")
            self.output(f"end date：\t{end_date}")

            self.output(f"total days：\t{total_days}")
            self.output(f"profit days：\t{profit_days}")
            self.output(f"loss days：\t{loss_days}")

            self.output(f"capital：\t{self.capital:,.2f}")
            self.output(f"end balance：\t{end_balance:,.2f}")

            self.output(f"total return：\t{total_return:,.2f}%")
            self.output(f"annual return：\t{annual_return:,.2f}%")
            self.output(f"linear regression annual return：\t{linear_regression_annual_return:,.2f}%")
            self.output(f"max drawdown: \t{max_drawdown:,.2f}")
            self.output(f"max drawdown percent: \t{max_ddpercent:,.2f}%")
            self.output(f"max drawdown duration: \t{max_drawdown_duration} days")

            self.output(f"total net pnl：\t{total_net_pnl:,.2f}")
            self.output(f"total commission：\t{total_commission:,.2f}")
            self.output(f"total slippage：\t{total_slippage:,.2f}")
            self.output(f"total turnover：\t{total_turnover:,.2f}")
            self.output(f"total trade count：\t{total_trade_count}")

            self.output(f"daily net pnl：\t{daily_net_pnl:,.2f}")
            self.output(f"daily commission：\t{daily_commission:,.2f}")
            self.output(f"daily slippage：\t{daily_slippage:,.2f}")
            self.output(f"daily turnover：\t{daily_turnover:,.2f}")
            self.output(f"daily trade count：\t{daily_trade_count}")

            self.output(f"daily return：\t{daily_return:,.2f}%")
            self.output(f"return std：\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"return drawdown ratio：\t{return_drawdown_ratio:,.2f}")
            self.output(f"mra ratio：\t{mra_ratio:,.2f}")
            self.output(f"r cubic：\t{r_cubic:,.2f}")

        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "linear_regression_annual_return": linear_regression_annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
            "mra_ratio": mra_ratio,
            "r_cubic": r_cubic,
        }

        # Filter potential error infinite value
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        self.output("finish calculating strategy's performance")
        return statistics

    def show_chart(self, df: DataFrame = None):
        """"""
        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            return

        fig = make_subplots(
            rows=5,
            cols=1,
            subplot_titles=["Balance", "Log Balance", "Drawdown", "Daily Pnl", "Pnl Distribution"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df.index,
            y=df["balance"],
            mode="lines",
            name="Balance"
        )
        log_balance_line = go.Scatter(
            x=df.index,
            y=df["log_balance"],
            mode="lines",
            name="Log Balance"
        )
        LinearRegression_log_balance_line = go.Scatter(
            x=df.index,
            y=df["LinearRegression_log_balance"],
            mode="lines",
            name="Linear Regression Log Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df.index,
            y=(df["drawdown"]/df["highlevel"])*100,
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(log_balance_line, row=2, col=1)
        fig.add_trace(LinearRegression_log_balance_line, row=2, col=1)
        fig.add_trace(drawdown_scatter, row=3, col=1)
        fig.add_trace(pnl_bar, row=4, col=1)
        fig.add_trace(pnl_histogram, row=5, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def run_optimization(self, optimization_setting: OptimizationSetting, output=True):
        """
        target name: end_balance, max_drawdown, max_ddpercent, max_drawdown_duration, total_net_pnl
        daily_net_pnl, total_commission, daily_commission, total_slippage, daily_slippage, total_turnover, daily_turnover
        total_trade_count, daily_trade_count, total_return, annual_return, daily_return, return_std, sharpe_ratio, return_drawdown_ratio
        """
        # Get optimization setting and target
        settings = optimization_setting.generate_setting()
        target_name = optimization_setting.target_name

        if not settings:
            self.output("produce parameters are empty, please check your parameters")
            return

        if not target_name:
            self.output("optimized target is not set, please check your target name")
            return

        # Use multiprocessing pool for running backtesting with different setting
        # Force to use spawn method to create new process (instead of fork on Linux)
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(multiprocessing.cpu_count())

        results = []
        for setting in settings:
            result = (pool.apply_async(optimize, (
                target_name,
                self.strategy_class,
                setting,
                self.vt_symbol,
                self.interval,
                self.start,
                self.rate,
                self.slippage,
                self.size,
                self.pricetick,
                self.capital,
                self.end,
                self.mode,
                self.inverse
            )))
            results.append(result)

        pool.close()
        pool.join()

        # Sort results and output
        result_values = [result.get() for result in results]
        result_values.sort(reverse=True, key=lambda result: result[1])

        if output:
            for value in result_values:
                msg = f"parameter：{value[0]}, target：{value[1]}"
                self.output(msg)

        return result_values

    def run_ga_optimization(self, optimization_setting: OptimizationSetting, population_size=100, ngen_size=30, output=True) -> list:
        """
        target name: end_balance, max_drawdown, max_ddpercent, max_drawdown_duration, total_net_pnl
        daily_net_pnl, total_commission, daily_commission, total_slippage, daily_slippage, total_turnover, daily_turnover
        total_trade_count, daily_trade_count, total_return, annual_return, daily_return, return_std, sharpe_ratio, return_drawdown_ratio
        """
        # Get optimization setting and target
        settings = optimization_setting.generate_setting_ga()
        target_name = optimization_setting.target_name

        if not settings:
            self.output("produce parameters are empty, please check your parameters")
            return

        if not target_name:
            self.output("optimized target is not set, please check your target name: ")
            return

        # Define parameter generation function
        def generate_parameter():
            """"""
            return random.choice(settings)

        def mutate_individual(individual, indpb):
            """"""
            size = len(individual)
            paramlist = generate_parameter()
            for i in range(size):
                if random.random() < indpb:
                    individual[i] = paramlist[i]
            return individual,

        # Create ga object function
        global ga_target_name
        global ga_strategy_class
        global ga_setting
        global ga_vt_symbol
        global ga_interval
        global ga_start
        global ga_rate
        global ga_slippage
        global ga_size
        global ga_pricetick
        global ga_capital
        global ga_end
        global ga_mode
        global ga_inverse

        ga_target_name = target_name
        ga_strategy_class = self.strategy_class
        ga_setting = settings[0]
        ga_vt_symbol = self.vt_symbol
        ga_interval = self.interval
        ga_start = self.start
        ga_rate = self.rate
        ga_slippage = self.slippage
        ga_size = self.size
        ga_pricetick = self.pricetick
        ga_capital = self.capital
        ga_end = self.end
        ga_mode = self.mode
        ga_inverse = self.inverse

        # Set up genetic algorithem
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, generate_parameter)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate_individual, indpb=1)
        toolbox.register("evaluate", ga_optimize)
        toolbox.register("select", tools.selNSGA2)

        total_size = len(settings)
        pop_size = population_size                      # number of individuals in each generation
        lambda_ = pop_size                              # number of children to produce at each generation
        mu = int(pop_size * 0.8)                        # number of individuals to select for the next generation

        cxpb = 0.95         # probability that an offspring is produced by crossover
        mutpb = 1 - cxpb    # probability that an offspring is produced by mutation
        ngen = ngen_size    # number of generation

        pop = toolbox.population(pop_size)
        hof = tools.ParetoFront()               # end result of pareto front

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        np.set_printoptions(suppress=True)
        stats.register("mean", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Multiprocessing is not supported yet.
        # pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # toolbox.register("map", pool.map)

        # Run ga optimization
        self.output(f"total size：{total_size}")
        self.output(f"population size：{pop_size}")
        self.output(f"selected next generation size：{mu}")
        self.output(f"number of generation：{ngen}")
        self.output(f"probability of crossover：{cxpb:.0%}")
        self.output(f"probability of mutation：{mutpb:.0%}")

        start = time()

        algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu,
            lambda_,
            cxpb,
            mutpb,
            ngen,
            stats,
            halloffame=hof
        )

        end = time()
        cost = int((end - start))

        self.output(f"finish optimization，cost {cost} seconds")

        # Return result list
        results = []

        for parameter_values in hof:
            setting = dict(parameter_values)
            target_value = ga_optimize(parameter_values)[0]
            results.append((setting, target_value, {}))

        return results

    def update_daily_close(self, price: float):
        """"""
        d = self.datetime.date()

        daily_result = self.daily_results.get(d, None)
        if daily_result:
            daily_result.close_price = price
        else:
            self.daily_results[d] = DailyResult(d, price)

    def new_bar(self, bar: BarData):
        """"""
        self.bar = bar
        self.datetime = bar.datetime

        self.cross_limit_order()
        self.cross_stop_order()
        self.strategy.on_bar(bar)

        self.update_daily_close(bar.close_price)

    def new_tick(self, tick: TickData):
        """"""
        self.tick = tick
        self.datetime = tick.datetime

        self.cross_limit_order()
        self.cross_stop_order()
        self.strategy.on_tick(tick)

        self.update_daily_close(tick.last_price)

    def cross_limit_order(self):
        """
        Cross limit order with last bar/tick data.
        """
        if self.mode == BacktestingMode.BAR:
            long_cross_price = self.bar.low_price
            short_cross_price = self.bar.high_price
            long_best_price = self.bar.open_price
            short_best_price = self.bar.open_price
        else:
            long_cross_price = self.tick.ask_price_1
            short_cross_price = self.tick.bid_price_1
            long_best_price = long_cross_price
            short_best_price = short_cross_price

        for order in list(self.active_limit_orders.values()):
            # Push order update with status "not traded" (pending).
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.on_order(order)

            # Check whether limit orders can be filled.
            long_cross = (Direction.LONG == order.direction and order.price >= long_cross_price > 0)
            short_cross = (Direction.SHORT == order.direction and order.price <= short_cross_price and short_cross_price > 0)

            if not long_cross and not short_cross:
                continue

            # Push order udpate with status "all traded" (filled).
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.on_order(order)

            self.active_limit_orders.pop(order.vt_orderid)

            # Push trade update
            self.trade_count += 1

            if long_cross:
                trade_price = min(order.price, Decimal(str(long_best_price)))
                pos_change = order.volume
            else:
                trade_price = max(order.price, Decimal(str(short_best_price)))
                pos_change = -order.volume

            trade = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            self.strategy.pos += pos_change
            self.strategy.on_trade(trade)

            self.trades[trade.vt_tradeid] = trade

    def cross_stop_order(self):
        """
        Cross stop order with last bar/tick data.
        """
        if self.mode == BacktestingMode.BAR:
            long_cross_price = self.bar.high_price
            short_cross_price = self.bar.low_price
            long_best_price = self.bar.open_price
            short_best_price = self.bar.open_price
        else:
            long_cross_price = self.tick.last_price
            short_cross_price = self.tick.last_price
            long_best_price = long_cross_price
            short_best_price = short_cross_price

        for stop_order in list(self.active_stop_orders.values()):
            # Check whether stop order can be triggered.
            # 加上long_cross_price > 0 和 short_cross_price > 0 两个条件，程序更加稳健
            long_cross = (
                stop_order.direction == Direction.LONG
                and stop_order.price <= long_cross_price
                and long_cross_price > 0
            )

            short_cross = (
                stop_order.direction == Direction.SHORT
                and stop_order.price >= short_cross_price
                and short_cross_price > 0
            )

            if not long_cross and not short_cross:
                continue

            # Create order data.
            self.limit_order_count += 1

            order: OrderData = OrderData(
                symbol=self.symbol,
                exchange=self.exchange,
                orderid=str(self.limit_order_count),
                direction=stop_order.direction,
                offset=stop_order.offset,
                price=stop_order.price,
                volume=stop_order.volume,
                traded=stop_order.volume,
                status=Status.ALLTRADED,
                gateway_name=self.gateway_name,
                datetime=self.datetime
            )

            self.limit_orders[order.vt_orderid] = order

            # Create trade data.
            if long_cross:
                trade_price = max(stop_order.price, Decimal(str(long_best_price)))
                pos_change = order.volume
            else:
                trade_price = min(stop_order.price, Decimal(str(short_best_price)))
                pos_change = -order.volume

            self.trade_count += 1

            trade = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            self.trades[trade.vt_tradeid] = trade

            # Update stop order.
            stop_order.vt_orderids.append(order.vt_orderid)
            stop_order.status = StopOrderStatus.TRIGGERED

            if stop_order.stop_orderid in self.active_stop_orders:
                self.active_stop_orders.pop(stop_order.stop_orderid)

            # Push update to strategy.
            self.strategy.on_stop_order(stop_order)
            self.strategy.on_order(order)

            self.strategy.pos += pos_change
            self.strategy.on_trade(trade)

    def load_bar(
        self,
        vt_symbol: str,
        days: int,
        interval: Interval,
        callback: Callable,
        use_database: bool
    ) -> List[BarData]:
        """"""
        self.days = days
        self.callback = callback
        return []

    def load_tick(self, vt_symbol: str, days: int, callback: Callable) -> List[TickData]:
        """"""
        self.days = days
        self.callback = callback
        return []

    def send_order(
        self,
        strategy: CtaTemplate,
        direction: Direction,
        offset: Offset,
        price: Decimal,
        volume: Decimal,
        stop: bool,
        lock: bool,
        net: bool,
        maker: bool = False
    ) -> List[str]:
        """"""
        price = round_to(price, Decimal(str(self.pricetick)))
        if stop:
            vt_orderid: str = self.send_stop_order(direction, offset, price, volume)
        else:
            vt_orderid: str = self.send_limit_order(direction, offset, price, volume)
        return [vt_orderid]

    def send_stop_order(
        self,
        direction: Direction,
        offset: Offset,
        price: Decimal,
        volume: Decimal
    ) -> str:
        """"""
        self.stop_order_count += 1

        stop_order: StopOrder = StopOrder(
            vt_symbol=self.vt_symbol,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            datetime=self.datetime,
            stop_orderid=f"{STOPORDER_PREFIX}.{self.stop_order_count}",
            strategy_name=self.strategy.strategy_name,
        )

        self.active_stop_orders[stop_order.stop_orderid] = stop_order
        self.stop_orders[stop_order.stop_orderid] = stop_order
        
        # 这里必须回调一下on_stop_order，才能把挂着的stop_order记录下来
        self.strategy.on_stop_order(stop_order)

        return stop_order.stop_orderid

    def send_limit_order(
        self,
        direction: Direction,
        offset: Offset,
        price: Decimal,
        volume: Decimal
    ) ->str:
        """"""
        self.limit_order_count += 1

        order: OrderData = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            gateway_name=self.gateway_name,
            datetime=self.datetime
        )

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order
        
        # 这里回调一下on_order，把刚刚提交的限价单记录下来
        self.strategy.on_order(order)

        return order.vt_orderid

    def cancel_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """
        Cancel order by vt_orderid.
        """
        if vt_orderid.startswith(STOPORDER_PREFIX):
            self.cancel_stop_order(strategy, vt_orderid)
        else:
            self.cancel_limit_order(strategy, vt_orderid)

    def cancel_stop_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """"""
        if vt_orderid not in self.active_stop_orders:
            return None
        stop_order = self.active_stop_orders.pop(vt_orderid)

        stop_order.status = StopOrderStatus.CANCELLED
        self.strategy.on_stop_order(stop_order)

    def cancel_limit_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """"""
        if vt_orderid not in self.active_limit_orders:
            return None
        order = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.on_order(order)

    def cancel_all(self, strategy: CtaTemplate) -> None:
        """
        Cancel all orders, both limit and stop.
        """
        vt_orderids = list(self.active_limit_orders.keys())
        for vt_orderid in vt_orderids:
            self.cancel_limit_order(strategy, vt_orderid)

        stop_orderids = list(self.active_stop_orders.keys())
        for vt_orderid in stop_orderids:
            self.cancel_stop_order(strategy, vt_orderid)

    def write_log(self, msg: str, strategy: CtaTemplate = None) -> None:
        """
        Write log message.
        """
        msg = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_email(self, msg: str, strategy: CtaTemplate = None) -> None:
        """
        Send email to default receiver.
        """
        pass

    def sync_strategy_data(self, strategy: CtaTemplate) -> None:
        """
        Sync strategy data into json file.
        """
        pass

    def get_engine_type(self) -> EngineType:
        """
        Return engine type.
        """
        return self.engine_type

    def get_pricetick(self, strategy: CtaTemplate) -> float:
        """
        Return contract pricetick data.
        """
        return self.pricetick

    def put_strategy_event(self, strategy: CtaTemplate) -> None:
        """
        Put an event to update strategy status.
        """
        pass

    def output(self, msg) -> None:
        """
        Output message of backtesting engine.
        """
        print(f"{datetime.now()}\t{msg}")

    def get_all_trades(self) -> List[TradeData]:
        """
        Return all trade data of current backtesting result.
        """
        return list(self.trades.values())

    def get_all_orders(self) -> List[OrderData]:
        """
        Return all limit order data of current backtesting result.
        """
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> List["DailyResult"]:
        """
        Return all daily result data.
        """
        return list(self.daily_results.values())


def optimize(
    target_name: str,
    strategy_class: CtaTemplate,
    setting: dict,
    vt_symbol: str,
    interval: Interval,
    start: datetime,
    rate: float,
    slippage: float,
    size: float,
    pricetick: float,
    capital: int,
    end: datetime,
    mode: BacktestingMode,
    inverse: bool
):
    """
    Function for running in multiprocessing.pool
    """
    engine = BacktestingEngine()

    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=interval,
        start=start,
        rate=rate,
        slippage=slippage,
        size=size,
        pricetick=pricetick,
        capital=capital,
        end=end,
        mode=mode,
        inverse=inverse
    )

    engine.add_strategy(strategy_class, setting)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    # statistics = engine.calculate_statistics(output=False)
    statistics = engine.calculate_statistics_for_optimization(output=False)

    target_value = statistics[target_name]
    return (str(setting), target_value, statistics)


@lru_cache(maxsize=1000000)
def _ga_optimize(parameter_values: tuple):
    """"""
    setting = dict(parameter_values)

    result = optimize(
        ga_target_name,
        ga_strategy_class,
        setting,
        ga_vt_symbol,
        ga_interval,
        ga_start,
        ga_rate,
        ga_slippage,
        ga_size,
        ga_pricetick,
        ga_capital,
        ga_end,
        ga_mode,
        ga_inverse
    )
    return (result[1],)


def ga_optimize(parameter_values: list):
    """"""
    return _ga_optimize(tuple(parameter_values))


@lru_cache(maxsize=999)
def load_bar_data(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime,
    end: datetime
):
    """"""
    return database.load_bar_data(
        symbol, exchange, interval, start, end
    )


@lru_cache(maxsize=999)
def load_tick_data(
    symbol: str,
    exchange: Exchange,
    start: datetime,
    end: datetime
):
    """"""
    return database.load_tick_data(
        symbol, exchange, start, end
    )


# GA related global value
ga_end = None
ga_mode = None
ga_target_name = None
ga_strategy_class = None
ga_setting = None
ga_vt_symbol = None
ga_interval = None
ga_start = None
ga_rate = None
ga_slippage = None
ga_size = None
ga_pricetick = None
ga_capital = None

@lru_cache(maxsize=999)
def load_cache(
    vt_symbol: str,
    interval: Interval,
    start: datetime,
    end: datetime
) -> Optional[Dict[Tuple[datetime, str], BarData]]:
    """加载缓存数据"""
    # 生成缓存文件名
    key = '_'.join([vt_symbol, interval.value, start.strftime("%Y%m%d"), end.strftime("%Y%m%d")])
    print("使用自定义加载函数加载", vt_symbol)

    # 检查文件是否存在
    path = data_path.joinpath(key)
    if not path.exists():
        return []

    # 存在则加载其中数据
    with open(str(path), "rb") as f:
        data = pickle.load(f)
        return data

def save_cache(
    vt_symbol: str,
    interval: Interval,
    start: datetime,
    end: datetime,
    data: list
) -> None:
    """保存缓存的数据"""
    # 生成缓存文件名
    key = "_".join([vt_symbol, interval.value, start.strftime("%Y%m%d"), end.strftime("%Y%m%d")])

    # 保存数据到文件
    path = data_path.joinpath(key)
    print("保存数据", vt_symbol)
    with open(str(path), "wb") as f:
        pickle.dump(data, f)
