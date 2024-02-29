from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Callable, List, Dict, Optional, Type
from functools import lru_cache, partial
import traceback

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pandas import DataFrame, Series
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from yubaotrader.trader.constant import (
    Direction,
    Offset,
    Exchange,
    Interval,
    Status
)
from yubaotrader.trader.database import get_database, BaseDatabase
from yubaotrader.trader.object import OrderData, TradeData, BarData, TickData
from yubaotrader.trader.utility import round_to, extract_vt_symbol
from yubaotrader.trader.optimize import (
    OptimizationSetting,
    check_optimization_setting,
    run_bf_optimization,
    run_ga_optimization
)
from decimal import Decimal

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


class BacktestingEngine:
    """"""

    engine_type: EngineType = EngineType.BACKTESTING
    gateway_name: str = "BACKTESTING"

    def __init__(self) -> None:
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
        self.capital: int = 1_000_000
        self.risk_free: float = 0
        self.annual_days: int = 365
        self.mode: BacktestingMode = BacktestingMode.BAR

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

        self.rm.clear_results()

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
        risk_free: float = 0,
        annual_days: int = 240
    ) -> None:
        """"""
        self.mode = mode
        self.vt_symbol = vt_symbol
        self.interval = Interval(interval)
        self.rate = rate
        self.slippage = slippage
        self.size = size
        self.pricetick = pricetick
        self.start = start

        self.symbol, exchange_str = self.vt_symbol.split(".")
        self.exchange = Exchange(exchange_str)

        self.capital = capital

        if not end:
            end = datetime.now()
        self.end = end.replace(hour=23, minute=59, second=59)

        self.mode = mode
        self.risk_free = risk_free
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

            progress += progress_days / total_days
            progress = min(progress, 1)

            start = end + interval_delta
            end += progress_delta

        self.output(f"历史数据加载完成，数据量：{len(self.history_data)}")

    def run_backtesting(self) -> None:
        """"""
        if self.mode == BacktestingMode.BAR:
            func = self.new_bar
        else:
            func = self.new_tick

        self.strategy.on_init()
        self.strategy.inited = True
        self.output("策略初始化完成")

        self.strategy.on_start()
        self.strategy.trading = True
        self.output("开始回放历史数据")

        total_size: int = len(self.history_data)
        batch_size: int = max(int(total_size / 10), 1)

        for ix, i in enumerate(range(0, total_size, batch_size)):
            batch_data: list = self.history_data[i: i + batch_size]
            for data in batch_data:
                try:
                    func(data)
                except Exception:
                    self.output("触发异常，回测终止")
                    self.output(traceback.format_exc())
                    return

            progress = min(ix / 10, 1)
            progress_bar: str = "=" * (ix + 1)
            self.output(f"回放进度：{progress_bar} [{progress:.0%}]")

        self.strategy.on_stop()
        self.output("历史数据回放结束")

    def calculate_result(self) -> DataFrame:
        """"""
        self.output("开始计算逐日盯市盈亏")

        if not self.trades:
            self.output("回测成交记录为空")

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
                self.slippage
            )

            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos

        # Generate dataframe
        results: defaultdict = defaultdict(list)

        for daily_result in self.daily_results.values():
            for key, value in daily_result.__dict__.items():
                results[key].append(value)

        self.daily_df = DataFrame.from_dict(results).set_index("date")

        self.output("逐日盯市盈亏计算完成")
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True):
        """"""
        self.output("开始计算策略统计指标")

        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df
        
        # Init all statistics default value
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

        # Check if balance is always positive
        positive_balance: bool = False

        if df is not None:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital  # 策略每天的净值
            # When balance falls below 0, set daily return to 0
            pre_balance: Series = df["balance"].shift(1)
            pre_balance.iloc[0] = self.capital
            x = df["balance"] / pre_balance
            x[x <= 0] = np.nan
            df["log_balance"] = np.log(df["balance"])  # 对数坐标下策略每天的净值
            # 对数坐标下策略每天的净值做线性回归
            x_ = df.reset_index().reset_index()["index"].values.reshape((-1, 1))
            y = df["log_balance"].values
            model = LinearRegression()
            model.fit(x_, y)
            y_pred = model.predict(x_)
            df["LinearRegression_log_balance"] = y_pred

            df["return"] = np.log(x).fillna(0)
            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )  # 从开始交易的第一天（策略完成初始化的那一天）到当前这天净值到达过的最高位
            df["drawdown"] = df["balance"] - df["highlevel"]  # 回撤
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100  # 百分比回撤

            # All balance value needs to be positive
            positive_balance = (df["balance"] > 0).all()
            if not positive_balance:
                self.output("回测中出现爆仓（资金小于等于0），无法计算策略统计指标")
            
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
        if positive_balance:
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
            self.output(f"首个交易日：\t{start_date}")
            self.output(f"最后交易日：\t{end_date}")

            self.output(f"总交易日：\t{total_days}")
            self.output(f"盈利交易日：\t{profit_days}")
            self.output(f"亏损交易日：\t{loss_days}")

            self.output(f"起始资金：\t{self.capital:,.2f}")
            self.output(f"结束资金：\t{end_balance:,.2f}")

            self.output(f"总收益率：\t{total_return:,.2f}%")
            self.output(f"年化收益：\t{annual_return:,.2f}%")
            self.output(f"线性回归年化收益：\t{linear_regression_annual_return:,.2f}%")
            self.output(f"最大回撤: \t{max_drawdown:,.2f}")
            self.output(f"百分比最大回撤: {max_ddpercent:,.2f}%")
            self.output(f"最长回撤天数: \t{max_drawdown_duration}")

            self.output(f"总盈亏：\t{total_net_pnl:,.2f}")
            self.output(f"总手续费：\t{total_commission:,.2f}")
            self.output(f"总滑点：\t{total_slippage:,.2f}")
            self.output(f"总成交金额：\t{total_turnover:,.2f}")
            self.output(f"总成交笔数：\t{total_trade_count}")

            self.output(f"日均盈亏：\t{daily_net_pnl:,.2f}")
            self.output(f"日均手续费：\t{daily_commission:,.2f}")
            self.output(f"日均滑点：\t{daily_slippage:,.2f}")
            self.output(f"日均成交金额：\t{daily_turnover:,.2f}")
            self.output(f"日均成交笔数：\t{daily_trade_count}")

            self.output(f"日均收益率：\t{daily_return:,.2f}%")
            self.output(f"收益标准差：\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"收益回撤比：\t{return_drawdown_ratio:,.2f}")
            self.output(f"mra比率：\t{mra_ratio:,.2f}")
            self.output(f"r立方：\t{r_cubic:,.2f}")

            self.output(f"平均持仓时间：\t{average_duration}")
            self.output(f"最小持仓时间：\t{min_duration}")
            self.output(f"最大持仓时间：\t{max_duration}")
            self.output(f"总交易次数（开仓后全部平仓算一次）：\t{total_open_to_close_trade_count}")
            self.output(f"盈利交易次数：\t{profit_trade_count}")
            self.output(f"亏损交易次数：\t{loss_trade_count}")
            self.output(f"最大连续亏损次数：\t{maximum_number_of_consecutive_losses}")
            self.output(f"最长衰退期：\t{longest_decline_period} days")
            self.output(f"胜率：\t{winning_percentage:,.2f}")
            self.output(f"总盈利：\t{total_profit:,.2f}")
            self.output(f"总亏损：\t{total_loss:,.2f}")
            self.output(f"盈亏比：\t{profit_loss_sharing_ratio:,.2f}")
            self.output(f"平均盈利幅度：\t{average_profit_return:,.2f}%")
            self.output(f"平均亏损幅度：\t{average_loss_return:,.2f}%")
            self.output(f"最大盈利幅度：\t{max_profit_return:,.2f}%")
            self.output(f"最大亏损幅度：\t{max_loss_return:,.2f}%")
            self.output(f"交易均值：\t{average_return:,.2f}%")
            self.output(f"e比率：\t{e_ratio:,.2f}")

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

        self.output("策略统计指标计算完成")
        return statistics

    def calculate_statistics_simplified(self, df: DataFrame = None, output=True) -> dict:
        """"""
        self.output("开始计算策略统计指标")

        # Check DataFrame input exterior
        if df is None:
            df: DataFrame = self.daily_df

        # Init all statistics default value
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

        # Check if balance is always positive
        positive_balance: bool = False

        if df is not None:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital

            # When balance falls below 0, set daily return to 0
            pre_balance: Series = df["balance"].shift(1)
            pre_balance.iloc[0] = self.capital
            x = df["balance"] / pre_balance
            x[x <= 0] = np.nan
            df["return"] = np.log(x).fillna(0)

            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # All balance value needs to be positive
            positive_balance = (df["balance"] > 0).all()
            if not positive_balance:
                self.output("回测中出现爆仓（资金小于等于0），无法计算策略统计指标")

        # Calculate statistics value
        if positive_balance:
            # Calculate statistics value
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days: int = len(df)
            profit_days: int = len(df[df["net_pnl"] > 0])
            loss_days: int = len(df[df["net_pnl"] < 0])

            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            max_drawdown_end = df["drawdown"].idxmin()

            if isinstance(max_drawdown_end, date):
                max_drawdown_start = df["balance"][:max_drawdown_end].idxmax()
                max_drawdown_duration: int = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration: int = 0

            total_net_pnl: float = df["net_pnl"].sum()
            daily_net_pnl: float = total_net_pnl / total_days

            total_commission: float = df["commission"].sum()
            daily_commission: float = total_commission / total_days

            total_slippage: float = df["slippage"].sum()
            daily_slippage: float = total_slippage / total_days

            total_turnover: float = df["turnover"].sum()
            daily_turnover: float = total_turnover / total_days

            total_trade_count: int = df["trade_count"].sum()
            daily_trade_count: int = total_trade_count / total_days

            total_return: float = (end_balance / self.capital - 1) * 100
            annual_return: float = total_return / total_days * self.annual_days
            daily_return: float = df["return"].mean() * 100
            return_std: float = df["return"].std() * 100

            if return_std:
                daily_risk_free: float = (np.exp(np.log(1+self.risk_free) / self.annual_days) - 1) * 100  # 由年化百分比收益率计算日均对数收益率，再乘100
                sharpe_ratio: float = (daily_return - daily_risk_free) / return_std * np.sqrt(self.annual_days) # 计算夏普比率
            else:
                sharpe_ratio = 0

            if max_ddpercent:
                return_drawdown_ratio: float = -total_return / max_ddpercent
            else:
                return_drawdown_ratio = 0

        # Output
        if output:
            self.output("-" * 30)
            self.output(f"首个交易日：\t{start_date}")
            self.output(f"最后交易日：\t{end_date}")

            self.output(f"总交易日：\t{total_days}")
            self.output(f"盈利交易日：\t{profit_days}")
            self.output(f"亏损交易日：\t{loss_days}")

            self.output(f"起始资金：\t{self.capital:,.2f}")
            self.output(f"结束资金：\t{end_balance:,.2f}")

            self.output(f"总收益率：\t{total_return:,.2f}%")
            self.output(f"年化收益：\t{annual_return:,.2f}%")
            self.output(f"最大回撤: \t{max_drawdown:,.2f}")
            self.output(f"百分比最大回撤: {max_ddpercent:,.2f}%")
            self.output(f"最长回撤天数: \t{max_drawdown_duration}")

            self.output(f"总盈亏：\t{total_net_pnl:,.2f}")
            self.output(f"总手续费：\t{total_commission:,.2f}")
            self.output(f"总滑点：\t{total_slippage:,.2f}")
            self.output(f"总成交金额：\t{total_turnover:,.2f}")
            self.output(f"总成交笔数：\t{total_trade_count}")

            self.output(f"日均盈亏：\t{daily_net_pnl:,.2f}")
            self.output(f"日均手续费：\t{daily_commission:,.2f}")
            self.output(f"日均滑点：\t{daily_slippage:,.2f}")
            self.output(f"日均成交金额：\t{daily_turnover:,.2f}")
            self.output(f"日均成交笔数：\t{daily_trade_count}")

            self.output(f"日均收益率：\t{daily_return:,.2f}%")
            self.output(f"收益标准差：\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"收益回撤比：\t{return_drawdown_ratio:,.2f}")

        statistics: dict = {
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
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        # Filter potential error infinite value
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        self.output("策略统计指标计算完成")
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
            y=df["drawdown"],
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

    def run_bf_optimization(
        self,
        optimization_setting: OptimizationSetting,
        output: bool = True,
        max_workers: int = None
    ) -> list:
        """"""
        if not check_optimization_setting(optimization_setting):
            return

        evaluate_func: callable = wrap_evaluate(self, optimization_setting.target_name)
        results: list = run_bf_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            max_workers=max_workers,
            output=self.output
        )

        if output:
            for result in results:
                msg: str = f"参数：{result[0]}, 目标：{result[1]}"
                self.output(msg)

        return results

    run_optimization = run_bf_optimization

    def run_ga_optimization(
        self,
        optimization_setting: OptimizationSetting,
        output: bool = True,
        max_workers: int = None,
        ngen_size: int = 30
    ) -> list:
        """"""
        if not check_optimization_setting(optimization_setting):
            return

        evaluate_func: callable = wrap_evaluate(self, optimization_setting.target_name)
        results: list = run_ga_optimization(
            evaluate_func,
            optimization_setting,
            get_target_value,
            max_workers=max_workers,
            ngen_size=ngen_size,
            output=self.output
        )

        if output:
            for result in results:
                msg: str = f"参数：{result[0]}, 目标：{result[1]}"
                self.output(msg)

        return results

    def update_daily_close(self, price: float) -> None:
        """"""
        d: date = self.datetime.date()

        daily_result: Optional[DailyResult] = self.daily_results.get(d, None)
        if daily_result:
            daily_result.close_price = price
        else:
            self.daily_results[d] = DailyResult(d, price)

    def new_bar(self, bar: BarData) -> None:
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

    def cross_limit_order(self) -> None:
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
            long_cross: bool = (
                order.direction == Direction.LONG
                and order.price >= long_cross_price
                and long_cross_price > 0
            )

            short_cross: bool = (
                order.direction == Direction.SHORT
                and order.price <= short_cross_price
                and short_cross_price > 0
            )

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
                trade_price = min(order.price, long_best_price)
                pos_change = order.volume
            else:
                trade_price = max(order.price, short_best_price)
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
        self.callback = callback

        init_end = self.start - INTERVAL_DELTA_MAP[interval]
        init_start = self.start - timedelta(days=days)

        symbol, exchange = extract_vt_symbol(vt_symbol)

        bars: List[BarData] = load_bar_data(
            symbol,
            exchange,
            interval,
            init_start,
            init_end
        )

        return bars

    def load_tick(self, vt_symbol: str, days: int, callback: Callable) -> List[TickData]:
        """"""
        self.callback = callback

        init_end = self.start - timedelta(seconds=1)
        init_start = self.start - timedelta(days=days)

        symbol, exchange = extract_vt_symbol(vt_symbol)

        ticks: List[TickData] = load_tick_data(
            symbol,
            exchange,
            init_start,
            init_end
        )

        return ticks

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

    def get_size(self, strategy: CtaTemplate) -> int:
        """
        Return contract size data.
        """
        return self.size

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


@lru_cache(maxsize=999)
def load_bar_data(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime,
    end: datetime
) -> List[BarData]:
    """"""
    database: BaseDatabase = get_database()

    return database.load_bar_data(
        symbol, exchange, interval, start, end
    )


@lru_cache(maxsize=999)
def load_tick_data(
    symbol: str,
    exchange: Exchange,
    start: datetime,
    end: datetime
) -> List[TickData]:
    """"""
    database: BaseDatabase = get_database()

    return database.load_tick_data(
        symbol, exchange, start, end
    )


def evaluate(
    target_name: str,
    strategy_class: CtaTemplate,
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
    setting: dict
) -> tuple:
    """
    Function for running in multiprocessing.pool
    """
    engine: BacktestingEngine = BacktestingEngine()

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
        mode=mode
    )

    engine.add_strategy(strategy_class, setting)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    statistics: dict = engine.calculate_statistics(output=False)

    target_value: float = statistics[target_name]
    return (setting, target_value, statistics)


def wrap_evaluate(engine: BacktestingEngine, target_name: str) -> callable:
    """
    Wrap evaluate function with given setting from backtesting engine.
    """
    func: callable = partial(
        evaluate,
        target_name,
        engine.strategy_class,
        engine.vt_symbol,
        engine.interval,
        engine.start,
        engine.rate,
        engine.slippage,
        engine.size,
        engine.pricetick,
        engine.capital,
        engine.end,
        engine.mode
    )
    return func


def get_target_value(result: list) -> float:
    """
    Get target value for sorting optimization results.
    """
    return result[1]
