from typing import Any
from yubaotrader.app.cta_strategy import (
    CtaTemplate,
    StopOrder
)

from yubaotrader.trader.object import TickData, BarData, TradeData, OrderData, Direction, Offset
from yubaotrader.trader.utility import BarGeneratorV2, ArrayManager, round_to
from decimal import Decimal

from yubaotrader.app.cta_strategy import PnlTracker, ResultManager, OrderRecorder
from yubaotrader.app.cta_strategy.base import StopOrderStatus


# import requests
# from urllib.parse import urlencode
# from typing import List, Dict
# from howtrader.trader.database import BaseDatabase, get_database
# from howtrader.trader.constant import LOCAL_TZ
# from howtrader.trader.object import Exchange, Interval
# from tzlocal import get_localzone_name
# from howtrader.trader.object import BarData
# from datetime import datetime, timedelta
# from time import sleep
import pdb

# database: BaseDatabase = get_database()

# import pytz
# tzinfo = pytz.timezone(get_localzone_name())


class TurtleSignal(CtaTemplate):
    """海龟信号"""
    author = "yubao"
    
    capital = 10000.0
    
    # 参数
    entry_window: int = 65
    exit_window: int = 28
    n_window: int = 5  # 计算atr的周期
    n_min_each_bar: int = 85  # k线周期
    risk_level = 0.0025
    capital_change_ratio = 0.4  # 涨跌多大比例后改变capital
    stop_loss: float = 2.0  # 开仓后几倍atr止损
    max_hold: int = 4  # 最多持有几份仓位
    # cci_window: int = 50
    # cci_signal: int = 20
    minimum_volume: float = 0.004   # 最小数量，
    max_gearing_ratio: float = 0.5  # 最大杠杆比例
    
    # 变量
    entry_up: float = 0.0
    entry_down: float = 0.0
    
    exit_up: float = 0.0
    exit_down: float = 0.0
    
    n: float = 0.0
    
    long_entry: float = 0.0
    short_entry: float = 0.0
    
    trading_size: float = 1.0
    
    parameters = [
        "entry_window",
        "exit_window",
        "n_window",
        "n_min_each_bar",
        "risk_level",
        "capital_change_ratio",
        "stop_loss",
        "max_hold",
        "minimum_volume",
        "max_gearing_ratio",
        # "cci_window",
        # "cci_signal",
    ]
    variables = [
        "entry_up",
        "entry_down",
        "exit_up",
        "exit_down",
        "n",
        "long_entry",
        "short_entry",
        "trading_size",
    ]
    
    def __init__(self, cta_engine: Any, strategy_name: str, vt_symbol: str, setting: dict) -> None:
        """构造函数"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
        self.bg = BarGeneratorV2(self.on_bar, self.n_min_each_bar, self.on_tmin_bar)
        self.am = ArrayManager()
        
        self.tracker = PnlTracker()
        self.bar_trackers = {}
        
        self.rm = ResultManager()
        
        self.pnl = 0.0 # 策略运行起来之后的盈亏
        
        self.order_recorder = OrderRecorder()
        
    def on_init(self):
        """初始化"""
        self.write_log("策略初始化")
        self.load_bar(20)
        
    def on_start(self):
        """启动"""
        self.write_log("策略启动")
        
    def on_stop(self):
        """停止"""
        self.write_log("策略停止")
        
    def on_min_each_bar(self, tick: TickData):
        """Tick推送"""
        self.bg.update_tick(tick)
        
    def on_bar(self, bar: BarData):
        """1分钟k线推送"""
        self.bg.update_bar(bar)
        
    def on_tmin_bar(self, bar: BarData):
        """收到一根新的t分钟K线"""
        self.bar = bar
        
        self.tracker.update_bar(bar)
        self.tracker.calculate_pnl()
        
        self.pnl += float(self.tracker.total_pnl)
        
        if abs(self.pnl) > self.capital * self.capital_change_ratio:
            self.capital += self.pnl
            self.pnl = 0.0
        
        self.bar_trackers[self.tracker.datetime] = self.tracker
        
        # 创建下一轮的PnlTracker
        self.tracker = PnlTracker(1, self.tracker)
        
        self.cancel_all()
        
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        if not round_to(self.pos * 1000, 1):
            self.entry_up, self.entry_down = self.am.donchian(self.entry_window)
            self.n = float(self.am.atr(self.n_window))
            if self.n == 0:
                self.trading_size = self.capital * self.max_gearing_ratio / bar.close_price
            else:
                self.trading_size = (self.capital * self.risk_level) / self.n

            self.trading_size = min(self.capital * self.max_gearing_ratio / bar.close_price, self.trading_size) # 最大杠杆倍数
            self.trading_size = max(self.trading_size, self.minimum_volume) # 最小报单量
            self.trading_size = round_to(self.trading_size, 0.001)

        self.exit_up, self.exit_down = self.am.donchian(self.exit_window)
        
        # self.cci = self.am.cci(self.cci_window)
        
        if not round_to(self.pos * 1000, 1):
            # if self.cci > self.cci_signal:
            self.send_long_orders()
            # elif self.cci < -self.cci_signal:
            self.send_short_orders()

        elif round_to(self.pos * 1000, 1) > 0:
            self.send_long_orders()

            long_stop = self.long_entry - self.stop_loss * self.n
            long_stop = max(long_stop, self.exit_down)
            self.sell(Decimal(str(long_stop)), Decimal(str(abs(self.pos))), stop=True)
            
        else:
            self.send_short_orders()
            
            short_stop = self.short_entry + self.stop_loss * self.n
            short_stop = min(short_stop, self.exit_up)
            self.cover(Decimal(str(short_stop)), Decimal(str(abs(self.pos))), stop=True)
            
    def on_trade(self, trade: TradeData) -> None:
        """成交推送"""
        trade.volume = float(round_to(trade.volume, 0.001))
        trade.price = float(round_to(trade.price, 0.01))
        # if self.bar.datetime > datetime(2018, 4, 9, tzinfo=tzinfo):
        #     print(f"{trade.datetime}成交{trade.volume}BTC在价格{trade.price},方向为{trade.direction}")
        # 记录开仓价格
        if trade.offset == Offset.OPEN:
            if trade.direction == Direction.LONG:
                self.long_entry = trade.price
                # 把sell止损单，即short，close的stop_order都撤了重新下单
                active_stop_orders = self.order_recorder.get_active_stop_orders()
                for stop_order in active_stop_orders.values():
                    if stop_order.status == StopOrderStatus.WAITING and stop_order.direction == Direction.SHORT and stop_order.offset == Offset.CLOSE:
                        self.cancel_order(stop_order.stop_orderid)
                
                # 重新计算止损位置然后下单
                long_stop = self.long_entry - self.stop_loss * self.n
                long_stop = max(long_stop, self.exit_down)
                self.sell(Decimal(str(long_stop)), Decimal(str(abs(self.pos))), stop=True)

            else:
                self.short_entry = trade.price
                # 把cover止损单，即long，close的stop_order都撤了重新下单
                active_stop_orders = self.order_recorder.get_active_stop_orders()
                for stop_order in active_stop_orders.values():
                    if stop_order.status == StopOrderStatus.WAITING and stop_order.direction == Direction.LONG and stop_order.offset == Offset.CLOSE:
                        self.cancel_order(stop_order.stop_orderid)
                        
                # 重新计算止损位置然后下单
                short_stop = self.short_entry + self.stop_loss * self.n
                short_stop = min(short_stop, self.exit_up)
                self.cover(Decimal(str(short_stop)), Decimal(str(abs(self.pos))), stop=True)
                
        # 更新到tracker和ResultManager
        self.tracker.update_trade(trade)
        self.rm.update_trade(trade)

    def on_order(self, order: OrderData) -> None:
        """委托推送"""
        self.order_recorder.update_limit_order(order)

    def on_stop_order(self, stop_order: StopOrder) -> None:
        """停止单推送"""
        self.order_recorder.update_stop_order(stop_order)

    def send_long_orders(self) -> None:
        """发送多头委托"""
        for i in range(1, self.max_hold+1):
            if round_to(self.pos * 1000, 1) < round_to(self.trading_size * 1000, 1) * i:
                self.buy(Decimal(str(self.entry_up + self.n * 0.5 * (i-1))), Decimal(str(self.trading_size)), stop=True)

    def send_short_orders(self) -> None:
        """发送空头委托"""
        for i in range(1, self.max_hold+1):
            if round_to(self.pos * 1000, 1) > -round_to(self.trading_size * 1000, 1) * i:
                self.short(Decimal(str(self.entry_down - self.n * 0.5 * (i-1))), Decimal(str(self.trading_size)), stop=True)
