from concurrent.futures.process import _ThreadWakeup
from datetime import datetime
from decimal import Decimal
from typing import Type, Dict, List
from copy import deepcopy

from howtrader.trader.object import TradeData, Direction
from howtrader.app.cta_strategy import StopOrder
from howtrader.trader.object import OrderData
from howtrader.trader.constant import Status
from howtrader.app.cta_strategy.base import (
    BacktestingMode,
    EngineType,
    STOPORDER_PREFIX,
    StopOrder,
    StopOrderStatus,
    INTERVAL_DELTA_MAP
)

import pdb


class TradeResult:
    """配对后的完整开平交易"""
    
    def __init__(self, size: int = 1) -> None:
        self.size = size

        # 交易成本
        self.long_cost = 0.0
        self.short_cost = 0.0

        # 交易仓位
        self.long_volume = 0.0
        self.short_volume = 0.0
        self.net_volume = 0.0

        # 开平时点
        self.open_dt: datetime = None
        self.close_dt: datetime = None

        # 成交记录
        self.trades: list[TradeData] = []

        # 交易盈亏
        self.pnl: float = 0
        
    def update_trade(self, trade: TradeData) -> bool:
        """更新成交"""
        # 添加成交记录
        trade.volume = round(float(trade.volume), 3)
        trade.price = round(float(trade.price), 2)
        
        self.trades.append(trade)

        # 更新成交数量和成本
        trade_cost = trade.price * trade.volume * self.size

        if trade.direction == Direction.LONG:
            self.long_volume += trade.volume
            self.long_cost += trade_cost
        else:
            self.short_volume += trade.volume
            self.short_cost += trade_cost

        self.net_volume = self.long_volume - self.short_volume

        if not round(round(float(self.net_volume), 3)*1000):
            self.calculate_result()
            return True
        else:
            return False

    def calculate_result(self) -> None:
        """计算盈亏"""
        # 卖出收到现金，买入付出现金
        self.pnl = self.short_cost - self.long_cost

        self.open_dt = self.trades[0].datetime
        self.close_dt = self.trades[-1].datetime


class ResultManager:
    """交易配对管理器"""

    def __init__(self, size: int = 1) -> None:
        """构造函数"""
        self.size = size

        # 第一条开平交易
        self.result: TradeResult = TradeResult(self.size)

        # 开平交易列表
        self.results: list[TradeResult] = []

    def update_trade(self, trade: TradeData) -> None:
        """更新成交"""
        trade_copy = deepcopy(trade)
        closed = self.result.update_trade(trade_copy)

        # 如果完成平仓，则创建下一条开平交易
        if closed:
            self.results.append(self.result)
            self.result = TradeResult(self.size)

    def get_results(self) -> list[TradeResult]:
        """获取记录"""
        return self.results


class OrderRecorder():
    """记录活跃的委托"""
    
    def __init__(self) -> None:
        """初始化构造函数"""
        # self.stop_orders: Dict[str, StopOrder] = {}
        # self.limit_orders: Dict[str, OrderData] = {}
        self.active_limit_orders: Dict[str, OrderData] = {}
        self.active_stop_orders: Dict[str, StopOrder] = {}
        
    def update_limit_order(self, order: OrderData) -> None:
        """更新限价单委托"""
        order = deepcopy(order)

        if order.status == Status.SUBMITTING or order.status == Status.NOTTRADED or order.status == Status.PARTTRADED:
            self.active_limit_orders[order.vt_orderid] = order

        elif order.status == Status.ALLTRADED or order.status == Status.CANCELLED:
            if order.vt_orderid in self.active_limit_orders:
                self.active_limit_orders.pop(order.vt_orderid)
                
        elif order.status == Status.REJECTED:
            print(f"报单{order.vt_orderid}被拒绝")
                
        else:
            print(f"异常：不存在的报单类型{order.status.value}")
            
    def get_active_limit_orders(self) -> Dict[str, OrderData]:
        active_limit_orders = deepcopy(self.active_limit_orders)
        return active_limit_orders

    def update_stop_order(self, stop_order: StopOrder) -> None:
        """更新停止单委托"""
        stop_order = deepcopy(stop_order)
        
        if stop_order.status == StopOrderStatus.WAITING:
            self.active_stop_orders[stop_order.stop_orderid] = stop_order
        
        elif stop_order.status == StopOrderStatus.TRIGGERED or stop_order.status == StopOrderStatus.CANCELLED:
            if stop_order.stop_orderid in self.active_stop_orders:
                self.active_stop_orders.pop(stop_order.stop_orderid)
                
        else:
            print(f"异常：不存在的报单类型{stop_order.status.value}")
        
    def get_active_stop_orders(self) -> Dict[str, StopOrder]:
        active_stop_orders = deepcopy(self.active_stop_orders)
        return active_stop_orders
