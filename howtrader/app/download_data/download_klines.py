import requests
from urllib.parse import urlencode
from typing import List, Dict
from howtrader.trader.database import BaseDatabase, get_database
from howtrader.trader.constant import LOCAL_TZ
from howtrader.trader.object import Exchange, Interval
from tzlocal import get_localzone_name
from howtrader.trader.object import BarData
from datetime import datetime, timedelta
from time import sleep
import pdb

database: BaseDatabase = get_database()

import pytz
tzinfo = pytz.timezone(get_localzone_name())

def parse_timestamp(timestamp: str) -> datetime:
    try:
        ts = float(timestamp)
        dt: datetime = datetime.fromtimestamp(ts / 1000)
        return dt.replace(tzinfo=LOCAL_TZ)
    except ValueError:
        return datetime.now(tz=LOCAL_TZ)

def get_klines_okx(instId, interval = Interval.MINUTE, start=None, end=None, limit=100) -> List[BarData]:
    """获取k线数据"""
    buf: Dict[datetime, BarData] = {}
    end_time: str = str(int(end.timestamp()//60) * 60 * 1000)
    start_ts: int = int(start.timestamp()//60) * 60 * 1000  # ts in millisecond
    path: str = "/api/v5/market/history-candles"
    
    while True:
        sleep(0.2)

        params: dict = {
            "instId": instId,
            "bar": interval.value,
            "limit": limit,
            "after": end_time
        }
        
        url = "https://www.okx.com" + path + '?' + urlencode(params)
        resp = requests.get(url)
        
        if resp.status_code // 100 != 2:
            msg = f"request failed，code：{resp.status_code} msg：{resp.text}"
            print(msg)
            break
        
        else:
            data: dict = resp.json()

            if not data["data"]:
                m = data["msg"]
                msg = f"request historical candles failed: {m}"
                print(msg)
                break
            
            for bar_list in data["data"]:
                ts, o, h, l, c, vol, _, _, confirmed = bar_list
                if confirmed:
                    dt = parse_timestamp(ts)
                    bar: BarData = BarData(
                        symbol=instId + '-OKX',
                        exchange=Exchange.OKX,
                        datetime=dt,
                        interval=interval,
                        volume=float(vol),
                        open_price=float(o),
                        high_price=float(h),
                        low_price=float(l),
                        close_price=float(c),
                        gateway_name="OKX"
                    )
                    buf[bar.datetime] = bar

            begin: str = data["data"][-1][0]
            end: str = data["data"][0][0]
            end_time = begin
            msg: str = f"request historical candles，{instId} - {interval.value}，{parse_timestamp(begin)} - {parse_timestamp(end)}"
            print(msg)
            if int(begin) < start_ts:
                break

    index: List[datetime] = list(buf.keys())
    index.sort()
    history: List[BarData] = [buf[i] for i in index]
    return history
