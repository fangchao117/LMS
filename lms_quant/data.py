"""
数据层 —— 读取玻璃日线，产出：
  · load_arrays()   -> (datetimes, close, returns)  给滤波器
  · build_bars()    -> list[BarData]                给 vnpy 回测
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval

import config


def _read() -> pd.DataFrame:
    df = pd.read_parquet(config.DATA_FILE)
    df = df.dropna(subset=["close"]).drop_duplicates("datetime").sort_values("datetime")
    return df.reset_index(drop=True)


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (datetime[ns], close, returns)。returns[0]=0（首根无前值）。"""
    df = _read()
    dt = df["datetime"].to_numpy()
    close = df["close"].to_numpy(dtype=float)

    if config.RETURN_MODE == "log":
        returns = np.diff(np.log(close), prepend=np.log(close[0]))
    else:
        returns = np.diff(close, prepend=close[0]) / np.concatenate([[close[0]], close[:-1]])
    returns[0] = 0.0
    return dt, close, returns


def build_bars() -> list[BarData]:
    df = _read()
    exchange = Exchange(config.EXCHANGE_STR)
    bars: list[BarData] = []
    for row in df.itertuples(index=False):
        bars.append(BarData(
            symbol=config.SYMBOL, exchange=exchange,
            datetime=row.datetime.to_pydatetime(), interval=Interval.DAILY,
            open_price=float(row.open), high_price=float(row.high),
            low_price=float(row.low), close_price=float(row.close),
            volume=float(row.volume), turnover=float(row.turnover),
            open_interest=float(row.open_interest), gateway_name="DB",
        ))
    return bars


if __name__ == "__main__":
    dt, close, r = load_arrays()
    print("bars:", len(close), "| return mean/std:", f"{r.mean():.2e}/{r.std():.2e}")
