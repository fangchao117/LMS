"""
第二梯队 —— 中国商品学术因子（时序版）

TSMOM、基差动量、期限结构代理、实现波动率。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from fundamental import _merge_fg00_daily, _oi_series
from ops import ts_mean, ts_rank, ts_std

FactorFn = Callable[[pd.DataFrame], pd.Series]
_REGISTRY: dict[str, FactorFn] = {}


def register(name: str):
    def wrap(fn: FactorFn) -> FactorFn:
        _REGISTRY[name] = fn
        return fn
    return wrap


def all_factors() -> dict[str, FactorFn]:
    return dict(_REGISTRY)


@register("comm_tsmom_20d")
def comm_tsmom_20d(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["date"] = d["datetime"].dt.date
    daily_c = d.groupby("date")["close"].last()
    mom_d = daily_c / daily_c.shift(20) - 1.0
    mom = d["date"].map(mom_d).astype(float)
    return np.sign(mom) * mom.abs().pow(0.5)


@register("comm_tsmom_60d")
def comm_tsmom_60d(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["date"] = d["datetime"].dt.date
    daily_c = d.groupby("date")["close"].last()
    mom_d = daily_c / daily_c.shift(60) - 1.0
    mom = d["date"].map(mom_d).astype(float)
    return np.sign(mom) * mom.abs().pow(0.5)


@register("comm_vol_20")
def comm_vol_20(df: pd.DataFrame) -> pd.Series:
    ret = df["close"].astype(float).pct_change()
    return ts_std(ret, 20)


@register("comm_vol_rank")
def comm_vol_rank(df: pd.DataFrame) -> pd.Series:
    ret = df["close"].astype(float).pct_change()
    vol = ts_std(ret, 20)
    return ts_rank(vol, 60) - 0.5


@register("comm_vol_carry")
def comm_vol_carry(df: pd.DataFrame) -> pd.Series:
    """低波环境偏多（carry / 趋势延续）"""
    ret = df["close"].astype(float).pct_change()
    vol = ts_std(ret, 20)
    vol_ma = ts_mean(vol, 60)
    return -(vol / (vol_ma + 1e-12) - 1.0)


@register("comm_basis_proxy")
def comm_basis_proxy(df: pd.DataFrame) -> pd.Series:
    """基差代理：现价相对 60 日均价"""
    c = df["close"].astype(float)
    ma = ts_mean(c, 60)
    return (c - ma) / (ma + 1e-12)


@register("comm_basis_mom")
def comm_basis_mom(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float)
    ma = ts_mean(c, 60)
    basis = (c - ma) / (ma + 1e-12)
    return basis - basis.shift(5)


@register("comm_term_slope")
def comm_term_slope(df: pd.DataFrame) -> pd.Series:
    """期限结构代理：30m 近月 20 日动量 - FG00 日线 20 日动量"""
    d = df.copy()
    d["date"] = d["datetime"].dt.date
    near_d = d.groupby("date")["close"].last()
    near_mom = near_d / near_d.shift(20) - 1.0
    near = d["date"].map(near_mom).astype(float)

    fg = _merge_fg00_daily(df)
    if "close_d" not in fg.columns:
        oi = _oi_series(df)
        oi_d = fg.copy()
        oi_d["date"] = oi_d["datetime"].dt.date
        oi_daily = oi_d.groupby("date")["oi_d"].last() if "oi_d" in oi_d.columns else None
        if oi_daily is not None:
            far_mom = oi_daily / oi_daily.shift(20) - 1.0
            far = d["date"].map(far_mom).astype(float)
            return near - far.fillna(0)
        return near

    fg["date"] = fg["datetime"].dt.date
    far_d = fg.groupby("date")["close_d"].last()
    far_mom = far_d / far_d.shift(20) - 1.0
    far = d["date"].map(far_mom).astype(float)
    return near - far.fillna(0)


@register("comm_roll_pressure")
def comm_roll_pressure(df: pd.DataFrame) -> pd.Series:
    """换月压力：symbol 变化前后价格跳空"""
    if "symbol" not in df.columns:
        return pd.Series(0.0, index=df.index)
    sym = df["symbol"].astype(str)
    roll = (sym != sym.shift(1)).astype(float)
    c = df["close"].astype(float)
    gap = c / c.shift(1) - 1.0
    return -roll * gap


@register("comm_skew_mom")
def comm_skew_mom(df: pd.DataFrame) -> pd.Series:
    ret = df["close"].astype(float).pct_change()
    sk = ret.rolling(40, min_periods=20).skew()
    return sk - sk.shift(10)


def compute_panel(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    reg = all_factors()
    keys = names or list(reg.keys())
    out = {}
    for k in keys:
        if k not in reg:
            continue
        try:
            out[k] = reg[k](df).astype(float)
        except Exception:
            out[k] = np.nan
    return pd.DataFrame(out, index=df.index)
