"""
第三梯队 —— 学术价量因子 + Alpha101 子集（时序版）

来源：Vibe-Trading academic zoo / WorldQuant alpha101 单品种适配。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ops import delay, delta, ts_corr, ts_max, ts_mean, ts_min, ts_rank, ts_std, ts_sum

FactorFn = Callable[[pd.DataFrame], pd.Series]
_REGISTRY: dict[str, FactorFn] = {}


def register(name: str):
    def wrap(fn: FactorFn) -> FactorFn:
        _REGISTRY[name] = fn
        return fn
    return wrap


def all_factors() -> dict[str, FactorFn]:
    return dict(_REGISTRY)


@register("acad_strev")
def acad_strev(df: pd.DataFrame) -> pd.Series:
    """短期反转"""
    c = df["close"].astype(float)
    return -(c / c.shift(5) - 1.0)


@register("acad_illiq")
def acad_illiq(df: pd.DataFrame) -> pd.Series:
    """Amihud 非流动性"""
    c = df["close"].astype(float)
    v = df["volume"].astype(float).replace(0, np.nan)
    ret = c.pct_change().abs()
    illiq = (ret / (c * v + 1e-12)).rolling(20).mean()
    return -np.log1p(illiq.fillna(0) * 1e6)


@register("acad_high52w")
def acad_high52w(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float)
    hi = ts_max(c, 260)
    return c / (hi + 1e-12) - 1.0


@register("acad_carhart_mom")
def acad_carhart_mom(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float)
    m12 = c / c.shift(240) - 1.0
    m1 = c / c.shift(20) - 1.0
    return m12 - m1


@register("acad_idiovol")
def acad_idiovol(df: pd.DataFrame) -> pd.Series:
    ret = df["close"].astype(float).pct_change()
    vol = ts_std(ret, 20)
    mkt = ts_mean(ret, 20)
    resid = ret - mkt
    return -ts_std(resid, 20)


@register("acad_rsi14")
def acad_rsi14(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float)
    d = delta(c, 1)
    up = ts_mean(d.clip(lower=0), 14)
    dn = ts_mean((-d).clip(lower=0), 14)
    rs = up / (dn + 1e-12)
    return 100.0 / (1.0 + rs) - 50.0


@register("alpha_001")
def alpha_001(df: pd.DataFrame) -> pd.Series:
    c, v = df["close"], df["volume"]
    ret = c.pct_change()
    inner = ret * np.log1p(v.astype(float))
    return ts_rank(inner, 20) - 0.5


@register("alpha_012")
def alpha_012(df: pd.DataFrame) -> pd.Series:
    c, v = df["close"], df["volume"]
    return np.sign(delta(v, 1)) * (-delta(c, 1))


@register("alpha_033")
def alpha_033(df: pd.DataFrame) -> pd.Series:
    c, o = df["close"], df["open"]
    return ts_rank(-(1.0 - c / (o + 1e-12)), 20)


@register("alpha_041")
def alpha_041(df: pd.DataFrame) -> pd.Series:
    c, h, l = df["close"], df["high"], df["low"]
    inner = np.sqrt(h * l) - c
    return ts_rank(inner, 20)


@register("alpha_054")
def alpha_054(df: pd.DataFrame) -> pd.Series:
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    num = (c - o) + 2.0 * (o - l) + 2.0 * (h - o)
    den = h - l + 1e-12
    return ts_rank(num / den, 20) - 0.5


@register("alpha_101")
def alpha_101(df: pd.DataFrame) -> pd.Series:
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    ret = c.pct_change()
    cond = ret < 0
    inner = np.where(cond, ts_std(ret, 20), c)
    return ts_rank(inner.astype(float), 20) - ts_rank(v.astype(float), 20)


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
