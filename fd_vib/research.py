"""
第四梯队 —— 扩展研究因子

GTJA 幸存者补充、动量加速、形态一致性等。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ops import decay_linear, delay, delta, sign, ts_corr, ts_max, ts_mean, ts_min, ts_rank, ts_std

FactorFn = Callable[[pd.DataFrame], pd.Series]
_REGISTRY: dict[str, FactorFn] = {}


def register(name: str):
    def wrap(fn: FactorFn) -> FactorFn:
        _REGISTRY[name] = fn
        return fn
    return wrap


def all_factors() -> dict[str, FactorFn]:
    return dict(_REGISTRY)


@register("res_gtja_054")
def res_gtja_054(df: pd.DataFrame) -> pd.Series:
    """GTJA191 #054 幸存者"""
    c, v = df["close"], df["volume"]
    corr = ts_corr(c, v, 20)
    rank_corr = ts_rank(corr, 20)
    rank_v = ts_rank(v.astype(float), 20)
    return -rank_corr * rank_v


@register("res_gtja_163")
def res_gtja_163(df: pd.DataFrame) -> pd.Series:
    """GTJA191 #163 幸存者"""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    ret = c.pct_change()
    hl = (h - l) / (ts_mean(h - l, 20) + 1e-12)
    return -ts_rank(ret, 20) * hl * ts_rank(v.astype(float), 20)


@register("res_mom_accel")
def res_mom_accel(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float)
    m5 = c / c.shift(5) - 1.0
    m20 = c / c.shift(20) - 1.0
    return m5 - m20


@register("res_breakout_strength")
def res_breakout_strength(df: pd.DataFrame) -> pd.Series:
    c, h, l = df["close"], df["high"], df["low"]
    hi = ts_max(h, 38)
    lo = ts_min(l, 38)
    width = hi - lo
    pos = (c - lo) / (width + 1e-12) - 0.5
    return pos * (width / (ts_mean(width, 38) + 1e-12))


@register("res_trend_consistency")
def res_trend_consistency(df: pd.DataFrame) -> pd.Series:
    """多周期趋势方向一致性（LLM 风格简化）"""
    c = df["close"].astype(float)
    dirs = []
    for w in (5, 10, 20, 40):
        dirs.append(np.sign(c - c.shift(w)))
    arr = np.column_stack(dirs)
    return pd.Series(arr.mean(axis=1), index=df.index)


@register("res_vol_breakout")
def res_vol_breakout(df: pd.DataFrame) -> pd.Series:
    c, v = df["close"], df["volume"]
    ret = c.pct_change()
    vol_spike = v.astype(float) / (ts_mean(v.astype(float), 20) + 1e-12)
    return np.sign(ret) * np.log1p(vol_spike)


@register("res_mean_revert_z")
def res_mean_revert_z(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float)
    m = ts_mean(c, 20)
    sd = ts_std(c, 20)
    return -(c - m) / (sd + 1e-12)


@register("res_decay_mom")
def res_decay_mom(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float)
    ret = c.pct_change()
    return decay_linear(ret, 20)


@register("res_intraday_pressure")
def res_intraday_pressure(df: pd.DataFrame) -> pd.Series:
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    body = (c - o) / (h - l + 1e-12)
    return ts_mean(body, 5)


@register("res_signed_vol")
def res_signed_vol(df: pd.DataFrame) -> pd.Series:
    c, v = df["close"], df["volume"]
    ret = c.pct_change()
    return np.sign(ret) * ts_rank(v.astype(float), 20)


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
