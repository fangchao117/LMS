"""
GTJA191 因子子集 —— 时序版（Vibe-Trading gtja191 zoo 适配单品种）

来源：国泰君安短周期价量因子，截面 rank 改为 rolling ts_rank。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ops import (
    decay_linear,
    delay,
    delta,
    sign,
    ts_corr,
    ts_max,
    ts_mean,
    ts_min,
    ts_rank,
    ts_std,
    ts_sum,
    vwap,
)

FactorFn = Callable[[pd.DataFrame], pd.Series]

_REGISTRY: dict[str, FactorFn] = {}


def register(name: str):
    def wrap(fn: FactorFn) -> FactorFn:
        _REGISTRY[name] = fn
        return fn
    return wrap


def all_factors() -> dict[str, FactorFn]:
    return dict(_REGISTRY)


@register("gtja_002")
def gtja_002(df: pd.DataFrame) -> pd.Series:
    c, h, l = df["close"], df["high"], df["low"]
    inner = ((c - l) - (h - c)) / (h - l + 1e-12)
    return -delta(inner, 1)


@register("gtja_012")
def gtja_012(df: pd.DataFrame) -> pd.Series:
    c, v = df["close"], df["volume"]
    return sign(delta(v, 1)) * (-delta(c, 1))


@register("gtja_018")
def gtja_018(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    return c / (delay(c, 5) + 1e-12) - 1.0


@register("gtja_031")
def gtja_031(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    m = ts_mean(c, 12)
    return (c - m) / (m + 1e-12)


@register("gtja_034")
def gtja_034(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    return ts_mean(c, 12) / (c + 1e-12) - 1.0


@register("gtja_041")
def gtja_041(df: pd.DataFrame) -> pd.Series:
    c, h, l = df["close"], df["high"], df["low"]
    vw = vwap(df, 20)
    return np.sqrt(h * l) - vw


@register("gtja_046")
def gtja_046(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    num = ts_mean(c, 3) + ts_mean(c, 6) + ts_mean(c, 12) + ts_mean(c, 24)
    return num / (4.0 * c + 1e-12) - 1.0


@register("gtja_053")
def gtja_053(df: pd.DataFrame) -> pd.Series:
    c, h, l = df["close"], df["high"], df["low"]
    inner = ((c - l) - (h - c)) / (h - l + 1e-12)
    return -delta(inner, 9)


@register("gtja_058")
def gtja_058(df: pd.DataFrame) -> pd.Series:
    c, v = df["close"], df["volume"]
    return ts_rank(c, 20) * v / (ts_mean(v, 20) + 1e-12)


@register("gtja_063")
def gtja_063(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    d = delta(c, 1)
    up = ts_mean(d.clip(lower=0), 6)
    dn = ts_mean(d.abs(), 6)
    return up / (dn + 1e-12) * 100 - 50


@register("gtja_088")
def gtja_088(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    return (c - delay(c, 20)) / (delay(c, 20) + 1e-12)


@register("gtja_101")
def gtja_101(df: pd.DataFrame) -> pd.Series:
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    return (c - o) / (h - l + 1e-12)


@register("gtja_111")
def gtja_111(df: pd.DataFrame) -> pd.Series:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    inner = v * ((c - l) - (h - c)) / (h - l + 1e-12)
    return ts_mean(inner, 11)


@register("gtja_127")
def gtja_127(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    return (ts_mean(c, 12) - c) / (ts_mean(c, 12) + 1e-12)


@register("gtja_133")
def gtja_133(df: pd.DataFrame) -> pd.Series:
    h, l = df["high"], df["low"]
    return (h - l) / (ts_mean(h - l, 20) + 1e-12) - 1.0


@register("gtja_143")
def gtja_143(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    return c / (delay(c, 1) + 1e-12) - 1.0


@register("gtja_151")
def gtja_151(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    return (c - ts_min(c, 20)) / (ts_max(c, 20) - ts_min(c, 20) + 1e-12) - 0.5


@register("gtja_171")
def gtja_171(df: pd.DataFrame) -> pd.Series:
    """反转：(-1 * delta(close,1)) * rank(volume) — 时序版"""
    c, v = df["close"], df["volume"]
    return -delta(c, 1) * ts_rank(v, 20)


@register("gtja_176")
def gtja_176(df: pd.DataFrame) -> pd.Series:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    inner = ((c - l) - (h - c)) / (h - l + 1e-12)
    return ts_corr(inner, v, 10)


@register("gtja_184")
def gtja_184(df: pd.DataFrame) -> pd.Series:
    c, o = df["close"], df["open"]
    return ts_rank(c - o, 10)


@register("gtja_188")
def gtja_188(df: pd.DataFrame) -> pd.Series:
    h, l, v = df["high"], df["low"], df["volume"]
    return ts_mean((h - l) / (v + 1e-12), 20)


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
