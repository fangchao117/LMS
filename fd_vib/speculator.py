"""
投机专用因子 —— 玻璃 30m 持仓量 / 量价微观结构（第五梯队补充）

针对「单合约持有、不移仓」场景，强化 OI 四象限与波动突破。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from fundamental import _oi_series
from ops import ts_corr, ts_mean, ts_rank, ts_std

FactorFn = Callable[[pd.DataFrame], pd.Series]
_REGISTRY: dict[str, FactorFn] = {}


def register(name: str):
    def wrap(fn: FactorFn) -> FactorFn:
        _REGISTRY[name] = fn
        return fn
    return wrap


def all_factors() -> dict[str, FactorFn]:
    return dict(_REGISTRY)


@register("spec_oi_accel")
def spec_oi_accel(df: pd.DataFrame) -> pd.Series:
    """持仓加速：二阶变化"""
    oi = _oi_series(df)
    d1 = oi.pct_change()
    return d1 - d1.shift(3)


@register("spec_smart_money")
def spec_smart_money(df: pd.DataFrame) -> pd.Series:
    """价涨增仓 + 放量：偏多头资金流入"""
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    oi = _oi_series(df)
    ret = c.pct_change()
    oi_up = oi.diff() > 0
    vol_hot = v > ts_mean(v, 20) * 1.1
    score = np.where(ret > 0, 1.0, np.where(ret < 0, -1.0, 0.0))
    score = score * oi_up.astype(float) * vol_hot.astype(float)
    return pd.Series(score, index=df.index)


@register("spec_short_squeeze")
def spec_short_squeeze(df: pd.DataFrame) -> pd.Series:
    """价涨 + OI 降：空头平仓推动"""
    c = df["close"].astype(float)
    oi = _oi_series(df)
    ret = c.pct_change()
    oi_dn = oi.diff() < 0
    return (ret * oi_dn.astype(float)).rolling(5).sum()


@register("spec_range_break")
def spec_range_break(df: pd.DataFrame) -> pd.Series:
    """窄幅后突破：波动压缩 + 方向"""
    c, h, l = df["close"], df["high"], df["low"]
    width = (h - l) / (c + 1e-12)
    comp = width / (ts_mean(width, 40) + 1e-12)
    ret = c.pct_change()
    return -comp * ts_rank(ret.abs(), 10) + ret * (1.0 / (comp + 0.1))


@register("spec_night_gap")
def spec_night_gap(df: pd.DataFrame) -> pd.Series:
    """夜盘跳空延续/反转（简化：大 gap 后反向）"""
    o, c = df["open"], df["close"]
    prev_c = c.shift(1)
    gap = o / (prev_c + 1e-12) - 1.0
    big = gap.abs() > gap.abs().rolling(60).quantile(0.85)
    return np.where(big, -gap, gap * 0.3)


@register("spec_vol_skew")
def spec_vol_skew(df: pd.DataFrame) -> pd.Series:
    """上涨量 vs 下跌量偏度"""
    c, v = df["close"], df["volume"].astype(float)
    ret = c.pct_change()
    up_vol = (v * (ret > 0).astype(float)).rolling(20).sum()
    dn_vol = (v * (ret < 0).astype(float)).rolling(20).sum()
    return (up_vol - dn_vol) / (up_vol + dn_vol + 1e-12)


@register("spec_trend_oi_confirm")
def spec_trend_oi_confirm(df: pd.DataFrame) -> pd.Series:
    """20 根趋势 + OI 同向确认"""
    c = df["close"].astype(float)
    oi = _oi_series(df)
    tr = np.sign(c - c.shift(20))
    oi_tr = np.sign(oi - oi.shift(20))
    return tr * oi_tr


@register("spec_reversal_extreme")
def spec_reversal_extreme(df: pd.DataFrame) -> pd.Series:
    """极端偏离均线后反转"""
    c = df["close"].astype(float)
    m = ts_mean(c, 30)
    sd = ts_std(c, 30)
    z = (c - m) / (sd + 1e-12)
    return -z.clip(-3, 3)


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
