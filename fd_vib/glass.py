"""
第六梯队 —— 玻璃期货专项因子

微观结构、时段、季节性、波动率状态、FG00 日线联动。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from fundamental import _merge_fg00_daily, _oi_series
from ops import ts_corr, ts_mean, ts_rank, ts_std

FactorFn = Callable[[pd.DataFrame], pd.Series]
_REGISTRY: dict[str, FactorFn] = {}

_FG00 = Path(__file__).resolve().parent.parent / "data" / "FG00.CZCE.parquet"


def register(name: str):
    def wrap(fn: FactorFn) -> FactorFn:
        _REGISTRY[name] = fn
        return fn
    return wrap


def all_factors() -> dict[str, FactorFn]:
    return dict(_REGISTRY)


def _hour(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt).dt.hour


@register("glass_amihud")
def glass_amihud(df: pd.DataFrame) -> pd.Series:
    c, v = df["close"].astype(float), df["volume"].astype(float).replace(0, np.nan)
    ret = c.pct_change().abs()
    illiq = (ret / (c * v + 1e-12)).rolling(20).mean()
    return -np.log1p(illiq.fillna(0) * 1e6)


@register("glass_turnover_ratio")
def glass_turnover_ratio(df: pd.DataFrame) -> pd.Series:
    if "turnover" in df.columns and df["turnover"].notna().any():
        t = df["turnover"].astype(float)
    else:
        t = df["close"].astype(float) * df["volume"].astype(float) * 20
    return t / (ts_mean(t, 20) + 1e-12) - 1.0


@register("glass_oi_price_corr")
def glass_oi_price_corr(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float).pct_change()
    oi = _oi_series(df).pct_change()
    return ts_corr(c, oi, 20)


@register("glass_vol_regime")
def glass_vol_regime(df: pd.DataFrame) -> pd.Series:
    """低波为正（趋势友好），高波为负"""
    ret = df["close"].astype(float).pct_change()
    vol = ts_std(ret, 20)
    vr = vol / (ts_mean(vol, 60) + 1e-12)
    return -(vr - 1.0)


@register("glass_range_compress")
def glass_range_compress(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    width = (h - l) / (c + 1e-12)
    return -(width / (ts_mean(width, 40) + 1e-12) - 1.0)


@register("glass_gap")
def glass_gap(df: pd.DataFrame) -> pd.Series:
    o, c = df["open"], df["close"]
    gap = o / (c.shift(1) + 1e-12) - 1.0
    return gap.rolling(5).mean()


@register("glass_night_bias")
def glass_night_bias(df: pd.DataFrame) -> pd.Series:
    """夜盘(21-23) vs 日盘收益差"""
    c = df["close"].astype(float)
    ret = c.pct_change()
    hr = _hour(df["datetime"])
    night = ret.where((hr >= 21) | (hr <= 2), 0.0)
    day = ret.where((hr >= 9) & (hr <= 15), 0.0)
    return ts_mean(night, 20) - ts_mean(day, 20)


@register("glass_weekday")
def glass_weekday(df: pd.DataFrame) -> pd.Series:
    wd = pd.to_datetime(df["datetime"]).dt.dayofweek
    c = df["close"].astype(float).pct_change()
    # 周一偏空、周五偏多（经验性，由 IC 检验）
    bias = np.where(wd == 0, -0.3, np.where(wd == 4, 0.3, 0.0))
    return pd.Series(bias, index=df.index) + c.rolling(5).mean()


@register("glass_ma_stack")
def glass_ma_stack(df: pd.DataFrame) -> pd.Series:
    c = df["close"].astype(float)
    mas = [ts_mean(c, w) for w in (10, 20, 40, 80)]
    stack = sum(np.sign(c - m) for m in mas) / 4.0
    return stack


@register("glass_pullback")
def glass_pullback(df: pd.DataFrame) -> pd.Series:
    """趋势中回调深度"""
    c = df["close"].astype(float)
    hi = c.rolling(40).max()
    lo = c.rolling(40).min()
    trend = np.sign(c - ts_mean(c, 40))
    pos = (c - lo) / (hi - lo + 1e-12) - 0.5
    return trend * (0.5 - pos.abs())


@register("glass_fg00_mom")
def glass_fg00_mom(df: pd.DataFrame) -> pd.Series:
    merged = _merge_fg00_daily(df)
    if "close_d" not in merged.columns:
        c = df["close"].astype(float)
        d = df.copy()
        d["date"] = d["datetime"].dt.date
        daily_c = d.groupby("date")["close"].last()
        mom = daily_c / daily_c.shift(10) - 1.0
        return d["date"].map(mom).astype(float)
    d = merged.copy()
    d["date"] = d["datetime"].dt.date
    daily_c = d.groupby("date")["close_d"].last()
    mom = daily_c / daily_c.shift(10) - 1.0
    return d["date"].map(mom).astype(float)


@register("glass_oi_extreme")
def glass_oi_extreme(df: pd.DataFrame) -> pd.Series:
    oi = _oi_series(df)
    z = (oi - ts_mean(oi, 60)) / (ts_std(oi, 60) + 1e-12)
    return -z.abs() + z * 0.2


@register("glass_liq_shock")
def glass_liq_shock(df: pd.DataFrame) -> pd.Series:
    v = df["volume"].astype(float)
    vr = v / (ts_mean(v, 20) + 1e-12)
    ret = df["close"].astype(float).pct_change()
    return np.sign(ret) * np.log1p(vr)


@register("glass_chop_filter")
def glass_chop_filter(df: pd.DataFrame) -> pd.Series:
    """震荡识别：均线缠绕 → 负值（应减仓）"""
    c = df["close"].astype(float)
    e1 = c.ewm(span=12, adjust=False).mean()
    e2 = c.ewm(span=48, adjust=False).mean()
    spread = (e1 - e2).abs() / (c + 1e-12)
    return spread / (ts_mean(spread, 40) + 1e-12) - 0.5


@register("glass_body_ratio")
def glass_body_ratio(df: pd.DataFrame) -> pd.Series:
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    body = (c - o).abs() / (h - l + 1e-12)
    return ts_mean(body, 10) - 0.5


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
