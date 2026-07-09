"""
Vibe-Trading / crypto-kol 形态因子 —— 时序版（玻璃 30m）

参考 Vibe-Trading pattern/regime capabilities，改为单品种滚动实现。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ops import delay, ts_max, ts_mean, ts_min, ts_rank, ts_std

FactorFn = Callable[[pd.DataFrame], pd.Series]
_REGISTRY: dict[str, FactorFn] = {}


def register(name: str):
    def wrap(fn: FactorFn) -> FactorFn:
        _REGISTRY[name] = fn
        return fn
    return wrap


def all_factors() -> dict[str, FactorFn]:
    return dict(_REGISTRY)


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * pd.Series(pdm, index=c.index).rolling(n).mean() / (atr + 1e-9)
    mdi = 100 * pd.Series(mdm, index=c.index).rolling(n).mean() / (atr + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    return dx.rolling(n).mean()


@register("vibe_fake_breakout")
def fake_breakout(df: pd.DataFrame) -> pd.Series:
    c, h, l = df["close"], df["high"], df["low"]
    hi20 = ts_max(h, 20).shift(1)
    lo20 = ts_min(l, 20).shift(1)
    fake_up = (delay(c, 1) > hi20) & (c < hi20)
    fake_dn = (delay(c, 1) < lo20) & (c > lo20)
    return np.where(fake_up, -0.7, np.where(fake_dn, 0.7, 0.0))


@register("vibe_sfp")
def sfp(df: pd.DataFrame) -> pd.Series:
    c, h, l = df["close"], df["high"], df["low"]
    hi20 = ts_max(h, 20).shift(1)
    lo20 = ts_min(l, 20).shift(1)
    wick_above = (h > hi20) & (c < hi20)
    wick_below = (l < lo20) & (c > lo20)
    return np.where(wick_above, -0.6, np.where(wick_below, 0.6, 0.0))


@register("vibe_bb_squeeze")
def bb_squeeze(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    m = ts_mean(c, 20)
    sd = c.rolling(20).std()
    width = 2 * sd / (m + 1e-12)
    w20 = width.rolling(100, min_periods=30).quantile(0.2)
    squeeze = delay(width, 1) < delay(w20, 1)
    upper, lower = m + 2 * sd, m - 2 * sd
    up = squeeze & (c > upper)
    dn = squeeze & (c < lower)
    return np.where(up, 0.6, np.where(dn, -0.6, 0.0))


@register("vibe_range_fade")
def range_fade(df: pd.DataFrame) -> pd.Series:
    c, h, l = df["close"], df["high"], df["low"]
    adx_v = _adx(h, l, c, 14)
    hi20, lo20 = ts_max(h, 20), ts_min(l, 20)
    pos = ((c - lo20) / (hi20 - lo20 + 1e-12)).clip(0, 1)
    in_range = adx_v < 20
    return np.where(in_range & (pos > 0.85), -0.5,
                    np.where(in_range & (pos < 0.15), 0.5, 0.0))


@register("vibe_wyckoff_spring")
def wyckoff_spring(df: pd.DataFrame) -> pd.Series:
    c, h, l = df["close"], df["high"], df["low"]
    lo50 = ts_min(l, 50)
    near = c < lo50 * 1.03
    wick = l < lo50.shift(1)
    close_above = c > lo50.shift(1)
    body = (c - df["open"]).abs()
    rng = (h - l).replace(0, np.nan)
    lower_wick = (np.minimum(c, df["open"]) - l) / (rng + 1e-12)
    trig = near & wick & close_above & (lower_wick > 0.4)
    return np.where(trig, 0.75, 0.0)


@register("vibe_regime_trend")
def regime_trend(df: pd.DataFrame) -> pd.Series:
    c, h, l = df["close"], df["high"], df["low"]
    ma50, ma200 = ts_mean(c, 50), ts_mean(c, 200)
    adx_v = _adx(h, l, c, 14)
    up = (c > ma200) & (ma50 > ma200) & (adx_v > 25)
    dn = (c < ma200) & (ma50 < ma200) & (adx_v > 25)
    return np.where(up, 1.0, np.where(dn, -1.0, 0.0))


@register("vibe_momentum_roc")
def momentum_roc(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    return c / delay(c, 10) - 1.0


@register("vibe_vol_breakout")
def vol_breakout(df: pd.DataFrame) -> pd.Series:
    c, v = df["close"], df["volume"]
    ret = c.pct_change()
    vol_ratio = v / (ts_mean(v, 20) + 1e-12)
    return ret * np.log1p(vol_ratio)


def compute_panel(df: pd.DataFrame, names: list[str] | None = None) -> pd.DataFrame:
    reg = all_factors()
    keys = names or list(reg.keys())
    out = {}
    for k in keys:
        if k not in reg:
            continue
        try:
            out[k] = pd.Series(reg[k](df), index=df.index).astype(float)
        except Exception:
            out[k] = np.nan
    return pd.DataFrame(out, index=df.index)
