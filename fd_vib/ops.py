"""时序算子 —— GTJA191 单品种适配（截面 rank → 滚动 ts_rank）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def delay(s: pd.Series, d: int) -> pd.Series:
    return s.shift(d)


def delta(s: pd.Series, d: int) -> pd.Series:
    return s.diff(d)


def ts_sum(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(1, w // 2)).sum()


def ts_mean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(1, w // 2)).mean()


def ts_std(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(1, w // 2)).std()


def ts_max(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).max()


def ts_min(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).min()


def ts_rank(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(3, w // 2)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def ts_corr(a: pd.Series, b: pd.Series, w: int) -> pd.Series:
    return a.rolling(w, min_periods=max(3, w // 2)).corr(b)


def decay_linear(s: pd.Series, w: int) -> pd.Series:
    weights = np.arange(1, w + 1, dtype=float)
    weights /= weights.sum()

    def _decay(x: np.ndarray) -> float:
        n = len(x)
        wts = weights[-n:]
        wts = wts / wts.sum()
        return float((x * wts).sum())

    return s.rolling(w, min_periods=max(2, w // 2)).apply(_decay, raw=True)


def sign(s: pd.Series) -> pd.Series:
    return np.sign(s)


def vwap(df: pd.DataFrame, w: int = 20) -> pd.Series:
    c = df["close"]
    v = df["volume"].replace(0, np.nan)
    tp = (df["high"] + df["low"] + c) / 3.0
    return (tp * v).rolling(w).sum() / (v.rolling(w).sum() + 1e-12)


def zscore_causal(s: pd.Series, min_periods: int = 40) -> pd.Series:
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    return (s - mu) / (sd + 1e-12)
