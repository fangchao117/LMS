"""
30 分钟多因子 —— 仅用 OHLCV，严格因果（shift 防未来）

因子：动量、波动率、偏度、峰度、EMA趋势、VWAP偏离、Amihud非流动性
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS: dict[str, float] = {
    "momentum": 0.25,
    "ema_trend": 0.25,
    "vwap_dev": 0.15,
    "volatility": 0.10,
    "skew": 0.10,
    "kurtosis": 0.05,
    "amihud": 0.10,
}


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def zscore_causal(raw: np.ndarray, min_periods: int = 40) -> np.ndarray:
    s = pd.Series(raw, dtype=float)
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    return ((s - mu) / (sd + 1e-12)).to_numpy()


def ts_mean(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=1).mean().to_numpy()


def compute_raw(df: pd.DataFrame) -> dict[str, np.ndarray]:
    c = df["close"].astype(float)
    v = df["volume"].astype(float).replace(0, np.nan)
    ret = c.pct_change().fillna(0.0)
    n = len(df)

    # 动量：多周期 ROC
    rocs = [c / c.shift(w) - 1 for w in (5, 10, 20)]
    momentum = pd.concat(rocs, axis=1).mean(axis=1).to_numpy()

    # 波动率：低波动略偏多（趋势延续），高波动降权
    vol = ret.rolling(20).std().to_numpy()
    vol_ma = pd.Series(vol).rolling(60, min_periods=20).mean().to_numpy()
    volatility = -(vol / (vol_ma + 1e-12) - 1.0)  # 相对低波为正

    # 偏度、峰度（滚动）
    skew = ret.rolling(40).skew().to_numpy()
    kurt = ret.rolling(40).kurt().to_numpy()
    # 极端峰度时降仓：用负的极端值
    kurtosis = -np.abs(kurt - 3.0)

    # EMA 趋势强度
    ef, es = _ema(c, 12), _ema(c, 48)
    ema_trend = ((ef - es) / (c + 1e-12)).to_numpy()

    # VWAP 偏离（滚动成交量加权）
    tp = (df["high"] + df["low"] + c) / 3.0
    pv = (tp * v).rolling(20).sum()
    vv = v.rolling(20).sum()
    vwap = (pv / (vv + 1e-12)).to_numpy()
    vwap_dev = ((c.to_numpy() - vwap) / (vwap + 1e-12))

    # Amihud：|ret|/成交额，流动性差为负向
    turnover = (c * v * 20).replace(0, np.nan)  # 近似名义成交额
    amihud_raw = (ret.abs() / (turnover + 1e-12)).rolling(20).mean()
    amihud = -np.log1p(amihud_raw.fillna(0).to_numpy() * 1e8)

    return {
        "momentum": momentum,
        "volatility": volatility,
        "skew": skew,
        "kurtosis": kurtosis,
        "ema_trend": ema_trend,
        "vwap_dev": vwap_dev,
        "amihud": amihud,
    }


def fuse(
    raw: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    z_min: int = 40,
    smooth: int = 3,
) -> np.ndarray:
    w = weights or DEFAULT_WEIGHTS
    n = len(next(iter(raw.values())))
    score = np.zeros(n, dtype=float)
    for name, arr in raw.items():
        wt = w.get(name, 0.0)
        if wt <= 0:
            continue
        z = zscore_causal(arr, min_periods=z_min)
        if smooth > 0:
            z = ts_mean(z, smooth)
        score += wt * z
    return score


def score_to_signal(score: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    return np.where(score > threshold, 1.0, np.where(score < -threshold, -1.0, 0.0))
