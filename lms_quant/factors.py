"""
多因子层 —— 与 LMS 自适应信号融合，全部严格因果（仅用截至 T 日数据）。

因子分组：
  · lms        —— NLMS 趋势预测（主因子）
  · momentum   —— 多周期 ROC 动量
  · ma_regime  —— 三均线趋势状态
  · volume     —— 量价配合
  · oi         —— 持仓量变化
  · rsi        —— RSI 摆动
  · atr_trend  —— ATR 归一化趋势强度
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def zscore_causal(raw: np.ndarray, min_periods: int = 20, smooth: int = 0) -> np.ndarray:
    """扩展窗 z-score + 可选 EMA 平滑。"""
    s = pd.Series(raw, dtype=float)
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std()
    z = (s - mu) / (sd + 1e-12)
    if smooth > 0:
        z = z.ewm(span=smooth, adjust=False).mean()
    return z.to_numpy()


def ma_regime(
    close: pd.Series,
    short: int = config.MA_SHORT,
    medium: int = config.MA_MEDIUM,
    long: int = config.MA_LONG,
) -> np.ndarray:
    ma_s = close.rolling(short).mean()
    ma_m = close.rolling(medium).mean()
    ma_l = close.rolling(long).mean()
    bull = (ma_s > ma_m) & (ma_m > ma_l) & (close > ma_s)
    bear = (ma_s < ma_m) & (ma_m < ma_l) & (close < ma_s)
    return np.where(bull, 1.0, np.where(bear, -1.0, 0.0))


def momentum(close: pd.Series, windows: tuple[int, ...] = config.MOM_WINDOWS) -> np.ndarray:
    rocs = pd.concat([close / close.shift(w) - 1 for w in windows], axis=1)
    return rocs.mean(axis=1, skipna=True).to_numpy()


def rsi_signal(close: pd.Series, period: int = config.RSI_PERIOD) -> np.ndarray:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-12))
    return ((rsi - 50) / 50).to_numpy()


def volume_signal(close: pd.Series, volume: pd.Series, window: int = config.VOL_WINDOW) -> np.ndarray:
    vma = volume.rolling(window).mean()
    vol_ratio = volume / (vma + 1e-12)
    direction = np.sign(close.diff().to_numpy())
    strength = np.clip(vol_ratio.to_numpy() - 1.0, -2.0, 2.0)
    return direction * strength


def oi_signal(close: pd.Series, oi: pd.Series, window: int = config.OI_WINDOW) -> np.ndarray:
    oi_chg = oi / oi.shift(window) - 1
    direction = np.sign(close.diff().to_numpy())
    strength = np.clip(oi_chg.to_numpy() * 5.0, -2.0, 2.0)
    return direction * strength


def atr_trend(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> np.ndarray:
    prev = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    move = close - close.shift(window)
    return (move / (atr + 1e-12)).to_numpy()


def compute_raw_factors(df: pd.DataFrame, lms_raw: np.ndarray) -> dict[str, np.ndarray]:
    """返回各因子原始序列（未 z-score）。"""
    close = df["close"]
    return {
        "lms": lms_raw,
        "momentum": momentum(close),
        "ma_regime": ma_regime(close),
        "volume": volume_signal(close, df["volume"]),
        "oi": oi_signal(close, df["open_interest"]),
        "rsi": rsi_signal(close),
        "atr_trend": atr_trend(df["high"], df["low"], close),
    }


def factor_directions(raw_factors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """离散方向：用于一致性过滤。"""
    dirs: dict[str, np.ndarray] = {}
    for name, arr in raw_factors.items():
        dirs[name] = np.sign(arr)
    return dirs


def fuse_factors(
    raw_factors: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    min_periods: int = config.FACTOR_ZSCORE_MIN,
    smooth: int = config.SIGNAL_SMOOTH,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    加权融合：各因子因果 z-score 后按权重求和，再对合成信号做一次 z-score。
    返回 (composite_z, factor_z_dict)。
    """
    w = weights or config.FACTOR_WEIGHTS
    factor_z: dict[str, np.ndarray] = {}
    composite = np.zeros(len(next(iter(raw_factors.values()))), dtype=float)

    for name, raw in raw_factors.items():
        weight = w.get(name, 0.0)
        if weight == 0:
            continue
        z = zscore_causal(raw, min_periods=min_periods, smooth=0)
        factor_z[name] = z
        composite += weight * np.nan_to_num(z, nan=0.0)

    final = zscore_causal(composite, min_periods=min_periods, smooth=smooth)
    return final, factor_z
