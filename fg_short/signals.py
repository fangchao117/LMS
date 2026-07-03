"""
短线信号 —— 在全历史上计算，严格因果。

模式：
  · trend_ma  —— 三均线趋势（推荐，小资金 1 手）
  · breakout  —— Donchian 突破 + EMA 过滤
  · ema_cross —— EMA 金叉/死叉 + 量能
  · rsi_momo  —— RSI 动量 + EMA 趋势
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

import config


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def _trend_ma(pdf: pd.DataFrame, s: int, m: int, l: int) -> np.ndarray:
    c = pdf["close"]
    ma_s, ma_m, ma_l = c.rolling(s).mean(), c.rolling(m).mean(), c.rolling(l).mean()
    long_c = (ma_s > ma_m) & (ma_m > ma_l) & (c > ma_m)
    short_c = (ma_s < ma_m) & (ma_m < ma_l) & (c < ma_m)
    return np.where(long_c, 1.0, np.where(short_c, -1.0, 0.0))


def generate_signals(
    df: pl.DataFrame,
    mode: str = config.SIGNAL_MODE,
    ema_fast: int = config.EMA_FAST,
    ema_slow: int = config.EMA_SLOW,
    donchian: int = config.DONCHIAN,
    ma_short: int = config.MA_SHORT,
    ma_medium: int = config.MA_MEDIUM,
    ma_long: int = config.MA_LONG,
    rsi_period: int = config.RSI_PERIOD,
) -> pl.DataFrame:
    pdf = df.sort("datetime").to_pandas()
    c, h, l, v = pdf["close"], pdf["high"], pdf["low"], pdf["volume"]

    if mode == "trend_ma":
        signal = _trend_ma(pdf, ma_short, ma_medium, ma_long)
        out_cols = {
            "datetime": pdf["datetime"],
            "vt_symbol": pdf["vt_symbol"],
            config.SIGNAL_COL: signal.astype(float),
            "ma_s": pdf["close"].rolling(ma_short).mean().values,
            "ma_m": pdf["close"].rolling(ma_medium).mean().values,
            "ma_l": pdf["close"].rolling(ma_long).mean().values,
        }
    else:
        ef, es = _ema(c, ema_fast), _ema(c, ema_slow)
        rsi = _rsi(c, rsi_period)
        vol_ma = v.rolling(10).mean()
        upper = h.rolling(donchian).max().shift(1)
        lower = l.rolling(donchian).min().shift(1)
        signal = np.zeros(len(pdf), dtype=float)

        if mode == "breakout":
            long_cond = (c > upper) & (ef > es) & (c > ef)
            short_cond = (c < lower) & (ef < es) & (c < ef)
            signal = np.where(long_cond, 1.0, np.where(short_cond, -1.0, 0.0))

        elif mode == "ema_cross":
            cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
            cross_dn = (ef < es) & (ef.shift(1) >= es.shift(1))
            vol_ok = v > vol_ma * 1.1
            signal = np.where(cross_up & vol_ok, 1.0, np.where(cross_dn & vol_ok, -1.0, np.nan))
            signal = pd.Series(signal).ffill().fillna(0.0).to_numpy()

        elif mode == "rsi_momo":
            long_cond = (rsi < config.RSI_OS) & (rsi.shift(1) >= config.RSI_OS) & (ef > es)
            short_cond = (rsi > config.RSI_OB) & (rsi.shift(1) <= config.RSI_OB) & (ef < es)
            signal = np.where(long_cond, 1.0, np.where(short_cond, -1.0, np.nan))
            signal = pd.Series(signal).ffill().fillna(0.0).to_numpy()

        else:
            raise ValueError(f"unknown mode: {mode}")

        out_cols = {
            "datetime": pdf["datetime"],
            "vt_symbol": pdf["vt_symbol"],
            config.SIGNAL_COL: signal.astype(float),
            "ema_fast": ef.values,
            "ema_slow": es.values,
            "rsi": rsi.values,
        }

    return pl.from_pandas(pd.DataFrame(out_cols)).with_columns(
        pl.col("datetime").cast(pl.Datetime("us"))
    )
