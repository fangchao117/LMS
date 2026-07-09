"""
日线趋势过滤 —— 供 30m 信号与回测共用

日线 EMA 定方向，30m EMA 执行（实测优于震荡+趋势分拆）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def load_daily(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    path = config.ROOT / "data" / "FG00.CZCE.parquet"
    pdf = pd.read_parquet(path)
    pdf["datetime"] = pd.to_datetime(pdf["datetime"])
    if start:
        pdf = pdf[pdf["datetime"] >= pd.Timestamp(start)]
    if end:
        pdf = pdf[pdf["datetime"] <= pd.Timestamp(end)]
    return pdf.sort_values("datetime").reset_index(drop=True)


def daily_trend_table(
    df_daily: pd.DataFrame,
    ema_fast: int | None = None,
    ema_slow: int | None = None,
) -> pd.DataFrame:
    """每日趋势方向：+1 多 / -1 空 / 0 中性"""
    ef = ema_fast or config.EMA_FAST
    es = ema_slow or config.EMA_SLOW
    c = df_daily["close"]
    e1, e2 = _ema(c, ef), _ema(c, es)
    direction = np.where(e1 > e2, 1, np.where(e1 < e2, -1, 0))
    return pd.DataFrame({
        "date": df_daily["datetime"].dt.date,
        "daily_trend": direction.astype(int),
    })


def align_daily_trend(df_30: pd.DataFrame, trend_tbl: pd.DataFrame) -> pd.Series:
    d = df_30.copy()
    d["date"] = d["datetime"].dt.date
    m = d.merge(trend_tbl, on="date", how="left")
    return m["daily_trend"].ffill().fillna(0).astype(int)


def apply_daily_filter(ltf_sig: pd.Series, daily_trend: pd.Series) -> pd.Series:
    """30m 信号仅保留与日线同向；日线中性则空仓。"""
    out = ltf_sig.copy()
    for i in range(len(out)):
        d = int(daily_trend.iloc[i])
        s = int(np.sign(out.iloc[i])) if out.iloc[i] != 0 else 0
        if d == 0:
            out.iloc[i] = 0.0
        elif d > 0 and s < 0:
            out.iloc[i] = 0.0
        elif d < 0 and s > 0:
            out.iloc[i] = 0.0
    return out


def generate_daily_filter_ema(
    df_30: pd.DataFrame,
    ema_fast: int | None = None,
    ema_slow: int | None = None,
    daily_df: pd.DataFrame | None = None,
) -> pd.Series:
    from signals import generate

    ef = ema_fast or config.EMA_FAST
    es = ema_slow or config.EMA_SLOW
    if daily_df is None:
        daily_df = load_daily(
            str(df_30["datetime"].min().date()),
            str(df_30["datetime"].max().date()),
        )
    trend_tbl = daily_trend_table(daily_df, ef, es)
    ltf = generate(df_30, "ema_cross", ema_fast=ef, ema_slow=es)
    dt = align_daily_trend(df_30, trend_tbl)
    return apply_daily_filter(ltf, dt)


def daily_donchian_table(
    df_daily: pd.DataFrame,
    donchian: int = 20,
) -> pd.DataFrame:
    """日线唐奇安方向：突破上轨多 / 突破下轨空，否则沿用前一日。"""
    h, l, c = df_daily["high"], df_daily["low"], df_daily["close"]
    upper = h.rolling(donchian).max().shift(1)
    lower = l.rolling(donchian).min().shift(1)
    raw = np.where(c > upper, 1, np.where(c < lower, -1, 0))
    direction = pd.Series(raw, dtype=float).replace(0, np.nan).ffill().fillna(0).astype(int)
    return pd.DataFrame({
        "date": df_daily["datetime"].dt.date,
        "daily_trend": direction.to_numpy(),
    })


def _donchian_channels(df: pd.DataFrame, donchian: int) -> tuple[pd.Series, pd.Series]:
    upper = df["high"].rolling(donchian).max().shift(1)
    lower = df["low"].rolling(donchian).min().shift(1)
    return upper, lower


def generate_ema_donchian_breakout(
    df: pd.DataFrame,
    ema_fast: int | None = None,
    ema_slow: int | None = None,
    donchian: int = 20,
) -> pd.Series:
    """EMA 定方向 + 唐奇安突破触发开仓，EMA 反向则平仓。"""
    ef = ema_fast or config.EMA_FAST
    es = ema_slow or config.EMA_SLOW
    c = df["close"]
    e1, e2 = _ema(c, ef), _ema(c, es)
    upper, lower = _donchian_channels(df, donchian)
    n = len(df)
    pos = np.zeros(n, dtype=float)
    cur = 0.0
    warmup = max(ef, es, donchian) + 2
    for i in range(warmup, n):
        bull = e1.iloc[i] > e2.iloc[i]
        bear = e1.iloc[i] < e2.iloc[i]
        long_brk = c.iloc[i] > upper.iloc[i]
        short_brk = c.iloc[i] < lower.iloc[i]
        if cur == 0:
            if bull and long_brk:
                cur = 1.0
            elif bear and short_brk:
                cur = -1.0
        elif cur > 0:
            if bear or short_brk:
                cur = 0.0
        elif cur < 0:
            if bull or long_brk:
                cur = 0.0
        pos[i] = cur
    return pd.Series(pos, index=df.index, dtype=float)


def generate_ema_donchian_both(
    df: pd.DataFrame,
    ema_fast: int | None = None,
    ema_slow: int | None = None,
    donchian: int = 20,
) -> pd.Series:
    """EMA 与唐奇安同向：趋势向上且价格在通道上方才多，反之才空。"""
    ef = ema_fast or config.EMA_FAST
    es = ema_slow or config.EMA_SLOW
    c = df["close"]
    e1, e2 = _ema(c, ef), _ema(c, es)
    upper, lower = _donchian_channels(df, donchian)
    long_c = (e1 > e2) & (c > upper)
    short_c = (e1 < e2) & (c < lower)
    raw = np.where(long_c, 1.0, np.where(short_c, -1.0, np.nan))
    return pd.Series(raw, index=df.index, dtype=float).ffill().fillna(0.0)


def generate_daily_filter_donchian(
    df_30: pd.DataFrame,
    ema_fast: int | None = None,
    ema_slow: int | None = None,
    donchian: int = 20,
    daily_df: pd.DataFrame | None = None,
) -> pd.Series:
    """日线 EMA 定方向 + 30m 唐奇安突破执行。"""
    ef = ema_fast or config.EMA_FAST
    es = ema_slow or config.EMA_SLOW
    if daily_df is None:
        daily_df = load_daily(
            str(df_30["datetime"].min().date()),
            str(df_30["datetime"].max().date()),
        )
    trend_tbl = daily_trend_table(daily_df, ef, es)
    ltf = generate_ema_donchian_breakout(df_30, ef, es, donchian)
    dt = align_daily_trend(df_30, trend_tbl)
    return apply_daily_filter(ltf, dt)


def generate_daily_donchian_filter_ema(
    df_30: pd.DataFrame,
    ema_fast: int | None = None,
    ema_slow: int | None = None,
    donchian: int = 20,
    daily_df: pd.DataFrame | None = None,
) -> pd.Series:
    """日线唐奇安定方向 + 30m EMA 执行。"""
    ef = ema_fast or config.EMA_FAST
    es = ema_slow or config.EMA_SLOW
    if daily_df is None:
        daily_df = load_daily(
            str(df_30["datetime"].min().date()),
            str(df_30["datetime"].max().date()),
        )
    trend_tbl = daily_donchian_table(daily_df, donchian)
    from signals import generate
    ltf = generate(df_30, "ema_cross", ema_fast=ef, ema_slow=es)
    dt = align_daily_trend(df_30, trend_tbl)
    return apply_daily_filter(ltf, dt)


def daily_adx_regime_table(
    df_daily: pd.DataFrame,
    adx_threshold: float = 25.0,
    ema_fast: int | None = None,
    ema_slow: int | None = None,
) -> pd.DataFrame:
    """日线趋势市判定：ADX>阈值 且 EMA 有方向。"""
    from signals import _adx

    ef = ema_fast or config.EMA_FAST
    es = ema_slow or config.EMA_SLOW
    c = df_daily["close"]
    e1, e2 = _ema(c, ef), _ema(c, es)
    adx = _adx(df_daily["high"], df_daily["low"], c, 14)
    direction = np.where(e1 > e2, 1, np.where(e1 < e2, -1, 0))
    is_trend = (adx > adx_threshold) & (direction != 0)
    sig = np.where(is_trend, direction, 0)
    return pd.DataFrame({
        "date": df_daily["datetime"].dt.date,
        "daily_trend": sig.astype(int),
    })


def generate_daily_adx_filter_ema(
    df_30: pd.DataFrame,
    adx_threshold: float = 25.0,
    ema_fast: int | None = None,
    ema_slow: int | None = None,
    daily_df: pd.DataFrame | None = None,
) -> pd.Series:
    """仅日线趋势市（ADX过滤）交易；30m EMA 执行。震荡市空仓。"""
    ef = ema_fast or config.EMA_FAST
    es = ema_slow or config.EMA_SLOW
    if daily_df is None:
        daily_df = load_daily(
            str(df_30["datetime"].min().date()),
            str(df_30["datetime"].max().date()),
        )
    trend_tbl = daily_adx_regime_table(daily_df, adx_threshold, ef, es)
    from signals import generate
    ltf = generate(df_30, "ema_cross", ema_fast=ef, ema_slow=es)
    dt = align_daily_trend(df_30, trend_tbl)
    return apply_daily_filter(ltf, dt)


def generate_daily_filter_ema_stable(
    df_30: pd.DataFrame,
    ema_fast: int | None = None,
    ema_slow: int | None = None,
    trend_ma: int = 50,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    ma_divergence_threshold: float = 0.015,
    daily_df: pd.DataFrame | None = None,
) -> pd.Series:
    """日线 EMA 定方向 + 30m EMA 稳定版（ADX/发散度过滤）。"""
    ef = ema_fast or config.EMA_FAST
    es = ema_slow or config.EMA_SLOW
    if daily_df is None:
        daily_df = load_daily(
            str(df_30["datetime"].min().date()),
            str(df_30["datetime"].max().date()),
        )
    trend_tbl = daily_trend_table(daily_df, ef, es)
    from signals import generate_ema_stable
    ltf = generate_ema_stable(
        df_30, ef, es, trend_ma, adx_period, adx_threshold, ma_divergence_threshold,
    )
    dt = align_daily_trend(df_30, trend_tbl)
    return apply_daily_filter(ltf, dt)
