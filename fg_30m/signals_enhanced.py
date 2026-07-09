"""
增强信号 —— BOLL / BBI / EMA + 稳定版过滤器（ADX、发散度、止损、日亏限）

供 compare_strategies.py 与回测使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def boll_bands(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma(close, n)
    sd = close.rolling(n).std()
    return mid - k * sd, mid, mid + k * sd


def bbi_line(close: pd.Series) -> pd.Series:
    return (_sma(close, 3) + _sma(close, 6) + _sma(close, 12) + _sma(close, 24)) / 4.0


def adx_series(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * pd.Series(pdm).rolling(n).mean() / (atr + 1e-9)
    mdi = 100 * pd.Series(mdm).rolling(n).mean() / (atr + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    return dx.rolling(n).mean()


def raw_signal(
    df: pd.DataFrame,
    mode: str,
    ema_fast: int = 26,
    ema_slow: int = 46,
    boll_n: int = 20,
    trend_ma: int = 50,
) -> pd.Series:
    c = df["close"]
    n = len(df)
    ef, es = _ema(c, ema_fast), _ema(c, ema_slow)
    _, boll_mid, _ = boll_bands(c, boll_n)
    bbi = bbi_line(c)
    ma_t = _sma(c, trend_ma)

    if mode == "ema":
        sig = np.where(ef > es, 1.0, np.where(ef < es, -1.0, 0.0))
    elif mode == "boll":
        sig = np.where(c > boll_mid, 1.0, np.where(c < boll_mid, -1.0, 0.0))
    elif mode == "bbi":
        sig = np.where(c > bbi, 1.0, np.where(c < bbi, -1.0, 0.0))
    elif mode == "triple":
        long_c = (ef > es) & (c > bbi) & (c > boll_mid)
        short_c = (ef < es) & (c < bbi) & (c < boll_mid)
        sig = np.where(long_c, 1.0, np.where(short_c, -1.0, 0.0))
    elif mode == "stable_ma":
        mf, ms = _sma(c, 20), _sma(c, 50)
        long_c = (mf > ms) & (c > ma_t)
        short_c = (mf < ms) & (c < ma_t)
        sig = np.where(long_c, 1.0, np.where(short_c, -1.0, 0.0))
    else:
        raise ValueError(mode)
    return pd.Series(sig, index=df.index, dtype=float)


def apply_filters(
    df: pd.DataFrame,
    sig: pd.Series,
    *,
    use_adx: bool = False,
    adx_threshold: float = 25.0,
    use_divergence: bool = False,
    div_threshold: float = 0.015,
    ema_fast: int = 26,
    trend_ma: int = 50,
) -> pd.Series:
    """稳定版：ADX + 均线发散过滤（仅开仓时生效，持仓跟随原信号平仓）。"""
    if not use_adx and not use_divergence:
        return sig

    c = df["close"]
    adx = adx_series(df)
    ma_f = _ema(c, ema_fast)
    ma_t = _sma(c, trend_ma)
    div = (ma_f - ma_t).abs() / (ma_t + 1e-9)

    out = sig.copy()
    for i in range(len(df)):
        if sig.iloc[i] == 0:
            continue
        ok = True
        if use_adx and (pd.isna(adx.iloc[i]) or adx.iloc[i] <= adx_threshold):
            ok = False
        if use_divergence and div.iloc[i] <= div_threshold:
            ok = False
        if not ok:
            out.iloc[i] = 0.0
    return out


def apply_risk_overlay(
    df: pd.DataFrame,
    sig: pd.Series,
    *,
    stop_points: float = 25.0,
    max_daily_loss: float = 300.0,
    contract_size: float = 20.0,
) -> pd.Series:
    """止损 + 日亏限制（bar 级模拟）。"""
    c = df["close"].to_numpy()
    target = sig.to_numpy()
    out = np.zeros(len(df))
    pos = 0
    entry = 0.0
    daily_loss = 0.0
    cur_day = None

    for i in range(len(df)):
        day = df["datetime"].iloc[i].date()
        if cur_day != day:
            cur_day = day
            daily_loss = 0.0
        price = c[i]
        tgt = int(np.sign(target[i])) if target[i] != 0 else 0

        if pos > 0 and entry > 0 and price <= entry - stop_points:
            daily_loss += (entry - price) * contract_size
            pos = 0
            entry = 0.0
        elif pos < 0 and entry > 0 and price >= entry + stop_points:
            daily_loss += (price - entry) * contract_size
            pos = 0
            entry = 0.0

        if abs(daily_loss) >= max_daily_loss and pos == 0:
            tgt = 0

        if tgt != pos:
            if tgt != 0:
                entry = price
            else:
                entry = 0.0
            pos = tgt
        out[i] = pos
    return pd.Series(out, index=df.index, dtype=float)


def build_signal(
    df: pd.DataFrame,
    mode: str,
    filters: str = "none",
    risk: bool = False,
    **kwargs,
) -> pd.Series:
    sig = raw_signal(df, mode, **kwargs)
    if filters == "adx":
        sig = apply_filters(df, sig, use_adx=True, **kwargs)
    elif filters == "adx_div":
        sig = apply_filters(df, sig, use_adx=True, use_divergence=True, **kwargs)
    elif filters == "none":
        pass
    else:
        raise ValueError(filters)
    if risk:
        sig = apply_risk_overlay(df, sig)
    return sig
