"""
30 分钟信号 —— 多模式，严格因果（T 根收盘出信号 → T+1 成交）

模式：
  ema_cross   EMA 金叉/死叉
  ma_trend    三均线趋势
  breakout    Donchian 突破
  rsi_momo    RSI 动量 + EMA 过滤
  rsi_revert  RSI 超买超卖反转
  macd        MACD 柱穿越零轴
  momentum    多周期 ROC 合成
  combo       多因子投票（≥2 同向才开仓）
  daily_filter_ema  日线 EMA 定方向 + 30m EMA 执行（推荐）
  ema_stable  EMA 金叉 + ADX/发散度/趋势过滤（稳定性增强）
  triple      BOLL + BBI + EMA 三指标同向
  mtf_ema     60 分钟 EMA 定方向 + 30 分钟 EMA 交易（需 df_60m）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ef, es = _ema(close, fast), _ema(close, slow)
    macd = ef - es
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def align_htf_to_ltf(
    df_ltf: pd.DataFrame,
    df_htf: pd.DataFrame,
    htf_sig: pd.Series,
) -> pd.Series:
    """将高周期信号对齐到低周期，高周期信号 shift(1) 防未来。"""
    htf = pd.DataFrame({
        "datetime": pd.to_datetime(df_htf["datetime"]),
        "htf_sig": htf_sig.shift(1).to_numpy(),
    }).dropna(subset=["datetime"]).sort_values("datetime")
    ltf = pd.DataFrame({"datetime": pd.to_datetime(df_ltf["datetime"])}).sort_values("datetime")
    merged = pd.merge_asof(ltf, htf, on="datetime", direction="backward")
    return pd.Series(merged["htf_sig"].fillna(0.0).to_numpy(), index=df_ltf.index, dtype=float)


def apply_htf_filter(ltf_sig: pd.Series, htf_sig: pd.Series) -> pd.Series:
    """仅保留与 60 分钟方向一致的 30 分钟信号。"""
    out = np.zeros(len(ltf_sig), dtype=float)
    l = ltf_sig.to_numpy()
    h = htf_sig.to_numpy()
    out = np.where((h > 0) & (l > 0), 1.0, np.where((h < 0) & (l < 0), -1.0, 0.0))
    return pd.Series(out, index=ltf_sig.index, dtype=float)


def generate_mtf(
    df_30m: pd.DataFrame,
    df_60m: pd.DataFrame,
    ltf_mode: str = "ema_cross",
    htf_mode: str = "ema_cross",
    **kwargs,
) -> pd.Series:
    ltf_kw = {k[4:]: v for k, v in kwargs.items() if k.startswith("ltf_")}
    htf_kw = {k[4:]: v for k, v in kwargs.items() if k.startswith("htf_")}
    ltf_sig = generate(df_30m, ltf_mode, **ltf_kw)
    htf_raw = generate(df_60m, htf_mode, **htf_kw)
    htf_aligned = align_htf_to_ltf(df_30m, df_60m, htf_raw)
    return apply_htf_filter(ltf_sig, htf_aligned)


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


def generate_ema_stable(
    df: pd.DataFrame,
    ema_fast: int = 26,
    ema_slow: int = 46,
    trend_ma: int = 50,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    ma_divergence_threshold: float = 0.015,
) -> pd.Series:
    """
    EMA 双均线 + 市场状态过滤（整合稳定版优点）：
      · 趋势 MA：价格在趋势线上方才做多 / 下方才做空
      · ADX > 阈值才允许新开仓
      · 快慢线与趋势线发散度 > 阈值才允许新开仓
      · 平仓：EMA 方向反转即平（不过滤）
    """
    c, h, l = df["close"], df["high"], df["low"]
    e1, e2 = _ema(c, ema_fast), _ema(c, ema_slow)
    e_trend = _ema(c, trend_ma)
    adx = _adx(h, l, c, adx_period)
    divergence = (e1 - e_trend).abs() / (e_trend + 1e-9)

    n = len(df)
    pos = np.zeros(n, dtype=float)
    cur = 0.0
    warmup = max(ema_fast, ema_slow, trend_ma, adx_period * 2)

    for i in range(n):
        if i < warmup:
            pos[i] = 0.0
            continue
        price = c.iloc[i]
        bull = e1.iloc[i] > e2.iloc[i]
        bear = e1.iloc[i] < e2.iloc[i]
        long_ok = price > e_trend.iloc[i]
        short_ok = price < e_trend.iloc[i]
        market_ok = (
            adx.iloc[i] > adx_threshold
            and divergence.iloc[i] > ma_divergence_threshold
        )

        if cur == 0:
            if market_ok:
                if bull and long_ok:
                    cur = 1.0
                elif bear and short_ok:
                    cur = -1.0
        elif cur > 0:
            if bear:
                cur = 0.0
        elif cur < 0:
            if bull:
                cur = 0.0
        pos[i] = cur

    return pd.Series(pos, index=df.index, dtype=float)


def generate(
    df: pd.DataFrame,
    mode: str,
    **kwargs,
) -> pd.Series:
    if mode == "mtf_ema":
        df_60m = kwargs.pop("df_60m")
        return generate_mtf(df, df_60m, **kwargs)
    c = df["close"]
    h, l, v = df["high"], df["low"], df["volume"]
    n = len(df)
    sig = np.zeros(n, dtype=float)

    if mode == "ema_cross":
        ef = kwargs.get("ema_fast", 5)
        es = kwargs.get("ema_slow", 20)
        e1, e2 = _ema(c, ef), _ema(c, es)
        sig = np.where(e1 > e2, 1.0, np.where(e1 < e2, -1.0, 0.0))

    elif mode == "ma_trend":
        s, m, lg = kwargs.get("ma_s", 8), kwargs.get("ma_m", 21), kwargs.get("ma_l", 55)
        ma_s, ma_m, ma_l = c.rolling(s).mean(), c.rolling(m).mean(), c.rolling(lg).mean()
        long_c = (ma_s > ma_m) & (ma_m > ma_l) & (c > ma_m)
        short_c = (ma_s < ma_m) & (ma_m < ma_l) & (c < ma_m)
        sig = np.where(long_c, 1.0, np.where(short_c, -1.0, 0.0))

    elif mode == "breakout":
        dc = kwargs.get("donchian", 20)
        upper = h.rolling(dc).max().shift(1)
        lower = l.rolling(dc).min().shift(1)
        ef, es = _ema(c, 8), _ema(c, 21)
        long_c = (c > upper) & (ef > es)
        short_c = (c < lower) & (ef < es)
        sig = np.where(long_c, 1.0, np.where(short_c, -1.0, 0.0))

    elif mode == "rsi_momo":
        rp = kwargs.get("rsi_period", 14)
        rsi = _rsi(c, rp)
        ef, es = _ema(c, 8), _ema(c, 21)
        long_c = (rsi > 50) & (rsi.shift(1) <= 50) & (ef > es)
        short_c = (rsi < 50) & (rsi.shift(1) >= 50) & (ef < es)
        sig = np.where(long_c, 1.0, np.where(short_c, -1.0, 0.0))

    elif mode == "rsi_revert":
        rp = kwargs.get("rsi_period", 7)
        ob, os_ = kwargs.get("rsi_ob", 72), kwargs.get("rsi_os", 28)
        rsi = _rsi(c, rp)
        long_c = (rsi < os_) & (rsi.shift(1) >= os_)
        short_c = (rsi > ob) & (rsi.shift(1) <= ob)
        sig = np.where(long_c, 1.0, np.where(short_c, -1.0, 0.0))
        sig = pd.Series(sig).ffill().fillna(0.0).to_numpy()

    elif mode == "macd":
        _, _, hist = _macd(c, kwargs.get("macd_fast", 12), kwargs.get("macd_slow", 26), kwargs.get("macd_sig", 9))
        sig = np.where(hist > 0, 1.0, np.where(hist < 0, -1.0, 0.0))

    elif mode == "momentum":
        windows = kwargs.get("mom_windows", (5, 10, 20))
        rocs = pd.concat([c / c.shift(w) - 1 for w in windows], axis=1)
        mom = rocs.mean(axis=1)
        th = kwargs.get("mom_th", 0.002)
        sig = np.where(mom > th, 1.0, np.where(mom < -th, -1.0, 0.0))

    elif mode == "combo":
        sub = [
            generate(df, "ema_cross", ema_fast=5, ema_slow=20),
            generate(df, "macd"),
            generate(df, "momentum", mom_th=0.001),
        ]
        vote = sum(sub)
        min_v = kwargs.get("combo_min", 2)
        sig = np.where(vote >= min_v, 1.0, np.where(vote <= -min_v, -1.0, 0.0))

    elif mode == "daily_filter_ema":
        import regime
        return regime.generate_daily_filter_ema(
            df,
            ema_fast=kwargs.get("ema_fast", 26),
            ema_slow=kwargs.get("ema_slow", 46),
        )

    elif mode == "ema_donchian_breakout":
        import regime
        return regime.generate_ema_donchian_breakout(
            df,
            ema_fast=kwargs.get("ema_fast", 26),
            ema_slow=kwargs.get("ema_slow", 46),
            donchian=kwargs.get("donchian", 20),
        )

    elif mode == "ema_donchian_both":
        import regime
        return regime.generate_ema_donchian_both(
            df,
            ema_fast=kwargs.get("ema_fast", 26),
            ema_slow=kwargs.get("ema_slow", 46),
            donchian=kwargs.get("donchian", 20),
        )

    elif mode == "daily_filter_donchian":
        import regime
        return regime.generate_daily_filter_donchian(
            df,
            ema_fast=kwargs.get("ema_fast", 26),
            ema_slow=kwargs.get("ema_slow", 46),
            donchian=kwargs.get("donchian", 20),
        )

    elif mode == "daily_donchian_filter_ema":
        import regime
        return regime.generate_daily_donchian_filter_ema(
            df,
            ema_fast=kwargs.get("ema_fast", 26),
            ema_slow=kwargs.get("ema_slow", 46),
            donchian=kwargs.get("donchian", 20),
        )

    elif mode == "daily_adx_filter_ema":
        import regime
        return regime.generate_daily_adx_filter_ema(
            df,
            adx_threshold=kwargs.get("adx_threshold", 25.0),
            ema_fast=kwargs.get("ema_fast", 26),
            ema_slow=kwargs.get("ema_slow", 46),
        )

    elif mode == "daily_filter_ema_stable":
        import regime
        return regime.generate_daily_filter_ema_stable(
            df,
            ema_fast=kwargs.get("ema_fast", 26),
            ema_slow=kwargs.get("ema_slow", 46),
            trend_ma=kwargs.get("trend_ma", 50),
            adx_period=kwargs.get("adx_period", 14),
            adx_threshold=kwargs.get("adx_threshold", 25.0),
            ma_divergence_threshold=kwargs.get("ma_divergence_threshold", 0.015),
        )

    elif mode == "ema_stable":
        sig = generate_ema_stable(
            df,
            ema_fast=kwargs.get("ema_fast", 26),
            ema_slow=kwargs.get("ema_slow", 46),
            trend_ma=kwargs.get("trend_ma", 50),
            adx_period=kwargs.get("adx_period", 14),
            adx_threshold=kwargs.get("adx_threshold", 25.0),
            ma_divergence_threshold=kwargs.get("ma_divergence_threshold", 0.015),
        ).to_numpy()

    elif mode == "triple":
        import signals_enhanced as se
        fl = kwargs.get("filters", "none")
        return se.build_signal(
            df, "triple", fl, kwargs.get("risk", False),
            ema_fast=kwargs.get("ema_fast", 26),
            ema_slow=kwargs.get("ema_slow", 46),
        )

    else:
        raise ValueError(f"unknown mode: {mode}")

    return pd.Series(sig, index=df.index, dtype=float)
