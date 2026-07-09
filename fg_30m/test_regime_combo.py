"""
日线趋势 + 30 分钟震荡 —— 组合策略回测

思路：
  · 日线 ADX/EMA 判断「趋势市 / 震荡市」
  · 趋势市：跟日线 EMA 方向（或 30m 顺势）
  · 震荡市：30m RSI 超买超卖 / 布林均值回归

    python test_regime_combo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
import data
import signals
from backtest import run as run_bt


def load_daily(start: str, end: str) -> pd.DataFrame:
    pdf = pd.read_parquet(config.ROOT / "data" / "FG00.CZCE.parquet")
    pdf["datetime"] = pd.to_datetime(pdf["datetime"])
    pdf = pdf[(pdf["datetime"] >= start) & (pdf["datetime"] <= end)].sort_values("datetime")
    pdf["vt_symbol"] = "FG00.CZCE"
    return pdf.reset_index(drop=True)


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * pd.Series(pdm, index=c.index).rolling(n).mean() / (atr + 1e-9)
    mdi = 100 * pd.Series(mdm, index=c.index).rolling(n).mean() / (atr + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    return dx.rolling(n).mean()


def daily_regime(df_d: pd.DataFrame, adx_th: float = 25.0) -> pd.DataFrame:
    """每日一行：trend_dir, is_trend, adx"""
    c = df_d["close"]
    e1, e2 = _ema(c, 26), _ema(c, 46)
    adx = _adx(df_d)
    trend_dir = np.where(e1 > e2, 1, np.where(e1 < e2, -1, 0))
    is_trend = adx > adx_th
    return pd.DataFrame({
        "date": df_d["datetime"].dt.date,
        "trend_dir": trend_dir,
        "is_trend": is_trend,
        "adx": adx,
    })


def align_daily_to_30m(df_30: pd.DataFrame, daily_info: pd.DataFrame) -> pd.DataFrame:
    d30 = df_30.copy()
    d30["date"] = d30["datetime"].dt.date
    merged = d30.merge(daily_info, on="date", how="left")
    merged["trend_dir"] = merged["trend_dir"].ffill().fillna(0)
    merged["is_trend"] = merged["is_trend"].ffill().fillna(False)
    return merged


def sig_daily_trend(df_30: pd.DataFrame, daily_info: pd.DataFrame) -> pd.Series:
    """趋势段：只跟日线 EMA 方向"""
    m = align_daily_to_30m(df_30, daily_info)
    sig = np.where(m["is_trend"], m["trend_dir"], 0.0)
    return pd.Series(sig, index=df_30.index, dtype=float)


def sig_30m_range(df_30: pd.DataFrame, daily_info: pd.DataFrame) -> pd.Series:
    """震荡段：30m RSI 超买超卖；趋势段空仓"""
    m = align_daily_to_30m(df_30, daily_info)
    rsi = signals._rsi(m["close"], 7)
    sig = np.zeros(len(m))
    for i in range(len(m)):
        if m["is_trend"].iloc[i]:
            sig[i] = 0.0
        elif rsi.iloc[i] < 28:
            sig[i] = 1.0
        elif rsi.iloc[i] > 72:
            sig[i] = -1.0
        else:
            sig[i] = sig[i - 1] if i > 0 else 0.0
    return pd.Series(sig, index=df_30.index, dtype=float)


def sig_regime_combo(df_30: pd.DataFrame, daily_info: pd.DataFrame) -> pd.Series:
    """核心：趋势市跟日线，震荡市做 30m RSI"""
    m = align_daily_to_30m(df_30, daily_info)
    rsi = signals._rsi(m["close"], 7)
    e1 = _ema(m["close"], 26)
    e2 = _ema(m["close"], 46)
    sig = np.zeros(len(m))
    for i in range(len(m)):
        if m["is_trend"].iloc[i]:
            sig[i] = m["trend_dir"].iloc[i]
        elif rsi.iloc[i] < 28:
            sig[i] = 1.0
        elif rsi.iloc[i] > 72:
            sig[i] = -1.0
        else:
            sig[i] = sig[i - 1] if i > 0 else 0.0
    return pd.Series(sig, index=df_30.index, dtype=float)


def sig_trend_filter_30m(df_30: pd.DataFrame, daily_info: pd.DataFrame) -> pd.Series:
    """30m EMA 仅在与日线同向时交易"""
    m = align_daily_to_30m(df_30, daily_info)
    ema30 = signals.generate(df_30, "ema_cross", ema_fast=26, ema_slow=46)
    out = ema30.copy()
    for i in range(len(m)):
        d = int(m["trend_dir"].iloc[i])
        if d > 0 and ema30.iloc[i] < 0:
            out.iloc[i] = 0
        elif d < 0 and ema30.iloc[i] > 0:
            out.iloc[i] = 0
        elif d == 0:
            out.iloc[i] = 0
    return out


def run_daily_only(df_d: pd.DataFrame) -> dict:
    sig = signals.generate(df_d, "ema_cross", ema_fast=26, ema_slow=46)
    return run_bt(df_d, sig, max_lots=1)[0]


def main() -> dict:
    config.ensure_dirs()
    start, end = config.BACKTEST_PERIOD
    mid = config.TUNE_PERIOD[1]

    df_30 = data.load_30m("FG609", use_cache=True)
    df_30 = df_30[(df_30["datetime"] >= start) & (df_30["datetime"] <= end)].reset_index(drop=True)
    df_d = load_daily(start, end)
    dinfo = daily_regime(df_d)

    strategies = [
        ("30m 纯EMA26/46", signals.generate(df_30, "ema_cross", ema_fast=26, ema_slow=46)),
        ("日线 纯EMA26/46", None),
        ("组合: 趋势跟日线+震荡RSI", sig_regime_combo(df_30, dinfo)),
        ("仅趋势段跟日线", sig_daily_trend(df_30, dinfo)),
        ("仅震荡段30m RSI", sig_30m_range(df_30, dinfo)),
        ("30m EMA + 日线方向过滤", sig_trend_filter_30m(df_30, dinfo)),
    ]

    rows = []
    for name, sig in strategies:
        if sig is None:
            st_f = run_daily_only(df_d)
            df_tune = df_d[df_d["datetime"] <= mid]
            df_test = df_d[df_d["datetime"] > mid]
            st_t = run_daily_only(df_tune)
            st_e = run_daily_only(df_test)
        else:
            st_f = run_bt(df_30, sig, max_lots=1)[0]
            mask_t = df_30["datetime"] <= mid
            st_t = run_bt(df_30[mask_t], sig[mask_t].reset_index(drop=True), max_lots=1)[0]
            st_e = run_bt(df_30[~mask_t], sig[~mask_t].reset_index(drop=True), max_lots=1)[0]
        rows.append({
            "strategy": name,
            "full": st_f["total_return_pct"],
            "full_dd": st_f["max_ddpercent"],
            "full_trades": st_f["total_trades"],
            "tune": st_t["total_return_pct"],
            "test": st_e["total_return_pct"],
            "test_dd": st_e["max_ddpercent"],
        })

    rows.sort(key=lambda x: -(x["full"] + 0.5 * x["test"]))
    out = {"period": [start, end], "results": rows}
    path = config.ARTIFACT_PATH / "regime_combo_test.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    trend_days = int(dinfo["is_trend"].sum())
    print(f"[regime] 日线趋势日 {trend_days}/{len(dinfo)}  震荡日 {len(dinfo)-trend_days}")
    print(f"{'策略':<32} {'全段':>7} {'前半':>7} {'后半':>7} {'回撤':>7} trades")
    for r in rows:
        print(
            f"{r['strategy']:<32} {r['full']:6.1f}% {r['tune']:6.1f}% {r['test']:6.1f}% "
            f"{r['full_dd']:6.1f}% {r['full_trades']:>5}"
        )
    print(f"\n[regime] 最优: {rows[0]['strategy']}")
    print(f"  -> {path}")
    return out


if __name__ == "__main__":
    main()
