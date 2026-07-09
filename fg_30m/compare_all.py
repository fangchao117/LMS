"""
全策略综合对比 —— 主力拼接 + 逐段稳定性 + TEST

    python compare_all.py
    python compare_all.py --quick
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

import config
import data
import dominant
import signals
import signals_enhanced as se
from backtest import run


def _ffill_event(sig: pd.Series) -> pd.Series:
    return sig.replace(0, np.nan).ffill().fillna(0.0)


def _build_specs(quick: bool) -> list[dict]:
    specs: list[dict] = []

    def add(name: str, fn: str, **kw: Any) -> None:
        specs.append({"name": name, "fn": fn, **kw})

    # --- 纯 30m ---
    for ef, es in [(26, 46), (12, 48), (5, 20)]:
        add(f"EMA{ef}/{es}", "signal", mode="ema_cross", ema_fast=ef, ema_slow=es)
    add("MA8/21/55", "signal", mode="ma_trend", ma_s=8, ma_m=21, ma_l=55)
    for dc in ([20] if quick else [12, 20, 40]):
        add(f"Donchian{dc}持仓", "signal", mode="breakout", donchian=dc, ffill=True)
    for rp in ([14] if quick else [10, 14]):
        add(f"RSI动量{rp}", "signal", mode="rsi_momo", rsi_period=rp, ffill=True)
    add("RSI反转7", "signal", mode="rsi_revert", rsi_period=7, rsi_ob=70, rsi_os=30, ffill=True)
    add("MACD", "signal", mode="macd")
    add("动量0.2%", "signal", mode="momentum", mom_th=0.002)
    if not quick:
        add("多因子投票", "signal", mode="combo", combo_min=2)
    for adx in ([35] if quick else [25, 35]):
        add(f"EMA稳定ADX{adx}", "signal", mode="ema_stable", ema_fast=26, ema_slow=46, adx_threshold=float(adx))

    # --- 增强 BOLL/BBI ---
    for label, mode, fl in [
        ("BOLL中轨", "boll", "none"),
        ("BBI", "bbi", "none"),
        ("三指标同向", "triple", "none"),
        ("三指标+ADX", "triple", "adx"),
        ("稳定MA20/50+ADX", "stable_ma", "adx_div"),
    ]:
        if quick and label not in ("三指标同向", "BOLL中轨"):
            continue
        add(label, "enhanced", mode=mode, filters=fl, risk=False)

    # --- 日线过滤 regime ---
    add("日线EMA+30mEMA26/46", "signal", mode="daily_filter_ema", ema_fast=26, ema_slow=46)
    if not quick:
        add("日线EMA+30mEMA12/48", "signal", mode="daily_filter_ema", ema_fast=12, ema_slow=48)
        add("日线EMA+30mDon20", "signal", mode="daily_filter_donchian", donchian=20)
        add("日线EMA+30mDon40", "signal", mode="daily_filter_donchian", donchian=40)
        add("日线Don+30mEMA", "signal", mode="daily_donchian_filter_ema", donchian=20)
        add("日线ADX25+30mEMA", "signal", mode="daily_adx_filter_ema", adx_threshold=25.0)
        add("日线ADX35+30mEMA", "signal", mode="daily_adx_filter_ema", adx_threshold=35.0)
        add("30mEMA+Don突破", "signal", mode="ema_donchian_breakout", donchian=20)
        add("30mEMA+Don同向", "signal", mode="ema_donchian_both", donchian=20)
    add("日线EMA+30m稳定ADX35", "signal", mode="daily_filter_ema_stable", adx_threshold=35.0)

    # --- 60m+30m ---
    if not quick:
        add("60mEMA+30mEMA", "mtf", ltf_ema_fast=26, ltf_ema_slow=46, htf_ema_fast=6, htf_ema_slow=24)

    return specs


def _make_signal(df: pd.DataFrame, spec: dict, df_60m: pd.DataFrame | None = None) -> pd.Series:
    fn = spec["fn"]
    kw = {k: v for k, v in spec.items() if k not in ("name", "fn", "ffill")}

    if fn == "signal":
        sig = signals.generate(df, kw.pop("mode"), **kw)
        if spec.get("ffill"):
            sig = _ffill_event(sig)
        return sig
    if fn == "enhanced":
        return se.build_signal(df, kw.pop("mode"), kw.pop("filters"), kw.pop("risk"), **kw)
    if fn == "mtf":
        if df_60m is None or df_60m.empty:
            return pd.Series(0.0, index=df.index)
        return signals.generate_mtf(
            df, df_60m,
            ltf_mode="ema_cross",
            htf_mode="ema_cross",
            ltf_ema_fast=kw.get("ltf_ema_fast", 26),
            ltf_ema_slow=kw.get("ltf_ema_slow", 46),
            htf_ema_fast=kw.get("htf_ema_fast", 6),
            htf_ema_slow=kw.get("htf_ema_slow", 24),
        )
    raise ValueError(fn)


def _seg_returns(spec: dict, df_60m: pd.DataFrame | None) -> list[dict]:
    rows = []
    for sym, start, end in dominant.DOMINANT_SEGMENTS:
        df = data.load_30m(sym, start=start, end=end, use_cache=True)
        if len(df) < 50:
            continue
        htf = None
        if df_60m is not None and not df_60m.empty:
            htf = df_60m[(df_60m["datetime"] >= start) & (df_60m["datetime"] <= end)]
        try:
            sig = _make_signal(df, spec, htf)
            st, _ = run(df, sig, max_lots=1)
        except Exception as e:
            rows.append({"symbol": sym, "return_pct": None, "error": str(e)})
            continue
        rows.append({
            "symbol": sym,
            "return_pct": st["total_return_pct"],
            "max_dd_pct": st["max_ddpercent"],
            "trades": st["total_trades"],
        })
    return rows


def _score(full_ret: float, test_ret: float, seg_rets: list[float], full_dd: float) -> dict:
    wins = sum(1 for r in seg_rets if r > 0)
    n = len(seg_rets)
    min_seg = min(seg_rets) if seg_rets else -999.0
    avg_seg = sum(seg_rets) / n if seg_rets else -999.0
    bal = full_ret * 0.4 + test_ret * 0.2 + min_seg * 0.25 + wins * 8 + full_dd * 0.15
    stab = min_seg * 3 + wins * 15 + test_ret * 0.5 + full_dd * 0.2
    return {
        "seg_wins": wins,
        "seg_n": n,
        "seg_min": min_seg,
        "seg_avg": avg_seg,
        "score_return": full_ret,
        "score_balanced": bal,
        "score_stable": stab,
    }


def main(quick: bool = False) -> dict:
    config.ensure_dirs()
    df_all = dominant.load_dominant_30m()
    t0, t1 = config.TEST_PERIOD
    df_test = df_all[(df_all["datetime"] >= t0) & (df_all["datetime"] <= t1)]
    df_60m = None
    if not quick:
        try:
            df_60m = data.load_60m("FG609", use_cache=True)
        except Exception:
            df_60m = None

    specs = _build_specs(quick)
    results: list[dict] = []

    print(f"全策略对比  主力 {dominant.dominant_period()[0]}~{dominant.dominant_period()[1]}  "
          f"共 {len(specs)} 种  1万1手")
    print("=" * 96)

    for i, spec in enumerate(specs):
        try:
            sig_full = _make_signal(df_all, spec, df_60m)
            st_full, _ = run(df_all, sig_full, max_lots=1)
            sig_test = _make_signal(df_test, spec, df_60m)
            st_test, _ = run(df_test, sig_test, max_lots=1)
        except Exception as e:
            results.append({"name": spec["name"], "error": str(e)})
            continue

        segs = _seg_returns(spec, df_60m)
        seg_rets = [s["return_pct"] for s in segs if s.get("return_pct") is not None]
        sc = _score(st_full["total_return_pct"], st_test["total_return_pct"], seg_rets, st_full["max_ddpercent"])

        rec = {
            "name": spec["name"],
            "spec": {k: v for k, v in spec.items() if k != "fn"},
            "full_ret": st_full["total_return_pct"],
            "full_dd": st_full["max_ddpercent"],
            "full_trades": st_full["total_trades"],
            "test_ret": st_test["total_return_pct"],
            "segments": segs,
            **sc,
        }
        results.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  … {i + 1}/{len(specs)}")

    valid = [r for r in results if "error" not in r]
    by_ret = sorted(valid, key=lambda x: x["full_ret"], reverse=True)
    by_stab = sorted(valid, key=lambda x: x["score_stable"], reverse=True)
    by_bal = sorted(valid, key=lambda x: x["score_balanced"], reverse=True)

    def _print_block(title: str, rows: list[dict], n: int = 10) -> None:
        print(f"\n### {title}")
        print(f"{'策略':<28} {'FULL':>7} {'TEST':>7} {'段赢':>5} {'段min':>7} {'dd':>7} {'笔':>4}")
        print("-" * 72)
        for r in rows[:n]:
            print(
                f"{r['name']:<28} {r['full_ret']:7.1f}% {r['test_ret']:7.1f}% "
                f"{r['seg_wins']}/{r['seg_n']:1} {r['seg_min']:7.1f}% {r['full_dd']:6.1f}% {r['full_trades']:4}"
            )

    _print_block("收益 TOP10", by_ret)
    _print_block("稳定性 TOP10", by_stab)
    _print_block("综合 TOP10", by_bal)

    out = {
        "period": list(dominant.dominant_period()),
        "n_strategies": len(valid),
        "top_return": by_ret[:10],
        "top_stable": by_stab[:10],
        "top_balanced": by_bal[:10],
        "all": results,
    }
    path = config.ARTIFACT_PATH / "compare_all.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)
