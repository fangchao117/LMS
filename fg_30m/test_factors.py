"""
多因子 A/B 测试 —— 相对 EMA12/48 基线，仅当 TEST 收益更高才建议接入

    python test_factors.py
"""
from __future__ import annotations

import json
from itertools import product

import pandas as pd

import config
import data
import factors
import signals
from backtest import run


def _eval(df: pd.DataFrame, sig: pd.Series, max_lots: int = 1) -> dict:
    st, _ = run(df, sig, max_lots=max_lots)
    return st


def main() -> None:
    config.ensure_dirs()
    sym = config.SYMBOL
    df_all = data.load_30m(sym)
    t0, t1 = config.TUNE_PERIOD
    e0, e1 = config.TEST_PERIOD
    df_v = df_all[(df_all["datetime"] >= t0) & (df_all["datetime"] <= t1)].reset_index(drop=True)
    df_t = df_all[(df_all["datetime"] >= e0) & (df_all["datetime"] <= e1)].reset_index(drop=True)

    # 基线：当前最优 EMA
    base_kw = dict(ema_fast=config.EMA_FAST, ema_slow=config.EMA_SLOW)
    sig_v_base = signals.generate(df_v, "ema_cross", **base_kw)
    sig_t_base = signals.generate(df_t, "ema_cross", **base_kw)
    base_v = _eval(df_v, sig_v_base)
    base_t = _eval(df_t, sig_t_base)

    print(f"[test_factors] 基线 EMA{config.EMA_FAST}/{config.EMA_SLOW} @ {sym}")
    print(f"  VALID 收益 {base_v['total_return_pct']:.1f}%  回撤 {base_v['max_ddpercent']:.1f}%  成交 {base_v['total_trades']}")
    print(f"  TEST  收益 {base_t['total_return_pct']:.1f}%  回撤 {base_t['max_ddpercent']:.1f}%  成交 {base_t['total_trades']}")

    thresholds = [0.2, 0.3, 0.4, 0.5]
    smooths = [1, 3, 5]
    weight_sets = [
        factors.DEFAULT_WEIGHTS,
        {"momentum": 0.3, "ema_trend": 0.3, "vwap_dev": 0.2, "volatility": 0.1, "skew": 0.05, "kurtosis": 0.0, "amihud": 0.05},
        {"momentum": 0.2, "ema_trend": 0.4, "vwap_dev": 0.15, "volatility": 0.15, "skew": 0.0, "kurtosis": 0.0, "amihud": 0.1},
        {"momentum": 0.15, "ema_trend": 0.25, "vwap_dev": 0.2, "volatility": 0.1, "skew": 0.15, "kurtosis": 0.1, "amihud": 0.05},
    ]

    results: list[dict] = []
    best: dict | None = None
    best_test = float("-inf")

    for th, sm, w in product(thresholds, smooths, weight_sets):
        raw_v = factors.compute_raw(df_v)
        raw_t = factors.compute_raw(df_t)
        sc_v = factors.fuse(raw_v, w, smooth=sm)
        sc_t = factors.fuse(raw_t, w, smooth=sm)
        sig_v = pd.Series(factors.score_to_signal(sc_v, th), index=df_v.index)
        sig_t = pd.Series(factors.score_to_signal(sc_t, th), index=df_t.index)
        st_v = _eval(df_v, sig_v)
        st_t = _eval(df_t, sig_t)
        row = {
            "threshold": th,
            "smooth": sm,
            "weights": w,
            "valid_return": st_v["total_return_pct"],
            "valid_dd": st_v["max_ddpercent"],
            "valid_trades": st_v["total_trades"],
            "test_return": st_t["total_return_pct"],
            "test_dd": st_t["max_ddpercent"],
            "test_trades": st_t["total_trades"],
            "beats_test": st_t["total_return_pct"] > base_t["total_return_pct"],
        }
        results.append(row)
        if st_t["total_return_pct"] > best_test:
            best_test = st_t["total_return_pct"]
            best = row

    results.sort(key=lambda x: x["test_return"], reverse=True)
    out = {
        "symbol": sym,
        "baseline": {"valid": base_v, "test": base_t, "ema": base_kw},
        "best_multi": best,
        "top10": results[:10],
        "recommend_add": bool(best and best["beats_test"]),
    }
    path = config.ARTIFACT_PATH / "factor_ab_test.json"

    print("\n[test_factors] ===== 多因子 TOP3（按 TEST 收益）=====")
    for r in results[:3]:
        tag = "beat baseline" if r["beats_test"] else "below baseline"
        print(
            f"  th={r['threshold']} sm={r['smooth']}  "
            f"VALID {r['valid_return']:.1f}%  TEST {r['test_return']:.1f}%  [{tag}]"
        )

    # EMA + 多因子方向过滤（仅当多因子与 EMA 强烈反向时平仓）
    def _hybrid(df: pd.DataFrame, th: float) -> pd.Series:
        ema = signals.generate(df, "ema_cross", **base_kw)
        sc = factors.fuse(factors.compute_raw(df), smooth=3)
        out = ema.copy()
        for i in range(len(df)):
            if ema.iloc[i] > 0 and sc[i] < th:
                out.iloc[i] = 0
            elif ema.iloc[i] < 0 and sc[i] > -th:
                out.iloc[i] = 0
        return out

    hybrid_rows: list[dict] = []
    for th in [-0.5, -0.2, 0.0, 0.2, 0.3]:
        st_v = _eval(df_v, _hybrid(df_v, th))
        st_t = _eval(df_t, _hybrid(df_t, th))
        hybrid_rows.append({
            "threshold": th,
            "valid_return": st_v["total_return_pct"],
            "test_return": st_t["total_return_pct"],
            "beats_test": st_t["total_return_pct"] > base_t["total_return_pct"],
        })
    best_hybrid = max(hybrid_rows, key=lambda x: x["test_return"])
    robust_hybrid = [
        r for r in hybrid_rows
        if r["test_return"] > base_t["total_return_pct"]
        and r["valid_return"] >= base_v["total_return_pct"]
    ]
    best_robust = max(robust_hybrid, key=lambda x: x["test_return"]) if robust_hybrid else None

    out["hybrid_filter"] = {
        "grid": hybrid_rows,
        "best_test": best_hybrid,
        "best_robust": best_robust,
    }
    out["recommend_add"] = bool(best and best["beats_test"])
    out["recommend_hybrid"] = bool(
        best_robust
        and best_robust["test_return"] - base_t["total_return_pct"] >= 2.0
    )
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n[test_factors] ===== EMA+因子过滤（按 TEST）=====")
    for r in hybrid_rows:
        tag = "beat" if r["beats_test"] else "below"
        print(f"  th={r['threshold']:4}  VALID {r['valid_return']:5.1f}%  TEST {r['test_return']:5.1f}%  [{tag}]")

    if best and best["beats_test"]:
        print(f"\n[test_factors] [OK] 纯多因子 TEST 提升: {base_t['total_return_pct']:.1f}% -> {best['test_return']:.1f}%")
    else:
        print(f"\n[test_factors] [NO] 纯多因子未超过基线 TEST {base_t['total_return_pct']:.1f}%")

    if out["recommend_hybrid"] and best_robust:
        print(
            f"[test_factors] [OK] 混合过滤 TEST 提升 >=2% 且 VALID 不劣化: "
            f"{base_t['total_return_pct']:.1f}% -> {best_robust['test_return']:.1f}% (th={best_robust['threshold']})"
        )
    elif best_robust:
        print(
            f"[test_factors] [NO] 混合过滤双段略优但 TEST 提升 <2% "
            f"({base_t['total_return_pct']:.1f}% -> {best_robust['test_return']:.1f}%)，不接入"
        )
    else:
        print("[test_factors] [NO] 无同时改善 VALID+TEST 的混合过滤参数，不接入")
    print(f"[test_factors] -> {path}")
    return out


if __name__ == "__main__":
    main()
