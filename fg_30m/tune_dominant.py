"""
主力合约 30m 扫参 —— 追求全段 + TEST 收益最大化

    python tune_dominant.py
    python tune_dominant.py --quick
"""
from __future__ import annotations

import argparse
import json
from itertools import product

import pandas as pd

import config
import dominant
import signals
from backtest import run


def _periods(df_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t0, t1 = config.TUNE_PERIOD
    e0, e1 = config.TEST_PERIOD
    full = df_all
    tune = df_all[(df_all["datetime"] >= t0) & (df_all["datetime"] <= t1)]
    test = df_all[(df_all["datetime"] >= e0) & (df_all["datetime"] <= e1)]
    return full, tune, test


def _grid(quick: bool) -> list[dict]:
    ema_pairs = [(26, 46), (12, 48), (8, 34), (5, 20)] if not quick else [(26, 46), (12, 48)]
    lots = [1, 2, 3] if not quick else [1, 2]
    out: list[dict] = []

    for (ef, es), ml in product(ema_pairs, lots):
        out.append({
            "label": f"daily_ema_{ef}_{es}_L{ml}",
            "mode": "daily_filter_ema",
            "max_lots": ml,
            "ema_fast": ef,
            "ema_slow": es,
        })

    for dc, ml in product([12, 20, 40], [1, 2] if not quick else [1]):
        out.append({
            "label": f"daily_ema_don_{dc}_L{ml}",
            "mode": "daily_filter_donchian",
            "max_lots": ml,
            "donchian": dc,
            "ema_fast": 26,
            "ema_slow": 46,
        })

    if not quick:
        for (ef, es), ml in product([(12, 48), (26, 46)], [1, 2, 3]):
            out.append({
                "label": f"ema_{ef}_{es}_L{ml}",
                "mode": "ema_cross",
                "max_lots": ml,
                "ema_fast": ef,
                "ema_slow": es,
            })
            out.append({
                "label": f"ema_don_both_{ef}_L{ml}",
                "mode": "ema_donchian_both",
                "max_lots": ml,
                "ema_fast": ef,
                "ema_slow": es,
                "donchian": 20,
            })

    return out


def _eval(df: pd.DataFrame, spec: dict) -> dict:
    g = dict(spec)
    label = g.pop("label")
    mode = g.pop("mode")
    ml = g.pop("max_lots")
    sig = signals.generate(df, mode, **g)
    st, _ = run(df, sig, max_lots=ml)
    return {"label": label, "mode": mode, "max_lots": ml, "params": g, **st}


def main(quick: bool = False) -> dict:
    config.ensure_dirs()
    print("[tune_dominant] 加载主力拼接 30m …")
    print(dominant.segment_summary().to_string(index=False))

    df_all = dominant.load_dominant_30m()
    s, e = dominant.dominant_period()
    config.BACKTEST_PERIOD = (s, e)
    config.TUNE_PERIOD = (s, "2026-03-31")
    config.TEST_PERIOD = ("2026-04-01", e)

    df_full, df_tune, df_test = _periods(df_all)
    print(f"[tune_dominant] 全段 {len(df_full)} bars  TUNE {len(df_tune)}  TEST {len(df_test)}")

    grid = _grid(quick)
    results: list[dict] = []
    best_score = float("-inf")
    best: dict | None = None

    for i, g in enumerate(grid):
        g = dict(g)
        row_tune = _eval(df_tune, g)
        row_full = _eval(df_full, g)
        row_test = _eval(df_test, g)
        score = row_tune["total_return_pct"] + 0.5 * row_test["total_return_pct"]
        if row_tune["total_return_pct"] > 0 and row_test["total_return_pct"] > 0:
            score += 15
        score += row_full["max_ddpercent"] * 0.2

        rec = {
            "label": row_tune["label"],
            "mode": row_tune["mode"],
            "max_lots": row_tune["max_lots"],
            "params": row_tune["params"],
            "tune_ret": row_tune["total_return_pct"],
            "full_ret": row_full["total_return_pct"],
            "test_ret": row_test["total_return_pct"],
            "full_dd": row_full["max_ddpercent"],
            "full_trades": row_full["total_trades"],
            "score": score,
        }
        results.append(rec)
        if score > best_score:
            best_score = score
            best = rec
        if (i + 1) % 10 == 0:
            print(f"  … {i + 1}/{len(grid)}")

    results.sort(key=lambda x: x["score"], reverse=True)
    out = {
        "dominant_period": [s, e],
        "segments": dominant.DOMINANT_SEGMENTS,
        "best": best,
        "top10": results[:10],
    }
    path = config.ARTIFACT_PATH / "tune_dominant.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== TOP 5（主力拼接，收益优先）===")
    for r in results[:5]:
        print(
            f"  {r['label']:<28} FULL {r['full_ret']:7.1f}%  "
            f"TUNE {r['tune_ret']:7.1f}%  TEST {r['test_ret']:7.1f}%  "
            f"dd {r['full_dd']:6.1f}%  trades {r['full_trades']}"
        )
    if best:
        print(f"\n[best] {best['label']}  FULL {best['full_ret']:.1f}%")
    print(f"[tune_dominant] -> {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)
