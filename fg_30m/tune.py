"""
30 分钟参数扫描 —— VALID 段收益最大化 → TEST 样本外

    python tune.py
    python tune.py --quick
"""
from __future__ import annotations

import argparse
import json
from itertools import product

import pandas as pd

import config
import data
import signals
from backtest import run


def _grid(quick: bool) -> list[dict]:
    modes_base = [
        ("ema_cross", {"ema_fast": 3, "ema_slow": 8}),
        ("ema_cross", {"ema_fast": 3, "ema_slow": 13}),
        ("ema_cross", {"ema_fast": 5, "ema_slow": 20}),
        ("ema_cross", {"ema_fast": 8, "ema_slow": 34}),
        ("ema_cross", {"ema_fast": 12, "ema_slow": 48}),
        ("ma_trend", {"ma_s": 5, "ma_m": 20, "ma_l": 60}),
        ("ma_trend", {"ma_s": 8, "ma_m": 21, "ma_l": 55}),
        ("ma_trend", {"ma_s": 10, "ma_m": 30, "ma_l": 90}),
        ("breakout", {"donchian": 12}),
        ("breakout", {"donchian": 24}),
        ("breakout", {"donchian": 40}),
        ("rsi_momo", {"rsi_period": 10}),
        ("rsi_momo", {"rsi_period": 14}),
        ("rsi_revert", {"rsi_period": 7, "rsi_ob": 70, "rsi_os": 30}),
        ("rsi_revert", {"rsi_period": 5, "rsi_ob": 75, "rsi_os": 25}),
        ("macd", {}),
        ("momentum", {"mom_th": 0.001}),
        ("momentum", {"mom_th": 0.003}),
        ("combo", {"combo_min": 2}),
    ]
    lots_list = [1, 2, 3] if not quick else [1, 2]
    if quick:
        modes_base = modes_base[:10]

    out: list[dict] = []
    for (mode, kw), ml in product(modes_base, lots_list):
        out.append({"mode": mode, "max_lots": ml, **kw})
    return out


def main(quick: bool = False, symbol: str | None = None) -> dict:
    config.ensure_dirs()
    sym = symbol or config.SYMBOL
    config.SYMBOL = sym
    config.VT_SYMBOL = f"{sym}.{config.EXCHANGE_STR}"

    print(f"[fg_30m/tune] 加载 {sym} 全量 30m …")
    df_all = data.load_30m(sym, use_cache=True)
    t0, t1 = config.TUNE_PERIOD
    e0, e1 = config.TEST_PERIOD
    df_tune = df_all[(df_all["datetime"] >= pd.Timestamp(t0)) & (df_all["datetime"] <= pd.Timestamp(t1))]
    df_test = df_all[(df_all["datetime"] >= pd.Timestamp(e0)) & (df_all["datetime"] <= pd.Timestamp(e1))]

    grid = _grid(quick)
    print(f"[fg_30m/tune] VALID {t0}~{t1} ({len(df_tune)} bars)  扫描 {len(grid)} 组 …")

    results: list[dict] = []
    best: dict | None = None
    best_ret = float("-inf")

    for i, g in enumerate(grid):
        mode = g.pop("mode")
        ml = g.pop("max_lots")
        sig_t = signals.generate(df_tune, mode, **g)
        st, _ = run(df_tune, sig_t, max_lots=ml)
        row = {
            "mode": mode,
            "max_lots": ml,
            **g,
            "valid_return": st["total_return_pct"],
            "valid_dd": st["max_ddpercent"],
            "valid_sharpe": st["sharpe_ratio"],
            "valid_trades": st["total_trades"],
        }
        results.append(row)
        if st["total_return_pct"] > best_ret:
            best_ret = st["total_return_pct"]
            best = row.copy()

        if (i + 1) % 15 == 0 or i + 1 == len(grid):
            print(f"    {i+1}/{len(grid)}  当前最优 VALID {best_ret:.1f}%  {best['mode'] if best else ''}")

    assert best is not None
    mode = best["mode"]
    ml = best["max_lots"]
    params = {k: v for k, v in best.items() if k not in (
        "mode", "max_lots", "valid_return", "valid_dd", "valid_sharpe", "valid_trades",
    )}
    sig_test = signals.generate(df_test, mode, **params)
    st_test, eq_test = run(df_test, sig_test, max_lots=ml)

    results.sort(key=lambda x: x["valid_return"], reverse=True)
    out = {
        "symbol": sym,
        "tune_period": config.TUNE_PERIOD,
        "test_period": config.TEST_PERIOD,
        "best": best,
        "test_stats": st_test,
        "top15": results[:15],
    }
    path = config.ARTIFACT_PATH / f"tune_{sym}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    eq_test.to_csv(config.ARTIFACT_PATH / f"equity_test_{sym}.csv", index=False)

    print("\n[fg_30m/tune] ===== VALID 最优 =====")
    print(f"    {mode}  max_lots={ml}  params={params}")
    print(f"    VALID 收益 {best['valid_return']:.1f}%  回撤 {best['valid_dd']:.1f}%  成交 {best['valid_trades']}")
    print("\n[fg_30m/tune] ===== TEST 样本外 =====")
    print(f"    收益 {st_test['total_return_pct']:.1f}%  回撤 {st_test['max_ddpercent']:.1f}%  成交 {st_test['total_trades']}")
    print(f"    期末 {st_test['end_equity']:,.0f}")
    print(f"\n[fg_30m/tune] -> {path}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--symbol", default=config.SYMBOL)
    args = p.parse_args()
    main(quick=args.quick, symbol=args.symbol)
