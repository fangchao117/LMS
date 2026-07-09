"""
30 分钟 vs 日线 —— BOLL+BBI+EMA + 稳定版过滤器 策略对比

    python compare_strategies.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
import data
import signals_enhanced as se
from backtest import run as run_30m


def load_daily_fg(start: str, end: str) -> pd.DataFrame:
    pdf = pd.read_parquet(config.ROOT / "data" / "FG00.CZCE.parquet")
    pdf["datetime"] = pd.to_datetime(pdf["datetime"])
    pdf = pdf[(pdf["datetime"] >= start) & (pdf["datetime"] <= end)]
    pdf = pdf.dropna(subset=["open", "high", "low", "close"]).sort_values("datetime")
    pdf["vt_symbol"] = "FG00.CZCE"
    return pdf.reset_index(drop=True)


def _eval(df: pd.DataFrame, sig: pd.Series) -> dict:
    st, _ = run_30m(df, sig, max_lots=1)
    return st


STRATEGIES: list[dict] = [
    {"name": "EMA26/46", "mode": "ema", "filters": "none", "risk": False},
    {"name": "EMA26/46+ADX", "mode": "ema", "filters": "adx", "risk": False},
    {"name": "EMA26/46+ADX+风控", "mode": "ema", "filters": "adx", "risk": True},
    {"name": "BOLL中轨", "mode": "boll", "filters": "none", "risk": False},
    {"name": "BBI", "mode": "bbi", "filters": "none", "risk": False},
    {"name": "BOLL+BBI+EMA", "mode": "triple", "filters": "none", "risk": False},
    {"name": "BOLL+BBI+EMA+ADX", "mode": "triple", "filters": "adx", "risk": False},
    {"name": "BOLL+BBI+EMA+ADX+发散", "mode": "triple", "filters": "adx_div", "risk": False},
    {"name": "BOLL+BBI+EMA+ADX+风控", "mode": "triple", "filters": "adx", "risk": True},
    {"name": "稳定版MA20/50+ADX", "mode": "stable_ma", "filters": "adx_div", "risk": True},
]


def _score(full_ret: float, test_ret: float, dd: float) -> float:
    """两段都正收益加分；大回撤惩罚。"""
    s = full_ret + 0.5 * test_ret
    if full_ret > 0 and test_ret > 0:
        s += 20
    s += dd * 0.3
    return s


def run_compare(
    df: pd.DataFrame,
    tune_end: str,
    label: str,
) -> list[dict]:
    df_tune = df[df["datetime"] <= tune_end].reset_index(drop=True)
    df_test = df[df["datetime"] > tune_end].reset_index(drop=True)
    rows: list[dict] = []

    for spec in STRATEGIES:
        sig_all = se.build_signal(df, spec["mode"], spec["filters"], spec["risk"])
        sig_tune = sig_all[df["datetime"] <= tune_end].reset_index(drop=True)
        sig_test = sig_all[df["datetime"] > tune_end].reset_index(drop=True)
        rf = _eval(df, sig_all)
        rt = _eval(df_tune, sig_tune)
        rs = _eval(df_test, sig_test)
        row = {
            "timeframe": label,
            "strategy": spec["name"],
            "full_return": rf["total_return_pct"],
            "full_dd": rf["max_ddpercent"],
            "full_trades": rf["total_trades"],
            "tune_return": rt["total_return_pct"],
            "test_return": rs["total_return_pct"],
            "test_dd": rs["max_ddpercent"],
            "test_trades": rs["total_trades"],
            "score": _score(rf["total_return_pct"], rs["total_return_pct"], rf["max_ddpercent"]),
        }
        rows.append(row)
    return rows


def main() -> dict:
    config.ensure_dirs()
    start, end = config.BACKTEST_PERIOD
    mid = config.TUNE_PERIOD[1]

    print(f"[compare] 区间 {start} ~ {end}  切分 TEST>{mid}")

    df_30 = data.load_30m("FG609", use_cache=True)
    df_30 = df_30[(df_30["datetime"] >= start) & (df_30["datetime"] <= end)].reset_index(drop=True)
    df_d = load_daily_fg(start, end)

    rows = run_compare(df_30, mid, "30m_FG609")
    rows += run_compare(df_d, mid, "daily_FG00")

    rows.sort(key=lambda x: -x["score"])
    out = {"period": [start, end], "tune_cutoff": mid, "results": rows, "best": rows[0]}
    path = config.ARTIFACT_PATH / "strategy_compare.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[compare] TOP 8（按综合分 score）")
    print(f"{'周期':<12} {'策略':<28} {'全段':>7} {'前半':>7} {'后半':>7} {'回撤':>7} score")
    for r in rows[:8]:
        print(
            f"{r['timeframe']:<12} {r['strategy']:<28} "
            f"{r['full_return']:6.1f}% {r['tune_return']:6.1f}% {r['test_return']:6.1f}% "
            f"{r['full_dd']:6.1f}% {r['score']:6.1f}"
        )

    best = rows[0]
    print(f"\n[compare] 最优: {best['timeframe']} / {best['strategy']}")
    print(f"  全段 {best['full_return']:.1f}%  前半 {best['tune_return']:.1f}%  后半 {best['test_return']:.1f}%")
    print(f"  -> {path}")
    return out


if __name__ == "__main__":
    main()
