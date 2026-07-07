"""
回测 —— LMS + 多因子融合（VALID 调阈值 + TEST 样本外）

    python backtest.py
    python backtest.py --no-tune
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import polars as pl

from vnpy.alpha import AlphaLab
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.trader.constant import Interval

import config
import data
from signal_gen import build_signal_df
from strategy import LmsFilterStrategy


def _to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _run(
    signal_df: pl.DataFrame,
    start: str,
    end: str,
    lab: AlphaLab,
    threshold: float,
) -> dict:
    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=[config.VT_SYMBOL],
        interval=Interval.DAILY,
        start=_to_dt(start),
        end=_to_dt(end),
        capital=config.CAPITAL,
        annual_days=240,
    )
    engine.add_strategy(
        LmsFilterStrategy,
        {
            "signal_threshold": threshold,
            "position_pct": config.POSITION_PCT,
            "price_add_ticks": config.PRICE_ADD_TICKS,
        },
        signal_df,
    )
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    return engine.calculate_statistics()


def tune_threshold(signal_df: pl.DataFrame, lab: AlphaLab) -> tuple[float, dict]:
    """VALID 段扫描阈值，取总收益最高者。"""
    best_th = config.SIGNAL_THRESHOLD
    best_stats: dict = {}
    best_ret = float("-inf")

    print("[backtest] VALID 段阈值扫描（收益最大化）…")
    for th in config.THRESHOLD_CANDIDATES:
        stats = _run(signal_df, config.VALID_PERIOD[0], config.VALID_PERIOD[1], lab, th)
        ret = float(stats.get("total_return", 0) or 0)
        print(f"    阈值 {th:.1f} -> 总收益 {ret:.1f}%  夏普 {stats.get('sharpe_ratio', 0):.2f}")
        if ret > best_ret:
            best_ret = ret
            best_th = th
            best_stats = stats

    print(f"[backtest] 最优阈值 {best_th}（VALID 总收益 {best_ret:.1f}%）")
    return best_th, best_stats


def main(tune: bool = True) -> dict:
    config.ensure_dirs()

    signal_df, _, _, _ = build_signal_df()

    lab = AlphaLab(str(config.LAB_PATH))
    lab.save_bar_data(data.build_bars())
    lab.add_contract_setting(
        config.VT_SYMBOL, config.LONG_RATE, config.SHORT_RATE,
        config.CONTRACT_SIZE, config.PRICE_TICK,
    )

    tune_info: dict | None = None
    threshold = config.SIGNAL_THRESHOLD
    if tune and config.TUNE_THRESHOLD:
        threshold, valid_stats = tune_threshold(signal_df, lab)
        tune_info = {"best_threshold": threshold, "valid_stats": valid_stats}

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=[config.VT_SYMBOL],
        interval=Interval.DAILY,
        start=_to_dt(config.TEST_PERIOD[0]),
        end=_to_dt(config.TEST_PERIOD[1]),
        capital=config.CAPITAL,
        annual_days=240,
    )
    engine.add_strategy(
        LmsFilterStrategy,
        {
            "signal_threshold": threshold,
            "position_pct": config.POSITION_PCT,
            "price_add_ticks": config.PRICE_ADD_TICKS,
        },
        signal_df,
    )
    engine.load_data()
    engine.run_backtesting()

    daily = engine.calculate_result()
    stats = engine.calculate_statistics()

    tag = config.SIGNAL_MODE
    if daily is not None and not daily.is_empty():
        daily.write_csv(config.ARTIFACT_PATH / f"lms_{tag}_daily.csv")

    def _fmt(v):
        return v.isoformat() if isinstance(v, datetime) else v

    out = {
        "signal_mode": config.SIGNAL_MODE,
        "factor_weights": config.FACTOR_WEIGHTS,
        "position_pct": config.POSITION_PCT,
        "threshold": threshold,
        "tune": tune_info,
        **{k: _fmt(v) for k, v in stats.items()},
    }
    (config.ARTIFACT_PATH / f"lms_{tag}_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\n[backtest] LMS 多因子（mode={tag}，阈值={threshold}，TEST 段）：")
    for k in (
        "start_date", "end_date", "total_return", "annual_return",
        "max_drawdown", "sharpe_ratio", "return_drawdown_ratio",
        "total_trade_count", "total_commission", "end_balance",
    ):
        if k in stats:
            print(f"    {k:<22} {stats[k]}")
    print(f"\n[backtest] 结果已存至 {config.ARTIFACT_PATH}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true", help="跳过 VALID 阈值扫描")
    args = parser.parse_args()
    main(tune=not args.no_tune)
