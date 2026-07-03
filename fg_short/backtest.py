"""
玻璃短线回测 —— TRAIN+VALID 调参 + TEST 样本外

调参目标：TRAIN 段（2014~2021），手数固定 1 手。
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
from signals import generate_signals
from strategy import ShortTermStrategy


def _to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _run(
    signal_df: pl.DataFrame,
    start: str,
    end: str,
    lab: AlphaLab,
    max_lots: int | None = None,
) -> dict:
    ml = max_lots if max_lots is not None else config.MAX_LOTS
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
        ShortTermStrategy,
        {
            "signal_threshold": config.SIGNAL_THRESHOLD,
            "position_pct": config.POSITION_PCT,
            "margin_rate": config.MARGIN_RATE,
            "max_lots": ml,
            "price_add_ticks": config.PRICE_ADD_TICKS,
        },
        signal_df,
    )
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    return engine.calculate_statistics()


def _score(valid_stats: dict) -> float:
    vr = float(valid_stats.get("total_return", 0) or 0)
    sharpe = float(valid_stats.get("sharpe_ratio", 0) or 0)
    if vr <= 0:
        return float("-inf")
    return vr + 5.0 * sharpe


def tune_params(lab: AlphaLab, bars_df: pl.DataFrame) -> dict:
    """在 TRAIN 段扫描三均线，手数固定 1（小资金风控）。"""
    ma_triples = [
        (3, 13, 34), (3, 21, 34), (5, 13, 34), (5, 21, 34),
        (8, 13, 34), (8, 21, 34), (8, 21, 55), (5, 34, 55),
        (10, 21, 55), (5, 13, 55),
    ]
    max_lots = 1
    start, end = config.TRAIN_PERIOD

    best: dict | None = None
    best_score = float("-inf")

    for ms, mm, ml_ma in ma_triples:
        sig = generate_signals(
            bars_df, mode="trend_ma", ma_short=ms, ma_medium=mm, ma_long=ml_ma
        )
        train_stats = _run(sig, start, end, lab, max_lots)
        sc = _score(train_stats)
        if sc > best_score:
            best_score = sc
            best = {
                "mode": "trend_ma",
                "ma_short": ms,
                "ma_medium": mm,
                "ma_long": ml_ma,
                "max_lots": max_lots,
                "train_stats": train_stats,
                "score": sc,
            }

    if best is None:
        return {
            "mode": "trend_ma",
            "ma_short": config.MA_SHORT,
            "ma_medium": config.MA_MEDIUM,
            "ma_long": config.MA_LONG,
            "max_lots": config.MAX_LOTS,
            "train_stats": {},
            "score": 0,
            "fallback": True,
        }
    return best


def main(tune: bool = True) -> dict:
    config.ensure_dirs()
    bars_df = data.load_polars_df()
    lab = AlphaLab(str(config.LAB_PATH))
    lab.save_bar_data(data.build_bar_data())
    lab.add_contract_setting(
        config.VT_SYMBOL,
        config.LONG_RATE,
        config.SHORT_RATE,
        config.CONTRACT_SIZE,
        config.PRICE_TICK,
    )

    tune_result: dict | None = None
    if tune:
        print("[fg_short] 三均线参数扫描（TRAIN 段，1 手）…")
        tune_result = tune_params(lab, bars_df)
        config.SIGNAL_MODE = tune_result["mode"]
        config.MA_SHORT = tune_result["ma_short"]
        config.MA_MEDIUM = tune_result["ma_medium"]
        config.MA_LONG = tune_result["ma_long"]
        config.MAX_LOTS = tune_result["max_lots"]
        if tune_result.get("fallback"):
            print("[fg_short] TRAIN 段无正收益组合，保留 config 默认参数")
        else:
            tr = float(tune_result["train_stats"].get("total_return", 0) or 0)
            print(
                f"[fg_short] TRAIN 最优: MA {config.MA_SHORT}/{config.MA_MEDIUM}/{config.MA_LONG} "
                f"return={tr:.1f}%"
            )

    signal_df = generate_signals(
        bars_df,
        mode=config.SIGNAL_MODE,
        ma_short=config.MA_SHORT,
        ma_medium=config.MA_MEDIUM,
        ma_long=config.MA_LONG,
    )

    stats = _run(signal_df, config.TEST_PERIOD[0], config.TEST_PERIOD[1], lab)

    daily_engine = BacktestingEngine(lab)
    daily_engine.set_parameters(
        vt_symbols=[config.VT_SYMBOL],
        interval=Interval.DAILY,
        start=_to_dt(config.TEST_PERIOD[0]),
        end=_to_dt(config.TEST_PERIOD[1]),
        capital=config.CAPITAL,
        annual_days=240,
    )
    daily_engine.add_strategy(
        ShortTermStrategy,
        {
            "signal_threshold": config.SIGNAL_THRESHOLD,
            "position_pct": config.POSITION_PCT,
            "margin_rate": config.MARGIN_RATE,
            "max_lots": config.MAX_LOTS,
            "price_add_ticks": config.PRICE_ADD_TICKS,
        },
        signal_df,
    )
    daily_engine.load_data()
    daily_engine.run_backtesting()
    daily = daily_engine.calculate_result()

    tag = f"ma{config.MA_SHORT}_{config.MA_MEDIUM}_{config.MA_LONG}"
    if daily is not None and not daily.is_empty():
        daily.write_csv(config.ARTIFACT_PATH / f"short_{tag}_daily.csv")

    def _fmt(v: object) -> object:
        return v.isoformat() if isinstance(v, datetime) else v

    out = {
        "capital": config.CAPITAL,
        "signal_mode": config.SIGNAL_MODE,
        "ma_short": config.MA_SHORT,
        "ma_medium": config.MA_MEDIUM,
        "ma_long": config.MA_LONG,
        "max_lots": config.MAX_LOTS,
        "margin_rate": config.MARGIN_RATE,
        "tune": tune_result,
        "test_stats": {k: _fmt(v) for k, v in stats.items()},
    }
    (config.ARTIFACT_PATH / f"short_{tag}_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    end_bal = float(stats.get("end_balance", 0) or 0)
    print(f"\n[fg_short] 玻璃短线 TEST 段（起始 {config.CAPITAL:,} 元 -> 约 {end_bal:,.0f} 元）：")
    print(
        f"    三均线 {config.MA_SHORT}/{config.MA_MEDIUM}/{config.MA_LONG}  "
        f"最多 {config.MAX_LOTS} 手  保证金率 {config.MARGIN_RATE:.0%}"
    )
    for k in (
        "start_date", "end_date", "total_return", "annual_return",
        "max_drawdown", "sharpe_ratio", "return_drawdown_ratio",
        "total_trade_count", "total_commission", "total_net_pnl",
    ):
        if k in stats:
            v = stats[k]
            if k == "total_return":
                print(f"    {k:<22} {float(v):.2f}%")
            elif k == "annual_return":
                print(f"    {k:<22} {float(v):.2f}%")
            else:
                print(f"    {k:<22} {v}")
    print(f"\n[fg_short] 结果已存至 {config.ARTIFACT_PATH}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="TRAIN 段扫描均线（实验性）")
    args = parser.parse_args()
    main(tune=args.tune)
