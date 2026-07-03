"""
回测 —— LMS 自适应滤波策略（TEST 段样本外）

流程：生成信号 -> 写入 AlphaLab 行情/合约 -> 回测 -> 打印绩效 + 存结果。
    python backtest.py
"""
from __future__ import annotations

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


def main() -> dict:
    config.ensure_dirs()

    signal_df, dt, close, returns = build_signal_df()

    lab = AlphaLab(str(config.LAB_PATH))
    lab.save_bar_data(data.build_bars())
    lab.add_contract_setting(config.VT_SYMBOL, config.LONG_RATE, config.SHORT_RATE,
                             config.CONTRACT_SIZE, config.PRICE_TICK)

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=[config.VT_SYMBOL], interval=Interval.DAILY,
        start=_to_dt(config.TEST_PERIOD[0]), end=_to_dt(config.TEST_PERIOD[1]),
        capital=config.CAPITAL, annual_days=240,
    )
    engine.add_strategy(LmsFilterStrategy, {
        "signal_threshold": config.SIGNAL_THRESHOLD,
        "position_pct": config.POSITION_PCT,
        "price_add_ticks": config.PRICE_ADD_TICKS,
    }, signal_df)

    engine.load_data()
    engine.run_backtesting()

    daily = engine.calculate_result()
    stats = engine.calculate_statistics()

    tag = config.SIGNAL_MODE
    if daily is not None and not daily.is_empty():
        daily.write_csv(config.ARTIFACT_PATH / f"lms_{tag}_daily.csv")

    def _fmt(v):
        return v.isoformat() if isinstance(v, datetime) else v

    (config.ARTIFACT_PATH / f"lms_{tag}_stats.json").write_text(
        json.dumps({k: _fmt(v) for k, v in stats.items()},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\n[backtest] LMS 自适应滤波（mode={tag}，TEST 段）：")
    for k in ("start_date", "end_date", "total_return", "annual_return",
              "max_drawdown", "sharpe_ratio", "return_drawdown_ratio",
              "total_trade_count", "total_commission"):
        if k in stats:
            print(f"    {k:<22} {stats[k]}")
    print(f"\n[backtest] 结果已存至 {config.ARTIFACT_PATH}")
    return stats


if __name__ == "__main__":
    main()
