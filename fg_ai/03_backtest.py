"""
步骤 03：回测

用 AlphaLab 中登记的合约参数与行情，回放 TEST 段，
驱动 GlassAlphaStrategy 按模型信号多空择时，输出绩效指标与净值曲线。
"""
from __future__ import annotations

import json
from datetime import datetime

import polars as pl

from vnpy.alpha import AlphaLab
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.trader.constant import Interval

import config
from strategy import GlassAlphaStrategy


def _to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main() -> dict:
    config.ensure_dirs()

    lab = AlphaLab(str(config.LAB_PATH))

    signal_df: pl.DataFrame | None = lab.load_signal(config.SIGNAL_NAME)
    if signal_df is None or signal_df.is_empty():
        raise RuntimeError("未找到信号，请先运行 02_train.py")

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=[config.VT_SYMBOL],
        interval=Interval.DAILY,
        start=_to_dt(config.TEST_PERIOD[0]),
        end=_to_dt(config.TEST_PERIOD[1]),
        capital=config.CAPITAL,
        annual_days=240,
    )

    setting: dict = {
        "signal_threshold": config.SIGNAL_THRESHOLD,
        "position_pct": config.POSITION_PCT,
        "price_add_ticks": config.PRICE_ADD_TICKS,
    }
    engine.add_strategy(GlassAlphaStrategy, setting, signal_df)

    engine.load_data()
    engine.run_backtesting()

    result_df = engine.calculate_result()
    stats: dict = engine.calculate_statistics()

    # 落地净值曲线与统计
    if result_df is not None and not result_df.is_empty():
        result_df.write_csv(config.ARTIFACT_PATH / "backtest_daily.csv")

    def _fmt(v: object) -> object:
        return v.isoformat() if isinstance(v, datetime) else v

    stats_clean = {k: _fmt(v) for k, v in stats.items()}
    (config.ARTIFACT_PATH / "backtest_stats.json").write_text(
        json.dumps(stats_clean, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n[03] 回测绩效（TEST 段样本外）：")
    for key in (
        "start_date", "end_date", "total_days",
        "total_return", "annual_return", "max_drawdown",
        "sharpe_ratio", "return_drawdown_ratio",
        "total_net_pnl", "total_commission",
        "total_trade_count", "daily_return", "return_std",
    ):
        if key in stats:
            print(f"    {key:<22} {stats[key]}")

    print(f"\n[03] 结果已保存至 {config.ARTIFACT_PATH}")
    return stats


if __name__ == "__main__":
    main()
