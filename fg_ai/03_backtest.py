"""
步骤 03：回测（趋势 + AI 否决混合策略）

默认使用 LmsAlphaStrategy：三均线定方向，AI z-score 仅在强烈反向时否决。
需先运行 02_train.py 生成信号；无信号时自动降级为纯规则。
"""
from __future__ import annotations

import json
from datetime import datetime

import polars as pl

from vnpy.alpha import AlphaLab
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.trader.constant import Interval

import config
import data_loader
import lms
from lms_strategy import LmsAlphaStrategy


def _to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main() -> dict:
    config.ensure_dirs()

    lab = AlphaLab(str(config.LAB_PATH))

    use_ai = config.LMS_USE_AI and config.LMS_AI_MODE != "off"
    model_signal: pl.DataFrame | None = None
    if use_ai:
        model_signal = lab.load_signal(config.SIGNAL_NAME)
        if model_signal is None or model_signal.is_empty():
            print("[03] ⚠️ 未找到 AI 信号，自动降级为纯规则（请先运行 02_train.py）")
            use_ai = False

    signal_df = lms.build_signal_df(
        data_loader.load_polars_df(),
        model_signal if use_ai else None,
    )

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
        "use_ai": use_ai,
        "ai_mode": config.LMS_AI_MODE if use_ai else "off",
        "signal_threshold": config.SIGNAL_THRESHOLD,
        "veto_threshold": config.LMS_VETO_THRESHOLD,
        "position_pct": config.POSITION_PCT,
        "price_add_ticks": config.PRICE_ADD_TICKS,
    }
    engine.add_strategy(LmsAlphaStrategy, setting, signal_df)

    engine.load_data()
    engine.run_backtesting()

    result_df = engine.calculate_result()
    stats: dict = engine.calculate_statistics()

    if result_df is not None and not result_df.is_empty():
        result_df.write_csv(config.ARTIFACT_PATH / "backtest_daily.csv")

    def _fmt(v: object) -> object:
        return v.isoformat() if isinstance(v, datetime) else v

    stats_clean = {k: _fmt(v) for k, v in stats.items()}
    (config.ARTIFACT_PATH / "backtest_stats.json").write_text(
        json.dumps(stats_clean, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if not use_ai:
        mode = "纯规则趋势"
    elif config.LMS_AI_MODE == "veto":
        mode = f"趋势+AI否决(>{config.LMS_VETO_THRESHOLD})"
    else:
        mode = f"趋势+AI确认(>{config.SIGNAL_THRESHOLD})"

    print(f"\n[03] 回测绩效（{mode}，TEST 段样本外）：")
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
