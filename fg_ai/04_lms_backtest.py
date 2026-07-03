"""
步骤 04：LMS 混合策略回测

两种模式（由 config.LMS_USE_AI 控制）：
  · 纯规则（False）：只用三均线趋势，无需训练，弱机器可直接跑。
  · 规则+AI（True）：并入 02_train 保存的 LightGBM 信号做双确认；
                     若找不到已保存信号，自动降级为纯规则并提示。

用法：
    python 04_lms_backtest.py
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

    # 确保行情与合约参数就绪（等价 01_prepare_lab，可重复执行）
    lab.save_bar_data(data_loader.build_bar_data())
    lab.add_contract_setting(config.VT_SYMBOL, config.LONG_RATE, config.SHORT_RATE,
                             config.CONTRACT_SIZE, config.PRICE_TICK)

    use_ai: bool = config.LMS_USE_AI
    model_signal: pl.DataFrame | None = None
    if use_ai:
        model_signal = lab.load_signal(config.SIGNAL_NAME)
        if model_signal is None or model_signal.is_empty():
            print("[04] ⚠️ 未找到已保存的模型信号，自动降级为『纯规则』模式"
                  "（如需融合请先运行 02_train.py）")
            use_ai = False

    # 组装信号表（规则 regime + 可选 AI signal）
    price_df = data_loader.load_polars_df()
    signal_df = lms.build_signal_df(price_df, model_signal if use_ai else None)

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

    tag = "hybrid" if use_ai else "rule"
    if result_df is not None and not result_df.is_empty():
        result_df.write_csv(config.ARTIFACT_PATH / f"lms_{tag}_daily.csv")

    def _fmt(v: object) -> object:
        return v.isoformat() if isinstance(v, datetime) else v

    (config.ARTIFACT_PATH / f"lms_{tag}_stats.json").write_text(
        json.dumps({k: _fmt(v) for k, v in stats.items()},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    mode = "纯规则趋势" if not use_ai else (
        f"趋势+AI否决(>{config.LMS_VETO_THRESHOLD})" if config.LMS_AI_MODE == "veto"
        else f"趋势+AI确认(>{config.SIGNAL_THRESHOLD})" if config.LMS_AI_MODE == "confirm"
        else "纯规则趋势"
    )
    print(f"\n[04] LMS 回测绩效（{mode}，TEST 段）：")
    for key in ("start_date", "end_date", "total_return", "annual_return",
                "max_drawdown", "sharpe_ratio", "return_drawdown_ratio",
                "total_trade_count", "total_commission"):
        if key in stats:
            print(f"    {key:<22} {stats[key]}")
    print(f"\n[04] 结果已保存至 {config.ARTIFACT_PATH}")
    return stats


if __name__ == "__main__":
    main()
