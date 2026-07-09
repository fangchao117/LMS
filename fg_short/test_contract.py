"""
单合约 / 分段回测 —— 用 FG00 连续数据近似（无 FG509 独立行情时）

    python test_contract.py --symbol FG509 --start 2025-05-16 --end 2025-08-14
    python test_contract.py --year 2025
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd
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


def fg509_dominant_window_2025() -> tuple[str, str]:
    """
    2025 年 9 月合约 FG509 作主力的大致窗口：
    · 5 月主力 FG505 于 4 月 15 日前后换月
    · 9 月合约于 8 月 15 日前后换至 FG601
    """
    return "2025-05-16", "2025-08-14"


def run_period(start: str, end: str, label: str) -> dict:
    bars_df = data.load_polars_df()
    pdf = bars_df.to_pandas()
    pdf = pdf[(pdf["datetime"] >= pd.Timestamp(start)) & (pdf["datetime"] <= pd.Timestamp(end))]
    if len(pdf) < config.MA_LONG + 5:
        return {"error": f"数据不足 {len(pdf)} 根", "label": label}

    sub = pl.from_pandas(pdf[["datetime", "vt_symbol", "open", "high", "low", "close", "volume", "open_interest"]])
    sig = generate_signals(sub, mode="trend_ma")

    lab = AlphaLab(str(config.LAB_PATH))
    # 仅保存区间内 bar
    from vnpy.trader.object import BarData
    from vnpy.trader.constant import Exchange
    exchange = Exchange(config.EXCHANGE_STR)
    bar_list = []
    for row in pdf.itertuples(index=False):
        bar_list.append(
            BarData(
                symbol=config.SYMBOL,
                exchange=exchange,
                datetime=row.datetime.to_pydatetime(),
                interval=Interval.DAILY,
                open_price=float(row.open),
                high_price=float(row.high),
                low_price=float(row.low),
                close_price=float(row.close),
                volume=float(row.volume),
                turnover=0.0,
                open_interest=float(row.open_interest),
                gateway_name="DB",
            )
        )
    lab.save_bar_data(bar_list)
    lab.add_contract_setting(
        config.VT_SYMBOL, config.LONG_RATE, config.SHORT_RATE,
        config.CONTRACT_SIZE, config.PRICE_TICK,
    )

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
            "max_lots": 1,
            "capital_max": config.CAPITAL_MAX,
            "price_add_ticks": config.PRICE_ADD_TICKS,
            "stop_loss_pct": config.STOP_LOSS_PCT,
            "max_drawdown_pct": config.MAX_DRAWDOWN_PCT,
        },
        sig,
    )
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics()
    return {
        "label": label,
        "start": start,
        "end": end,
        "bars": len(pdf),
        "total_return": float(stats.get("total_return", 0) or 0),
        "max_ddpercent": float(stats.get("max_ddpercent", 0) or 0),
        "sharpe": float(stats.get("sharpe_ratio", 0) or 0),
        "trades": int(stats.get("total_trade_count", 0) or 0),
        "end_balance": float(stats.get("end_balance", 0) or 0),
        "note": "价格来自 FG00 连续合约，非 FG509 真实独立行情",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="FG509")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--year", default="")
    args = parser.parse_args()

    print(f"[test_contract] 数据说明：本地仅有 FG00 连续 parquet，无 {args.symbol} 独立日线。")
    print("              以下用 FG00 同区间价格近似，与真实 FG509 会有基差/换月误差。\n")

    runs: list[tuple[str, str, str]] = []
    if args.start and args.end:
        runs.append((args.start, args.end, f"{args.symbol} 指定区间"))
    elif args.year:
        y = args.year
        runs.append((f"{y}-01-01", f"{y}-12-31", f"FG00 全年 {y}"))
        if args.symbol.upper() == "FG509" and y == "2025":
            s, e = fg509_dominant_window_2025()
            runs.append((s, e, f"FG509 主力窗口(估) {s}~{e}"))
    else:
        s, e = fg509_dominant_window_2025()
        runs.append((s, e, f"FG509 主力窗口(估) {s}~{e}"))
        runs.append(("2025-01-01", "2025-12-31", "FG00 全年 2025"))

    for start, end, label in runs:
        r = run_period(start, end, label)
        print(f"===== {label} =====")
        if "error" in r:
            print(f"  {r['error']}")
            continue
        print(f"  区间: {r['start']} ~ {r['end']}  ({r['bars']} 根K线)")
        print(f"  起始资金: {config.CAPITAL:,}  期末: {r['end_balance']:,.0f}")
        print(f"  总收益:   {r['total_return']:.2f}%")
        print(f"  最大回撤: {r['max_ddpercent']:.2f}%")
        print(f"  Sharpe:   {r['sharpe']:.2f}")
        print(f"  成交笔数: {r['trades']}")
        print(f"  {r['note']}\n")


if __name__ == "__main__":
    main()
