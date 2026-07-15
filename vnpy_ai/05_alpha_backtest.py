"""用导出的 ML signal 做 vnpy.alpha 单合约回测。"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import polars as pl
from vnpy.trader.constant import Interval
from vnpy.trader.object import BarData, TradeData

from vnpy.alpha import AlphaLab, AlphaStrategy, BacktestingEngine

LAB_PATH = Path(__file__).resolve().parent / "lab"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


class FgSignalStrategy(AlphaStrategy):
    """单合约：预测值 > 阈值做多，< -阈值做空，否则平仓。"""

    threshold: float = 0.00015
    max_lots: float = 1.0
    price_add: float = 0.0001

    def on_init(self) -> None:
        self.write_log(
            f"FgSignalStrategy init threshold={self.threshold} max_lots={self.max_lots}"
        )

    def on_trade(self, trade: TradeData) -> None:
        pass

    def on_bars(self, bars: dict[str, BarData]) -> None:
        sig_df = self.get_signal()
        if sig_df.is_empty():
            return

        for row in sig_df.iter_rows(named=True):
            vt_symbol = row["vt_symbol"]
            signal = float(row["signal"])
            if signal > self.threshold:
                target = self.max_lots
            elif signal < -self.threshold:
                target = -self.max_lots
            else:
                target = 0.0
            self.set_target(vt_symbol, target)

        self.execute_trading(bars, price_add=self.price_add)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="FG609")
    parser.add_argument("--signal", default=None, help="signal 名称，默认 {symbol}_lgb_test")
    parser.add_argument("--start", default="2026-05-16")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--capital", type=int, default=300_000)
    parser.add_argument("--threshold", type=float, default=0.00015)
    parser.add_argument("--lab", default=str(LAB_PATH))
    args = parser.parse_args()

    vt_symbol = f"{args.symbol}.CZCE"
    signal_name = args.signal or f"{args.symbol}_lgb_test"

    lab = AlphaLab(args.lab)
    signal_df = lab.load_signal(signal_name)
    if signal_df is None:
        raise SystemExit(
            f"找不到 signal={signal_name}\n"
            f"请先运行: python 04_export_signal.py --symbol {args.symbol}"
        )

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=[vt_symbol],
        interval=Interval.MINUTE,
        start=datetime.fromisoformat(args.start),
        end=datetime.fromisoformat(args.end),
        capital=args.capital,
    )
    engine.add_strategy(
        FgSignalStrategy,
        {"threshold": args.threshold, "max_lots": 1.0},
        signal_df,
    )
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / f"alpha_backtest_{args.symbol}.json"
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"回测完成: {vt_symbol}  signal={signal_name}")
    for k in ("total_return", "sharpe_ratio", "max_drawdown", "total_trade_count"):
        if k in stats:
            print(f"  {k}: {stats[k]}")
    print(f"详细统计: {out}")


if __name__ == "__main__":
    main()
