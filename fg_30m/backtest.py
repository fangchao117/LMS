"""
30 分钟回测 —— 保证金模型，T 信号 → T+1 开盘价成交

    python backtest.py
    python backtest.py --symbol FG609 --start 2025-09-01 --end 2026-07-01
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

import config
import data
import signals


def _calc_lots(equity: float, price: float, max_lots: int) -> int:
    if price <= 0:
        return 0
    margin = price * config.CONTRACT_SIZE * config.MARGIN_RATE
    lots = int(equity * config.POSITION_PCT / margin)
    return min(max(lots, 0), max_lots)


def run(
    df: pd.DataFrame,
    sig: pd.Series,
    capital: float | None = None,
    max_lots: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    cap0 = float(capital or config.CAPITAL)
    ml = max_lots if max_lots is not None else config.MAX_LOTS
    comm = config.COMMISSION_YUAN_PER_LOT
    slip = config.SLIPPAGE_TICKS * config.PRICE_TICK
    size = config.CONTRACT_SIZE

    target = sig.shift(1).fillna(0).clip(-ml, ml).astype(int).to_numpy()
    cash = cap0
    pos = 0
    equity_curve = []
    trades = 0

    for idx in range(len(df)):
        row = df.iloc[idx]
        price_open = row["open"]
        price_close = row["close"]
        tgt = int(target[idx]) if idx < len(target) else 0

        if tgt != pos and price_open > 0:
            traded = abs(tgt - pos)
            cash -= traded * comm * 2
            cash -= traded * slip * size
            pos = tgt
            trades += traded

        if pos != 0:
            cash += pos * (price_close - price_open) * size

        mtm = cash + pos * price_close * size * config.MARGIN_RATE if pos else cash
        equity_curve.append({"datetime": row["datetime"], "equity": mtm, "pos": pos})

    eq = pd.DataFrame(equity_curve)
    end_eq = float(eq["equity"].iloc[-1]) if len(eq) else cap0
    ret = (end_eq / cap0 - 1) * 100

    peak = eq["equity"].cummax()
    dd = (eq["equity"] - peak) / peak.replace(0, np.nan)
    max_dd_pct = float(dd.min() * 100) if len(dd) else 0.0

    daily_ret = eq.set_index("datetime")["equity"].resample("D").last().pct_change(fill_method=None).dropna()
    sharpe = 0.0
    if len(daily_ret) > 5 and daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(240))

    stats = {
        "capital": cap0,
        "end_equity": end_eq,
        "total_return_pct": ret,
        "max_ddpercent": max_dd_pct,
        "sharpe_ratio": sharpe,
        "total_trades": trades,
        "bars": len(df),
        "max_lots": ml,
    }
    return stats, eq


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=config.SYMBOL)
    parser.add_argument("--start", default=config.BACKTEST_PERIOD[0])
    parser.add_argument("--end", default=config.BACKTEST_PERIOD[1])
    parser.add_argument("--mode", default=config.SIGNAL_MODE)
    parser.add_argument(
        "--dominant",
        action="store_true",
        default=config.USE_DOMINANT_BACKTEST,
        help="使用主力合约拼接 30m（默认开）",
    )
    parser.add_argument("--no-dominant", action="store_false", dest="dominant")
    args = parser.parse_args()

    config.ensure_dirs()

    sig_kw = dict(
        ema_fast=config.EMA_FAST,
        ema_slow=config.EMA_SLOW,
        donchian=config.DONCHIAN,
        adx_threshold=config.ADX_THRESHOLD,
        trend_ma=config.TREND_MA,
        ma_divergence_threshold=config.MA_DIVERGENCE_THRESHOLD,
    )

    if args.dominant:
        import dominant as dom
        print(f"[fg_30m] 主力拼接 30m {args.start} ~ {args.end} …")
        print(dom.segment_summary().to_string(index=False))
        df = dom.load_dominant_30m()
        df = df[(df["datetime"] >= args.start) & (df["datetime"] <= args.end)]
        tag = f"dominant_{args.mode}"
    else:
        config.SYMBOL = args.symbol
        config.VT_SYMBOL = f"{args.symbol}.{config.EXCHANGE_STR}"
        print(f"[fg_30m] 加载 {args.symbol} 30m {args.start} ~ {args.end} …")
        df = data.load_30m(args.symbol, args.start, args.end, use_cache=True)
        tag = f"{args.symbol}_{args.mode}"

    sig = signals.generate(df, args.mode, **sig_kw)
    stats, eq = run(df, sig)
    eq.to_csv(config.ARTIFACT_PATH / f"equity_{tag}.csv", index=False)
    (config.ARTIFACT_PATH / f"stats_{tag}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[fg_30m] {args.mode}  bars={stats['bars']}  trades={stats['total_trades']}")
    print(f"    收益 {stats['total_return_pct']:.2f}%  回撤 {stats['max_ddpercent']:.2f}%  Sharpe {stats['sharpe_ratio']:.2f}")
    print(f"    期末权益 {stats['end_equity']:,.0f}")
    return stats


if __name__ == "__main__":
    main()
