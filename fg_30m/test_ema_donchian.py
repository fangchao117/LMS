"""EMA + 唐奇安组合对比 —— python test_ema_donchian.py"""
from __future__ import annotations

import numpy as np

import config
import data
import signals
from backtest import run


def breakout_hold(df, dc: int = 20):
    sig = signals.generate(df, "breakout", donchian=dc)
    return sig.replace(0, np.nan).ffill().fillna(0)


CASES = [
    ("daily+30m EMA(基准)", lambda d: signals.generate(d, "daily_filter_ema")),
    ("30m EMA", lambda d: signals.generate(d, "ema_cross", ema_fast=26, ema_slow=46)),
    ("30m EMA+Donchian突破", lambda d: signals.generate(d, "ema_donchian_breakout", donchian=20)),
    ("30m EMA+Donchian同向", lambda d: signals.generate(d, "ema_donchian_both", donchian=20)),
    ("日线EMA+30m Donchian", lambda d: signals.generate(d, "daily_filter_donchian", donchian=20)),
    ("日线Donchian+30m EMA", lambda d: signals.generate(d, "daily_donchian_filter_ema", donchian=20)),
    ("Donchian20持仓", breakout_hold),
]


def main() -> None:
    df_all = data.load_30m("FG609")
    periods = [
        ("FULL", config.BACKTEST_PERIOD),
        ("TUNE", config.TUNE_PERIOD),
        ("TEST", config.TEST_PERIOD),
    ]
    print("FG609  1万1手  EMA26/46  Donchian20")
    print("-" * 72)
    for pname, (s, e) in periods:
        df = df_all[(df_all["datetime"] >= s) & (df_all["datetime"] <= e)].reset_index(drop=True)
        print(f"=== {pname} {s} ~ {e} ===")
        for label, fn in CASES:
            st = run(df, fn(df), max_lots=1)[0]
            print(
                f"  {label:22} ret={st['total_return_pct']:7.1f}%  "
                f"dd={st['max_ddpercent']:6.1f}%  trades={st['total_trades']:4}"
            )
        print()


if __name__ == "__main__":
    main()
