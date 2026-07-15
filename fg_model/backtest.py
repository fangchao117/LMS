"""ML 信号简易回测（T 信号 → 下一根开盘）。"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

import config


def run_backtest(df: pd.DataFrame, position: pd.Series) -> tuple[dict[str, Any], pd.DataFrame]:
    cap0 = float(config.CAPITAL)
    ml = int(config.MAX_LOTS)
    size = config.CONTRACT_SIZE
    comm = config.COMMISSION_PER_LOT
    slip = config.SLIPPAGE_TICKS * config.PRICE_TICK

    cash = cap0
    pos = 0
    entry = 0.0
    rows = []

    o = df["open"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    sig = position.fillna(0).astype(int).to_numpy()
    dt = df["datetime"].to_numpy()

    for i in range(1, len(df)):
        target = int(np.clip(sig[i - 1], -ml, ml))
        price = o[i]

        if target != pos:
            if pos != 0:
                cash += pos * (price - entry) * size
                cash -= abs(pos) * comm
                cash -= abs(pos) * slip * size
            pos = target
            if pos != 0:
                entry = price
                cash -= abs(pos) * comm
                cash -= abs(pos) * slip * size

        mtm = cash + pos * (c[i] - entry) * size if pos else cash
        rows.append({"datetime": dt[i], "cash": cash, "pos": pos, "equity": mtm})

    eq = pd.DataFrame(rows)
    if eq.empty:
        return {"total_return_pct": 0.0, "max_ddpercent": 0.0, "sharpe_ratio": 0.0, "total_trades": 0}, eq

    eq["peak"] = eq["equity"].cummax()
    eq["dd"] = (eq["equity"] - eq["peak"]) / eq["peak"] * 100
    ret = eq["equity"].pct_change().fillna(0)
    sharpe = float(ret.mean() / (ret.std() + 1e-12) * np.sqrt(252 * 8))

    trades = int((eq["pos"].diff().fillna(0) != 0).sum())
    stats = {
        "total_return_pct": float((eq["equity"].iloc[-1] / cap0 - 1) * 100),
        "max_ddpercent": float(eq["dd"].min()),
        "sharpe_ratio": sharpe,
        "total_trades": trades,
        "final_equity": float(eq["equity"].iloc[-1]),
    }
    return stats, eq
