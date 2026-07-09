"""ATR 风控回测 —— T 信号 → T+1 开盘；换月强制平旧开新"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.Series:
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _close_pos(
    pos: int,
    price: float,
    entry_price: float,
    cash: float,
    comm: float,
    slip: float,
    size: float,
    roll_extra: bool = False,
) -> tuple[float, int, int]:
    trades = 0
    if pos == 0:
        return cash, 0, trades
    pnl = pos * (price - entry_price) * size
    cash += pnl
    cash -= abs(pos) * comm
    cash -= abs(pos) * slip * size
    if roll_extra:
        cash -= abs(pos) * comm * config.ROLL_EXTRA_COMM_MULT
    trades += abs(pos)
    return cash, 0, trades


def run(
    df: pd.DataFrame,
    sig: pd.Series,
    capital: float | None = None,
    max_lots: int | None = None,
    atr_stop_mult: float = 3.5,
    trail_atr_mult: float = 2.62,
    max_loss_pct: float = 6.0,
    cooldown_bars: int = 0,
    handle_roll: bool = True,
) -> tuple[dict, pd.DataFrame]:
    cap0 = float(capital or config.CAPITAL)
    ml = max_lots if max_lots is not None else config.MAX_LOTS
    comm = config.COMMISSION_YUAN_PER_LOT
    slip = config.SLIPPAGE_TICKS * config.PRICE_TICK
    size = config.CONTRACT_SIZE

    target = sig.shift(1).fillna(0).clip(-ml, ml).astype(int).to_numpy()
    atr_v = _atr(df["high"], df["low"], df["close"]).to_numpy()
    symbols = df["symbol"].astype(str).to_numpy() if "symbol" in df.columns else None

    cash = cap0
    pos = 0
    entry_price = 0.0
    peak_pnl = 0.0
    cooldown = 0
    stop_hits = 0
    roll_closes = 0
    trades = 0
    rows = []

    for i in range(len(df)):
        row = df.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        tgt = int(target[i]) if i < len(target) else 0
        atr_i = float(atr_v[i]) if i < len(atr_v) and np.isfinite(atr_v[i]) else 0.0

        # 换月：旧合约强制平仓（不平移）
        if handle_roll and symbols is not None and i > 0 and symbols[i] != symbols[i - 1]:
            if pos != 0 and o > 0:
                cash, pos, tc = _close_pos(pos, o, entry_price, cash, comm, slip, size, roll_extra=True)
                trades += tc
                roll_closes += 1
                entry_price = 0.0
                peak_pnl = 0.0
            cooldown = 0

        if cooldown > 0:
            cooldown -= 1
            tgt = 0

        if pos != 0 and atr_i > 0:
            if pos > 0:
                pnl_low = (l - entry_price) * size * pos
                pnl_high = (h - entry_price) * size * pos
            else:
                pnl_low = (entry_price - h) * size * (-pos)
                pnl_high = (entry_price - l) * size * (-pos)
            peak_pnl = max(peak_pnl, pnl_high)
            hard_stop = -atr_stop_mult * atr_i * size * abs(pos)
            trail_stop = peak_pnl - trail_atr_mult * atr_i * size * abs(pos)
            max_loss = -cap0 * max_loss_pct / 100.0
            if pnl_low <= hard_stop or pnl_low <= trail_stop or pnl_low <= max_loss:
                cash, pos, tc = _close_pos(pos, l if pos > 0 else h, entry_price, cash, comm, slip, size)
                trades += tc
                entry_price = 0.0
                peak_pnl = 0.0
                stop_hits += 1
                cooldown = cooldown_bars
                tgt = 0

        if tgt != pos and o > 0:
            if pos != 0:
                cash, pos, tc = _close_pos(pos, o, entry_price, cash, comm, slip, size)
                trades += tc
            traded = abs(tgt - pos)
            if traded > 0:
                cash -= traded * comm
                cash -= traded * slip * size
            if tgt != 0:
                entry_price = o
                peak_pnl = 0.0
            pos = tgt
            trades += traded

        if pos != 0:
            cash += pos * (c - o) * size

        sym = symbols[i] if symbols is not None else ""
        mtm = cash + pos * c * size * config.MARGIN_RATE if pos else cash
        rows.append({
            "datetime": row["datetime"],
            "equity": mtm,
            "pos": pos,
            "symbol": sym,
        })

    eq = pd.DataFrame(rows)
    end_eq = float(eq["equity"].iloc[-1]) if len(eq) else cap0
    ret = (end_eq / cap0 - 1) * 100
    peak = eq["equity"].cummax()
    dd = (eq["equity"] - peak) / peak.replace(0, np.nan)
    max_dd = float(dd.min() * 100) if len(dd) else 0.0
    daily_ret = eq.set_index("datetime")["equity"].resample("D").last().pct_change(fill_method=None).dropna()
    sharpe = 0.0
    if len(daily_ret) > 5 and daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(240))

    stats = {
        "capital": cap0,
        "end_equity": end_eq,
        "total_return_pct": ret,
        "max_ddpercent": max_dd,
        "sharpe_ratio": sharpe,
        "total_trades": trades,
        "stop_hits": stop_hits,
        "roll_closes": roll_closes,
        "bars": len(df),
        "max_lots": ml,
    }
    return stats, eq
