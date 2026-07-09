"""
玻璃主力 30m 拼接 —— 按换月窗口串联具体月份合约

换月规则（与 vnpy_sim/dominant_contract 一致）：交割月前一月 15 日。
价格做向后比例复权，避免换月跳空污染 PnL。
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

import config
import data

# 2025-06 起本地有 1m 数据的窗口（估）
DOMINANT_SEGMENTS: list[tuple[str, str, str]] = [
    ("FG509", "2025-06-20", "2025-08-14"),
    ("FG601", "2025-08-15", "2025-12-14"),
    ("FG605", "2025-12-15", "2026-04-14"),
    ("FG609", "2026-04-15", "2026-07-08"),
]


def dominant_period() -> tuple[str, str]:
    return DOMINANT_SEGMENTS[0][1], DOMINANT_SEGMENTS[-1][2]


def load_dominant_30m(
    segments: list[tuple[str, str, str]] | None = None,
    adjust: bool = True,
) -> pd.DataFrame:
    segs = segments or DOMINANT_SEGMENTS
    parts: list[pd.DataFrame] = []
    factor = 1.0

    for sym, start, end in segs:
        try:
            chunk = data.load_30m(sym, start=start, end=end, use_cache=True)
        except RuntimeError:
            continue
        if chunk.empty:
            continue
        chunk = chunk.sort_values("datetime").drop_duplicates("datetime").copy()
        chunk["symbol"] = sym

        if adjust and parts:
            prev_close = float(parts[-1]["close"].iloc[-1])
            first_open = float(chunk["open"].iloc[0])
            if first_open > 0:
                factor *= prev_close / first_open
            for col in ("open", "high", "low", "close"):
                chunk[col] = chunk[col] * factor

        parts.append(chunk)

    if not parts:
        raise RuntimeError("主力拼接无数据，请先 refresh_cache / EDB 下载")

    out = pd.concat(parts, ignore_index=True)
    out["vt_symbol"] = "FG_dominant.CZCE"
    out = out.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return out.reset_index(drop=True)


def segment_summary() -> pd.DataFrame:
    rows = []
    for sym, start, end in DOMINANT_SEGMENTS:
        try:
            df = data.load_30m(sym, start=start, end=end, use_cache=True)
            n = len(df)
        except RuntimeError:
            n = 0
        rows.append({"symbol": sym, "start": start, "end": end, "bars_30m": n})
    return pd.DataFrame(rows)
