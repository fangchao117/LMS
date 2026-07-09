"""
玻璃主力 —— 换月日历、未复权拼接、自动识别当期主力

换月规则：交割月(1/5/9)前一月 15 日收盘后切换。
投机客户：旧合约平仓 → 新合约开仓（不平移），回测扣双倍手续费+滑点。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from bridge import load_30m

DELIVERY_MONTHS: tuple[int, ...] = (1, 5, 9)

# 本地 30m 数据覆盖窗口（可按 refresh 更新）
DOMINANT_SEGMENTS: list[tuple[str, str, str]] = [
    ("FG509", "2025-06-20", "2025-08-14"),
    ("FG601", "2025-08-15", "2025-12-14"),
    ("FG605", "2025-12-15", "2026-04-14"),
    ("FG609", "2026-04-15", "2026-08-14"),  # 换月日 2026-08-15 → 下一主力
]


def _year_digit(y: int) -> int:
    return y % 10


def contract_symbol(year: int, month: int) -> str:
    return f"FG{_year_digit(year)}{month:02d}"


def roll_date(year: int, delivery_month: int) -> date:
    if delivery_month == 1:
        return date(year - 1, 12, 15)
    return date(year, delivery_month - 1, 15)


def iter_contracts(start_year: int, end_year: int) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for y in range(start_year, end_year + 1):
        for m in DELIVERY_MONTHS:
            out.append((contract_symbol(y, m), y, m))
    return out


def build_segments(
    start: str,
    end: str,
) -> list[tuple[str, str, str]]:
    sy = pd.Timestamp(start).year - 1
    ey = pd.Timestamp(end).year + 1
    contracts = iter_contracts(sy, ey)
    segs: list[tuple[str, str, str]] = []
    for i, (sym, y, m) in enumerate(contracts):
        r = roll_date(y, m)
        start_d = r + timedelta(days=1)
        if i + 1 < len(contracts):
            end_d = roll_date(contracts[i + 1][1], contracts[i + 1][2]) - timedelta(days=1)
        else:
            end_d = date(y, 12, 31)
        if start_d > end_d:
            continue
        segs.append((sym, start_d.isoformat(), end_d.isoformat()))

    ds, de = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    out = []
    for sym, st, en in segs:
        if pd.Timestamp(en).date() < ds or pd.Timestamp(st).date() > de:
            continue
        out.append((sym, max(st, start), min(en, end)))
    return out


def dominant_period(segments: list[tuple[str, str, str]] | None = None) -> tuple[str, str]:
    segs = segments or DOMINANT_SEGMENTS
    return segs[0][1], segs[-1][2]


def load_dominant_30m(
    segments: list[tuple[str, str, str]] | None = None,
    adjust: bool = False,
) -> pd.DataFrame:
    """adjust=False：未复权真实价格，换月有跳空（实盘一致）"""
    segs = segments or DOMINANT_SEGMENTS
    parts: list[pd.DataFrame] = []
    factor = 1.0

    for sym, start, end in segs:
        try:
            chunk = load_30m(sym, start=start, end=end, use_cache=True)
        except RuntimeError:
            continue
        if chunk.empty:
            continue
        chunk = chunk.sort_values("datetime").drop_duplicates("datetime").copy()
        chunk["symbol"] = sym
        chunk["vt_symbol"] = f"{sym}.CZCE"

        if adjust and parts:
            prev_close = float(parts[-1]["close"].iloc[-1])
            first_open = float(chunk["open"].iloc[0])
            if first_open > 0:
                factor *= prev_close / first_open
            for col in ("open", "high", "low", "close"):
                chunk[col] = chunk[col] * factor

        parts.append(chunk)

    if not parts:
        raise RuntimeError("主力无数据，请先下载 1m 并 refresh_cache")

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return out.reset_index(drop=True)


def current_dominant(as_of: date | datetime | None = None) -> tuple[str, str, str]:
    """返回 (symbol, segment_start, segment_end)"""
    dt = as_of or date.today()
    if isinstance(dt, datetime):
        dt = dt.date()
    for sym, start, end in DOMINANT_SEGMENTS:
        if pd.Timestamp(start).date() <= dt <= pd.Timestamp(end).date():
            return sym, start, end
    # 数据窗口外：沿用最后一段或日历推断
    if DOMINANT_SEGMENTS and dt > pd.Timestamp(DOMINANT_SEGMENTS[-1][2]).date():
        return DOMINANT_SEGMENTS[-1]
    if DOMINANT_SEGMENTS and dt < pd.Timestamp(DOMINANT_SEGMENTS[0][1]).date():
        return DOMINANT_SEGMENTS[0]
    return DOMINANT_SEGMENTS[-1]


def _parse_symbol(sym: str) -> tuple[int, int]:
    """FG609 -> (2026, 9)  仅支持 2020s"""
    digit = int(sym[2])
    month = int(sym[4:6])
    year = 2020 + digit if digit >= 0 else 2010 + digit
    return year, month


def next_contract_symbol(sym: str) -> str | None:
    y, m = _parse_symbol(sym)
    idx = DELIVERY_MONTHS.index(m)
    if idx + 1 < len(DELIVERY_MONTHS):
        return contract_symbol(y, DELIVERY_MONTHS[idx + 1])
    return contract_symbol(y + 1, DELIVERY_MONTHS[0])


def next_roll_info(as_of: date | datetime | None = None) -> dict:
    sym, start, end = current_dominant(as_of)
    end_d = pd.Timestamp(end).date()
    dt = as_of.date() if isinstance(as_of, datetime) else (as_of or date.today())
    days = (end_d - dt).days
    idx = next((i for i, s in enumerate(DOMINANT_SEGMENTS) if s[0] == sym), len(DOMINANT_SEGMENTS) - 1)
    if idx + 1 < len(DOMINANT_SEGMENTS):
        nxt = DOMINANT_SEGMENTS[idx + 1][0]
    else:
        nxt = next_contract_symbol(sym)
    return {
        "current": sym,
        "segment_start": start,
        "segment_end": end,
        "days_to_roll": days,
        "next_symbol": nxt,
        "action_on_roll": "平旧仓 → 新合约按信号开仓",
    }


def roll_bar_indices(df: pd.DataFrame) -> list[int]:
    if "symbol" not in df.columns:
        return []
    sym = df["symbol"].astype(str)
    return sym.ne(sym.shift(1)).to_numpy().nonzero()[0].tolist()
